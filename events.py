"""
Modular Event Detection System for Proctor AI.

Provides an extensible EventDetector pipeline that combines multiple sub-detectors
(head pose, gaze, voice activity, and future plugins like object detection or multi-face)
into a unified event stream.
"""


from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from facetracking import PoseResult


@dataclass
class EventConfig:
    # Head Pose & Gaze Thresholds
    yaw_threshold_deg: float = 25.0
    pitch_threshold_deg: float = 20.0
    sustained_seconds: float = 2.5
    frequency_window_seconds: float = 300.0
    frequency_threshold_count: int = 6
    gaze_only_threshold: float = 0.5
    gaze_head_still_yaw_deg: float = 8.0
    isCellphone_usage_seconds: float = 3.0

    # Voice Activity Thresholds
    speaking_sustained_seconds: float = 2.0



# ---------------------------------------------------------------------------
# Event metadata & rendering constants (labels & colors)
# ---------------------------------------------------------------------------
FLAG_LABELS: Dict[str, str] = {
    "sustained_look_away":        "Sustained look-away",
    "frequent_look_away":         "Frequent look-aways",
    "gaze_without_head_movement": "Gaze shift (head still)",
    "voice_detected":             "Voice detected",
    "sustained_speaking":         "Sustained speaking",
    "cellphone_detected":         "Cellphone detected",
    "sustained_cellphone_usage":  "Sustained cellphone usage",
}

# Colors in BGR format for OpenCV overlay rendering
FLAG_COLORS: Dict[str, tuple[int, int, int]] = {
    "sustained_look_away":        (0,  60, 235),
    "frequent_look_away":         (0,  20, 200),
    "gaze_without_head_movement": (0, 130, 235),
    "voice_detected":             (0, 165, 255),
    "sustained_speaking":         (0, 100, 255),
    "cellphone_detected":         (0,   0, 255),
    "sustained_cellphone_usage":  (0,   0, 200),
}


class BaseSubDetector:
    """Base interface for modular proctoring sub-detectors."""

    def update(
        self, pose: Optional[PoseResult], state: Optional[Dict[str, Any]], ts: float
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class HeadPoseSubDetector(BaseSubDetector):
    """Detects suspicious head pose and gaze movements."""

    def __init__(self, config: EventConfig):
        self.cfg = config
        self._away_start: Optional[float] = None
        self._recent: deque[float] = deque()

    def update(
        self, pose: Optional[PoseResult], state: Optional[Dict[str, Any]], ts: float
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        if pose is None or not pose.face_found:
            self._away_start = None
            return events

        turned = (
            abs(pose.yaw) > self.cfg.yaw_threshold_deg
            or abs(pose.pitch) > self.cfg.pitch_threshold_deg
        )

        # --- Sustained look-away ---
        if turned:
            if self._away_start is None:
                self._away_start = ts
            elif ts - self._away_start >= self.cfg.sustained_seconds:
                events.append(
                    {
                        "type": "sustained_look_away",
                        "ts": ts,
                        "yaw": pose.yaw,
                        "pitch": pose.pitch,
                    }
                )
                self._recent.append(ts)
                self._away_start = None
        else:
            self._away_start = None

        # --- Frequency flag (rolling window) ---
        cutoff = ts - self.cfg.frequency_window_seconds
        while self._recent and self._recent[0] < cutoff:
            self._recent.popleft()
        if len(self._recent) >= self.cfg.frequency_threshold_count:
            events.append(
                {
                    "type": "frequent_look_away",
                    "ts": ts,
                    "count": len(self._recent),
                }
            )
            self._recent.clear()

        # --- Gaze without head movement ---
        head_still = abs(pose.yaw) < self.cfg.gaze_head_still_yaw_deg
        gaze_off = (
            abs(pose.gaze_x) > self.cfg.gaze_only_threshold
            or abs(pose.gaze_y) > self.cfg.gaze_only_threshold
        )
        if head_still and gaze_off:
            events.append(
                {
                    "type": "gaze_without_head_movement",
                    "ts": ts,
                    "gaze_x": pose.gaze_x,
                    "gaze_y": pose.gaze_y,
                }
            )

        return events


class VoiceActivitySubDetector(BaseSubDetector):
    """Detects voice activity and sustained speaking."""

    def __init__(self, config: EventConfig):
        self.cfg = config
        self._speaking_start: Optional[float] = None

    def update(
        self, pose: Optional[PoseResult], state: Optional[Dict[str, Any]], ts: float
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        if state is None:
            return events

        is_speaking = state.get("is_speaking", False)

        if is_speaking:
            events.append(
                {
                    "type": "voice_detected",
                    "ts": ts,
                }
            )

            if self._speaking_start is None:
                self._speaking_start = ts
            elif ts - self._speaking_start >= self.cfg.speaking_sustained_seconds:
                events.append(
                    {
                        "type": "sustained_speaking",
                        "ts": ts,
                        "duration": ts - self._speaking_start,
                    }
                )
                self._speaking_start = None
        else:
            self._speaking_start = None

        return events


class CellphoneSubDetector(BaseSubDetector):
    """Detects cellphone presence and sustained usage."""

    def __init__(self, config: EventConfig):
        self.cfg = config
        self._phone_start: Optional[float] = None

    def update(
        self, pose: Optional[PoseResult], state: Optional[Dict[str, Any]], ts: float
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        if state is None:
            return events

        is_phone_detected = state.get("isCellphone_detected", False)

        if is_phone_detected:
            events.append({"type": "cellphone_detected", "ts": ts})

            if self._phone_start is None:
                self._phone_start = ts
            elif ts - self._phone_start >= self.cfg.isCellphone_usage_seconds:
                events.append(
                    {
                        "type": "sustained_cellphone_usage",
                        "ts": ts,
                        "duration": ts - self._phone_start,
                    }
                )
                self._phone_start = None
        else:
            self._phone_start = None

        return events


class EventDetector:
    """
    Central Event Detector Pipeline.

    Manages multiple modular sub-detectors and yields aggregated proctoring events.
    """

    def __init__(self, config: Optional[EventConfig] = None):
        self.cfg = config or EventConfig()
        self.sub_detectors: List[BaseSubDetector] = [
            HeadPoseSubDetector(self.cfg),
            VoiceActivitySubDetector(self.cfg),
            CellphoneSubDetector(self.cfg),
        ]

    def register_sub_detector(self, detector: BaseSubDetector) -> None:
        """Register a new sub-detector (e.g. object detection, multi-face)."""
        self.sub_detectors.append(detector)

    def update(
        self,
        pose: Optional[PoseResult] = None,
        state: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run all registered sub-detectors and collect events."""
        if ts is None:
            import time

            ts = time.time()

        all_events: List[Dict[str, Any]] = []
        for detector in self.sub_detectors:
            all_events.extend(detector.update(pose, state, ts))

        return all_events
