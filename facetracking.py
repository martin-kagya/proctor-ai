"""
Head-pose + gaze estimation for a proctoring vision pipeline.

Uses the modern MediaPipe Tasks FaceLandmarker API (replaces the deprecated
solutions.face_mesh).  Key improvements over the old approach:

  - output_facial_transformation_matrixes=True gives a 4×4 rigid-body matrix
    per face directly -- no manual solvePnP, no generic 3-D face model needed.
  - FaceLandmarker is actively maintained; solutions.face_mesh is frozen/deprecated.
  - VIDEO running mode is synchronous and works naturally in a cv2 capture loop.

Model file (~1 MB) is auto-downloaded on first run from Google's CDN.

Install:
    pip install mediapipe opencv-python numpy
"""

from __future__ import annotations

import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from events import EventConfig, EventDetector


# ---------------------------------------------------------------------------
# Model asset -- auto-downloaded if absent
# ---------------------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path("face_landmarker.task")


def ensure_model() -> Path:
    """Download the FaceLandmarker .task file if it is not already present."""
    if not MODEL_PATH.exists():
        print(f"Downloading {MODEL_PATH.name} from Google CDN ...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.", flush=True)
    return MODEL_PATH


# ---------------------------------------------------------------------------
# Iris / eye landmark indices (478-point FaceLandmarker model)
# Indices 468-477 are the iris points added on top of the base 468.
# ---------------------------------------------------------------------------

RIGHT_IRIS        = [468, 469, 470, 471, 472]
LEFT_IRIS         = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS  = (33, 133)   # outer, inner
RIGHT_EYE_CORNERS = (362, 263)  # inner, outer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """3×3 rotation matrix → (yaw, pitch, roll) in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2( R[1, 0], R[0, 0])
        roll  = math.atan2( R[2, 1], R[2, 2])
    else:                           # gimbal-lock fallback
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
        roll  = math.atan2(-R[1, 2], R[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class EMA:
    """Exponential moving average -- smooths per-frame webcam noise."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._v: Optional[float] = None

    def update(self, x: float) -> float:
        self._v = x if self._v is None else self.alpha * x + (1 - self.alpha) * self._v
        return self._v


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------

@dataclass
class PoseResult:
    yaw:        float
    pitch:      float
    roll:       float
    gaze_x:     float   # normalized horizontal iris offset  ≈ -1 … +1
    gaze_y:     float   # normalized vertical   iris offset  ≈ -1 … +1
    face_found: bool = True


# ---------------------------------------------------------------------------
# Head-pose + gaze estimator
# ---------------------------------------------------------------------------

class HeadPoseEstimator:
    """
    Wraps MediaPipe Tasks FaceLandmarker in VIDEO mode to produce smoothed
    yaw / pitch / roll and a normalized gaze offset per frame.

    VIDEO mode is fully synchronous -- call detect_for_video() with a
    monotonically increasing millisecond timestamp and get results back
    immediately, no callbacks needed.
    """

    def __init__(self, frame_width: int, frame_height: int,
                 smoothing_alpha: float = 0.3):
        self.w = frame_width
        self.h = frame_height

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(ensure_model())
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            # The 4×4 facial transformation matrix replaces the manual
            # solvePnP + generic-3D-model approach used in the old code.
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        self._yaw   = EMA(smoothing_alpha)
        self._pitch = EMA(smoothing_alpha)
        self._roll  = EMA(smoothing_alpha)
        self._gx    = EMA(smoothing_alpha)
        self._gy    = EMA(smoothing_alpha)

    def process_frame(self, frame_bgr: np.ndarray,
                      timestamp_ms: int) -> PoseResult:
        """
        Run detection on one BGR frame.  timestamp_ms must increase
        monotonically (use int(time.time() * 1000)).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return PoseResult(0.0, 0.0, 0.0, 0.0, 0.0, face_found=False)

        landmarks = result.face_landmarks[0]  # NormalizedLandmark list

        # Head pose -- directly from the 4×4 rigid-body transform matrix.
        # Top-left 3×3 sub-matrix is the rotation component.
        mat = np.array(result.facial_transformation_matrixes[0])
        yaw, pitch, roll = rotation_matrix_to_euler(mat[:3, :3])

        gaze_x, gaze_y = self._estimate_gaze(landmarks)

        return PoseResult(
            yaw    = self._yaw.update(yaw),
            pitch  = self._pitch.update(pitch),
            roll   = self._roll.update(roll),
            gaze_x = self._gx.update(gaze_x),
            gaze_y = self._gy.update(gaze_y),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _px(self, lm) -> tuple[float, float]:
        """Normalized landmark → pixel coordinates."""
        return lm.x * self.w, lm.y * self.h

    def _estimate_gaze(self, landmarks) -> tuple[float, float]:
        """
        Rough gaze offset: iris center relative to the eye's corner span,
        normalized so 0 = straight ahead.  Averaged across both eyes.
        Intentionally simple -- good enough to catch eyes-moved-without-
        head-moving; not a precision gaze tracker.
        """
        def offset(iris_ids, corner_ids):
            iris_pts = np.array([self._px(landmarks[i]) for i in iris_ids])
            center   = iris_pts.mean(axis=0)
            c1 = np.array(self._px(landmarks[corner_ids[0]]))
            c2 = np.array(self._px(landmarks[corner_ids[1]]))
            eye_width = np.linalg.norm(c2 - c1) + 1e-6
            return (center - (c1 + c2) / 2) / eye_width

        avg = (offset(LEFT_IRIS, LEFT_EYE_CORNERS) +
               offset(RIGHT_IRIS, RIGHT_EYE_CORNERS)) / 2
        # 4× scale is an empirical fit; recalibrate against your own footage.
        return (float(np.clip(avg[0] * 4, -1, 1)),
                float(np.clip(avg[1] * 4, -1, 1)))

    def close(self):
        self._landmarker.close()


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

# How long (seconds) a flag badge stays visible after firing.
FLAG_TTL = 3.0

_FLAG_LABELS = {
    "sustained_look_away":        "Sustained look-away",
    "frequent_look_away":         "Frequent look-aways",
    "gaze_without_head_movement": "Gaze shift (head still)",
    "voice_detected":             "Voice detected",
    "sustained_speaking":         "Sustained speaking",
}
# BGR
_FLAG_COLORS = {
    "sustained_look_away":        (0,  60, 235),
    "frequent_look_away":         (0,  20, 200),
    "gaze_without_head_movement": (0, 130, 235),
    "voice_detected":             (0, 165, 255),
    "sustained_speaking":         (0, 100, 255),
}



def draw_overlay(frame: np.ndarray, pose: PoseResult,
                 active_flags: dict, cfg: EventConfig) -> None:
    """
    Renders a semi-transparent side panel with live pose meters, gaze meters,
    face status, active flag list, and a bottom alert banner when flags fire.
    """
    fh, fw = frame.shape[:2]
    PANEL_W = 310
    LINE_H  = 26

    # ── Semi-transparent dark side panel ──────────────────────────────────
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (PANEL_W, fh), (12, 12, 12), -1)
    cv2.addWeighted(panel, 0.62, frame, 0.38, 0, frame)
    # Thin right border
    cv2.line(frame, (PANEL_W, 0), (PANEL_W, fh), (60, 60, 60), 1)

    y = [28]   # mutable so closures can advance it

    def put(msg: str, color=(210, 210, 210), scale=0.50, bold=False):
        cv2.putText(frame, msg, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2 if bold else 1, cv2.LINE_AA)
        y[0] += LINE_H

    def divider():
        y[0] += 4
        cv2.line(frame, (8, y[0]), (PANEL_W - 8, y[0]), (60, 60, 60), 1)
        y[0] += 12

    def meter(label: str, value: float, maxv: float,
              threshold: float, warn_color=(0, 55, 230)):
        """Labeled bar with a threshold tick-mark."""
        ratio   = min(abs(value) / maxv, 1.0)
        BAR_X   = 118
        BAR_W   = PANEL_W - BAR_X - 10
        filled  = int(ratio * BAR_W)
        thr_x   = BAR_X + int((threshold / maxv) * BAR_W)
        over    = abs(value) > threshold
        bar_clr = warn_color if over else (40, 170, 40)
        lbl_clr = warn_color if over else (170, 170, 170)
        # trough
        cv2.rectangle(frame, (BAR_X, y[0] - 13), (BAR_X + BAR_W, y[0] + 2),
                      (50, 50, 50), -1)
        # fill
        if filled > 0:
            cv2.rectangle(frame, (BAR_X, y[0] - 13),
                          (BAR_X + filled, y[0] + 2), bar_clr, -1)
        # threshold tick
        cv2.line(frame, (thr_x, y[0] - 15), (thr_x, y[0] + 4), (170, 170, 0), 1)
        # label
        cv2.putText(frame, f"{label}: {value:+.1f}", (8, y[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, lbl_clr, 1, cv2.LINE_AA)
        y[0] += LINE_H

    # ── Header ────────────────────────────────────────────────────────────
    put("PROCTOR MONITOR", (110, 195, 255), 0.60, bold=True)
    divider()

    # ── Face status ───────────────────────────────────────────────────────
    if pose.face_found:
        put("[OK]  Face detected", (50, 210, 50), 0.48)
    else:
        put("[!!]  No face detected", (0, 55, 230), 0.48)
    divider()

    # ── Head pose meters ──────────────────────────────────────────────────
    put("HEAD POSE", (120, 120, 120), 0.38)
    meter("Yaw  ", pose.yaw,   90.0, cfg.yaw_threshold_deg)
    meter("Pitch", pose.pitch, 60.0, cfg.pitch_threshold_deg)
    meter("Roll ", pose.roll,  45.0, 30.0)
    divider()

    # ── Gaze meters ───────────────────────────────────────────────────────
    put("GAZE", (120, 120, 120), 0.38)
    meter("Gaze X", pose.gaze_x, 1.0, cfg.gaze_only_threshold, (0, 130, 230))
    meter("Gaze Y", pose.gaze_y, 1.0, cfg.gaze_only_threshold, (0, 130, 230))
    divider()

    # ── Active flags ──────────────────────────────────────────────────────
    put("ACTIVE FLAGS", (120, 120, 120), 0.38)
    now  = time.time()
    live = {k: exp for k, exp in active_flags.items() if exp > now}

    if not live:
        put("  -- None --", (70, 70, 70), 0.43)
    else:
        for key, expiry in live.items():
            # Fade colour as TTL runs down (stays full-bright for first 80 %)
            fade  = min((expiry - now) / FLAG_TTL, 1.0)
            base  = _FLAG_COLORS.get(key, (0, 80, 220))
            color = tuple(int(c * (0.35 + 0.65 * fade)) for c in base)
            label = _FLAG_LABELS.get(key, key)
            put(f"  [!] {label}", color, 0.45)

    # ── Bottom alert banner (only when flags are live) ────────────────────
    if live:
        BANNER_H = 46
        banner = frame.copy()
        cv2.rectangle(banner, (0, fh - BANNER_H), (fw, fh), (0, 0, 175), -1)
        cv2.addWeighted(banner, 0.78, frame, 0.22, 0, frame)
        cv2.putText(
            frame,
            "[!] SUSPICIOUS BEHAVIOUR DETECTED",
            (PANEL_W + 12, fh - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    estimator    = HeadPoseEstimator(w, h)
    detector     = EventDetector()
    active_flags: dict[str, float] = {}   # event_type -> expiry timestamp

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts_ms  = int(time.time() * 1000)
            pose   = estimator.process_frame(frame, ts_ms)
            events = detector.update(pose, ts_ms / 1000.0)

            for ev in events:
                # Stamp the flag; it stays on-screen for FLAG_TTL seconds.
                active_flags[ev["type"]] = time.time() + FLAG_TTL
                print(ev)   # -> feed into your risk-scoring pipeline

            draw_overlay(frame, pose, active_flags, detector.cfg)

            cv2.imshow("proctor -- head pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        estimator.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()