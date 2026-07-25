from datetime import datetime
import cv2
import time
from facetracking import HeadPoseEstimator, draw_overlay
from events import EventDetector, EventConfig
from vad import VoiceActivityDectector
import threading

vad = VoiceActivityDectector()
detector = EventDetector()
head = HeadPoseEstimator(640, 480)

state = {"is_speaking": False, "yaw": 0, "pitch": 0, "roll": 0}
active_flags = {}

def audio_worker(vad, state, running_flag):
    while running_flag["active"]:
        state["is_speaking"] = vad.read_chunk()

def main():
    running_flag = {"active": True}
    t = threading.Thread(target=audio_worker, args=(vad, state, running_flag), daemon=True)
    t.start()
    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            print("Error: Could not access webcam")
            exit()
        print("Press q to exit the video stream")

        while True:
            ret, frame = cap.read()
            current_timestamp_ms = int(time.time() * 1000)

            if not ret:
                print("Error: Could not grab frames")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            pose = head.process_frame(frame, current_timestamp_ms)
            state["yaw"] = pose.yaw
            state["pitch"] = pose.pitch
            state["roll"] = pose.roll

            # Unified event detection (head pose + gaze + voice activity)
            events = detector.update(pose=pose, state=state, ts=time.time())
            for ev in events:
                active_flags[ev["type"]] = time.time() + 3.0
                print(f"🚨 [Suspicious Event]: {ev}")

            draw_overlay(frame, pose, active_flags, detector.cfg)
            cv2.imshow("Proctor-ai", frame)

    finally:
        running_flag["active"] = False
        t.join(timeout=1.0)
        
        cap.release()
        cv2.destroyAllWindows()
        vad.close()
        head.close()


if __name__ == "__main__": 
    main()

