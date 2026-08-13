
from mediapipe.tasks.python.core import base_options
import urllib.request
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from pathlib import Path
import cv2
from ultralytics import YOLO
import urllib

model = YOLO("yolo26n.pt")
class objectDetection:

    def __init__(self):
        pass
    def predict(self, frame):
        result = model.predict(source=frame, conf=0.2, classes=[67], verbose=False)
        return len(result[0].boxes) > 0
        




