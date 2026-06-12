"""
core/detector.py – YOLOv8 person detection.
Tuned for small/distant students in a large classroom.
"""

from ultralytics import YOLO
import numpy as np
import config as C


class PersonDetector:
    def __init__(self):
        print(f"[Detector] Loading {C.YOLO_MODEL} ...")
        self.model = YOLO(C.YOLO_MODEL)
        self.model.to("cuda")
        print("[Detector] Ready.")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on one frame.
        Returns list of dicts: {bbox:(x1,y1,x2,y2), conf:float}

        Tuning for large classrooms:
        - agnostic_nms=True  → avoids duplicate boxes on same person
        - imgsz kept at network default (640) for speed
        - MIN_BOX_AREA filters out background clutter
        """
        results = self.model(
            frame,
            conf=C.CONF_THRESH,
            iou=C.NMS_IOU,
            classes=[0],          # person only
            agnostic_nms=True,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area < C.MIN_BOX_AREA:
                continue
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(box.conf[0]),
            })
        return detections
