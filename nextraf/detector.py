"""Shared YOLO26n detector used by both NEXTRAF modules.

Loads the model once and returns plain Detection objects, so the crowd and
vehicle modules never touch Ultralytics directly.
"""
import time
from dataclasses import dataclass

COCO_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON = 0
VEHICLES = (2, 3, 5, 7)


@dataclass
class Detection:
    cls_id: int
    label: str
    conf: float
    box: tuple  # (x1, y1, x2, y2) in pixels of the original frame

    @property
    def foot(self):
        """Bottom-centre of the box: where a person/vehicle touches the ground."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)


class Detector:
    def __init__(self, model="yolo26n_executorch_model", conf=0.4, imgsz=640,
                 classes=tuple(COCO_NAMES)):
        from ultralytics import YOLO  # imported here so geometry code can be tested without it
        self.model = YOLO(model, task="detect")
        self.model_path = model
        self.conf, self.imgsz, self.classes = conf, imgsz, list(classes)

    def detect(self, frame):
        """Return (detections, latency_ms) for one BGR frame."""
        t0 = time.perf_counter()
        r = self.model.predict(frame, classes=self.classes, conf=self.conf,
                               imgsz=self.imgsz, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000.0
        dets = [Detection(int(c), COCO_NAMES.get(int(c), str(int(c))), float(s),
                          tuple(float(v) for v in b))
                for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                   r.boxes.conf.tolist())]
        return dets, ms
