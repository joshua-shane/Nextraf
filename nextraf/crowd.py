"""Module A - Crowd Intelligence.

Counts people per frame inside an optional ROI and compares the count with a
configured capacity. Density (people per m^2) is reported ONLY when the real
ground area of the ROI is configured - pixel area is never treated as m^2.

Note: a per-frame count is people visible in that frame, not unique individuals.
"""
import json
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import PERSON


@dataclass
class CrowdConfig:
    roi: list = None            # [[x, y], ...] polygon in pixels; None = whole frame
    capacity: int = None        # max people allowed in the ROI; None = no alerting
    warn_ratio: float = 0.8     # NEAR_CAPACITY at this share of capacity
    area_m2: float = None       # measured ground area of the ROI; None = no density
    site: str = "unnamed-site"
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        if not path:
            return cls()
        with open(path) as f:
            data = json.load(f)
        known = {k: data.pop(k) for k in list(data) if k in cls.__dataclass_fields__}
        return cls(**known, extra=data)


def in_roi(point, roi):
    """True if point is inside (or on the edge of) the ROI polygon."""
    if not roi:
        return True
    contour = np.asarray(roi, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def analyse(detections, cfg):
    """ROI inclusion rule: a person counts if their FOOT point (bottom-centre of
    the box) is inside the ROI - that is where they stand on the ground."""
    people = [d for d in detections if d.cls_id == PERSON]
    inside = [d for d in people if in_roi(d.foot, cfg.roi)]
    n = len(inside)

    if cfg.capacity is None:
        status, occupancy = "NO_CAPACITY_SET", None
    else:
        occupancy = round(100.0 * n / cfg.capacity, 1) if cfg.capacity else None
        if n > cfg.capacity:
            status = "OVERCROWDED"
        elif n >= cfg.warn_ratio * cfg.capacity:
            status = "NEAR_CAPACITY"
        else:
            status = "OK"

    density = round(n / cfg.area_m2, 3) if cfg.area_m2 else None
    return {
        "people_detected": len(people),
        "people_in_roi": n,
        "capacity": cfg.capacity,
        "occupancy_pct": occupancy,
        "density_per_m2": density,
        "status": status,
        "_inside": inside,  # for drawing only; dropped before writing records
    }


STATUS_COLOURS = {"OK": (0, 170, 0), "NEAR_CAPACITY": (0, 165, 255),
                  "OVERCROWDED": (0, 0, 255), "NO_CAPACITY_SET": (200, 200, 200)}


def draw(frame, detections, result, cfg):
    img = frame.copy()
    if cfg.roi:
        cv2.polylines(img, [np.asarray(cfg.roi, np.int32)], True, (255, 255, 0), 2)
    inside_ids = {id(d) for d in result["_inside"]}
    for d in detections:
        if d.cls_id != PERSON:
            continue
        x1, y1, x2, y2 = map(int, d.box)
        col = (0, 200, 0) if id(d) in inside_ids else (160, 160, 160)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{d.conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.circle(img, tuple(map(int, d.foot)), 3, col, -1)
    text = f"People in ROI: {result['people_in_roi']}"
    if result["capacity"] is not None:
        text += f" / {result['capacity']}"
    if result["density_per_m2"] is not None:
        text += f"  ({result['density_per_m2']}/m2)"
    text += f"  [{result['status']}]"
    col = STATUS_COLOURS.get(result["status"], (255, 255, 255))
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return img
