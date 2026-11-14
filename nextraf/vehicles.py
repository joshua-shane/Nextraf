"""Module B - Vehicle and Parking Intelligence.

Per frame: count vehicles by class, keep vehicle IDs across frames with a small
IoU tracker, measure how long each vehicle has been stationary, and check how
much of its ground footprint is on the carriageway side of a road-edge line.

A POTENTIAL_VIOLATION is raised only when BOTH hold:
  1. stationary for >= park_seconds, and
  2. >= encroach share of its footprint is on the carriageway side.
It is labelled "potential" because a line crossing alone does not prove illegal
parking (signage, loading zones, breakdowns, camera perspective).
"""
import json
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import COCO_NAMES, VEHICLES


# ---------------- configuration ----------------
@dataclass
class RoadConfig:
    p1: tuple = None            # road-edge line point 1 (pixels)
    p2: tuple = None            # road-edge line point 2
    allowed_point: tuple = None # any point on the side where parking is fine
    frame_size: list = None     # [w, h] of the frame the line was drawn on
    park_seconds: float = 5.0
    encroach: float = 0.15      # share of footprint on carriageway to count
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

    @property
    def has_line(self):
        return self.p1 is not None and self.p2 is not None and self.allowed_point is not None

    def scaled(self, w, h):
        """Line in the coordinates of a (w, h) frame."""
        if not self.has_line:
            return None
        fw, fh = self.frame_size or [w, h]
        sx, sy = w / fw, h / fh
        s = lambda p: (p[0] * sx, p[1] * sy)
        return s(self.p1), s(self.p2), s(self.allowed_point)


# ---------------- geometry ----------------
def side_of_line(p1, p2, pts):
    """Signed cross product: >0 one side of line p1->p2, <0 the other side."""
    (x1, y1), (x2, y2) = p1, p2
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return (x2 - x1) * (pts[:, 1] - y1) - (y2 - y1) * (pts[:, 0] - x1)


def footprint(box, n=9, strip=0.15):
    """Points along the bottom strip of the box - where the tyres meet the road.
    Using this (not the box centre) avoids flagging a car whose roof merely
    appears over the line because of camera perspective."""
    x1, y1, x2, y2 = box
    xs = np.linspace(x1, x2, n)
    return np.array([(x, y) for y in (y2, y2 - strip * (y2 - y1)) for x in xs])


def fraction_on_carriageway(box, line):
    p1, p2, allowed = line
    allowed_sign = np.sign(side_of_line(p1, p2, [allowed])[0])
    s = np.sign(side_of_line(p1, p2, footprint(box)))
    return float(np.mean(s == -allowed_sign))


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------- tracking ----------------
@dataclass
class Track:
    id: int
    label: str
    box: tuple
    anchor: tuple        # box where the vehicle last came to rest
    still_since: float
    last_seen: float
    alerted: bool = False


class IoUTracker:
    """Lightweight tracker: a detection keeps the ID of the previous-frame box it
    overlaps most (IoU >= match_iou). Stationary time resets when the box drifts
    away from its resting 'anchor' (IoU < still_iou). No extra dependencies."""

    def __init__(self, match_iou=0.3, still_iou=0.7, max_age=2.0):
        self.match_iou, self.still_iou, self.max_age = match_iou, still_iou, max_age
        self.tracks, self._next_id = [], 1

    def update(self, dets, t):
        pairs = sorted(((iou(d.box, tr.box), di, ti) for di, d in enumerate(dets)
                        for ti, tr in enumerate(self.tracks)), reverse=True)
        used_d, used_t, assigned = set(), set(), {}
        for v, di, ti in pairs:
            if v < self.match_iou:
                break
            if di in used_d or ti in used_t:
                continue
            used_d.add(di); used_t.add(ti); assigned[di] = self.tracks[ti]
        out = []
        for di, d in enumerate(dets):
            tr = assigned.get(di)
            if tr is None:
                tr = Track(self._next_id, d.label, d.box, d.box, t, t)
                self._next_id += 1
                self.tracks.append(tr)
            else:
                tr.box, tr.label, tr.last_seen = d.box, d.label, t
                if iou(d.box, tr.anchor) < self.still_iou:
                    tr.anchor, tr.still_since = d.box, t
            out.append(tr)
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.max_age]
        return out


# ---------------- per-frame analysis ----------------
def analyse(detections, tracker, cfg, frame_w, frame_h, t):
    """t = video time in seconds, or None for a single image (no duration known)."""
    vdets = [d for d in detections if d.cls_id in VEHICLES]
    counts = {COCO_NAMES[c]: 0 for c in VEHICLES}
    for d in vdets:
        counts[d.label] += 1
    line = cfg.scaled(frame_w, frame_h)
    tracks = tracker.update(vdets, t) if t is not None else [None] * len(vdets)

    vehicles, new_alerts = [], []
    for d, tr in zip(vdets, tracks):
        frac = fraction_on_carriageway(d.box, line) if line else None
        still = None if tr is None else round(t - tr.still_since, 1)
        if tr is None:
            status = "NO_DURATION"           # single image: cannot judge parking
        elif still < cfg.park_seconds:
            status = "MOVING_OR_BRIEF_STOP"
        elif frac is None:
            status = "STATIONARY"            # no road line configured
        elif frac >= cfg.encroach:
            status = "POTENTIAL_VIOLATION"
        else:
            status = "STATIONARY_OK"
        v = {"id": None if tr is None else tr.id, "label": d.label, "conf": round(d.conf, 2),
             "box": [round(x) for x in d.box], "stationary_s": still,
             "on_carriageway_frac": None if frac is None else round(frac, 2), "status": status}
        vehicles.append(v)
        if status == "POTENTIAL_VIOLATION" and not tr.alerted:
            tr.alerted = True
            new_alerts.append(v)
    summary = {
        "vehicle_counts": counts,
        "vehicles_total": len(vdets),
        "stationary": sum(v["status"] in ("STATIONARY", "STATIONARY_OK", "POTENTIAL_VIOLATION") for v in vehicles),
        "potential_violations": sum(v["status"] == "POTENTIAL_VIOLATION" for v in vehicles),
    }
    return summary, vehicles, new_alerts


STATUS_COLOURS = {"POTENTIAL_VIOLATION": (0, 0, 255), "STATIONARY_OK": (0, 200, 0),
                  "STATIONARY": (0, 200, 200), "MOVING_OR_BRIEF_STOP": (170, 170, 170),
                  "NO_DURATION": (255, 160, 0)}


def draw(frame, vehicles, summary, cfg):
    img = frame.copy()
    line = cfg.scaled(img.shape[1], img.shape[0])
    if line:
        p1, p2, a = (tuple(map(int, p)) for p in line)
        cv2.line(img, p1, p2, (0, 255, 255), 2)
        cv2.putText(img, "parking OK side", a, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for v in vehicles:
        x1, y1, x2, y2 = v["box"]
        col = STATUS_COLOURS.get(v["status"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        tag = v["label"] if v["id"] is None else f"#{v['id']} {v['label']}"
        if v["stationary_s"] is not None:
            tag += f" {v['stationary_s']:.0f}s"
        if v["status"] == "POTENTIAL_VIOLATION":
            tag += " POTENTIAL VIOLATION"
        cv2.putText(img, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    c = summary["vehicle_counts"]
    text = (f"car {c['car']}  moto {c['motorcycle']}  bus {c['bus']}  truck {c['truck']}"
            f"  | potential violations: {summary['potential_violations']}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 255) if summary["potential_violations"] else (255, 255, 255), 2)
    return img
