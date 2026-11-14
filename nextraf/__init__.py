"""NEXTRAF - offline crowd and roadside vehicle intelligence.

Edge-AI system for the Namma AI Hackathon (PS2). Runs fully offline on a
Raspberry Pi 5 using a YOLO26n model exported to ExecuTorch (.pte) with the
XNNPACK CPU backend.

Package layout
--------------
detector  : loads the ExecuTorch model and returns per-frame detections
crowd     : Module A - people counting, ROI occupancy, overcrowding alerts
vehicles  : Module B - vehicle counting, IoU tracking, road-edge line and
            potential parking-violation detection
sources   : frame iterator for images, folders and video files

Typical use
-----------
    from nextraf import Detector, CrowdConfig, RoadConfig, IoUTracker
    from nextraf import crowd, vehicles

    det = Detector()                        # YOLO26n ExecuTorch, conf 0.4
    dets, latency_ms = det.detect(frame)
    result = crowd.analyse(dets, CrowdConfig.load("config/crowd_site.json"))
"""

__version__ = "1.0.0"
__author__ = "Team NEXTRAF"

# COCO class IDs the system detects.
PERSON = 0
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
VEHICLE_CLASSES = (CAR, MOTORCYCLE, BUS, TRUCK)

# Public API: the names a user of this package is expected to import.
from .detector import Detection, Detector
from .crowd import CrowdConfig
from .vehicles import RoadConfig, IoUTracker
from .sources import iter_frames

__all__ = [
    "Detection",
    "Detector",
    "CrowdConfig",
    "RoadConfig",
    "IoUTracker",
    "iter_frames",
    "PERSON",
    "CAR",
    "MOTORCYCLE",
    "BUS",
    "TRUCK",
    "VEHICLE_CLASSES",
    "__version__",
]
