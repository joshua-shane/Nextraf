"""Mark the road-edge line for the parking module (one time per camera position).

Usage:  .venv\\Scripts\\python.exe draw_road_line.py street.jpg config\\road_line.json
Click 1: road-edge line point 1
Click 2: road-edge line point 2
Click 3: any point on the side where parking is ALLOWED (shoulder / parking bay)
Keys:    S = save, R = redo, Q = quit
Existing config values (park_seconds, encroach, site) are kept.
"""
import json
import os
import sys

import cv2

from nextraf.sources import iter_frames

src = sys.argv[1] if len(sys.argv) > 1 else "street.jpg"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("config", "road_line.json")
_, _, _, frame = next(iter_frames(src))
pts = []
prompts = ["Click road-edge point 1", "Click road-edge point 2",
           "Click a point on the PARKING-ALLOWED side", "S = save   R = redo   Q = quit"]


def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
        pts.append([x, y])


win = "NEXTRAF - mark road edge"
cv2.namedWindow(win)
cv2.setMouseCallback(win, on_click)
while True:
    img = frame.copy()
    if len(pts) >= 2:
        cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 255, 255), 2)
    for i, p in enumerate(pts):
        cv2.circle(img, tuple(p), 6, (0, 200, 0) if i == 2 else (0, 255, 255), -1)
    cv2.putText(img, prompts[len(pts)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow(win, img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord("r"):
        pts.clear()
    elif k == ord("q"):
        sys.exit("Cancelled - nothing saved")
    elif k == ord("s") and len(pts) == 3:
        break
cv2.destroyAllWindows()

cfg = {}
if os.path.exists(out):
    with open(out) as f:
        cfg = json.load(f)
cfg.update({"p1": pts[0], "p2": pts[1], "allowed_point": pts[2],
            "frame_size": [frame.shape[1], frame.shape[0]]})
cfg.setdefault("park_seconds", 5.0)
cfg.setdefault("encroach", 0.15)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Saved road line to {out}")
