"""Draw the crowd ROI polygon on a frame and save it into a site config.

Usage:  .venv\\Scripts\\python.exe draw_roi.py street2.jpg config\\crowd_site.json
Left-click to add corners. Keys: S = save, U = undo, R = reset, Q = quit.
Existing config values (capacity, area_m2, ...) are kept; only "roi" is replaced.
"""
import json
import os
import sys

import cv2
import numpy as np

from nextraf.sources import iter_frames

src = sys.argv[1] if len(sys.argv) > 1 else "street2.jpg"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("config", "crowd_site.json")
_, _, _, frame = next(iter_frames(src))
pts = []


def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])


win = "NEXTRAF - draw ROI (S save, U undo, R reset, Q quit)"
cv2.namedWindow(win)
cv2.setMouseCallback(win, on_click)
while True:
    img = frame.copy()
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], len(pts) > 2, (255, 255, 0), 2)
    for p in pts:
        cv2.circle(img, tuple(p), 4, (0, 255, 255), -1)
    cv2.imshow(win, img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord("u") and pts:
        pts.pop()
    elif k == ord("r"):
        pts.clear()
    elif k == ord("q"):
        sys.exit("Cancelled - nothing saved")
    elif k == ord("s") and len(pts) >= 3:
        break
cv2.destroyAllWindows()

cfg = {}
if os.path.exists(out):
    with open(out) as f:
        cfg = json.load(f)
cfg["roi"] = pts
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Saved ROI with {len(pts)} corners to {out}")
