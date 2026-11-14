"""NEXTRAF Module A - crowd counting, ROI occupancy and overcrowding alerts.

Examples (PowerShell, from the Nextraf folder):
  .venv\\Scripts\\python.exe run_crowd.py --source street2.jpg
  .venv\\Scripts\\python.exe run_crowd.py --source clip.mp4 --config config\\crowd_site.json
  .venv\\Scripts\\python.exe run_crowd.py --source clip.mp4 --mode deployment

Outputs (in --out, default out\\crowd):
  crowd_counts.csv  one row per processed frame      (for the dashboard / MAE)
  latest.json       newest status snapshot            (dashboard polls this)
  alerts.jsonl      one line each time status becomes OVERCROWDED
  *.jpg             annotated frames (validation mode only)
"""
import argparse
import csv
import json
import os
import time

import cv2
import psutil

from nextraf.crowd import CrowdConfig, analyse, draw
from nextraf.detector import PERSON, Detector
from nextraf.sources import iter_frames


def main():
    ap = argparse.ArgumentParser(description="NEXTRAF crowd module")
    ap.add_argument("--source", required=True, help="image, folder of images, or video")
    ap.add_argument("--config", default=None, help="site JSON: roi, capacity, warn_ratio, area_m2")
    ap.add_argument("--model", default="yolo26n_executorch_model")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=1, help="video: process every Nth frame")
    ap.add_argument("--mode", choices=["validation", "deployment"], default="validation")
    ap.add_argument("--save-every", type=int, default=15, help="video: save every Nth processed frame")
    ap.add_argument("--out", default=os.path.join("out", "crowd"))
    args = ap.parse_args()

    cfg = CrowdConfig.load(args.config)
    det = Detector(args.model, conf=args.conf, imgsz=args.imgsz, classes=[PERSON])
    os.makedirs(args.out, exist_ok=True)
    proc = psutil.Process()

    lat, prev_status, n_alerts, n_saved = [], None, 0, 0
    fields = ["site", "source", "frame", "video_t_sec", "wall_time", "people_detected",
              "people_in_roi", "capacity", "occupancy_pct", "density_per_m2", "status", "latency_ms"]
    with open(os.path.join(args.out, "crowd_counts.csv"), "w", newline="") as fcsv, \
         open(os.path.join(args.out, "alerts.jsonl"), "a") as falert:
        wr = csv.DictWriter(fcsv, fieldnames=fields)
        wr.writeheader()
        for k, (name, idx, t, frame) in enumerate(iter_frames(args.source, args.stride)):
            dets, ms = det.detect(frame)
            res = analyse(dets, cfg)
            lat.append(ms)
            rec = {"site": cfg.site, "source": name, "frame": idx,
                   "video_t_sec": None if t is None else round(t, 2),
                   "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   **{f: res[f] for f in fields if f in res}, "latency_ms": round(ms, 1)}
            wr.writerow(rec)
            with open(os.path.join(args.out, "latest.json"), "w") as fj:
                json.dump(rec, fj, indent=2)
            if res["status"] == "OVERCROWDED" and prev_status != "OVERCROWDED":
                falert.write(json.dumps({"type": "OVERCROWDING", **rec}) + "\n")
                n_alerts += 1
            prev_status = res["status"]
            print(f"{name} frame {idx}: {res['people_in_roi']} in ROI "
                  f"({res['people_detected']} detected) [{res['status']}] {ms:.0f} ms")
            if args.mode == "validation" and (t is None or k % args.save_every == 0):
                stem = os.path.splitext(name)[0]
                cv2.imwrite(os.path.join(args.out, f"{stem}_f{idx:05d}.jpg"), draw(frame, dets, res, cfg))
                n_saved += 1

    if lat:
        steady = lat[1:] or lat  # first frame includes model warm-up
        avg = sum(steady) / len(steady)
        print(f"\nframes={len(lat)}  warm-up={lat[0]:.0f} ms  steady avg latency={avg:.1f} ms "
              f"(~{1000/avg:.1f} FPS)  RAM={proc.memory_info().rss/1e6:.0f} MB  "
              f"alerts={n_alerts}  images_saved={n_saved}")
        print("Note: laptop numbers are not Raspberry Pi numbers.")
    print(f"Outputs in {args.out}")


if __name__ == "__main__":
    main()
