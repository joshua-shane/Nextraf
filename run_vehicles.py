"""NEXTRAF Module B - vehicle counts, tracking, stationary time and potential
roadside parking violations.

Examples (PowerShell, from the Nextraf folder):
  .venv\\Scripts\\python.exe run_vehicles.py --source street.jpg
  .venv\\Scripts\\python.exe run_vehicles.py --source clip.mp4 --config config\\road_line.json

Single images give counts and footprint overlap only (no duration, so no
violations). Parking checks need a video from a FIXED camera.

Outputs (in --out, default out\\vehicles):
  vehicle_counts.csv  one row per processed frame
  latest.json         newest snapshot incl. per-vehicle status (dashboard)
  alerts.jsonl        one line per vehicle ID when it first becomes a potential violation
  *.jpg               annotated frames (validation mode only)
"""
import argparse
import csv
import json
import os
import time

import cv2
import psutil

from nextraf.detector import VEHICLES, Detector
from nextraf.sources import iter_frames
from nextraf.vehicles import IoUTracker, RoadConfig, analyse, draw


def main():
    ap = argparse.ArgumentParser(description="NEXTRAF vehicle & parking module")
    ap.add_argument("--source", required=True, help="image, folder of images, or video")
    ap.add_argument("--config", default=None, help="road line JSON from draw_road_line.py")
    ap.add_argument("--model", default="yolo26n_executorch_model")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=1, help="video: process every Nth frame")
    ap.add_argument("--mode", choices=["validation", "deployment"], default="validation")
    ap.add_argument("--save-every", type=int, default=15, help="video: save every Nth processed frame")
    ap.add_argument("--out", default=os.path.join("out", "vehicles"))
    args = ap.parse_args()

    cfg = RoadConfig.load(args.config)
    det = Detector(args.model, conf=args.conf, imgsz=args.imgsz, classes=list(VEHICLES))
    tracker = IoUTracker()
    os.makedirs(args.out, exist_ok=True)
    proc = psutil.Process()

    fields = ["site", "source", "frame", "video_t_sec", "wall_time", "car", "motorcycle", "bus",
              "truck", "vehicles_total", "stationary", "potential_violations", "latency_ms"]
    lat, n_alerts, n_saved = [], 0, 0
    with open(os.path.join(args.out, "vehicle_counts.csv"), "w", newline="") as fcsv, \
         open(os.path.join(args.out, "alerts.jsonl"), "a") as falert:
        wr = csv.DictWriter(fcsv, fieldnames=fields)
        wr.writeheader()
        for k, (name, idx, t, frame) in enumerate(iter_frames(args.source, args.stride)):
            dets, ms = det.detect(frame)
            h, w = frame.shape[:2]
            summary, vehicles, new_alerts = analyse(dets, tracker, cfg, w, h, t)
            lat.append(ms)
            rec = {"site": cfg.site, "source": name, "frame": idx,
                   "video_t_sec": None if t is None else round(t, 2),
                   "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   **summary["vehicle_counts"], "vehicles_total": summary["vehicles_total"],
                   "stationary": summary["stationary"],
                   "potential_violations": summary["potential_violations"],
                   "latency_ms": round(ms, 1)}
            wr.writerow(rec)
            with open(os.path.join(args.out, "latest.json"), "w") as fj:
                json.dump({**rec, "vehicles": vehicles}, fj, indent=2)
            for v in new_alerts:
                falert.write(json.dumps({"type": "POTENTIAL_PARKING_VIOLATION", "site": cfg.site,
                                         "source": name, "frame": idx, "video_t_sec": rec["video_t_sec"],
                                         "wall_time": rec["wall_time"], **v}) + "\n")
                n_alerts += 1
            c = summary["vehicle_counts"]
            print(f"{name} frame {idx}: car={c['car']} moto={c['motorcycle']} bus={c['bus']} "
                  f"truck={c['truck']} stationary={summary['stationary']} "
                  f"potential_violations={summary['potential_violations']} {ms:.0f} ms")
            if args.mode == "validation" and (t is None or k % args.save_every == 0 or new_alerts):
                stem = os.path.splitext(name)[0]
                cv2.imwrite(os.path.join(args.out, f"{stem}_f{idx:05d}.jpg"),
                            draw(frame, vehicles, summary, cfg))
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
