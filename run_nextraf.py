"""NEXTRAF combined pipeline: crowd + vehicles + parking in ONE detection pass.
Writes one stable, frontend-friendly output folder.

  .venv\\Scripts\\python.exe run_nextraf.py --source clip.mp4 ^
      --crowd-config config\\crowd_site.json --road-config config\\road_line.json ^
      --model yolo26n_1280_executorch_model --imgsz 1280 --stride 2 --out out\\demo

Outputs (schema_version 1) in --out:
  summary.json   run metadata, peaks, totals, latency          (final, written at end)
  timeline.json  list of per-frame records incl. vehicle list  (final, written at end)
  frames.csv     flat per-frame table (people + vehicles)      (streamed)
  alerts.json    all alerts: OVERCROWDING and PARKING          (final, written at end)
  latest.json    newest frame record - poll this for live view (streamed)
  *.jpg          annotated frames (validation mode only)
"""
import argparse
import csv
import json
import os
import time

import cv2
import psutil

from nextraf.crowd import CrowdConfig
from nextraf.crowd import analyse as crowd_analyse
from nextraf.crowd import draw as crowd_draw
from nextraf.detector import COCO_NAMES, PERSON, VEHICLES, Detector
from nextraf.sources import iter_frames
from nextraf.vehicles import IoUTracker, RoadConfig
from nextraf.vehicles import analyse as veh_analyse
from nextraf.vehicles import draw as veh_draw

SCHEMA = 1
CSV_FIELDS = ["frame", "video_t_sec", "people_in_roi", "people_detected", "capacity", "occupancy_pct",
              "density_per_m2", "crowd_status", "car", "motorcycle", "bus", "truck", "vehicles_total",
              "stationary", "potential_violations", "latency_ms"]


def dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic, so a polling frontend never reads half a file


def main():
    ap = argparse.ArgumentParser(description="NEXTRAF combined pipeline")
    ap.add_argument("--source", required=True)
    ap.add_argument("--crowd-config", default=None)
    ap.add_argument("--road-config", default=None)
    ap.add_argument("--model", default="yolo26n_1280_executorch_model")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--mode", choices=["validation", "deployment"], default="validation")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("out", "nextraf"))
    a = ap.parse_args()

    ccfg, rcfg = CrowdConfig.load(a.crowd_config), RoadConfig.load(a.road_config)
    det = Detector(a.model, conf=a.conf, imgsz=a.imgsz, classes=[PERSON, *VEHICLES])
    tracker = IoUTracker()
    os.makedirs(a.out, exist_ok=True)
    proc = psutil.Process()

    timeline, alerts, lat = [], [], []
    prev_crowd, peak_people, peak_veh, n_saved = None, 0, 0, 0
    source_name, started = None, time.strftime("%Y-%m-%dT%H:%M:%S")

    def write_final():
        avg = sum(lat) / len(lat) if lat else None
        steady = lat[1:] if len(lat) > 1 else lat
        dump(os.path.join(a.out, "timeline.json"), timeline)
        dump(os.path.join(a.out, "alerts.json"), alerts)
        dump(os.path.join(a.out, "summary.json"), {
            "schema_version": SCHEMA, "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": source_name, "model": a.model, "imgsz": a.imgsz, "stride": a.stride,
            "crowd_site": ccfg.site, "road_site": rcfg.site,
            "capacity": ccfg.capacity, "roi_configured": bool(ccfg.roi),
            "road_line_configured": rcfg.has_line,
            "frames_processed": len(lat), "peak_people_in_roi": peak_people,
            "peak_vehicles": peak_veh, "n_alerts": len(alerts),
            "n_crowd_alerts": sum(x["type"] == "OVERCROWDING" for x in alerts),
            "n_parking_alerts": sum(x["type"] == "POTENTIAL_PARKING_VIOLATION" for x in alerts),
            "avg_latency_ms": None if avg is None else round(avg, 1),
            "steady_latency_ms": round(sum(steady) / len(steady), 1) if steady else None,
            "ram_mb": round(proc.memory_info().rss / 1e6),
            "notes": ["People count is per frame (visible people), not unique individuals.",
                      "Parking alerts are POTENTIAL: line crossing + stationary time only.",
                      "Latency measured on a laptop CPU, not on the target device."]})

    try:
        with open(os.path.join(a.out, "frames.csv"), "w", newline="") as fcsv:
            wr = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS)
            wr.writeheader()
            for k, (name, idx, t, frame) in enumerate(iter_frames(a.source, a.stride)):
                source_name = name
                dets, ms = det.detect(frame)
                lat.append(ms)
                h, w = frame.shape[:2]
                cres = crowd_analyse(dets, ccfg)
                vsum, vehicles, new_alerts = veh_analyse(dets, tracker, rcfg, w, h, t)
                wall = time.strftime("%Y-%m-%dT%H:%M:%S")
                vt = None if t is None else round(t, 2)
                c = vsum["vehicle_counts"]
                rec = {"frame": idx, "video_t_sec": vt, "wall_time": wall,
                       "crowd": {k2: cres[k2] for k2 in ("people_detected", "people_in_roi", "capacity",
                                                         "occupancy_pct", "density_per_m2", "status")},
                       "vehicles": {"counts": c, "total": vsum["vehicles_total"],
                                    "stationary": vsum["stationary"],
                                    "potential_violations": vsum["potential_violations"],
                                    "items": vehicles},
                       "latency_ms": round(ms, 1)}
                timeline.append(rec)
                dump(os.path.join(a.out, "latest.json"), {"source": name, **rec})
                wr.writerow({"frame": idx, "video_t_sec": vt, "people_in_roi": cres["people_in_roi"],
                             "people_detected": cres["people_detected"], "capacity": cres["capacity"],
                             "occupancy_pct": cres["occupancy_pct"], "density_per_m2": cres["density_per_m2"],
                             "crowd_status": cres["status"], **c, "vehicles_total": vsum["vehicles_total"],
                             "stationary": vsum["stationary"],
                             "potential_violations": vsum["potential_violations"],
                             "latency_ms": round(ms, 1)})
                if cres["status"] == "OVERCROWDED" and prev_crowd != "OVERCROWDED":
                    alerts.append({"type": "OVERCROWDING", "frame": idx, "video_t_sec": vt, "wall_time": wall,
                                   "people_in_roi": cres["people_in_roi"], "capacity": ccfg.capacity})
                prev_crowd = cres["status"]
                for v in new_alerts:
                    alerts.append({"type": "POTENTIAL_PARKING_VIOLATION", "frame": idx, "video_t_sec": vt,
                                   "wall_time": wall, **v})
                peak_people = max(peak_people, cres["people_in_roi"])
                peak_veh = max(peak_veh, vsum["vehicles_total"])
                print(f"{name} f{idx}: people={cres['people_in_roi']} [{cres['status']}] "
                      f"vehicles={vsum['vehicles_total']} stationary={vsum['stationary']} "
                      f"violations={vsum['potential_violations']} {ms:.0f} ms")
                if a.mode == "validation" and (t is None or k % a.save_every == 0 or new_alerts):
                    # Both draw() helpers paint a 34 px banner at the top and would cover each other,
                    # so keep the vehicle banner as a separate strip stacked above the frame.
                    img = veh_draw(frame, vehicles, vsum, rcfg)
                    veh_banner = img[:34].copy()
                    img = crowd_draw(img, [d for d in dets if d.cls_id == PERSON], cres, ccfg)
                    img = cv2.vconcat([veh_banner, img])
                    cv2.imwrite(os.path.join(a.out, f"{os.path.splitext(name)[0]}_f{idx:05d}.jpg"), img)
                    n_saved += 1
    finally:
        write_final()
    print(f"\nframes={len(lat)} alerts={len(alerts)} images={n_saved}  -> {a.out}")


if __name__ == "__main__":
    main()
