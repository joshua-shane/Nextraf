# NEXTRAF - frontend data contract (schema_version 1)

Produced by `run_nextraf.py`. The frontend only reads files from the `--out` folder. Samples are
taken from real runs of this code (on Windows).

## Files

| File | Written | Use |
|---|---|---|
| `latest.json` | after every frame (atomic replace) | live view: poll this |
| `frames.csv` | streamed, one row per frame | charts / spreadsheets |
| `timeline.json` | at end of run (also on Ctrl+C or a crash) | full history incl. per-vehicle detail |
| `alerts.json` | at end of run (same) | list of alerts |
| `summary.json` | at end of run (same) | run totals and metadata |
| `<source>_fNNNNN.jpg` | `--mode validation` only | annotated frames. **`--mode deployment` saves no images** |

Handle missing files: `latest.json` does not exist until the first frame is processed, and the
three "end of run" files do not exist while a run is in progress. Re-running with the same
`--out` overwrites everything. Boxes are `[x1, y1, x2, y2]` in source-frame pixels.

## Frame record (`latest.json` = one `timeline.json` element plus `source`)
```json
{
  "source": "synthetic_static_street.mp4",
  "frame": 10, "video_t_sec": 5.0, "wall_time": "2026-11-14T13:06:48",
  "crowd": {
    "people_detected": 12, "people_in_roi": 12, "capacity": 10,
    "occupancy_pct": 120.0, "density_per_m2": null, "status": "OVERCROWDED"
  },
  "vehicles": {
    "counts": {"car": 26, "motorcycle": 3, "bus": 1, "truck": 2},
    "total": 32, "stationary": 30, "potential_violations": 6,
    "items": [
      {"id": 5, "label": "car", "conf": 0.87, "box": [234, 477, 378, 639],
       "stationary_s": 5.0, "on_carriageway_frac": 0.0, "status": "STATIONARY_OK"}
    ]
  },
  "latency_ms": 214.9
}
```
(`items` trimmed to one vehicle.) `frame` is the source frame index (it skips by `--stride`).

### Enumerations
- `crowd.status`: `OK` (< 80 % of capacity), `NEAR_CAPACITY` (80-100 %), `OVERCROWDED` (> capacity), `NO_CAPACITY_SET`.
- `vehicles.items[].status`: `MOVING_OR_BRIEF_STOP`, `STATIONARY`, `STATIONARY_OK`, `POTENTIAL_VIOLATION`, `NO_DURATION`.
  - `NO_DURATION`: single image - stationary time unknown.
  - `STATIONARY`: stationary, but no road line configured.
  - `POTENTIAL_VIOLATION`: stationary >= `park_seconds` (default 5) **and** >= `encroach` (default 15 %) of its footprint is on the road side of the line.

### Nulls to handle
- `video_t_sec`: null for still images.
- `density_per_m2`: null unless `area_m2` is set in the crowd config. `occupancy_pct`: null if no capacity.
- `items[].id`, `stationary_s`: null for still images. `on_carriageway_frac`: null with no road line.

## frames.csv
```
frame,video_t_sec,people_in_roi,people_detected,capacity,occupancy_pct,density_per_m2,crowd_status,car,motorcycle,bus,truck,vehicles_total,stationary,potential_violations,latency_ms
0,0.0,11,11,10,110.0,,OVERCROWDED,25,4,1,2,32,0,0,3830.5
1,0.5,13,13,10,130.0,,OVERCROWDED,27,3,1,2,33,0,0,213.7
```
Empty cell = null. The first row's latency includes model warm-up.

## alerts.json
A list; each alert is raised once. Crowd alerts fire when the status *enters* `OVERCROWDED`;
parking alerts fire once per tracked vehicle ID.
```json
{"type": "OVERCROWDING", "frame": 0, "video_t_sec": 0.0, "wall_time": "2026-11-14T13:06:45",
 "people_in_roi": 11, "capacity": 10}
```
```json
{"type": "POTENTIAL_PARKING_VIOLATION", "frame": 10, "video_t_sec": 5.0,
 "wall_time": "2026-11-14T13:06:48", "id": 4, "label": "car", "conf": 0.86,
 "box": [722, 352, 807, 432], "stationary_s": 5.0, "on_carriageway_frac": 1.0,
 "status": "POTENTIAL_VIOLATION"}
```

## summary.json
```json
{
  "schema_version": 1, "started": "2026-11-14T13:05:42", "finished": "2026-11-14T13:05:47",
  "source": "synthetic_static_street.mp4", "model": "yolo26n_1280_executorch_model",
  "imgsz": 1280, "stride": 1, "crowd_site": "test-street2", "road_site": "synthetic-test",
  "capacity": 10, "roi_configured": false, "road_line_configured": true,
  "frames_processed": 14, "peak_people_in_roi": 17, "peak_vehicles": 35,
  "n_alerts": 7, "n_crowd_alerts": 1, "n_parking_alerts": 6,
  "avg_latency_ms": 335.5, "steady_latency_ms": 212.5, "ram_mb": 619,
  "notes": ["People count is per frame ...", "Parking alerts are POTENTIAL ...", "Latency measured on a laptop CPU ..."]
}
```
`steady_latency_ms` excludes the first (warm-up) frame. The latency/RAM figures in these samples are
from a Windows laptop, **not a Raspberry Pi**.

## Limitations to show in the UI
- People count = people visible in that frame, not unique people. Motorcycle riders count as people.
- Parking alerts are **potential** (line crossing + stationary time), not legal findings.
- Vehicle IDs come from a lightweight IoU tracker; they can change after occlusion or fast motion and differ between runs.
- The samples above come from a static test image repeated as video, so every vehicle looks stationary and all parking alerts fire at t = 5 s. Real footage is not yet validated.
