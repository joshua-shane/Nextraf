# NEXTRAF

Offline, privacy-preserving edge-AI prototype for a Raspberry Pi 5: **crowd intelligence**
(people count, ROI occupancy, overcrowding alerts, density when calibrated) and **roadside
vehicle intelligence** (vehicle counts, tracking, stationary time, potential parking violations).
One pretrained YOLO26n model (COCO) runs through ExecuTorch + XNNPACK. Only counts and alerts
are written in deployment mode - never footage.

## Status (what is real today)

| Feature | Status |
|---|---|
| Person + vehicle detection, exported fp32 `.pte` (640 and 1280) | Done, tested on laptop CPU |
| Per-frame people count, ROI (foot-point rule), capacity alerts, density if `area_m2` set | Done, logic unit-tested, run on images |
| Vehicle counts by class | Done, run on images and a real video |
| IoU tracker, stationary timer, road-line footprint check, potential-violation alerts | Done, unit-tested + synthetic clip. **Not validated on real fixed-camera footage** |
| Combined app + frontend output (`run_nextraf.py`) | Done (contract below) |
| Backup model YOLO11n, exported fp32 `.pte` (640 and 1280), runs in the same pipeline | Done, tested on laptop CPU (not in the Pi ZIP) |
| MAE evaluation tool (`eval_crowd_mae.py`) | Done, tested with dummy counts only. **No real MAE yet** - needs hand counts |
| Results viewer (`frontend/`): opens `run_nextraf.py` output, 9 screens, offline site report | Done, 175 tests; [live demo](https://nextraf-frontend.vercel.app) with synthetic data |
| Benchmark and parity tools (`nextraf_pipeline/`): temperature, throttling, peak memory, MAE per configuration | Done, tested on laptop CPU with `yolo11n.pt`. **Not run on the Pi yet** |
| Raspberry Pi 5 deployment and benchmarks | **Not done** |
| INT8 export, fine-tuning on Indian data | **Not done** |

## Setup (Windows, already done on the dev laptop)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\yolo.exe export model=yolo26n.pt format=executorch imgsz=640
```
For the 1280 model: copy `yolo26n.pt` to `yolo26n_1280.pt` first (so the 640 export is not
overwritten), then export with `imgsz=1280`. **Windows only:** if export fails with
`FileNotFoundError: [WinError 2]` from `flatc`, ExecuTorch cannot find its bundled flatc - run
`$env:FLATC_EXECUTABLE = "$PWD\.venv\Lib\site-packages\executorch\data\bin\flatc.exe"` first.

### Backup model: YOLO11n
YOLO26n is the default. YOLO11n (listed in the original proposal) is kept as a backup; it uses
the same COCO class IDs, so no code changes are needed. Create the exports the same way:
```powershell
.venv\Scripts\yolo.exe export model=yolo11n.pt format=executorch imgsz=640
copy yolo11n.pt yolo11n_1280.pt
.venv\Scripts\yolo.exe export model=yolo11n_1280.pt format=executorch imgsz=1280
```
(`yolo11n.pt` downloads automatically from Ultralytics on first use.) To switch, pass
`--model yolo11n_1280_executorch_model --imgsz 1280` (or `yolo11n_executorch_model` with
`--imgsz 640`) to any `run_*.py`.

## Running

| Command | What it does |
|---|---|
| `.venv\Scripts\python.exe draw_roi.py <image> config\crowd_site.json` | click the monitored area (S to save) |
| `.venv\Scripts\python.exe draw_road_line.py <image or video> config\road_line.json` | click 2 road-edge points, then 1 point on the parking-allowed side |
| `.venv\Scripts\python.exe run_crowd.py --source <src> --config config\crowd_site.json` | crowd module only |
| `.venv\Scripts\python.exe run_vehicles.py --source <src> --config config\road_line.json` | vehicle/parking module only |
| `.venv\Scripts\python.exe run_nextraf.py --source <src> ...` | **both modules, one detection pass** (see below) |

`<src>` can be an image, a folder of images, or a video. Add `--mode deployment` to write counts
and alerts only. Parking needs a video from a **fixed** camera; single images give counts only.
`config\crowd_site.json` keys: `site`, `roi` (polygon or null), `capacity`, `warn_ratio`, `area_m2`
(real ground area in m^2, or null - pixel area is never used as m^2). Road config keys: `p1`, `p2`,
`allowed_point`, `frame_size`, `park_seconds` (default 5), `encroach` (default 0.15).

## Results viewer

`frontend/` is an offline Streamlit app that opens a `run_nextraf.py` output folder: people and vehicles over
time, crowding verdict, replay, validation images, device numbers and a one-page site report. It has its own
environment (run it on a laptop, not on the Pi while benchmarking):

```powershell
python -m venv .venv-viewer
.venv-viewer\Scripts\pip install -r frontend\requirements.txt
.venv-viewer\Scripts\streamlit run frontend\app.py -- --results-dir out\demo
```
Details in [frontend/README.md](frontend/README.md). How it maps `frames.csv` and `summary.json` is in
[docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md).

## Benchmark and parity tools

`nextraf_pipeline/` measures the pipeline on the Pi: ExecuTorch against PyTorch parity, and latency, peak memory,
temperature, throttling and MAE for every model, input size and threshold. Its files feed the viewer's
Affordability envelope and Validation screens. See [nextraf_pipeline/README.md](nextraf_pipeline/README.md).

## Repository layout

| Path | What it is |
|---|---|
| `nextraf/`, `run_nextraf.py`, `run_crowd.py`, `run_vehicles.py`, `draw_*.py`, `config/` | The pipeline (detection, crowd, vehicles) |
| `eval_crowd_mae.py` | MAE against hand counts for one run |
| `frontend/` | Results viewer (Streamlit) and its tests |
| `nextraf_pipeline/` | Benchmark and parity tools |
| `nextraf_common/` | Hash-chained CSV log shared by `nextraf_pipeline/` and the viewer |
| `docs/DATA_CONTRACT.md` | Every file the viewer reads |

## Crowd-count accuracy (MAE)

`eval_crowd_mae.py` compares the model's per-frame people count with counts you make by hand.
1. Run `run_crowd.py` on a clip, then create a counting template (10-20 evenly spaced frames,
   saved as plain images without model boxes, plus a CSV with an empty `gt_people` column):
   ```powershell
   .venv\Scripts\python.exe eval_crowd_mae.py template --pred out\<run>\crowd_counts.csv --source <clip> --n 15 --out out\eval\<name>
   ```
2. Count the people in each image (same rule as the model: everyone whose feet are inside the
   ROI; with no ROI, every visible person) and type the numbers into `gt_people`.
3. Evaluate:
   ```powershell
   .venv\Scripts\python.exe eval_crowd_mae.py evaluate --pred out\<run>\crowd_counts.csv --gt out\eval\<name>\ground_truth.csv --out out\eval\<name>
   ```
Rows match on `source` + `frame` (the original video frame index), or `frame` only for
`run_nextraf.py`'s `frames.csv`. It reports MAE, mean signed error (negative = undercount), max
error, % of frames within +-1 and a per-frame table (`mae_results.csv`, `mae_summary.json`).
Empty, invalid, duplicate and unmatched rows are labelled and excluded; with no valid counts it
prints "MAE NOT COMPUTED" instead of a number.

## Findings so far (laptop CPU, i7-13650HX; not Raspberry Pi numbers)

- `bus.jpg`: 4/4 people and the bus found (exported model). A partly visible person scored 0.56,
  so the confidence threshold directly changes crowd counts.
- `street.jpg` (dense Bengaluru traffic, top-down): **0 motorcycles at conf 0.4** (~25 visible),
  4 at conf 0.15. Exported `.pte` and original `.pt` gave the same counts, so this is a model
  limitation, not an export bug.
- `hyderabad_480p.webm` (handheld, close range): ~1-2 of 6-8 visible two-wheelers detected;
  auto-rickshaws are labelled truck/bus (COCO has no auto-rickshaw class).
  -> Fine-tuning on Indian traffic data (with an auto-rickshaw class) is justified future work.
- `street2.jpg` (small distant pedestrians): 640 input finds 1-2 people, **1280 finds 10**.
  Resolution vs cost is the main crowd-accuracy lever.
- Steady latency at 640: ~105-110 ms/frame (~9 FPS), RAM ~410-420 MB; first frame adds 2.5-5 s
  ExecuTorch warm-up. Single-run timings vary; benchmark on the Pi with many frames.
- YOLO26n vs YOLO11n (exported `.pte`, conf 0.4): similar laptop latency (640: ~125 vs ~119 ms;
  1280: ~220 vs ~214 ms on `street.jpg`). YOLO11n reports somewhat more objects on the street
  images, but without ground truth that is **not** evidence of better accuracy. At 1280 both
  models missed the bus in `bus.jpg` that they find at 640 (the bus fills most of the frame).

## Media credits (test inputs, not committed)

- `street.jpg` - "TRAFFIC VIEW IN BANGALORE", Divya Chinnasamy, Wikimedia Commons, CC0.
- `street2.jpg` - "Traffic in Bangalore (India, 2017)", Neet Bhadra, Wikimedia Commons, CC BY-SA 4.0.
- `hyderabad_480p.webm` - "Traffic in Hyderabad" (480p transcode), Oleg Yunakov, Wikimedia Commons, CC BY-SA 4.0.
- `bus.jpg` - Ultralytics sample image.

Models: YOLO26n (default) and YOLO11n (backup) by Ultralytics, AGPL-3.0. Model approval: the
original proposal lists SSDLite320-MobileNetV3 / YOLO11n; using YOLO26n needs organiser confirmation.

---

# Frontend data contract

`run_nextraf.py` runs crowd counting and vehicle/parking analysis on one detection pass
(YOLO26n, ExecuTorch) and writes JSON/CSV files into one output folder. The frontend only
needs to read those files. All samples below come from a real run (schema_version 1).

```powershell
.venv\Scripts\python.exe run_nextraf.py --source clip.mp4 `
  --crowd-config config\crowd_site.json --road-config <road_line.json> `
  --model yolo26n_1280_executorch_model --imgsz 1280 --stride 1 --out out\demo
```

`--imgsz` must equal the model's export size (640 or 1280). Without `--road-config` there is no
parking analysis; without `--crowd-config` there is no capacity or alerting.

## Files in the output folder

| File | Written | Use |
|---|---|---|
| `latest.json` | every frame (atomic replace) | live view: poll this |
| `frames.csv` | streamed, one row per frame | charts, spreadsheets |
| `timeline.json` | at end of run | full history incl. per-vehicle detail |
| `alerts.json` | at end of run | alert list |
| `summary.json` | at end of run | run totals and metadata |
| `*.jpg` | validation mode only | annotated frames (image is 34 px taller than the source: vehicle banner on top; JSON box coordinates are in source-frame pixels) |

`timeline.json`, `alerts.json` and `summary.json` are also written if the run is interrupted.

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

`box` is `[x1, y1, x2, y2]` in source-frame pixels. `items` is trimmed to one vehicle here.

### Enumerations
- `crowd.status`: `OK` (< 80 % of capacity), `NEAR_CAPACITY` (80-100 %), `OVERCROWDED` (> capacity), `NO_CAPACITY_SET`.
- `vehicles.items[].status`: `MOVING_OR_BRIEF_STOP`, `STATIONARY`, `STATIONARY_OK`, `POTENTIAL_VIOLATION`, `NO_DURATION`.
  - `NO_DURATION`: single image, so stationary time is unknown.
  - `STATIONARY`: stationary but no road line configured.
  - `POTENTIAL_VIOLATION`: stationary >= 5 s **and** >= 15 % of its footprint is on the road side of the line.

### Nulls to handle
- `video_t_sec`: `null` for still images.
- `density_per_m2`: `null` unless `area_m2` is set in the crowd config.
- `occupancy_pct`: `null` when no capacity is set.
- `items[].id` and `stationary_s`: `null` for still images. `on_carriageway_frac`: `null` with no road line.

## frames.csv

```
frame,video_t_sec,people_in_roi,people_detected,capacity,occupancy_pct,density_per_m2,crowd_status,car,motorcycle,bus,truck,vehicles_total,stationary,potential_violations,latency_ms
0,0.0,11,11,10,110.0,,OVERCROWDED,25,4,1,2,32,0,0,3830.5
1,0.5,13,13,10,130.0,,OVERCROWDED,27,3,1,2,33,0,0,213.7
```
Empty cells are nulls. The first row's latency includes model warm-up.

## alerts.json
A list. Each alert is raised once: crowd alerts when the status *enters* `OVERCROWDED`; parking
alerts once per tracked vehicle ID.

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
  "notes": ["..."]
}
```
`steady_latency_ms` excludes the first (warm-up) frame.

## Limitations to show in the UI
- People count is **people visible in that frame**, not unique people.
- Parking alerts are **potential** violations (line crossing + stationary time). They are not legal findings.
- Vehicle IDs come from a lightweight IoU tracker; IDs can change after occlusion or fast motion and may differ between runs.
- Latency and RAM were measured on a laptop CPU, not on the target device.
- Samples come from a static test image repeated as video, so every vehicle looks stationary and all parking alerts fire at t = 5 s. Real footage is still unvalidated.
