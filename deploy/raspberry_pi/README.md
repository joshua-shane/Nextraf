# NEXTRAF - Raspberry Pi 5 handover package

NEXTRAF counts people (crowd occupancy vs capacity) and vehicles, tracks vehicles with a light
IoU tracker, and flags **potential** roadside parking violations. One YOLO26n model (ExecuTorch
`.pte`) runs once per frame; the crowd and vehicle modules share its detections. Results are
written as JSON/CSV files for a dashboard. See `FRONTEND_SCHEMA.md`.

## Status - read this first
| | |
|---|---|
| Tested on | **Windows 11, x64, Python 3.13.3 only** |
| Tested on Raspberry Pi 5 | **NO. Nothing here has been run on a Pi.** |
| Package contents verified | extracted to a clean folder; imports resolve; model loads and the pipeline runs there (on Windows) |
| ARM64 wheels | exist per PyPI / PyTorch CPU index metadata - **not installed on ARM** |
| ExecuTorch XNNPACK on ARM | **unverified** (the model uses the XNNPACK delegate) |
| Camera input | **not supported by the current code** (video files and images only) - see `DEPLOYMENT_GUIDE.md` |
| Real-footage accuracy | **not validated** (no ground-truth counts exist yet) |

## Contents
- `run_nextraf.py` - **main entry point**: crowd + vehicles + parking, unified outputs.
- `run_crowd.py`, `run_vehicles.py` - the two modules on their own (older, separate output formats).
- `draw_roi.py`, `draw_road_line.py` - OpenCV click tools to mark the crowd ROI / road-edge line. They need a display; run them on a desktop and copy the JSON to the Pi.
- `nextraf/` - library: `detector.py` (model wrapper), `sources.py` (image/folder/video input), `crowd.py`, `vehicles.py`.
- `config/crowd_site.json` - site config (capacity, warn ratio, optional ROI/area). `config/road_line.example.json` - example road-line config (placeholder values).
- `yolo26n_1280_executorch_model/` - **recommended** model, 1280x1280 input. `yolo26n_executorch_model/` - 640x640 model (faster, detects far fewer small/distant people).
- `pi_smoke_test.py` - quick model test to run first on the Pi.
- `requirements-pi.txt`, `DEPLOYMENT_GUIDE.md`, `FRONTEND_SCHEMA.md`, `PACKAGE_MANIFEST.txt`.
- `HANDOVER.md` - team handover notes, copied from the git repo (commit `0c6d89c`); "README" in it means the repo README.

## Quick start (details in DEPLOYMENT_GUIDE.md)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pi.txt
python pi_smoke_test.py            # quick model test - must end with RESULT: PASS
python run_nextraf.py --source your_clip.mp4 --crowd-config config/crowd_site.json \
  --model yolo26n_1280_executorch_model --imgsz 1280 --stride 30 --mode deployment --out out/run1
```
`--imgsz` must equal the model's export size (1280 or 640). Docstrings inside the scripts show
Windows (`.venv\Scripts\python.exe`, backslash) examples; on the Pi use the Linux forms above.
