# NEXTRAF - handover notes for teammates

## For the Raspberry Pi teammate

**Goal:** run `run_nextraf.py` offline on the Pi 5 and measure real performance. Nothing has
been run on a Pi yet - every number in the README is from a laptop.

1. **OS:** Raspberry Pi OS 64-bit (aarch64). Check the Python version with `python3 --version`.
2. **Copy from the laptop** (USB stick or `scp`):
   - `nextraf\`, `run_nextraf.py`, `run_crowd.py`, `run_vehicles.py`, `requirements.txt`, `config\`
   - `yolo26n_executorch_model\` (640) and, if testing it, `yolo26n_1280_executorch_model\` (1280)
   - a test clip (for example `hyderabad_480p.webm`, or a clip from a fixed camera)
3. **Environment on the Pi:**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   - Not yet verified: that `executorch==1.6.0` and `torch==2.15.0` install on the Pi's Python
     version. If pip cannot find them, report the exact error - do not change versions blindly,
     because the `.pte` files were exported with these versions.
   - On a Pi without a screen, `opencv-python-headless` can replace `opencv-python`.
   - The `FLATC_EXECUTABLE` fix in the README is for **exporting on Windows** only; it is not
     needed to run inference.
4. **Draw the ROI and road line on the laptop** (`draw_roi.py`, `draw_road_line.py` open a
   window), using a frame from the Pi camera, then copy the JSON files into `config/` on the Pi.
5. **Benchmark run:**
   ```bash
   python run_nextraf.py --source clip.mp4 --model yolo26n_executorch_model --imgsz 640 --mode deployment --out out/pi_640
   ```
   Then repeat with `--model yolo26n_1280_executorch_model --imgsz 1280 --out out/pi_1280`.
   `out/.../summary.json` contains `steady_latency_ms` (excludes warm-up), `ram_mb` and frame count.
   - `--imgsz` must match the export size of the model.
   - Not measured by the code yet: **CPU utilisation** and **temperature/throttling**. Watch them
     in a second terminal with `top` and `vcgencmd measure_temp`.
   - Also record the size of each `model.pte`.
6. **Report back:** the `summary.json` files, the Pi model/RAM, the Python version and any errors.

## For the frontend teammate

- The data contract is in the README section **"Frontend data contract"** (schema_version 1).
  Field names in it are what the code writes today.
- **Live view:** poll `latest.json` (replaced atomically, so it is never half-written).
- **History / charts:** `frames.csv` (streamed) or `timeline.json` (written at the end of a run).
- **Alerts:** `alerts.json` - types `OVERCROWDING` and `POTENTIAL_PARKING_VIOLATION`.
- **Sample data to build against:** run
  ```powershell
  .venv\Scripts\python.exe run_nextraf.py --source out\test\synthetic_static_street.mp4 --crowd-config config\crowd_site.json --road-config out\test\road_line_test.json --out out\demo
  ```
  `out\` is not in git, so the clip and test road line exist only on the dev laptop - ask for
  the `out\test\` folder, or copy a finished `out\demo\` folder instead.
  This clip is a still image repeated as video, so every vehicle looks parked and all parking
  alerts fire at t = 5 s. It is fine for building the UI, not for judging accuracy.
- **Wording the UI must keep:**
  - "People in view" (a per-frame count, not unique people).
  - "Potential parking violation" (not a legal finding).
  - Laptop timings are not Raspberry Pi timings.
