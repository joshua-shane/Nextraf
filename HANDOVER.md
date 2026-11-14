# NEXTRAF - handover notes for teammates

## For the Raspberry Pi teammate

**Goal:** run `run_nextraf.py` offline on the Pi 5 and measure real performance. Nothing has
been run on a Pi yet - every number in the README is from a laptop.

1. **Use the package, not the git repo:** copy `NEXTRAF_RaspberryPi5_Package.zip` (on the dev
   laptop, not in git - it contains both `.pte` models) to the Pi and follow its
   `DEPLOYMENT_GUIDE.md`. Its code is byte-identical to commit `5dce800`.
2. **Install with `requirements-pi.txt` from the package - never the repo's `requirements.txt`.**
   On Linux, plain PyPI torch pulls gigabytes of CUDA packages; `requirements-pi.txt` pins the
   CPU-only `torch==2.15.0+cpu` from the PyTorch CPU index. Linux ARM64 wheels exist for every
   pin on Python 3.11-3.13 (checked from index metadata, not installed). numpy 2.5.4 needs
   Python 3.12+, so prefer a Raspberry Pi OS with Python 3.12 or newer.
3. **Main unverified risk:** whether the ARM64 ExecuTorch runtime loads these XNNPACK `.pte`
   files. Run the guide's smoke test first and report the full error if it fails. The
   `FLATC_EXECUTABLE` fix in the README is for exporting on Windows only.
   Also bring a test clip (for example `hyderabad_480p.webm`, or a clip from a fixed camera).
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
