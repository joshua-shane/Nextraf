# NEXTRAF - Raspberry Pi 5 deployment guide

> **Not tested on a Raspberry Pi.** Everything in this package was run only on Windows 11 x64
> (Python 3.13.3). The commands below are the intended Pi procedure, derived from package
> metadata and standard Raspberry Pi OS practice. Treat each step as unverified until you run it.

## 1. What was verified vs. not

| Item | State |
|---|---|
| Pipeline runs end to end from an extracted copy of this package | Verified (Windows only) |
| Model files: `model.pte` format `ET12`, XNNPACK-delegated, input 1280x1280, fp32, 80 COCO classes | Verified by inspecting the files |
| aarch64 Linux wheels exist for torch, torchvision, executorch, numpy, opencv-python, psutil | Verified from PyPI / PyTorch CPU index metadata (versions pinned in `requirements-pi.txt`) |
| Installing those wheels on a Pi | **Unverified** |
| ExecuTorch aarch64 runtime includes the XNNPACK backend and loads this `.pte` | **Unverified** - the key hardware/runtime risk |
| Speed on Pi 5 | **Unknown.** Windows laptop (i7-13650HX): ~215-330 ms/frame at 1280, ~600 MB RAM. The Pi 5 will be substantially slower - measure it |
| Camera input | **Not supported by the code** (see section 6) |
| Detection accuracy on real footage | Not validated |

## 2. Prerequisites
- Raspberry Pi 5 (4 GB RAM or more recommended; ~600 MB used on Windows), 64-bit Raspberry Pi OS (`uname -m` must print `aarch64`).
- **Python >= 3.12 recommended** (3.13 is what was tested). `python3 --version` to check. Raspberry Pi OS based on Debian 13 "Trixie" ships Python 3.13; Debian 12 "Bookworm" ships 3.11, where numpy 2.5.4 cannot install (see `requirements-pi.txt`) - that setup is untested.
- Active cooling recommended; sustained inference will throttle a bare Pi 5.
- ~3 GB free disk (torch is large). Internet access for the install.
- System packages:
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0 unzip
```
(`libgl1`/`libglib2.0-0` are needed by the OpenCV wheel on Pi OS Lite.)

## 3. Install
```bash
unzip NEXTRAF_RaspberryPi5_Package.zip && cd NEXTRAF_RaspberryPi5
python3 -m venv .venv            # required: Pi OS blocks system-wide pip installs (PEP 668)
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt
```
A dry-run resolution of this file for Linux aarch64 (Python 3.12 and 3.13, run on Windows with
pip's `--platform` option, nothing installed) resolved 63 packages, all aarch64 or pure-Python
wheels, except `antlr4-python3-runtime 4.9.3` (pulled in by executorch -> hydra-core), which has no
wheel and is **built from source by pip on the Pi** - it is pure Python, so this is quick and needs
no compiler. `nvidia-ml-py` also appears; it is a tiny pure-Python NVML binding (an ultralytics
dependency), not a CUDA library. Python 3.11 could not be dry-run checked.

`requirements-pi.txt` pins `torch==2.15.0+cpu` from the PyTorch CPU index on purpose: the plain
PyPI torch declares CUDA dependencies for every Linux machine. If `pip` starts downloading
`nvidia-*` or `cuda-*` packages, **cancel** (Ctrl+C) and check you are using that file.

## 4. Model path configuration
Models are folders next to the scripts; there is nothing else to configure.

| Model | Folder | `--imgsz` |
|---|---|---|
| **Recommended** | `yolo26n_1280_executorch_model` (`model.pte` + `metadata.yaml`) | `1280` |
| Faster, lower accuracy | `yolo26n_executorch_model` | `640` |

Pass `--model <folder>` and the matching `--imgsz`. **A mismatch fails or gives wrong detections.**
Paths are relative to the folder you launch from, so `cd` into the package first, or pass absolute
paths. On a test image the 640 model found 1 person where the 1280 model found 10, so use 640 only
if 1280 is too slow, and expect lower counts. No `.pt` file is needed at runtime.

**Quick model test - run this first** (checks platform, imports, that ExecuTorch has the
XNNPACK backend, then loads both models and runs one blank-frame inference each; no media,
no internet, writes nothing):
```bash
python pi_smoke_test.py
```
Expected last line: `RESULT: PASS (...)`. On `RESULT: FAIL`, stop and send the whole output back.

One-line alternative (1280 model only):
```bash
python -c "import numpy as np; from nextraf.detector import Detector; \
d=Detector('yolo26n_1280_executorch_model', imgsz=1280); print(d.detect(np.zeros((720,1280,3),np.uint8)))"
```
Expected: `([], <milliseconds>)`. If this fails, stop and see Troubleshooting.

## 5. Configuration files
- `config/crowd_site.json`: `capacity` (max people), `warn_ratio` (default 0.8), optional `roi` (polygon, pixels) and `area_m2` (real ground area; density is reported only if set), `site` name. Shipped values (`capacity: 10`, no ROI) are test placeholders - set real ones.
- Road line for parking: copy `config/road_line.example.json` to `config/road_line.json` **and redraw it for your camera view** (the example numbers are meaningless for another scene). Parking alerts need a fixed camera and are off without `--road-config`.
- `draw_roi.py` / `draw_road_line.py` use `cv2.imshow` and need a display. Easiest: run them on a desktop with a saved frame, e.g. `python draw_road_line.py frame.jpg config/road_line.json`, then copy the JSON to the Pi (the line is scaled automatically if the frame size differs).

## 6. Video / camera input
`nextraf/sources.py` accepts exactly: **an image file, a folder of images, or a video file** (anything OpenCV can decode; mp4 and webm were tested on Windows). **Webcam indices, CSI cameras and RTSP/HTTP URLs are rejected** (`Source not found`). There is no live-camera mode.

Workable today: record clips on the Pi and process them.
```bash
rpicam-hello --list-cameras                      # check the camera is detected
rpicam-vid -t 30000 --codec libav -o clip.mp4    # 30 s clip (command from Pi OS docs; untested here)
python run_nextraf.py --source clip.mp4 ...
```
For true live use, `sources.py` needs a small addition (for a USB/V4L2 camera, roughly:
accept `source.isdigit()` and call `cv2.VideoCapture(int(source))`; CSI cameras would need
`picamera2` or a GStreamer pipeline). This is **untested, not implemented** - a development task for the team.

Choose `--stride` from the video's FPS: `--stride 30` on a 30 fps clip analyses one frame per second.
Frame times (`video_t_sec`) come from the video's FPS; stationary-time (parking) logic depends on them.

## 7. Launch commands
Run from inside the package folder with the venv active (`source .venv/bin/activate`).
```bash
# Main app: crowd + vehicles + parking, unified output for the dashboard
python run_nextraf.py --source clip.mp4 \
  --crowd-config config/crowd_site.json --road-config config/road_line.json \
  --model yolo26n_1280_executorch_model --imgsz 1280 \
  --stride 30 --mode deployment --out out/run1

# Single image, with annotated output
python run_nextraf.py --source photo.jpg --crowd-config config/crowd_site.json \
  --model yolo26n_1280_executorch_model --imgsz 1280 --out out/photo

# Individual modules (older, separate output formats)
python run_crowd.py    --source clip.mp4 --config config/crowd_site.json --model yolo26n_1280_executorch_model --imgsz 1280 --stride 30
python run_vehicles.py --source clip.mp4 --config config/road_line.json --model yolo26n_1280_executorch_model --imgsz 1280 --stride 30
```
Options: `--conf` (default 0.4), `--mode validation|deployment` (`deployment` writes no images - counts
and alerts only), `--save-every N`, `--out DIR`. Output files are described in `FRONTEND_SCHEMA.md`;
the dashboard should poll `<out>/latest.json`. Note that `run_nextraf.py` defaults to the 1280 model,
but `run_crowd.py` / `run_vehicles.py` default to the **640** model and `--imgsz 640`.

Long runs: use `tmux` or `nohup python run_nextraf.py ... &` so the run survives an SSH disconnect.
Check throttling with `vcgencmd get_throttled` (`0x0` = fine).

## 8. Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `externally-managed-environment` | You ran pip outside a venv. Use the venv from section 3. |
| pip downloads `nvidia-*` / `cuda-*` (GBs) | Plain PyPI torch is being used. Cancel; use `requirements-pi.txt` (`+cpu` pins, PyTorch CPU index). |
| `No matching distribution` for `torch==2.15.0+cpu` | Not 64-bit OS (`uname -m` must be `aarch64`), Python version without a wheel (wheels exist for 3.11-3.15), or no internet. |
| numpy `requires Python >=3.12` | You are on Python 3.11. Use Python 3.12+, or accept the untested unpinned-numpy fallback. |
| `ImportError: libGL.so.1` / `libgthread` | `sudo apt install libgl1 libglib2.0-0` |
| Ultralytics tries to `pip install executorch` at startup | The ExecuTorch import/check failed. Run `python -c "from executorch.runtime import Runtime; print(Runtime.get())"` and fix the install; an offline Pi cannot auto-install. |
| ExecuTorch error naming `XnnpackBackend` (not registered / failed to init) | The aarch64 runtime may lack XNNPACK (**the main unverified risk**). Report the full error. Fallbacks to evaluate (not provided here): export the model in another format Ultralytics can load (e.g. NCNN/ONNX) on a dev machine - `Detector` accepts any Ultralytics-loadable path - then test it. |
| Error mentioning tensor shape / `1280` / `640` | `--imgsz` does not match the model folder. |
| `Source not found` / `Could not open video` | Wrong path, or a camera/URL (unsupported). For codec problems, re-encode: `ffmpeg -i in.x -c:v libx264 out.mp4` (`sudo apt install ffmpeg`). |
| Very slow / first frame takes seconds | First frame includes model warm-up (~2-5 s on Windows). Judge speed by the `steady_latency_ms` in `summary.json`. Raise `--stride`, or try the 640 model. |
| Process killed / out of memory | Check `free -h`; close other apps; use the 640 model or a smaller video. |
| `draw_*.py`: cannot connect to display / `cv2.imshow` error | No GUI on the Pi. Draw on a desktop and copy the JSON. |
| No parking alerts | Needs `--road-config`, a **video** (not a single image), a fixed camera, and vehicles stationary >= 5 s with >= 15 % of their footprint over the line. |
| `.pt` file missing | Not needed. Only the `*_executorch_model` folders are used at runtime. |

## 9. Remaining limitations
Unvalidated on Pi hardware; no live camera input; accuracy unvalidated on real footage (only a
static test image and one short sample video were used on Windows); per-frame people counts are
not unique people; IoU tracking is basic; parking alerts are potential, not legal findings; the
shipped crowd config is a placeholder.
