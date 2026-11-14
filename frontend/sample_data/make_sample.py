"""Generate the bundled sample ``results`` folder (synthetic: no real footage, no real people).

Run:  python frontend/sample_data/make_sample.py [output_dir]
The output follows docs/DATA_CONTRACT.md. All numbers are invented for the demo.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nextraf_common import hashlog  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "results_sample"
SEED = 2026
N_FRAMES = 20
STEP_S = 142  # 20 frames from 15:30:00 to 16:14:58

# Hand-counted people per frame (ground truth), rising into school dismissal.
TRUE_COUNTS = [6, 7, 5, 8, 9, 11, 10, 12, 13, 14, 15, 15, 14, 13, 13, 16, 19, 22, 25, 25]
SPIKE_FRAME, SPIKE_VALUE = 8, 48      # implausible detection at 15:48:56
NO_CAMERA_FRAME = 11                  # frame too dark/blocked
RESOLUTIONS = (320, 480, 640)
PRECISIONS = ("fp32", "int8")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
FAST_ID, ACCURATE_ID = "yolo26n_int8_320_t0.5", "yolo26n_fp32_640_t0.5"
SWITCH_COUNT = 15

# (model_ms, e2e_ms) per (precision, resolution); the threshold does not change latency.
LATENCY = {("fp32", 320): (62, 95), ("fp32", 480): (128, 170), ("fp32", 640): (225, 275),
           ("int8", 320): (34, 62), ("int8", 480): (70, 105), ("int8", 640): (125, 170)}
PEAK_MEM = {("fp32", 320): 210, ("fp32", 480): 260, ("fp32", 640): 330,
            ("int8", 320): 150, ("int8", 480): 185, ("int8", 640): 235}
SIZE_MB = {"fp32": 10.2, "int8": 3.1}
POWER_W = {("fp32", 320): 6.6, ("fp32", 480): 7.1, ("fp32", 640): 7.8,
           ("int8", 320): 5.7, ("int8", 480): 6.0, ("int8", 640): 6.4}
BASE_UNDERCOUNT = {320: 0.15, 480: 0.08, 640: 0.04}


def config_id(precision: str, res: int, thr: float) -> str:
    return f"yolo26n_{precision}_{res}_t{thr:.1f}"


def _predict(true: np.ndarray, precision: str, res: int, thr: float, rng: np.random.Generator) -> np.ndarray:
    """Synthetic detector: undercounts more in dense frames, at low resolution, int8 and high thresholds."""
    rate = BASE_UNDERCOUNT[res] + (0.03 if precision == "int8" else 0.0) + (thr - 0.3) * 0.12
    miss = true * rate * (0.4 + true / 25.0)
    false_pos = max(0.0, 0.5 - thr) * 1.5
    noise = rng.normal(0, 0.35 + 0.25 * (320 / res), size=true.shape)
    return np.clip(np.rint(true - miss + false_pos + noise), 0, None).astype(int)


def _per_frame_and_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    true = np.array(TRUE_COUNTS)
    frames, bench = [], []
    for precision in PRECISIONS:
        for res in RESOLUTIONS:
            model_ms, e2e_ms = LATENCY[(precision, res)]
            for thr in THRESHOLDS:
                cid = config_id(precision, res, thr)
                pred = _predict(true, precision, res, thr, rng)
                err = pred - true
                frames.append(pd.DataFrame({"config_id": cid, "frame_idx": range(N_FRAMES),
                                            "pred_count": pred, "true_count": true, "class": "person"}))
                throttled = (precision, res, thr) == ("fp32", 640, 0.3)
                bench.append({
                    "config_id": cid, "model": "YOLO26n", "resolution": res, "precision": precision,
                    "threshold": thr, "threads": 4,
                    "model_latency_median_ms": model_ms, "model_latency_p95_ms": round(model_ms * 1.14, 1),
                    "e2e_median_ms": e2e_ms, "e2e_p95_ms": round(e2e_ms * 1.14, 1),
                    "fps": round(1000 / e2e_ms, 2), "model_size_mb": SIZE_MB[precision],
                    "peak_mem_mb": PEAK_MEM[(precision, res)],
                    "mae": round(float(np.abs(err).mean()), 3),
                    "within1_pct": round(100 * float((np.abs(err) <= 1).mean()), 1),
                    "mean_signed_error": round(float(err.mean()), 3),
                    "temp_before_c": 47.5, "temp_after_c": 83.0 if throttled else round(55 + e2e_ms / 25, 1),
                    "throttled": throttled, "power_w": POWER_W[(precision, res)],
                    "energy_j_per_frame": round(POWER_W[(precision, res)] * e2e_ms / 1000, 3),
                })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(bench)


def _write_counts(path: Path, per_frame: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED + 1)
    fast = per_frame[per_frame["config_id"] == FAST_ID].set_index("frame_idx")["pred_count"]
    accurate = per_frame[per_frame["config_id"] == ACCURATE_ID].set_index("frame_idx")["pred_count"]
    cols = ["time", "person", "bicycle", "car", "motorcycle", "bus", "truck", "vehicles_total",
            "vehicles_on_footpath", "cars_on_footpath", "frame_idx", "config_id", "model_used",
            "mode", "camera_ok", "count_low", "count_high"]
    start = datetime(2026, 11, 14, 15, 30, 0)
    prev_count = 0
    path.unlink(missing_ok=True)
    for i in range(N_FRAMES):
        model_used = ACCURATE_ID if prev_count >= SWITCH_COUNT else FAST_ID  # adaptive switching
        person = SPIKE_VALUE if i == SPIKE_FRAME else int((accurate if model_used == ACCURATE_ID else fast)[i])
        late = i >= 9
        motorcycle, car = int(rng.integers(1, 4)), int(rng.integers(0, 3))
        bicycle, bus, truck = int(rng.integers(0, 2)), int(i in (6, 17)), int(i in (4, 15))
        total = bicycle + car + motorcycle + bus + truck
        if i >= 15:  # dismissal rush: many vehicles at the same time as many people
            motorcycle, total = motorcycle + 3, total + 3
        on_fp = int(rng.integers(2, 5)) if late else int(rng.integers(0, 2))
        cars_fp = 1 if i == 12 else 0
        ok = i != NO_CAMERA_FRAME
        row = {
            "time": (start + timedelta(seconds=STEP_S * i)).strftime("%H:%M:%S"),
            "person": person if ok else 0, "bicycle": bicycle, "car": car, "motorcycle": motorcycle,
            "bus": bus, "truck": truck, "vehicles_total": total,
            "vehicles_on_footpath": min(on_fp, total), "cars_on_footpath": min(cars_fp, on_fp),
            "frame_idx": i, "config_id": FAST_ID,
            "model_used": model_used,
            "mode": "deployment", "camera_ok": ok,
            "count_low": max(person - 1, 0) if ok else None, "count_high": person + 1 if ok else None,
        }
        hashlog.append_row(path, row, columns=cols)
        prev_count = person if ok else prev_count


def _site() -> dict:
    colours = ["#cfe8f3", "#a3d3e0", "#f3e3a0", "#f2c27a", "#e8955f", "#c8553d"]
    spans = [(5.0, None), (4.0, 5.0), (3.0, 4.0), (2.0, 3.0), (1.0, 2.0), (0.0, 1.0)]
    letters = "ABCDEF"
    bands = [{"name": f"PLACEHOLDER {letters[i]}" + (" (best)" if i == 0 else " (worst)" if i == 5 else ""),
              "min_m2_per_person": lo, "max_m2_per_person": hi, "color": colours[i]}
             for i, (lo, hi) in enumerate(spans)]
    return {
        "site_name": "Example Primary School – main gate", "region": "IN", "area_m2": 20.0,
        "area_is_assumed": True, "time_window": "15:30–16:15",
        "camera_note": "Synthetic sample data. Fixed webcam, 3 m high, facing the gate (assumed)",
        "los_bands": bands, "los_bands_are_placeholders": True,
        "vehicle_footprint_m2": {"two_wheeler": 1.5, "car": 7.0},
        # Sample-only surge thresholds: the 20-frame sample is far sparser than a real run.
        "surge": {"warn_density_ppm2": 1.2, "warn_rise_ppm2_per_min": 0.05,
                  "critical_density_ppm2": 4.0, "window_frames": 5},
        "conflict": {"min_people": 15, "min_vehicles": 5},
        "adaptive": {"fast_config_id": FAST_ID, "accurate_config_id": ACCURATE_ID, "switch_count": SWITCH_COUNT},
        "hardware": {"os_overhead_mb": 350, "headroom_fraction": 0.2, "currency": "INR",
                     "boards": [{"ram_gb": 1, "price": 5044}, {"ram_gb": 2, "price": 7869},
                                {"ram_gb": 4, "price": 13409}, {"ram_gb": 8, "price": 19999}],
                     "prices_note": "Example retail prices, Robocraze India, Nov 2026 – edit as needed"},
    }


def _sample_image(path: Path, seed: int, n: int) -> None:
    """Placeholder frame: grey scene with simple stick shapes and boxes (no real people)."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (640, 360), (200, 205, 210))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 250, 640, 360), fill=(150, 150, 155))
    for _ in range(n):
        x, y = int(rng.integers(40, 580)), int(rng.integers(150, 300))
        d.ellipse((x, y, x + 14, y + 14), fill=(70, 70, 80))
        d.rectangle((x + 3, y + 14, x + 11, y + 46), fill=(70, 70, 80))
        d.rectangle((x - 8, y - 6, x + 22, y + 52), outline=(220, 40, 40), width=2)
    d.text((10, 10), f"PLACEHOLDER IMAGE - {n} boxes (synthetic shapes)", fill=(20, 20, 20))
    img.save(path)


def build(out_dir: str | Path = DEFAULT_OUT) -> Path:
    """Write the full sample results folder to ``out_dir`` (replaced if it exists)."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "samples").mkdir(parents=True)
    per_frame, bench = _per_frame_and_benchmark()
    per_frame.to_csv(out / "per_frame_eval.csv", index=False, lineterminator="\n")
    bench.to_csv(out / "benchmark.csv", index=False, lineterminator="\n")
    _write_counts(out / "counts.csv", per_frame)
    (out / "site.example.json").write_text(json.dumps(_site(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "device.json").write_text(json.dumps(
        {"board": "Raspberry Pi 5 (sample)", "ram_gb": 4, "os": "Raspberry Pi OS 64-bit (sample)",
         "executorch_version": "placeholder", "cooler": True, "threads": 4}, indent=2) + "\n", encoding="utf-8")
    _sample_image(out / "samples" / "validation_01.png", 1, 8)
    _sample_image(out / "samples" / "validation_02.png", 2, 19)
    return out


if __name__ == "__main__":
    print(f"Wrote sample data to {build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)}")
