r"""Benchmark and evaluation in one pass.

Runs every (model, resolution) over frames that have a hand-counted truth, then derives every threshold from the
same detections (the threshold changes only the counting, not the inference). Writes benchmark.csv and
per_frame_eval.csv in the layout the viewer reads (see docs/DATA_CONTRACT.md).

  python -m nextraf_pipeline.benchmark --input clip.mp4 --truth truth.csv \
      --model yolo26n=models/yolo26n.pt:fp32 --model yolo11n=models/yolo11n_int8:int8 \
      --resolutions 320,416,640 --thresholds 0.3,0.4,0.5 --threads 4 --out results/

An ExecuTorch export has a fixed input size, so give it with @<size>; that model then runs only at that size:

  python -m nextraf_pipeline.benchmark --input clip.mp4 --truth truth.csv \
      --model yolo26n=yolo26n_executorch_model:fp32@640 --model yolo26n=yolo26n_1280_executorch_model:fp32@1280 \
      --thresholds 0.3,0.4,0.5 --out results/

truth.csv has two columns: frame_idx,true_count (people). Run it on the Pi with nothing else running.
Peak memory is the process-wide peak, so list models from smallest to largest.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .core import count_classes


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    precision: str  # fp32 | int8
    imgsz: int | None = None  # fixed input size of an export (ExecuTorch); None = run at every --resolutions value


def config_id(name: str, precision: str, resolution: int, threshold: float) -> str:
    return f"{name}_{precision}_{resolution}_t{threshold:g}"


def compare_counts(a: dict[int, int], b: dict[int, int]) -> dict:
    """Frame-by-frame agreement between two models' counts (e.g. PyTorch vs ExecuTorch)."""
    common = sorted(set(a) & set(b))
    diffs = [abs(a[i] - b[i]) for i in common]
    same = sum(d == 0 for d in diffs)
    return {"frames": len(common), "identical_frames": same,
            "agreement_pct": round(100 * same / len(common), 2) if common else 0.0,
            "max_abs_diff": max(diffs) if diffs else 0,
            "mean_abs_diff": round(sum(diffs) / len(diffs), 3) if diffs else 0.0,
            "mismatches": [(i, a[i], b[i]) for i in common if a[i] != b[i]]}


class PiSensors:
    """Best-effort readings. Anything unavailable returns None rather than a made-up value."""

    def temp_c(self) -> float | None:
        try:
            return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
        except Exception:
            return None

    def throttled(self) -> bool | None:
        try:
            out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout
            return (int(out.split("=")[1], 16) & 0xF000F) != 0  # under-voltage or throttled, now or since boot
        except Exception:
            return None

    def peak_mem_mb(self) -> float | None:
        try:
            import psutil
            peak = getattr(psutil.Process().memory_info(), "peak_wset", None)  # Windows only
            if peak is not None:
                return round(peak / 1e6, 1)
        except Exception:
            pass
        try:
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # Linux reports KB
        except Exception:
            return None


def _size_mb(path: str) -> float:
    p = Path(path)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return round(total / 1e6, 2)


def run_benchmark(frames: dict[int, object], truth: dict[int, int], specs: list[ModelSpec], resolutions: list[int],
                  thresholds: list[float], threads: int, factory: Callable[[ModelSpec, int], object],
                  sensors=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensors = sensors or PiSensors()
    idx = sorted(i for i in frames if i in truth)
    if not idx:
        raise ValueError("No frame has a matching truth count; check the frame_idx values in truth.csv.")
    bench, per_frame = [], []
    for spec in specs:
        for res in [spec.imgsz] if spec.imgsz else resolutions:
            det = factory(spec, res)
            det.detect(frames[idx[0]])  # warm-up, not timed
            t_before = sensors.temp_c()
            e2e, model, dets = [], [], {}
            for i in idx:
                t0 = time.perf_counter()
                dets[i] = det.detect(frames[i])
                dt = (time.perf_counter() - t0) * 1000
                e2e.append(dt)
                model.append(getattr(det, "last_model_ms", None) or dt)
            t_after, thr_flag, peak = sensors.temp_c(), sensors.throttled(), sensors.peak_mem_mb()
            for th in thresholds:
                cid = config_id(spec.name, spec.precision, res, th)
                pred = {i: count_classes(dets[i], th)["person"] for i in idx}
                err = np.array([pred[i] - truth[i] for i in idx], dtype=float)
                bench.append({
                    "config_id": cid, "model": spec.name, "resolution": res, "precision": spec.precision,
                    "threshold": th, "threads": threads,
                    "model_latency_median_ms": round(float(np.median(model)), 2),
                    "model_latency_p95_ms": round(float(np.percentile(model, 95)), 2),
                    "e2e_median_ms": round(float(np.median(e2e)), 2),
                    "e2e_p95_ms": round(float(np.percentile(e2e, 95)), 2),
                    "fps": round(1000 / float(np.median(e2e)), 2),
                    "model_size_mb": _size_mb(spec.path), "peak_mem_mb": peak,
                    "mae": round(float(np.abs(err).mean()), 3),
                    "within1_pct": round(float((np.abs(err) <= 1).mean() * 100), 1),
                    "mean_signed_error": round(float(err.mean()), 3),
                    "temp_before_c": t_before, "temp_after_c": t_after, "throttled": thr_flag})
                per_frame += [{"config_id": cid, "frame_idx": i, "pred_count": pred[i], "true_count": truth[i],
                               "class": "person"} for i in idx]
    cols = ["config_id", "frame_idx", "pred_count", "true_count", "class"]
    return pd.DataFrame(bench), pd.DataFrame(per_frame, columns=cols)


def read_truth(path: str | Path) -> dict[int, int]:
    df = pd.read_csv(path)
    if not {"frame_idx", "true_count"} <= set(df.columns):
        raise SystemExit("truth.csv needs the columns frame_idx,true_count")
    return {int(r.frame_idx): int(r.true_count) for r in df.itertuples()}


def _parse_model(text: str) -> ModelSpec:
    name, _, rest = text.partition("=")
    path, _, precision = rest.rpartition(":")
    precision, _, size = precision.partition("@")
    if not name or not path or precision not in ("fp32", "int8") or (size and not size.isdigit()):
        raise SystemExit(f"--model must look like name=path:fp32 or name=path:int8, optionally with @<export size> "
                         f"(name=path:fp32@640), got {text!r}")
    return ModelSpec(name, path, precision, int(size) if size else None)


def _write_device_json(out: Path, threads: int) -> None:
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        ram = None
    try:
        board = Path("/proc/device-tree/model").read_text().strip("\x00 \n")
    except Exception:
        board = platform.node() or "unknown"
    info = {"board": board, "ram_gb": ram, "os": platform.platform(), "threads": threads}
    (out / "device.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _named(det, spec: ModelSpec):
    det.name = spec.name
    return det


def main() -> None:
    from .run import UltralyticsDetector, _frames
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="name=path:fp32|int8, plus @<size> for a fixed-size export such as ExecuTorch (repeatable)")
    ap.add_argument("--resolutions", default="320,416,640")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="recorded in benchmark.csv")
    ap.add_argument("--out", default="results/")
    a = ap.parse_args()
    truth = read_truth(a.truth)
    it, _, _ = _frames(Path(a.input))
    frames = {i: f for i, f in enumerate(it) if i in truth}
    specs = [_parse_model(m) for m in a.model]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench, per_frame = run_benchmark(
        frames, truth, specs, [int(x) for x in a.resolutions.split(",")], [float(x) for x in a.thresholds.split(",")],
        a.threads, lambda spec, res: _named(UltralyticsDetector(spec.path, res), spec))
    bench.to_csv(out / "benchmark.csv", index=False)
    per_frame.to_csv(out / "per_frame_eval.csv", index=False)
    _write_device_json(out, a.threads)
    best = bench.sort_values(["mae", "e2e_median_ms"]).iloc[0]
    print(f"{len(bench)} configurations on {len(frames)} frames. Lowest MAE: {best.config_id} "
          f"(MAE {best.mae}, {best.e2e_median_ms} ms/frame)")
    print(f"wrote benchmark.csv, per_frame_eval.csv, device.json in {out}")


if __name__ == "__main__":
    main()
