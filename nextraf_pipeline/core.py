"""Detector-agnostic counting loop for the Pi / laptop pipeline.

Pure Python: no OpenCV or model imports here, so it is fast to test.
Writes the hash-chained counts.csv the viewer reads, plus inference_summary.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from nextraf_common import hashlog

VEHICLES = ("bicycle", "car", "motorcycle", "bus", "truck")
CLASSES = ("person", *VEHICLES)
COLUMNS = ["time", *CLASSES, "vehicles_total", "frame_idx", "model_used", "mode", "camera_ok"]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class Detector(Protocol):
    name: str

    def detect(self, frame) -> list[Detection]: ...


def count_classes(dets: list[Detection], threshold: float) -> dict[str, int]:
    """Count detections above ``threshold`` per tracked class (people and vehicles only)."""
    counts = {c: 0 for c in CLASSES}
    for d in dets:
        if d.confidence >= threshold and d.label in counts:
            counts[d.label] += 1
    counts["vehicles_total"] = sum(counts[v] for v in VEHICLES)
    return counts


def _clock(seconds: float) -> str:
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{ms:03d}"


def _sample_indices(total: int, wanted: int) -> set[int]:
    wanted = min(wanted, total)
    if wanted <= 0:
        return set()
    if wanted == 1:
        return {0}
    return {round(k * (total - 1) / (wanted - 1)) for k in range(wanted)}


def run_pipeline(frames: Iterable, detector: Detector, out_dir: str | Path, *, fps: float,
                 threshold: float = 0.5, annotate_n: int = 2, total: int | None = None,
                 mode: str = "validation",
                 save_annotated: Callable[[object, list[Detection], Path], None] | None = None) -> dict:
    """Run every frame through the detector and write counts.csv, samples/ and inference_summary.json.

    Deployment mode never saves images: frames are only held in memory.
    The average inference time excludes the first (warm-up) frame; it is reported separately as first_frame_ms.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts_path = out / "counts.csv"
    if counts_path.exists():
        counts_path.unlink()
    if total is None:
        total = len(frames)  # type: ignore[arg-type]
    keep = _sample_indices(total, annotate_n) if mode == "validation" and save_annotated else set()
    (out / "samples").mkdir(exist_ok=True) if keep else None
    infer_s: list[float] = []
    people = 0
    n = 0
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        infer_s.append(time.perf_counter() - t0)
        c = count_classes(dets, threshold)
        people += c["person"]
        hashlog.append_row(counts_path, {"time": _clock(i / fps), **c, "frame_idx": i,
                                         "model_used": detector.name, "mode": mode, "camera_ok": True},
                           columns=COLUMNS)
        if i in keep:
            save_annotated(frame, [d for d in dets if d.confidence >= threshold], out / "samples" / f"frame_{i:05d}.jpg")  # type: ignore[misc]
        n += 1
    # The first call includes model warm-up, so average the rest when there are any.
    steady = infer_s[1:] if n > 1 else infer_s
    avg = sum(steady) / len(steady) if steady else 0.0
    summary = {"model": detector.name, "frames": n, "threshold": threshold, "mode": mode,
               "first_frame_ms": round(infer_s[0] * 1000, 2) if n else 0.0,
               "avg_inference_ms": round(avg * 1000, 2),
               "throughput_fps": round(1 / avg, 2) if avg > 0 else 0.0,
               "total_people": people}
    (out / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
