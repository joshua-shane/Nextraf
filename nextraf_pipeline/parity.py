"""Parity check: do two models (e.g. PyTorch .pt and its ExecuTorch export) give the same counts on the same frames?

  python -m nextraf_pipeline.parity --input frames/ --model-a yolo26n.pt --model-b yolo26n_executorch --threshold 0.5

The brief requires this before any benchmark number is trusted. Exit code 1 if any frame differs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import compare_counts
from .core import count_classes


def main() -> None:
    from .run import UltralyticsDetector, _frames
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=50)
    a = ap.parse_args()
    frames, _, _ = _frames(Path(a.input))
    cache = [f for _, f in zip(range(a.max_frames), frames)]
    counts = []
    for path in (a.model_a, a.model_b):
        det = UltralyticsDetector(path, a.imgsz)
        counts.append({i: count_classes(det.detect(f), a.threshold)["person"] for i, f in enumerate(cache)})
    r = compare_counts(*counts)
    print(f"{r['identical_frames']}/{r['frames']} frames identical ({r['agreement_pct']}%), "
          f"max difference {r['max_abs_diff']}, mean {r['mean_abs_diff']}")
    for i, x, y in r["mismatches"][:10]:
        print(f"  frame {i}: A={x} B={y}")
    raise SystemExit(0 if not r["mismatches"] else 1)


if __name__ == "__main__":
    main()
