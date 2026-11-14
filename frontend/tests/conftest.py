import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "frontend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Header of frames.csv as written by the team pipeline's run_nextraf.py (README "Frontend data contract", schema 1).
RUN_HEADER = ("frame,video_t_sec,people_in_roi,people_detected,capacity,occupancy_pct,density_per_m2,crowd_status,"
              "car,motorcycle,bus,truck,vehicles_total,stationary,potential_violations,latency_ms")


@pytest.fixture
def run_folder(tmp_path):
    """Factory for a run_nextraf.py output folder: frames.csv, summary.json and one annotated frame."""
    def make(still: bool = False, density: bool = True, frames: int = 12) -> Path:
        folder = tmp_path / ("run_still" if still else "run")
        folder.mkdir()
        rows = [RUN_HEADER]
        for i in range(frames):
            people = 5 + i
            t = "" if still else f"{i * 0.5:.1f}"
            d = f"{people / 20:.4f}" if density else ""
            car, moto = 3 + i % 4, 2
            rows.append(f"{i},{t},{people},{people + 1},10,{people * 10:.1f},{d},OK,{car},{moto},0,1,{car + moto + 1},"
                        f"0,0,{3830.5 if i == 0 else 212.0}")
        (folder / "frames.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        summary = {"schema_version": 1, "started": "2026-11-14T13:05:42", "finished": "2026-11-14T13:05:49",
                   "source": "frames/" if still else "gate.mp4", "model": "yolo26n_executorch_model", "imgsz": 640,
                   "stride": 1, "crowd_site": "school-gate", "capacity": 10, "frames_processed": frames,
                   "avg_latency_ms": 335.5, "steady_latency_ms": 212.5, "ram_mb": 619,
                   "notes": ["People count is per frame (visible people), not unique individuals.",
                             "Latency measured on a laptop CPU, not on the target device."]}
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        from PIL import Image
        Image.new("RGB", (32, 24), (200, 200, 200)).save(folder / "frame_000003.jpg")
        return folder
    return make
