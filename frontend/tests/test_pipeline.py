import json

import pandas as pd

from nextraf_common import hashlog
from nextraf_pipeline.core import Detection, count_classes, run_pipeline


class FakeDetector:
    name = "fake_model"

    def __init__(self, per_frame):
        self.per_frame = per_frame
        self.i = 0

    def detect(self, frame):
        dets = self.per_frame[self.i]
        self.i += 1
        return dets


def D(label, conf=0.9):
    return Detection(label, conf, (0, 0, 10, 10))


def test_count_classes_applies_threshold_and_totals_vehicles():
    dets = [D("person"), D("person", 0.2), D("car"), D("motorcycle"), D("dog")]
    c = count_classes(dets, threshold=0.5)
    assert c["person"] == 1 and c["car"] == 1 and c["motorcycle"] == 1
    assert c["vehicles_total"] == 2 and c["bus"] == 0


def _run(tmp_path, n=5, annotate_n=2):
    frames = [f"frame{i}" for i in range(n)]
    det = FakeDetector([[D("person")] * (i + 1) for i in range(n)])
    saved = []
    summary = run_pipeline(frames, det, tmp_path, fps=2.0, threshold=0.5, annotate_n=annotate_n,
                           save_annotated=lambda frame, dets, path: (saved.append(path), path.write_bytes(b"x")))
    return summary, saved


def test_pipeline_writes_chained_counts_with_one_row_per_frame(tmp_path):
    _run(tmp_path)
    ok, bad = hashlog.verify(tmp_path / "counts.csv")
    assert ok and bad is None
    df = pd.read_csv(tmp_path / "counts.csv")
    assert list(df["person"]) == [1, 2, 3, 4, 5]
    assert df["model_used"].iloc[0] == "fake_model"
    assert df["time"].iloc[1] == "00:00:00.500" or df["time"].iloc[1].startswith("00:00:00")


def test_pipeline_saves_at_least_two_annotated_samples(tmp_path):
    _, saved = _run(tmp_path, n=6, annotate_n=3)
    assert len(saved) == 3
    assert all(p.parent.name == "samples" and p.exists() for p in saved)


def test_summary_reports_model_frames_and_timing(tmp_path):
    summary, _ = _run(tmp_path)
    on_disk = json.loads((tmp_path / "inference_summary.json").read_text())
    assert on_disk == summary
    assert summary["model"] == "fake_model" and summary["frames"] == 5
    assert summary["avg_inference_ms"] >= 0 and summary["throughput_fps"] > 0
    assert summary["total_people"] == 15


def test_annotated_count_never_exceeds_frames(tmp_path):
    _, saved = _run(tmp_path, n=2, annotate_n=5)
    assert len(saved) == 2


def test_annotated_images_only_get_detections_above_threshold(tmp_path):
    det = FakeDetector([[D("person", 0.9), D("person", 0.1), D("dog", 0.95)]] * 2)
    seen = []
    run_pipeline(["a", "b"], det, tmp_path, fps=1.0, threshold=0.5, annotate_n=1,
                 save_annotated=lambda frame, dets, path: (seen.append(dets), path.write_bytes(b"x")))
    assert [d.confidence for d in seen[0]] == [0.9, 0.95] and all(d.confidence >= 0.5 for d in seen[0])


def test_summary_excludes_warmup_first_frame_from_average(tmp_path, monkeypatch):
    import nextraf_pipeline.core as core
    ticks = iter([0.0, 1.0,  1.0, 1.01,  2.0, 2.01])  # frame 0 takes 1.0 s, frames 1 and 2 take 0.01 s
    monkeypatch.setattr(core.time, "perf_counter", lambda: next(ticks))
    det = FakeDetector([[D("person")]] * 3)
    s = run_pipeline(["a", "b", "c"], det, tmp_path, fps=1.0, annotate_n=0, total=3)
    assert s["first_frame_ms"] == 1000.0 and s["avg_inference_ms"] == 10.0
