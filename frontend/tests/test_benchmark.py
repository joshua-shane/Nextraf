import pandas as pd
import pytest

from nextraf_pipeline.benchmark import ModelSpec, compare_counts, config_id, run_benchmark
from nextraf_pipeline.core import Detection

TRUTH = {0: 5, 1: 8, 2: 3}
FRAMES = {0: "f0", 1: "f1", 2: "f2"}


class FakeDet:
    """Counts every true person at 640 px but misses one person at 320 px."""

    def __init__(self, res):
        self.res, self.name, self.last_model_ms = res, f"fake{res}", 4.0

    def detect(self, frame):
        n = TRUTH[int(frame[1:])] - (1 if self.res == 320 else 0)
        return [Detection("person", 0.9, (0, 0, 1, 1))] * n + [Detection("person", 0.3, (0, 0, 1, 1))]


class Sensors:
    def temp_c(self): return 50.0
    def throttled(self): return False
    def peak_mem_mb(self): return 123.0


def _run(tmp_path, thresholds=(0.5, 0.2)):
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x" * 2_000_000)
    return run_benchmark(FRAMES, TRUTH, [ModelSpec("fake", str(weights), "fp32")], resolutions=[320, 640],
                         thresholds=list(thresholds), threads=4, factory=lambda spec, res: FakeDet(res),
                         sensors=Sensors())


def test_config_id_format():
    assert config_id("yolo26n", "int8", 320, 0.5) == "yolo26n_int8_320_t0.5"


def test_one_benchmark_row_per_configuration_with_required_columns(tmp_path):
    bench, _ = _run(tmp_path)
    assert len(bench) == 4  # 1 model x 2 resolutions x 2 thresholds
    required = ["config_id", "model", "resolution", "precision", "threshold", "threads", "model_latency_median_ms",
                "model_latency_p95_ms", "e2e_median_ms", "e2e_p95_ms", "fps", "model_size_mb", "peak_mem_mb",
                "mae", "within1_pct", "mean_signed_error"]
    assert all(c in bench.columns for c in required)
    assert bench["model_size_mb"].iloc[0] == pytest.approx(2.0)
    assert bench["peak_mem_mb"].iloc[0] == 123.0 and bench["model_latency_median_ms"].iloc[0] == 4.0


def test_accuracy_metrics_use_the_threshold(tmp_path):
    bench, _ = _run(tmp_path)
    row = bench.set_index("config_id")
    assert row.loc["fake_fp32_640_t0.5", "mae"] == 0 and row.loc["fake_fp32_640_t0.5", "within1_pct"] == 100
    assert row.loc["fake_fp32_320_t0.5", "mae"] == 1 and row.loc["fake_fp32_320_t0.5", "mean_signed_error"] == -1
    # at threshold 0.2 the low-confidence extra person is counted, so 640 over-counts by one
    assert row.loc["fake_fp32_640_t0.2", "mean_signed_error"] == 1


def test_per_frame_eval_has_every_config_on_every_frame(tmp_path):
    bench, ev = _run(tmp_path)
    assert len(ev) == len(bench) * len(TRUTH)
    assert list(ev.columns) == ["config_id", "frame_idx", "pred_count", "true_count", "class"]
    one = ev[ev["config_id"] == "fake_fp32_320_t0.5"].set_index("frame_idx")
    assert list(one["pred_count"]) == [4, 7, 2] and list(one["true_count"]) == [5, 8, 3]


def test_throttled_run_is_flagged(tmp_path):
    class Hot(Sensors):
        def throttled(self): return True
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x")
    bench, _ = run_benchmark(FRAMES, TRUTH, [ModelSpec("fake", str(weights), "fp32")], [320], [0.5], 4,
                             lambda s, r: FakeDet(r), Hot())
    assert bool(bench["throttled"].iloc[0]) is True


def test_compare_counts_reports_agreement_and_mismatches():
    r = compare_counts({0: 5, 1: 8, 2: 3}, {0: 5, 1: 7, 2: 3})
    assert r["frames"] == 3 and r["identical_frames"] == 2 and r["max_abs_diff"] == 1
    assert r["mismatches"] == [(1, 8, 7)] and r["agreement_pct"] == pytest.approx(66.67, abs=0.01)


def test_benchmark_frames_must_have_truth(tmp_path):
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x")
    with pytest.raises(ValueError, match="truth"):
        run_benchmark(FRAMES, {}, [ModelSpec("fake", str(weights), "fp32")], [320], [0.5], 4,
                      lambda s, r: FakeDet(r), Sensors())
    assert isinstance(pd.DataFrame(), pd.DataFrame)


def test_model_spec_reads_a_fixed_export_size():
    from nextraf_pipeline.benchmark import _parse_model
    assert _parse_model("yolo26n=yolo26n_executorch_model:fp32@640") == ModelSpec(
        "yolo26n", "yolo26n_executorch_model", "fp32", 640)
    assert _parse_model("yolo11n=models/yolo11n.pt:int8").imgsz is None
    with pytest.raises(SystemExit):
        _parse_model("yolo26n=m:fp32@big")


def test_a_fixed_size_export_runs_only_at_its_own_size(tmp_path):
    # ExecuTorch exports have a fixed input size; run_nextraf.py's --imgsz must match it, and so must the benchmark
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x")
    specs = [ModelSpec("pte", str(weights), "fp32", 640), ModelSpec("pt", str(weights), "fp32")]
    bench, _ = run_benchmark(FRAMES, TRUTH, specs, [320, 640], [0.5], 4, lambda spec, res: FakeDet(res),
                             sensors=Sensors())
    assert set(zip(bench["model"], bench["resolution"])) == {("pte", 640), ("pt", 320), ("pt", 640)}
