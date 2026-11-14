from nextraf_common import hashlog
from nextraf_viz import crowding, io, trust
from sample_data import make_sample


def test_sample_folder_matches_the_data_contract(tmp_path):
    make_sample.build(tmp_path)
    res = io.load_results(tmp_path)
    assert len(res.counts) == 20
    assert len(res.benchmark) == 30
    assert res.per_frame.groupby("config_id").size().eq(20).all() and res.per_frame["config_id"].nunique() == 30
    assert len(res.samples) == 2 and res.device["board"]
    assert res.site["los_bands_are_placeholders"] is True
    assert set(res.site["adaptive"].values()) - {15} <= set(res.benchmark["config_id"])


def test_sample_is_deterministic(tmp_path):
    make_sample.build(tmp_path / "a")
    make_sample.build(tmp_path / "b")
    for name in ("counts.csv", "benchmark.csv", "per_frame_eval.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_sample_counts_have_the_designed_features(tmp_path):
    make_sample.build(tmp_path)
    c = io.load_counts(tmp_path / "counts.csv")
    assert c["person"].min() >= 5 and c["person"].max() <= 48
    assert (~c["camera_ok"]).sum() == 1
    assert crowding.stability_flags(c["person"], window=5, k=3).sum() == 1
    assert c["vehicles_on_footpath"].dropna().between(0, 4).all()
    assert trust.chain_status(tmp_path / "counts.csv")["state"] == "valid"


def test_tampering_with_sample_counts_is_detected(tmp_path):
    make_sample.build(tmp_path)
    p = tmp_path / "counts.csv"
    lines = p.read_text(encoding="utf-8").split("\n")
    cells = lines[6].split(",")
    cells[1] = str(int(cells[1]) + 5)
    lines[6] = ",".join(cells)
    p.write_text("\n".join(lines), encoding="utf-8", newline="")
    assert hashlog.verify(p) == (False, 7)


def test_sample_benchmark_tells_the_designed_story(tmp_path):
    make_sample.build(tmp_path)
    b = io.load_benchmark(tmp_path / "benchmark.csv").set_index("config_id")
    assert b["throttled"].sum() == 1
    assert b["power_w"].notna().all()
    fast, slow = b.loc["yolo26n_int8_320_t0.5"], b.loc["yolo26n_fp32_320_t0.5"]
    assert fast["e2e_median_ms"] < slow["e2e_median_ms"] and fast["mae"] >= slow["mae"]
    hi = b.loc["yolo26n_fp32_640_t0.5"]
    assert hi["e2e_median_ms"] > slow["e2e_median_ms"] and hi["mae"] < slow["mae"]
