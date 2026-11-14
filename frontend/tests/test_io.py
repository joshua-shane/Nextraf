import json

import pandas as pd
import pytest

from nextraf_viz import io


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_counts_accepts_count_alias_and_ignores_extra_columns(tmp_path):
    p = _w(tmp_path, "counts.csv", "time,count,whatever\n15:30:00,4,x\n15:30:10,6,y\n")
    df = io.load_counts(p)
    assert list(df["person"]) == [4, 6]
    assert df["camera_ok"].all()


def test_counts_parses_hms_and_iso_times(tmp_path):
    hms = io.load_counts(_w(tmp_path, "a.csv", "time,person\n15:30:00,1\n15:31:00,2\n"))
    iso = io.load_counts(_w(tmp_path, "b.csv", "time,person\n2026-11-14T15:30:00,1\n2026-11-14T15:31:00,2\n"))
    assert hms["minutes"].tolist() == [0.0, 1.0]
    assert iso["minutes"].tolist() == [0.0, 1.0]


def test_counts_missing_person_column_gives_friendly_error(tmp_path):
    p = _w(tmp_path, "counts.csv", "time,car\n15:30:00,1\n")
    with pytest.raises(io.DataError, match="person.*count"):
        io.load_counts(p)


def test_counts_missing_time_column(tmp_path):
    with pytest.raises(io.DataError, match="time"):
        io.load_counts(_w(tmp_path, "counts.csv", "person\n1\n"))


def test_counts_rejects_negative_and_non_integer(tmp_path):
    with pytest.raises(io.DataError, match="line 3"):
        io.load_counts(_w(tmp_path, "c.csv", "time,person\n15:30:00,1\n15:30:10,-2\n"))
    with pytest.raises(io.DataError, match="person"):
        io.load_counts(_w(tmp_path, "d.csv", "time,person\n15:30:00,abc\n"))


def test_counts_missing_file_is_a_data_error(tmp_path):
    with pytest.raises(io.DataError, match="counts.csv"):
        io.load_counts(tmp_path / "counts.csv")


def test_camera_ok_false_blanks_counts_to_nan(tmp_path):
    p = _w(tmp_path, "c.csv", "time,person,camera_ok\n15:30:00,5,true\n15:30:10,0,false\n")
    df = io.load_counts(p)
    assert df["camera_ok"].tolist() == [True, False]
    assert pd.isna(df.loc[1, "person"]) and df.loc[0, "person"] == 5


def test_vehicles_total_is_derived_when_absent(tmp_path):
    p = _w(tmp_path, "c.csv", "time,person,car,bus\n15:30:00,5,2,1\n")
    assert io.load_counts(p).loc[0, "vehicles_total"] == 3


BENCH_HEAD = ("config_id,model,resolution,precision,threshold,threads,model_latency_median_ms,"
              "model_latency_p95_ms,e2e_median_ms,e2e_p95_ms,fps,model_size_mb,peak_mem_mb,mae,"
              "within1_pct,mean_signed_error")
BENCH_ROW = "a,YOLO26n,320,fp32,0.5,4,50,60,80,95,12.5,9.0,150,0.8,70,-0.2"


def test_benchmark_loads_and_flags_untrustworthy(tmp_path):
    p = _w(tmp_path, "benchmark.csv", BENCH_HEAD + ",temp_after_c,throttled\n"
           + BENCH_ROW + ",70,false\n" + BENCH_ROW.replace("a,", "b,", 1) + ",85,false\n"
           + BENCH_ROW.replace("a,", "c,", 1) + ",60,true\n")
    df = io.load_benchmark(p)
    assert df.set_index("config_id")["not_trustworthy"].to_dict() == {"a": False, "b": True, "c": True}


def test_benchmark_missing_required_columns_listed(tmp_path):
    p = _w(tmp_path, "benchmark.csv", "config_id,model\na,x\n")
    with pytest.raises(io.DataError, match="resolution"):
        io.load_benchmark(p)


def test_per_frame_defaults_class_to_person(tmp_path):
    p = _w(tmp_path, "pf.csv", "config_id,frame_idx,pred_count,true_count\na,0,3,4\n")
    assert io.load_per_frame(p).loc[0, "class"] == "person"


def test_site_json_fills_defaults_and_validates_region(tmp_path):
    p = _w(tmp_path, "site.json", json.dumps({"site_name": "X", "region": "UK"}))
    site = io.load_site(p)
    assert site["region"] == "UK" and site["area_m2"] is None
    assert site["surge"]["window_frames"] == 5
    with pytest.raises(io.DataError, match="region"):
        io.load_site(_w(tmp_path, "s2.json", json.dumps({"region": "FR"})))


def test_site_json_bad_json_is_friendly(tmp_path):
    with pytest.raises(io.DataError, match="site.json"):
        io.load_site(_w(tmp_path, "site.json", "{nope"))


def test_load_results_only_counts_required_and_notes_missing(tmp_path):
    _w(tmp_path, "counts.csv", "time,person\n15:30:00,1\n")
    res = io.load_results(tmp_path)
    assert res.benchmark is None and res.per_frame is None and res.device is None
    assert any("benchmark.csv" in n for n in res.notes)
    assert res.site["region"] == "IN"


def test_load_results_counts_missing_raises(tmp_path):
    with pytest.raises(io.DataError, match="counts.csv"):
        io.load_results(tmp_path)


def test_resolve_falls_back_to_sample_when_folder_missing(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    path, is_sample = io.resolve_results_dir(tmp_path / "nope", sample)
    assert path == sample and is_sample
    real = tmp_path / "real"
    real.mkdir()
    assert io.resolve_results_dir(real, sample) == (real, False)


def test_resolve_rejects_folders_outside_the_allowed_base(tmp_path):
    base = tmp_path / "results"
    (base / "siteA").mkdir(parents=True)
    outside = tmp_path / "secret"
    outside.mkdir()
    sample = tmp_path / "sample"
    sample.mkdir()
    assert io.resolve_results_dir(base / "siteA", sample, base=base) == ((base / "siteA").resolve(), False)
    with pytest.raises(io.DataError, match="inside"):
        io.resolve_results_dir(outside, sample, base=base)
    with pytest.raises(io.DataError, match="inside"):
        io.resolve_results_dir(base / ".." / "secret", sample, base=base)


def test_resolve_missing_folder_still_falls_back_to_sample_with_base(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    assert io.resolve_results_dir(tmp_path / "results", sample, base=tmp_path / "results") == (sample, True)


# ---- uploads ----

def test_save_uploads_accepts_only_known_files_and_flattens_names(tmp_path):
    files = [("counts.csv", b"time,person\n15:30:00,1\n"), ("../../evil.csv", b"x"), ("clip.mp4", b"video"),
             ("site.json", b"{}"), ("notes.txt", b"hi"), ("sub/dir/benchmark.csv", b"a")]
    saved, rejected = io.save_uploads(files, tmp_path / "up")
    assert sorted(saved) == ["benchmark.csv", "counts.csv", "site.json"]
    assert sorted(rejected) == ["../../evil.csv", "clip.mp4", "notes.txt"]
    assert not (tmp_path / "evil.csv").exists() and (tmp_path / "up" / "counts.csv").is_file()


def test_save_uploads_puts_images_under_samples(tmp_path):
    saved, rejected = io.save_uploads([("annotated.PNG", b"1234"), ("b.jpg", b"5")], tmp_path)
    assert sorted(saved) == ["samples/annotated.PNG", "samples/b.jpg"] and not rejected
    assert (tmp_path / "samples" / "annotated.PNG").read_bytes() == b"1234"


def test_save_uploads_enforces_a_size_limit(tmp_path):
    saved, rejected = io.save_uploads([("counts.csv", b"x" * 50)], tmp_path, max_bytes=10)
    assert not saved and rejected == ["counts.csv"]


# ---- run_nextraf.py output folders (the team pipeline: frames.csv + summary.json, schema 1) ----

def test_run_folder_loads_people_vehicles_times_and_model(run_folder):
    c = io.load_results(run_folder()).counts
    assert c["person"].tolist()[:3] == [5, 6, 7]  # people_in_roi
    assert c["car"].tolist()[:2] == [3, 4] and c["vehicles_total"].iloc[0] == 6
    assert c["time"].iloc[0] == pd.Timestamp("2026-11-14T13:05:42")
    assert c["time"].iloc[1] - c["time"].iloc[0] == pd.Timedelta(seconds=0.5)
    assert c["model_used"].iloc[0] == "yolo26n_executorch_model" and c["camera_ok"].all()


def test_run_folder_fills_site_device_and_samples(run_folder):
    res = io.load_results(run_folder())
    assert res.site["site_name"] == "school-gate"
    assert res.site["area_m2"] == pytest.approx(20.0)  # people_in_roi / density_per_m2
    assert res.device["model"] == "yolo26n_executorch_model" and res.device["steady_latency_ms"] == 212.5
    assert res.device["throughput_fps"] == pytest.approx(1000 / 212.5, abs=0.01)
    assert [p.name for p in res.samples] == ["frame_000003.jpg"]


def test_run_folder_of_still_images_spaces_frames_one_second_apart(run_folder):
    res = io.load_results(run_folder(still=True))
    assert (res.counts["time"].diff().dropna() == pd.Timedelta(seconds=1)).all()
    assert any("still images" in n for n in res.notes)


def test_run_folder_without_density_leaves_the_area_unset(run_folder):
    assert io.load_results(run_folder(density=False)).site["area_m2"] is None


def test_folder_without_counts_or_run_files_names_both_options(tmp_path):
    with pytest.raises(io.DataError, match="frames.csv"):
        io.load_results(tmp_path)


def test_is_results_folder_and_uploads_accept_run_files(tmp_path, run_folder):
    assert io.is_results_folder(run_folder()) and not io.is_results_folder(tmp_path)
    saved, rejected = io.save_uploads([("frames.csv", b"x"), ("summary.json", b"{}")], tmp_path / "up")
    assert sorted(saved) == ["frames.csv", "summary.json"] and not rejected


def test_time_format_shows_seconds_for_short_runs():
    short = pd.Series(pd.to_datetime(["2026-11-14 16:20:00", "2026-11-14 16:20:15"]))
    long = pd.Series(pd.to_datetime(["2026-11-14 15:30:00", "2026-11-14 16:15:00"]))
    assert io.time_format(short) == "%H:%M:%S" and io.time_format(long) == "%H:%M"


def test_run_folder_keeps_the_pipeline_notes_with_the_device_numbers(run_folder):
    # summary.json says where latency and RAM were measured; the Device table must not drop that caveat
    res = io.load_results(run_folder())
    assert "Latency measured on a laptop CPU, not on the target device." in res.device["pipeline_notes"]
