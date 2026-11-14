import pytest

from nextraf_viz import insights, io
from nextraf_viz.context import Ctx
from sample_data import make_sample


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sample")
    make_sample.build(folder)
    return Ctx(res=io.load_results(folder), is_sample=True, region="IN", col="person", tolerance=0.5,
               stability_k=3.0, window=5, exclude_untrusted=True, switch_count=15)


def test_verdict_for_sample_is_surge_risk_with_text_and_icon(ctx):
    v = insights.verdict(ctx)
    assert v["level"] == "Surge risk" and v["icon"] and "surge" in v["text"].lower()


def test_verdict_is_unknown_without_a_walking_area(ctx):
    no_area = insights.with_site(ctx, area_m2=None)
    assert insights.verdict(no_area)["level"] == "Unknown"


def test_verdict_calm_when_thresholds_are_far_away(ctx):
    calm = insights.with_site(ctx, surge={"warn_density_ppm2": 50.0, "critical_density_ppm2": 99.0},
                              conflict={"min_people": 500, "min_vehicles": 500}, los_bands=[])
    assert insights.verdict(calm)["level"] == "Calm"


def test_verdict_critical_beats_surge_risk(ctx):
    crit = insights.with_site(ctx, surge={"critical_density_ppm2": 0.5, "warn_density_ppm2": 0.2})
    assert insights.verdict(crit)["level"] == "Critical"


def test_busiest_window_is_late_in_the_sample_and_ignores_the_spike(ctx):
    start, end, mean = insights.busiest_window(ctx, frames=3)
    assert start.strftime("%H:%M") >= "16:05" and mean < 30 and start < end


def test_findings_are_plain_english_and_mention_key_facts(ctx):
    text = " ".join(insights.findings(ctx)).lower()
    for needle in ("peak", "busiest", "surge", "conflict", "footpath", "no data"):
        assert needle in text, needle


def test_suggestions_use_regional_wording_and_are_labelled(ctx):
    uk = Ctx(**{**ctx.__dict__, "region": "UK"})
    assert "ward office" in " ".join(insights.suggestions(ctx)).lower()
    assert "council" in " ".join(insights.suggestions(uk)).lower()
    assert insights.suggestions(ctx)[0].lower().startswith("suggestion")


def test_what_if_changes_results_without_touching_the_original(ctx):
    tight = insights.with_site(ctx, area_m2=8.0)
    assert tight.site["area_m2"] == 8.0 and ctx.site["area_m2"] == 20.0
    assert insights.minutes_in_worst_band(tight) > insights.minutes_in_worst_band(ctx)


def test_replay_state_reports_frame_values_and_flags(ctx):
    s = insights.replay_state(ctx, 19)
    assert s["time"] == "16:14:58" and s["people"] == ctx.counts["person"].iloc[19] and s["camera_ok"]
    assert insights.replay_state(ctx, 11)["camera_ok"] is False and insights.replay_state(ctx, 11)["people"] is None
    assert insights.replay_state(ctx, 8)["implausible"] is True
    assert insights.replay_state(ctx, 19)["band"] and insights.replay_state(ctx, 19)["density"] > 0


def test_density_scale_places_marker_and_zone_on_a_linear_scale():
    s = insights.density_scale(1.0, warn=2.0, critical=4.0)
    assert s["top"] == 5.0  # critical * 1.25 when the value is low
    assert s["warn_pct"] == pytest.approx(40.0) and s["critical_pct"] == pytest.approx(80.0)
    assert s["marker_pct"] == pytest.approx(20.0) and s["zone"] == "Below warning"
    assert insights.density_scale(2.5, 2.0, 4.0)["zone"] == "Warning"
    assert insights.density_scale(4.0, 2.0, 4.0)["zone"] == "Critical"


def test_density_scale_grows_to_fit_high_values_and_handles_no_data():
    high = insights.density_scale(9.0, 2.0, 4.0)
    assert high["top"] == pytest.approx(9.9) and high["marker_pct"] < 100
    none = insights.density_scale(None, 2.0, 4.0)
    assert none["marker_pct"] is None and none["zone"] == "No data"


def test_replay_state_clamps_out_of_range_index(ctx):
    assert insights.replay_state(ctx, 999)["index"] == len(ctx.counts) - 1
    assert insights.replay_state(ctx, -5)["index"] == 0


def test_compare_configs_reports_both_and_the_differences(ctx):
    a, b = "yolo26n_int8_320_t0.5", "yolo26n_fp32_640_t0.5"
    c = insights.compare_configs(ctx.bench, ctx.res.per_frame, a, b)
    assert c["a"]["config_id"] == a and c["b"]["config_id"] == b
    assert c["delta"]["e2e_median_ms"] == c["b"]["e2e_median_ms"] - c["a"]["e2e_median_ms"]
    assert c["delta"]["mae"] < 0 and c["frames"]["a_pred"].shape == c["frames"]["b_pred"].shape


def test_data_checklist_lists_every_file_and_what_it_unlocks(ctx):
    rows = insights.data_checklist(ctx)
    names = {r["file"]: r for r in rows}
    assert names["counts.csv"]["present"] and "required" in names["counts.csv"]["unlocks"].lower()
    assert names["benchmark.csv"]["present"] and not any(
        r["present"] for r in insights.data_checklist(Ctx(**{**ctx.__dict__, "res": io.Results(
            folder=ctx.res.folder, counts=ctx.counts, site=ctx.site)})) if r["file"] == "benchmark.csv")


def test_surge_ignores_the_implausible_spike_frame(ctx):
    from nextraf_viz.context import surge
    levels = surge(ctx)[2]["level"]
    assert levels.iloc[8] == "no data"          # the injected 48-person spike
    first_warning = ctx.counts["time"][levels == "warning"].iloc[0]
    assert first_warning.strftime("%H:%M") >= "16:00"
    assert insights.verdict(ctx)["level"] == "Surge risk"


def test_checklist_names_the_run_files_for_a_run_folder(run_folder):
    res = io.load_results(run_folder())
    run_ctx = Ctx(res=res, is_sample=False, region="IN", col="person", tolerance=0.5,
                  stability_k=3.0, window=5, exclude_untrusted=True, switch_count=15)
    rows = insights.data_checklist(run_ctx)
    assert rows[0]["file"] == "frames.csv + summary.json" and rows[0]["present"]


def test_findings_for_a_short_clip_name_the_second(run_folder):
    res = io.load_results(run_folder())
    run_ctx = Ctx(res=res, is_sample=False, region="IN", col="person", tolerance=0.5,
                  stability_k=3.0, window=5, exclude_untrusted=True, switch_count=15)
    assert any("13:05:47" in line for line in insights.findings(run_ctx))
