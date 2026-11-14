import pandas as pd

from nextraf_viz import recommend


def bench(rows):
    df = pd.DataFrame(rows, columns=["config_id", "e2e_median_ms", "mae", "not_trustworthy"])
    return df


def test_recommend_picks_fastest_within_tolerance_of_best_mae():
    df = bench([("slow_best", 300, 0.5, False), ("fast_ok", 100, 0.9, False),
                ("faster_bad", 50, 1.2, False)])
    assert recommend.recommend(df, tolerance=0.5)["config_id"] == "fast_ok"


def test_recommend_tolerance_boundary_is_inclusive():
    df = bench([("best", 300, 0.5, False), ("edge", 100, 1.0, False)])
    assert recommend.recommend(df, tolerance=0.5)["config_id"] == "edge"


def test_recommend_tie_on_latency_prefers_lower_mae():
    df = bench([("a", 100, 0.8, False), ("b", 100, 0.6, False)])
    assert recommend.recommend(df, tolerance=0.5)["config_id"] == "b"


def test_recommend_excludes_untrustworthy_runs_when_asked():
    df = bench([("hot_fast", 50, 0.5, True), ("ok", 100, 0.6, False)])
    assert recommend.recommend(df, 0.5, exclude_untrustworthy=True)["config_id"] == "ok"
    assert recommend.recommend(df, 0.5, exclude_untrustworthy=False)["config_id"] == "hot_fast"


def test_recommend_none_when_no_rows():
    assert recommend.recommend(bench([("x", 1, 1, True)]), 0.5, exclude_untrustworthy=True) is None


def test_pareto_frontier_minimises_both_axes():
    df = bench([("a", 100, 1.0, False), ("b", 200, 0.5, False),
                ("c", 150, 1.2, False), ("d", 300, 0.5, False)])
    front = recommend.pareto_frontier(df, "e2e_median_ms", "mae")
    assert list(front["config_id"]) == ["a", "b"]


def test_pareto_frontier_keeps_equal_points_once_each():
    df = bench([("a", 100, 1.0, False), ("b", 100, 1.0, False)])
    assert len(recommend.pareto_frontier(df, "e2e_median_ms", "mae")) == 2


# ---- adaptive switching (A), hardware fit and energy (D) ----

def _adaptive_inputs():
    pf = pd.DataFrame({
        "config_id": ["fast"] * 4 + ["acc"] * 4, "frame_idx": [0, 1, 2, 3] * 2,
        "pred_count": [10, 16, 5, 20, 11, 15, 5, 19], "true_count": [10, 15, 5, 19] * 2, "class": "person"})
    bench = pd.DataFrame({"config_id": ["fast", "acc"], "e2e_median_ms": [60, 200],
                          "peak_mem_mb": [150, 330], "model_size_mb": [3.0, 10.0]})
    return pf, bench


def test_adaptive_uses_accurate_model_only_after_a_crowded_frame():
    pf, bench = _adaptive_inputs()
    r = recommend.adaptive_simulation(pf, bench, "fast", "acc", switch_count=15)
    assert r["chosen"] == ["fast", "fast", "acc", "fast"]
    assert r["mae"] == 0.5 and r["accurate_share"] == 0.25
    assert r["e2e_median_ms"] == 95.0


def test_adaptive_memory_estimate_loads_both_models():
    pf, bench = _adaptive_inputs()
    r = recommend.adaptive_simulation(pf, bench, "fast", "acc", switch_count=15)
    assert r["peak_mem_mb"] == 333.0


def test_adaptive_never_switches_with_an_unreachable_switch_count():
    pf, bench = _adaptive_inputs()
    r = recommend.adaptive_simulation(pf, bench, "fast", "acc", switch_count=999)
    assert set(r["chosen"]) == {"fast"} and r["e2e_median_ms"] == 60.0


BOARDS = [{"ram_gb": 1, "price": 5044}, {"ram_gb": 2, "price": 7869},
          {"ram_gb": 4, "price": 13409}, {"ram_gb": 8, "price": 19999}]


def test_hardware_fit_picks_the_cheapest_board_that_fits_with_headroom():
    assert recommend.cheapest_board(330, BOARDS, os_overhead_mb=350, headroom_fraction=0.2)["ram_gb"] == 1
    assert recommend.cheapest_board(500, BOARDS, 350, 0.2)["ram_gb"] == 2


def test_hardware_fit_picks_lowest_price_not_smallest_ram_and_none_if_nothing_fits():
    odd = [{"ram_gb": 2, "price": 100}, {"ram_gb": 4, "price": 50}]
    assert recommend.cheapest_board(500, odd, 350, 0.2)["price"] == 50
    assert recommend.cheapest_board(50000, BOARDS, 350, 0.2) is None


def test_energy_estimates_per_frame_hour_window_and_battery():
    e = recommend.energy(power_w=6.0, e2e_ms=62, window_hours=0.75)
    assert round(e["j_per_frame"], 3) == 0.372
    assert e["wh_per_hour"] == 6.0 and e["wh_window"] == 4.5
    assert round(e["battery_wh"], 3) == 5.625


def test_window_hours_parses_dash_variants_and_rejects_garbage():
    assert recommend.window_hours("15:30–16:15") == 0.75
    assert recommend.window_hours("15:30-16:15") == 0.75
    assert recommend.window_hours("") is None and recommend.window_hours("soon") is None
