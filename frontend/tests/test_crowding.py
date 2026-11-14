import numpy as np
import pandas as pd

from nextraf_viz import crowding

BANDS = [
    {"name": "A", "min_m2_per_person": 5.0, "max_m2_per_person": None},
    {"name": "B", "min_m2_per_person": 2.0, "max_m2_per_person": 5.0},
    {"name": "F", "min_m2_per_person": 0.0, "max_m2_per_person": 2.0},
]


def test_space_per_person_divides_area_by_count_and_nan_for_zero():
    out = crowding.space_per_person(pd.Series([10, 0, np.nan]), 30.0)
    assert out.iloc[0] == 3.0 and pd.isna(out.iloc[1]) and pd.isna(out.iloc[2])


def test_los_band_maps_boundaries_to_the_roomier_band():
    names = [crowding.los_band(v, BANDS) for v in (6.0, 5.0, 3.0, 2.0, 0.5)]
    assert names == ["A", "A", "B", "B", "F"]


def test_los_band_empty_space_counts_as_best_band_and_nan_is_none():
    assert crowding.los_band(float("inf"), BANDS) == "A"
    assert crowding.los_band(float("nan"), BANDS) is None


def test_time_in_band_sums_minutes_and_skips_gaps():
    minutes = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    space = pd.Series([6.0, 3.0, 1.0, np.nan, 1.0])
    out = crowding.time_in_band(minutes, space, BANDS)
    got = dict(zip(out["band"], out["minutes"]))
    assert got == {"A": 1.0, "B": 1.0, "F": 2.0}


def test_stability_flags_spike_but_not_steady_series():
    counts = pd.Series([10, 11, 10, 12, 11, 60, 11, 10, 12, 11], dtype=float)
    flags = crowding.stability_flags(counts, window=5, k=3)
    assert flags[5] and flags.sum() == 1


def test_stability_flags_constant_series_has_no_flags():
    assert not crowding.stability_flags(pd.Series([5.0] * 8), window=5, k=3).any()


def test_stability_flags_ignore_nan_gaps():
    counts = pd.Series([5, 5, np.nan, 5, 5, 5], dtype=float)
    assert not crowding.stability_flags(counts, window=3, k=3).any()


def test_rolling_mean_and_peak():
    s = pd.Series([1, 2, 9, 4], dtype=float)
    assert crowding.rolling_mean(s, 2).tolist()[1:] == [1.5, 5.5, 6.5]
    assert crowding.peak(s) == (2, 9.0)


def test_stability_flags_tolerate_one_person_wobble_when_neighbours_are_identical():
    counts = pd.Series([11, 11, 12, 11, 11, 11], dtype=float)
    assert not crowding.stability_flags(counts, window=5, k=3).any()


def test_worst_band_is_the_one_with_least_space_regardless_of_list_order():
    shuffled = [BANDS[2], BANDS[0], BANDS[1]]
    assert crowding.worst_band(shuffled)["name"] == "F"
    assert crowding.worst_band([]) is None


# ---- encroachment (B), surge (C), conflict ----

FOOT = {"two_wheeler": 1.5, "car": 7.0}


def test_usable_area_subtracts_vehicle_footprints():
    out = crowding.usable_area(20.0, pd.Series([3, 0]), pd.Series([1, 0]), FOOT)
    assert out.tolist() == [20 - 3 * 1.5 - 7.0, 20.0]


def test_usable_area_never_drops_below_ten_percent_of_area():
    assert crowding.usable_area(10.0, pd.Series([3]), pd.Series([3]), FOOT).iloc[0] == 1.0


def test_usable_area_treats_missing_vehicle_counts_as_zero():
    assert crowding.usable_area(20.0, pd.Series([float("nan")]), pd.Series([0]), FOOT).iloc[0] == 20.0


SURGE = dict(warn=1.0, rise_per_min=0.3, critical=2.3, window=3)


def test_surge_warns_only_when_dense_and_rising_and_critical_on_density_alone():
    minutes = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    dens = pd.Series([0.5, 0.8, 1.3, 1.8, 2.4])
    out = crowding.surge_levels(minutes, dens, **SURGE)
    assert out["level"].tolist() == ["ok", "ok", "warning", "warning", "critical"]


def test_surge_does_not_warn_for_dense_but_flat_crowd():
    out = crowding.surge_levels(pd.Series([0.0, 1.0, 2.0, 3.0]), pd.Series([1.5] * 4), **SURGE)
    assert set(out["level"]) == {"ok"}


def test_surge_marks_gaps_as_no_data_and_does_not_bridge_them_with_zero():
    out = crowding.surge_levels(pd.Series([0.0, 1.0, 2.0]), pd.Series([0.5, float("nan"), 1.5]), **SURGE)
    assert out["level"].tolist()[1] == "no data"


def test_surge_events_lists_only_warning_and_critical_rows():
    minutes = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    levels = crowding.surge_levels(minutes, pd.Series([0.5, 0.8, 1.3, 1.8, 2.4]), **SURGE)
    ev = crowding.surge_events(levels)
    assert ev["level"].tolist() == ["warning", "warning", "critical"] and ev["index"].tolist() == [2, 3, 4]


def test_conflict_moments_need_many_people_and_many_vehicles_together():
    people = pd.Series([20, 20, 5, 20])
    vehicles = pd.Series([6, 2, 9, 5])
    assert crowding.conflict_moments(people, vehicles, 15, 5).tolist() == [True, False, False, True]


def test_conflict_moments_ignore_no_data_frames():
    assert crowding.conflict_moments(pd.Series([float("nan")]), pd.Series([9.0]), 15, 5).tolist() == [False]
