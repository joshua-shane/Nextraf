"""Crowding maths: space per person, LOS bands, stability flags, rolling stats."""
from __future__ import annotations

import numpy as np
import pandas as pd


def space_per_person(counts: pd.Series, area_m2: float) -> pd.Series:
    """m2 per person. Zero or missing counts give NaN (no crowding to measure)."""
    c = counts.astype(float).where(counts > 0)
    return area_m2 / c


def los_band(space: float, bands: list[dict]) -> str | None:
    """Name of the band with min <= space < max (max None = unbounded); None for NaN."""
    if space is None or np.isnan(space):
        return None
    for b in bands:
        hi = b["max_m2_per_person"]
        if b["min_m2_per_person"] <= space and (hi is None or space < hi):
            return b["name"]
    return None


def time_in_band(minutes: pd.Series, space: pd.Series, bands: list[dict]) -> pd.DataFrame:
    """Minutes spent in each band. Each frame is credited with the gap to the next frame
    (the last frame reuses the median gap); gap frames (NaN space) are skipped."""
    m = minutes.to_numpy(dtype=float)
    if len(m) == 0:
        return pd.DataFrame({"band": [b["name"] for b in bands], "minutes": 0.0})
    gaps = np.diff(m)
    step = np.append(gaps, np.median(gaps) if len(gaps) else 0.0)
    names = [los_band(v, bands) for v in space]
    totals = {b["name"]: 0.0 for b in bands}
    for name, dt in zip(names, step):
        if name is not None:
            totals[name] += float(dt)
    return pd.DataFrame({"band": list(totals), "minutes": list(totals.values())})


def rolling_mean(counts: pd.Series, window: int) -> pd.Series:
    """Trailing rolling mean that ignores missing frames."""
    return counts.rolling(window, min_periods=1).mean()


def peak(counts: pd.Series) -> tuple[int, float]:
    """(row position, value) of the maximum count."""
    pos = int(np.nanargmax(counts.to_numpy(dtype=float)))
    return pos, float(counts.iloc[pos])


def stability_flags(counts: pd.Series, window: int = 5, k: float = 3.0) -> pd.Series:
    """True where |count - rolling median| > k * MAD over a centred window (implausible frames).

    MAD is floored at 1 person: counts are integers, so a window of identical values
    must not flag a one-person wobble.
    """
    x = counts.astype(float)
    med = x.rolling(window, center=True, min_periods=3).median()
    mad = x.rolling(window, center=True, min_periods=3).apply(
        lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True)
    flags = (x - med).abs() > k * mad.clip(lower=1.0)
    return flags.fillna(False).astype(bool)


def worst_band(bands: list[dict]) -> dict | None:
    """The most crowded band (smallest minimum space per person)."""
    return min(bands, key=lambda b: b["min_m2_per_person"]) if bands else None


def usable_area(area_m2: float, two_wheelers: pd.Series, cars: pd.Series, footprint: dict) -> pd.Series:
    """Walking area left after vehicles parked on the footpath (feature B).

    Missing counts are treated as zero. The result never drops below 10% of the area,
    so density stays finite.
    """
    used = (two_wheelers.fillna(0) * footprint["two_wheeler"] + cars.fillna(0) * footprint["car"])
    return (area_m2 - used).clip(lower=0.1 * area_m2)


def surge_levels(minutes: pd.Series, density: pd.Series, warn: float, rise_per_min: float,
                 critical: float, window: int) -> pd.DataFrame:
    """Per-frame surge level (feature C): 'ok', 'warning', 'critical' or 'no data'.

    warning = density >= warn AND density rose at least ``rise_per_min`` per minute over the last
    ``window`` frames; critical = density >= critical. Counts-only: no tracking.
    """
    m, d = minutes.to_numpy(dtype=float), density.to_numpy(dtype=float)
    levels, rises = [], []
    for i in range(len(d)):
        j = max(0, i - window + 1)
        rise = (d[i] - d[j]) / (m[i] - m[j]) if i > j and m[i] > m[j] else 0.0
        rises.append(0.0 if np.isnan(rise) else rise)
        if np.isnan(d[i]):
            levels.append("no data")
        elif d[i] >= critical:
            levels.append("critical")
        elif d[i] >= warn and rises[-1] >= rise_per_min:
            levels.append("warning")
        else:
            levels.append("ok")
    return pd.DataFrame({"level": levels, "rise_per_min": rises})


def surge_events(levels: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``surge_levels`` that are warnings or criticals, keeping the frame position as ``index``."""
    ev = levels[levels["level"].isin(["warning", "critical"])]
    return ev.reset_index()


def conflict_moments(people: pd.Series, vehicles: pd.Series, min_people: float, min_vehicles: float) -> pd.Series:
    """True where many people and many vehicles are present at the same time (NaN counts never conflict)."""
    return ((people >= min_people) & (vehicles >= min_vehicles)).fillna(False).astype(bool)
