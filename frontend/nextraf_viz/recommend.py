"""Configuration recommendation and Pareto frontier for the affordability envelope."""
from __future__ import annotations

import re

import pandas as pd

_EPS = 1e-9


def recommend(bench: pd.DataFrame, tolerance: float = 0.5,
              exclude_untrustworthy: bool = True) -> pd.Series | None:
    """Fastest configuration whose MAE is at most best MAE + tolerance.

    Ties on latency go to the lower MAE. Returns None if nothing is eligible.
    """
    pool = bench[~bench["not_trustworthy"]] if exclude_untrustworthy else bench
    pool = pool.dropna(subset=["mae", "e2e_median_ms"])
    if pool.empty:
        return None
    eligible = pool[pool["mae"] <= pool["mae"].min() + tolerance + _EPS]
    return eligible.sort_values(["e2e_median_ms", "mae"], kind="stable").iloc[0]


def pareto_frontier(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Rows not dominated on (x, y), both minimised, ordered by x. Identical points are all kept."""
    ordered = df.dropna(subset=[x, y]).sort_values([x, y], kind="stable")
    keep, best_y, last = [], float("inf"), None
    for idx, row in ordered.iterrows():
        if row[y] < best_y or (last is not None and (row[x], row[y]) == last):
            keep.append(idx)
            best_y, last = min(best_y, row[y]), (row[x], row[y])
    return ordered.loc[keep]


def adaptive_simulation(per_frame: pd.DataFrame, bench: pd.DataFrame, fast_id: str, accurate_id: str,
                        switch_count: int) -> dict:
    """Simulate density-adaptive switching offline (feature A).

    Each frame uses the fast configuration unless the previous frame's chosen prediction was
    >= ``switch_count``, then the accurate one. Latency is the mean of the chosen configs'
    ``e2e_median_ms``. Peak memory is an ESTIMATE: both models are loaded, so it is the larger
    peak plus the model size of the other one.
    """
    def frames(cid: str) -> pd.DataFrame:
        sub = per_frame[(per_frame["config_id"] == cid) & (per_frame["class"] == "person")]
        return sub.sort_values("frame_idx").set_index("frame_idx")

    fast, acc = frames(fast_id), frames(accurate_id)
    chosen, preds, prev = [], [], None
    for idx in fast.index:
        use_acc = prev is not None and prev >= switch_count
        pred = acc.loc[idx, "pred_count"] if use_acc else fast.loc[idx, "pred_count"]
        chosen.append(accurate_id if use_acc else fast_id)
        preds.append(float(pred))
        prev = pred
    err = pd.Series(preds, index=fast.index) - fast["true_count"]
    b = bench.set_index("config_id")
    latency = float(sum(b.loc[c, "e2e_median_ms"] for c in chosen) / len(chosen))
    bigger, other = (fast_id, accurate_id) if b.loc[fast_id, "peak_mem_mb"] >= b.loc[accurate_id, "peak_mem_mb"] \
        else (accurate_id, fast_id)
    return {"chosen": chosen, "mae": float(err.abs().mean()), "within1_pct": float((err.abs() <= 1).mean() * 100),
            "mean_signed_error": float(err.mean()), "e2e_median_ms": latency,
            "accurate_share": chosen.count(accurate_id) / len(chosen),
            "peak_mem_mb": float(b.loc[bigger, "peak_mem_mb"] + b.loc[other, "model_size_mb"])}


def cheapest_board(peak_mem_mb: float, boards: list[dict], os_overhead_mb: float,
                   headroom_fraction: float) -> dict | None:
    """Lowest-priced board where peak memory + OS overhead <= (1 - headroom) x RAM (1 GB = 1024 MB)."""
    fits = [b for b in boards if peak_mem_mb + os_overhead_mb <= (1 - headroom_fraction) * b["ram_gb"] * 1024]
    return min(fits, key=lambda b: b["price"]) if fits else None


def energy(power_w: float, e2e_ms: float, window_hours: float | None) -> dict:
    """Energy estimates from measured power: J per frame, Wh per hour and per site window, battery size.

    Assumes continuous running at ``power_w``. Battery = Wh needed / 0.8 usable (an estimate).
    """
    wh_window = power_w * window_hours if window_hours is not None else None
    return {"j_per_frame": power_w * e2e_ms / 1000.0, "wh_per_hour": power_w, "wh_window": wh_window,
            "battery_wh": wh_window / 0.8 if wh_window is not None else None}


def window_hours(time_window: str) -> float | None:
    """Hours in a ``HH:MM–HH:MM`` site window (en dash or hyphen); None if unparseable."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})\s*", time_window or "")
    if not m:
        return None
    h1, m1, h2, m2 = map(int, m.groups())
    hours = ((h2 * 60 + m2) - (h1 * 60 + m1)) / 60.0
    return hours if hours > 0 else None


TARGETS = [("Time per frame", "e2e_median_ms", "≤ 200 ms", lambda v: v <= 200, "{:.0f} ms"),
           ("Throughput", "fps", "≥ 5 FPS", lambda v: v >= 5, "{:.1f} FPS"),
           ("Peak memory", "peak_mem_mb", "≤ 300 MB", lambda v: v <= 300, "{:.0f} MB"),
           ("Count error (MAE)", "mae", "≤ 1 person/frame", lambda v: v <= 1, "{:.2f} people/frame")]


def targets_table(rec: pd.Series) -> pd.DataFrame:
    """Measured value of the recommended configuration against each proposal target."""
    rows = [{"Measure": name, "Target": target, "Measured": fmt.format(rec[col]),
             "Result": "✓ Pass" if ok(rec[col]) else "✗ Fail"} for name, col, target, ok, fmt in TARGETS]
    return pd.DataFrame(rows)
