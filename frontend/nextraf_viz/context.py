"""Run context (results + sidebar settings) and series derived from it. No Streamlit imports."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nextraf_viz import crowding, io, recommend, strings

CLASS_LABELS = {"person": "People", "bicycle": "Bicycles", "car": "Cars", "motorcycle": "Motorcycles",
                "bus": "Buses", "truck": "Trucks", "vehicles_total": "All vehicles"}


@dataclass
class Ctx:
    res: io.Results
    is_sample: bool
    region: str
    col: str
    tolerance: float
    stability_k: float
    window: int
    exclude_untrusted: bool
    switch_count: int

    @property
    def site(self) -> dict:
        return self.res.site

    @property
    def label(self) -> str:
        return CLASS_LABELS[self.col]

    @property
    def counts(self) -> pd.DataFrame:
        return self.res.counts

    @property
    def tf(self) -> str:
        """Clock format for this run: with seconds for short clips, hours and minutes otherwise."""
        return io.time_format(self.res.counts["time"])

    @property
    def bench(self) -> pd.DataFrame | None:
        return self.res.benchmark

    @property
    def rec(self) -> pd.Series | None:
        if self.bench is None:
            return None
        return recommend.recommend(self.bench, self.tolerance, self.exclude_untrusted)

    def t(self, key: str, **fmt):
        return strings.text(key, self.region, **fmt)

    @property
    def currency(self) -> str:
        code = self.site["hardware"].get("currency") or strings.text("currency_default", self.region)
        return strings.CURRENCY_SYMBOLS.get(code, code + " ")


def available_classes(counts: pd.DataFrame) -> list[str]:
    return [c for c in CLASS_LABELS if c in counts.columns]


def has_encroachment(ctx: Ctx) -> bool:
    return bool(ctx.site["area_m2"]) and "vehicles_on_footpath" in ctx.counts.columns


def usable_area(ctx: Ctx) -> pd.Series | None:
    """Walking area left after vehicles on the footpath (None if no area is set)."""
    area = ctx.site["area_m2"]
    if not area:
        return None
    c = ctx.counts
    cars = c["cars_on_footpath"] if "cars_on_footpath" in c.columns else pd.Series(0.0, index=c.index)
    on_fp = c["vehicles_on_footpath"] if "vehicles_on_footpath" in c.columns else pd.Series(0.0, index=c.index)
    two_wheelers = (on_fp - cars).clip(lower=0)
    return crowding.usable_area(area, two_wheelers, cars, ctx.site["vehicle_footprint_m2"])


def surge(ctx: Ctx) -> tuple[pd.Series, pd.Series, pd.DataFrame] | None:
    """(usable area, density, surge levels), or None if no area is set. Implausible frames count as no data."""
    usable = usable_area(ctx)
    if usable is None:
        return None
    people = ctx.counts["person"]
    flags = crowding.stability_flags(people, window=max(ctx.window, 3), k=ctx.stability_k)
    density = people.where(~flags) / usable  # implausible frames carry no density, like no-data frames
    cfg = ctx.site["surge"]
    levels = crowding.surge_levels(ctx.counts["minutes"], density, cfg["warn_density_ppm2"],
                                   cfg["warn_rise_ppm2_per_min"], cfg["critical_density_ppm2"], cfg["window_frames"])
    return usable, density, levels


def adaptive_ids(ctx: Ctx) -> tuple[str, str, bool] | None:
    """(fast id, accurate id, auto_picked). Uses site.json ids, else the fastest and most accurate trusted runs."""
    bench = ctx.bench
    if bench is None or ctx.res.per_frame is None:
        return None
    cfg = ctx.site["adaptive"]
    known = set(bench["config_id"]) & set(ctx.res.per_frame["config_id"])
    if cfg["fast_config_id"] in known and cfg["accurate_config_id"] in known:
        return cfg["fast_config_id"], cfg["accurate_config_id"], False
    pool = bench[bench["config_id"].isin(known) & ~bench["not_trustworthy"]]
    if len(pool) < 2:
        return None
    fast = pool.sort_values("e2e_median_ms").iloc[0]["config_id"]
    accurate = pool.sort_values(["mae", "e2e_median_ms"]).iloc[0]["config_id"]
    return (fast, accurate, True) if fast != accurate else None


def adaptive_result(ctx: Ctx) -> dict | None:
    """Simulated adaptive-switching result for the sidebar switch count, or None if data is missing."""
    ids = adaptive_ids(ctx)
    if ids is None:
        return None
    result = recommend.adaptive_simulation(ctx.res.per_frame, ctx.bench, ids[0], ids[1], ctx.switch_count)
    return {**result, "fast_id": ids[0], "accurate_id": ids[1], "auto_picked": ids[2]}


def conflict(ctx: Ctx) -> pd.Series:
    """Conflict moments (many people and many vehicles), ignoring frames flagged as implausible."""
    cfg = ctx.site["conflict"]
    people = ctx.counts["person"]
    flags = crowding.stability_flags(people, window=max(ctx.window, 3), k=ctx.stability_k)
    return crowding.conflict_moments(people.where(~flags), ctx.counts["vehicles_total"],
                                     cfg["min_people"], cfg["min_vehicles"])
