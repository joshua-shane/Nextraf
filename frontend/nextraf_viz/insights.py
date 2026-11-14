"""Plain-English insights, verdict, replay state, what-if and comparison helpers (counts only, no Streamlit)."""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd

from . import crowding
from .io import is_run_folder
from .context import Ctx, conflict, has_encroachment, surge, usable_area

ICONS = {"Calm": "●", "Busy": "●", "Surge risk": "●", "Critical": "●", "Unknown": "○"}


def with_site(ctx: Ctx, **overrides) -> Ctx:
    """Copy of ``ctx`` with site settings overridden (dict values are merged). For what-if tests; the original is untouched."""
    site = copy.deepcopy(ctx.site)
    for key, value in overrides.items():
        site[key] = {**site[key], **value} if isinstance(value, dict) and isinstance(site.get(key), dict) else value
    return dataclasses.replace(ctx, res=dataclasses.replace(ctx.res, site=site))


def _flags(ctx: Ctx) -> pd.Series:
    return crowding.stability_flags(ctx.counts["person"], window=max(ctx.window, 3), k=ctx.stability_k)


def minutes_in_worst_band(ctx: Ctx) -> float:
    """Minutes the site spent in the most crowded band (0 if no area or bands are set)."""
    area, bands = ctx.site["area_m2"], ctx.site["los_bands"]
    if not area or not bands:
        return 0.0
    table = crowding.time_in_band(ctx.counts["minutes"], crowding.space_per_person(ctx.counts["person"], area), bands)
    return float(table.loc[table["band"] == crowding.worst_band(bands)["name"], "minutes"].sum())


def _first_time(ctx: Ctx, mask: pd.Series) -> str:
    return f"{ctx.counts['time'][mask].iloc[0]:{ctx.tf}}"


def verdict(ctx: Ctx) -> dict:
    """One-word crowding verdict with an icon and a sentence: Calm, Busy, Surge risk, Critical or Unknown."""
    s = surge(ctx)
    if s is None:
        return {"level": "Unknown", "icon": ICONS["Unknown"],
                "text": "Add a walking area in site.json to get a crowding verdict."}
    levels = s[2]["level"]
    if (levels == "critical").any():
        level, text = "Critical", f"Density reached the critical level, first at {_first_time(ctx, levels == 'critical')}."
    elif (levels == "warning").any():
        n = int((levels == "warning").sum())
        level, text = "Surge risk", (f"Surge risk: density was high and rising in {n} frame(s), "
                                     f"first at {_first_time(ctx, levels == 'warning')}.")
    elif conflict(ctx).any() or minutes_in_worst_band(ctx) > 0:
        level, text = "Busy", "Busy: many people, and vehicles at the same time or time in the most crowded band."
    else:
        level, text = "Calm", "Calm: no surge, no conflict moments and no time in the most crowded band."
    return {"level": level, "icon": ICONS[level], "text": text}


def busiest_window(ctx: Ctx, frames: int = 3) -> tuple[pd.Timestamp, pd.Timestamp, float] | None:
    """(start, end, mean people) of the busiest run of ``frames`` frames, ignoring implausible and no-data frames."""
    people = ctx.counts["person"].where(~_flags(ctx))
    mean = people.rolling(frames, min_periods=min(2, frames)).mean()
    if mean.notna().sum() == 0:
        return None
    end = int(np.nanargmax(mean.to_numpy(dtype=float)))
    start = max(0, end - frames + 1)
    times = ctx.counts["time"]
    return times.iloc[start], times.iloc[end], float(mean.iloc[end])


def findings(ctx: Ctx) -> list[str]:
    """Plain-English findings from the counts, in reading order."""
    counts, site, out = ctx.counts, ctx.site, []
    flags = _flags(ctx)
    people = counts["person"].where(~flags)
    if people.notna().any():
        pos, peak = crowding.peak(people)
        out.append(f"Peak occupancy was {peak:.0f} people at {counts['time'].iloc[pos]:{ctx.tf}}.")
    busy = busiest_window(ctx)
    if busy:
        out.append(f"The busiest stretch was {busy[0]:{ctx.tf}} to {busy[1]:{ctx.tf}}, averaging {busy[2]:.0f} people per frame.")
    if site["area_m2"] and site["los_bands"]:
        note = " (walking area is assumed)" if site["area_is_assumed"] else ""
        out.append(f"The {ctx.t('footpath')} spent {minutes_in_worst_band(ctx):.1f} minutes in the most crowded band{note}.")
    s = surge(ctx)
    if s is not None:
        ev = crowding.surge_events(s[2])
        if ev.empty:
            out.append("No surge warnings: density never rose fast towards a high level.")
        else:
            out.append(f"{len(ev)} surge warning frame(s), the first at {counts['time'].iloc[int(ev['index'].iloc[0])]:{ctx.tf}}: "
                       "density was high and still rising.")
    moments = conflict(ctx)
    if moments.any():
        cfg = site["conflict"]
        out.append(f"{int(moments.sum())} conflict moment(s), at least {cfg['min_people']} people and "
                   f"{cfg['min_vehicles']} vehicles together, the first at {_first_time(ctx, moments)}.")
    if has_encroachment(ctx):
        lost = float((site["area_m2"] - usable_area(ctx)).max())
        out.append(f"Up to {int(counts['vehicles_on_footpath'].max())} vehicle(s) were on the {ctx.t('footpath')} at once, "
                   f"taking up to {lost:.1f} m² (assumed footprints).")
    no_data = int((~counts["camera_ok"]).sum())
    if no_data:
        out.append(f"{no_data} frame(s) had no data (camera too dark, blurred or blocked) and are shown as gaps, not zero people.")
    if flags.any():
        out.append(f"{int(flags.sum())} frame(s) looked implausible and are left out of the peak.")
    return out


def suggestions(ctx: Ctx) -> list[str]:
    """Generic, clearly-labelled ideas for the local authority, from counts only. Never an instruction or an alert."""
    who, foot, out = ctx.t("authority"), ctx.t("footpath"), []
    v = verdict(ctx)["level"]
    if v in ("Surge risk", "Critical"):
        out.append(f"Suggestion: the {who} could consider a marshal, a barrier or staggered exits around the surge time.")
    if conflict(ctx).any():
        out.append(f"Suggestion: the {who} could look at keeping vehicles off the {foot} when many people are present.")
    if has_encroachment(ctx) and ctx.counts["vehicles_on_footpath"].max() > 0:
        out.append(f"Suggestion: repeated encroachment can be reported to the {who} using these counts and times (no plates or footage).")
    return out or [f"Suggestion: nothing stands out; the {who} can keep monitoring this site."]


def replay_state(ctx: Ctx, index: int) -> dict:
    """Everything shown for one frame of the live replay."""
    i = int(min(max(index, 0), len(ctx.counts) - 1))
    row = ctx.counts.iloc[i]
    ok = bool(row["camera_ok"])
    people = None if pd.isna(row["person"]) else int(row["person"])
    vehicles = None if pd.isna(row["vehicles_total"]) else int(row["vehicles_total"])
    s = surge(ctx)
    density = band = None
    level = "no data" if not ok else "ok"
    if s is not None:
        d = s[1].iloc[i]
        density = None if pd.isna(d) else float(d)
        level = s[2]["level"].iloc[i]
        if people:
            band = crowding.los_band(ctx.site["area_m2"] / people, ctx.site["los_bands"]) if ctx.site["los_bands"] else None
    return {"index": i, "time": f"{row['time']:%H:%M:%S}", "people": people, "vehicles": vehicles, "camera_ok": ok,
            "density": density, "band": band, "level": level, "conflict": bool(conflict(ctx).iloc[i]),
            "implausible": bool(_flags(ctx).iloc[i])}


def density_scale(density: float | None, warn: float, critical: float) -> dict:
    """Geometry of the replay's linear density scale: where the zones split and where the marker sits (percent)."""
    top = max(critical * 1.25, (density or 0) * 1.1)
    if density is None:
        zone, marker = "No data", None
    else:
        zone = "Critical" if density >= critical else "Warning" if density >= warn else "Below warning"
        marker = 100 * density / top
    return {"top": top, "warn_pct": 100 * warn / top, "critical_pct": 100 * critical / top,
            "marker_pct": marker, "zone": zone}


_COMPARE_COLS = ["config_id", "model", "resolution", "precision", "threshold", "e2e_median_ms", "fps",
                 "peak_mem_mb", "model_size_mb", "mae", "within1_pct", "mean_signed_error"]


def compare_configs(bench: pd.DataFrame, per_frame: pd.DataFrame | None, a: str, b: str) -> dict:
    """Two configurations side by side: their benchmark rows, B minus A deltas, and per-frame predictions."""
    rows = bench.set_index("config_id")
    ra, rb = rows.loc[a].copy(), rows.loc[b].copy()
    ra["config_id"], rb["config_id"] = a, b
    keys = ["e2e_median_ms", "fps", "peak_mem_mb", "model_size_mb", "mae", "within1_pct", "mean_signed_error"]
    result = {"a": ra[[c for c in _COMPARE_COLS if c in ra.index]].to_dict(),
              "b": rb[[c for c in _COMPARE_COLS if c in rb.index]].to_dict(),
              "delta": {k: rb[k] - ra[k] for k in keys}, "frames": None}
    if per_frame is not None:
        pf = per_frame[per_frame["class"] == "person"]
        fa = pf[pf["config_id"] == a].set_index("frame_idx")
        fb = pf[pf["config_id"] == b].set_index("frame_idx")
        result["frames"] = pd.DataFrame({"true": fa["true_count"], "a_pred": fa["pred_count"],
                                         "b_pred": fb["pred_count"]}).reset_index()
    return result


def data_checklist(ctx: Ctx) -> list[dict]:
    """Which input files are present and what each one unlocks in the app."""
    res, folder = ctx.res, ctx.res.folder
    site_present = (folder / "site.json").is_file() or (folder / "site.example.json").is_file()
    run = not (folder / "counts.csv").is_file() and is_run_folder(folder)
    return [
        {"file": "frames.csv + summary.json" if run else "counts.csv", "present": True,
         "unlocks": "Required. Overview, timeline, replay and the report."},
        {"file": "benchmark.csv", "present": res.benchmark is not None,
         "unlocks": "Affordability envelope, recommendation, targets, board fit, energy."},
        {"file": "per_frame_eval.csv", "present": res.per_frame is not None,
         "unlocks": "Validation, adaptive switching, comparing two configurations."},
        {"file": "site.json", "present": site_present, "unlocks": "Walking area, crowding bands, surge and conflict settings."},
        {"file": "device.json", "present": res.device is not None, "unlocks": "Device details on the Trust screen."},
        {"file": "samples/", "present": bool(res.samples), "unlocks": "Annotated image gallery on Validation."},
    ]
