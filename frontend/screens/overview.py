"""Overview: the site at a glance."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from nextraf_viz import charts, crowding, insights, recommend, trust

from . import common, ui
from .common import Ctx


def _board(ctx: Ctx, rec: pd.Series | None) -> dict:
    hw = ctx.site["hardware"]
    item = {"label": "Cheapest board that fits"}
    if rec is None or not hw["boards"]:
        return {**item, "value": "n/a", "note": "needs benchmark.csv and board prices"}
    board = recommend.cheapest_board(rec["peak_mem_mb"], hw["boards"], hw["os_overhead_mb"], hw["headroom_fraction"])
    if board is None:
        return {**item, "value": "None fits", "note": "peak memory too high", "tone": "bad"}
    return {**item, "value": f"Raspberry Pi 5, {board['ram_gb']} GB", "note": f"{ctx.currency}{board['price']:,.0f} per node"}


def render(ctx: Ctx) -> None:
    site, counts = ctx.site, ctx.counts
    ui.page_header(ctx.t("nav_overview"), ctx.t("help_overview"))

    people = counts["person"]
    flags = crowding.stability_flags(people, window=max(ctx.window, 3), k=ctx.stability_k)
    unflagged = people.where(~flags)
    health = trust.camera_health(counts)
    pos = None
    if unflagged.notna().any():
        pos, peak = crowding.peak(unflagged)
        left_out = f", {int(flags.sum())} implausible frame(s) left out" if flags.any() else ""
        peak_item = {"label": "Peak occupancy", "value": f"{peak:.0f}", "unit": "people",
                     "note": f"at {counts['time'].iloc[pos]:{ctx.tf}}{left_out}"}
    else:
        peak_item = {"label": "Peak occupancy", "value": "n/a", "note": "no usable frames"}
    busy = insights.busiest_window(ctx)
    if site["area_m2"] and site["los_bands"]:
        band_item = {"label": "Time in most crowded band", "value": f"{insights.minutes_in_worst_band(ctx):.1f}",
                     "unit": "min", "note": "walking area assumed" if site["area_is_assumed"] else None}
    else:
        missing = "crowding bands" if site["area_m2"] else "a walking area"
        band_item = {"label": "Time in most crowded band", "value": "n/a", "note": f"needs {missing} in site.json"}
    ui.readouts([
        peak_item,
        {"label": "Busiest stretch", "value": f"{busy[0]:{ctx.tf}}–{busy[1]:{ctx.tf}}" if busy else "n/a",
         "note": f"about {busy[2]:.0f} people per frame" if busy else None},
        band_item,
        {"label": "Frames processed", "value": f"{health['frames']}", "note": f"{health['no_data']} with no camera data"},
    ])

    st.markdown("#### People per frame")
    ui.chart(charts.timeline_fig(counts, "person", "People", crowding.rolling_mean(people, ctx.window), pos, flags,
                                 None, height=320))
    st.caption(ctx.t("cap_timeline"))

    with st.container(border=True, key="findings"):
        st.markdown("#### What happened")
        st.markdown(f"**{insights.verdict(ctx)['text']}**")
        st.markdown("\n".join(f"- {line}" for line in insights.findings(ctx)))
        for line in insights.suggestions(ctx):
            st.caption(line)

    rec = ctx.rec
    chain = trust.chain_status(ctx.res.folder / "counts.csv")
    videos = trust.find_video_files(ctx.res.folder)
    ui.specs([
        {"label": "Recommended setup", "value": rec["config_id"] if rec is not None else "Needs benchmark.csv",
         "note": f"{rec['fps']:.1f} FPS, MAE {rec['mae']:.2f} people/frame" if rec is not None else None},
        _board(ctx, rec),
        {"label": "Data integrity",
         "value": {"valid": "Verified", "broken": "Broken", "absent": "No hash chain"}[chain["state"]],
         "note": {"valid": "hash chain intact", "broken": f"first bad line {chain['first_bad_line']}",
                  "absent": "edits cannot be detected"}[chain["state"]],
         "tone": {"valid": "good", "broken": "bad", "absent": ""}[chain["state"]]},
        {"label": "Footage stored", **trust.footage_card(ctx.res.folder)},
    ])

    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown("#### Targets from the proposal")
        if rec is None:
            st.info(ctx.res.notes[0] if ctx.res.benchmark is None and ctx.res.notes else
                    "No eligible configuration. Try turning off the not-trustworthy filter in Analysis settings.")
        else:
            st.dataframe(recommend.targets_table(rec), hide_index=True, width="stretch")
    with b:
        _events(ctx)
        if videos:
            st.warning(ctx.t("privacy_warn", files=", ".join(videos)))
        if site["los_bands_are_placeholders"] and site["los_bands"]:
            st.warning(ctx.t("placeholder_los"))
        if site["area_is_assumed"] and site["area_m2"]:
            st.caption(ctx.t("area_assumed"))
    with st.expander("Which files are loaded and what each one unlocks"):
        table = pd.DataFrame(insights.data_checklist(ctx))
        table["present"] = table["present"].map({True: "Loaded", False: "Not found"})
        st.dataframe(table.rename(columns={"file": "File", "present": "Status", "unlocks": "Unlocks"}),
                     hide_index=True, width="stretch")
        for note in ctx.res.notes:
            st.caption(ctx.t("missing_section", note=note))


def _events(ctx: Ctx) -> None:
    """Surge and conflict events found in the counts."""
    st.markdown("#### Events")
    s = common.surge(ctx)
    lines = []
    if s is not None:
        times = ctx.counts["time"]
        ev = crowding.surge_events(s[2])
        for level in ("critical", "warning"):
            sub = ev[ev["level"] == level]
            if len(sub):
                lines.append(f"**{level.capitalize()}:** {len(sub)} surge {level} frame(s), "
                             f"last at {times.iloc[int(sub['index'].iloc[-1])]:{ctx.tf}}")
    cfg = ctx.site["conflict"]
    conflict = common.conflict(ctx)
    if conflict.any():
        lines.append(f"**Conflict:** {int(conflict.sum())} conflict moment(s), at least {cfg['min_people']} people "
                     f"and {cfg['min_vehicles']} vehicles together")
    if lines:
        st.markdown("\n".join(f"- {line}" for line in lines))
    else:
        st.caption("No surge or conflict events in this window.")
