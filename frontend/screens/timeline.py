"""Screen 2: Crowding timeline."""
from __future__ import annotations

import streamlit as st

from nextraf_viz import charts, crowding

from . import common, ui
from .common import Ctx


def render(ctx: Ctx) -> None:
    ui.page_header(ctx.t("nav_timeline"), ctx.t("help_timeline"))
    counts, site, col = ctx.counts, ctx.site, ctx.col
    series = counts[col]
    rolling = crowding.rolling_mean(series, ctx.window)
    flags = crowding.stability_flags(series, window=max(ctx.window, 3), k=ctx.stability_k)
    unflagged = series.where(~flags)
    pos = crowding.peak(unflagged)[0] if unflagged.notna().any() else None

    band = None
    if col == "person" and {"count_low", "count_high"} <= set(counts.columns):
        band = (counts["count_low"], counts["count_high"], "Uncertainty band (from counts.csv)")
    elif col == "person" and ctx.rec is not None:
        mae = float(ctx.rec["mae"])
        band = ((series - mae).clip(lower=0), series + mae, f"± MAE of {ctx.rec['config_id']} (estimate)")

    st.markdown(f"#### {ctx.label} per frame")
    ui.chart(charts.timeline_fig(counts, col, ctx.label, rolling, pos, flags, band))
    st.caption(ctx.t("cap_timeline"))
    if flags.any():
        st.caption(f"{int(flags.sum())} frame(s) flagged as implausible "
                   f"(beyond {ctx.stability_k:g} × MAD of the neighbours).")

    _vehicles(ctx)
    area, bands = site["area_m2"], site["los_bands"]
    st.markdown("#### Space per person")
    if not area or not bands:
        st.info(ctx.t("no_area"))
        return
    st.caption(ctx.t("crowding_ref"))
    space = crowding.space_per_person(counts["person"], area)
    if site["los_bands_are_placeholders"]:
        st.warning(ctx.t("placeholder_los"))
    if site["area_is_assumed"]:
        st.caption(ctx.t("area_assumed"))
    ui.chart(charts.space_fig(counts["time"], space, bands))
    st.caption(ctx.t("cap_space"))
    table = crowding.time_in_band(counts["minutes"], space, bands).rename(
        columns={"band": "Crowding band", "minutes": "Minutes in band"})
    st.dataframe(table.round({"Minutes in band": 1}), hide_index=True, width="stretch")

    if common.has_encroachment(ctx):
        _encroachment(ctx, space, bands)
    _surge(ctx)


def _vehicles(ctx: Ctx) -> None:
    counts = ctx.counts
    if not (counts["vehicles_total"].fillna(0) > 0).any():
        return
    st.markdown("#### People and vehicles")
    cfg = ctx.site["conflict"]
    conflict = common.conflict(ctx)
    ui.chart(charts.overlay_fig(counts, conflict))
    st.caption(ctx.t("cap_overlay"))
    st.caption(f"Conflict = at least {cfg['min_people']} people and {cfg['min_vehicles']} vehicles together "
               f"(assumed thresholds; implausible frames ignored). {int(conflict.sum())} moment(s) found. " + ctx.t("rickshaw_note"))


def _encroachment(ctx: Ctx, space_before, bands) -> None:
    st.markdown(f"#### Encroachment: space before and after vehicles on the {ctx.t('footpath')}")
    usable = common.usable_area(ctx)
    after = crowding.space_per_person(ctx.counts["person"], 1.0) * usable
    ui.chart(charts.encroachment_fig(ctx.counts["time"], space_before, after, bands))
    st.caption(ctx.t("cap_encroach", footpath=ctx.t("footpath")))
    st.caption(ctx.t("assumed_footprint"))
    lost = float((ctx.site["area_m2"] - usable).max())
    st.caption(f"Most area taken by vehicles in any frame: {lost:.1f} m² of {ctx.site['area_m2']:g} m².")


def _surge(ctx: Ctx) -> None:
    s = common.surge(ctx)
    if s is None:
        return
    _, density, levels = s
    cfg = ctx.site["surge"]
    st.markdown("#### Crowd-surge early warning")
    ui.chart(charts.surge_fig(ctx.counts["time"], density, levels, cfg["warn_density_ppm2"],
                              cfg["critical_density_ppm2"]))
    st.caption(ctx.t("cap_surge"))
    st.caption(ctx.t("assumed_surge"))
    ev = crowding.surge_events(levels)
    if ev.empty:
        st.info("No surge warnings in this window.")
        return
    ev["Time"] = ctx.counts["time"].iloc[ev["index"]].dt.strftime("%H:%M:%S").to_numpy()
    ev["Density (people/m²)"] = density.iloc[ev["index"]].round(2).to_numpy()
    ev["Rise (people/m² per min)"] = ev["rise_per_min"].round(3)
    st.dataframe(ev[["Time", "level", "Density (people/m²)", "Rise (people/m² per min)"]].rename(
        columns={"level": "Level"}), hide_index=True, width="stretch")
