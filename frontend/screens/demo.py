"""Guided demo: a five-step story for the 2-minute pitch, laid out like slides."""
from __future__ import annotations

import streamlit as st

from nextraf_viz import charts, crowding, insights, recommend, trust

from . import common, ui
from .common import Ctx

STEPS = ["The problem", "This site", "Cheapest setup", "Trust and privacy", "Honest limits"]


def render(ctx: Ctx) -> None:
    step = st.session_state.setdefault("demo_step", 0)
    ui.stepper(STEPS, step)
    with st.container(key="slide"):
        [_problem, _site, _setup, _trust, _limits][step](ctx)
    st.divider()
    back, nxt, note = st.columns([1, 1, 4], vertical_alignment="center")
    if back.button("Previous", icon=":material/arrow_back:", disabled=step == 0, width="stretch"):
        st.session_state["demo_step"] = step - 1
        st.rerun()
    if nxt.button("Next", icon=":material/arrow_forward:", icon_position="right", disabled=step == len(STEPS) - 1,
                  type="primary", width="stretch"):
        st.session_state["demo_step"] = step + 1
        st.rerun()
    note.caption(f"Step {step + 1} of {len(STEPS)}. Every number on these slides comes from the loaded results folder.")


def _problem(ctx: Ctx) -> None:
    st.subheader(ctx.t("demo_problem_title"))
    a, b = st.columns([1, 1.45], gap="large")
    with a:
        ui.stat(ctx.t("demo_stat_value"), ctx.t("demo_stat_label"), ctx.t("demo_stat_source"))
    with b:
        ui.lede(ctx.t("demo_problem"), ctx.t("demo_stadium"))
    st.subheader(ctx.t("demo_solution_title"))
    ui.lede(ctx.t("demo_solution"))
    ui.compare(("Cloud CCTV", "NEXTRAF on a Raspberry Pi 5"), ctx.t("demo_compare"))


def _site(ctx: Ctx) -> None:
    site, counts = ctx.site, ctx.counts
    st.subheader(site["site_name"])
    v = insights.verdict(ctx)
    st.markdown(f'{ui.chip(v["level"], on_paper=True)}&nbsp;&nbsp;<span class="nx-body">{ui._esc(v["text"])}</span>',
                unsafe_allow_html=True)
    st.write("")
    flags = crowding.stability_flags(counts["person"], window=max(ctx.window, 3), k=ctx.stability_k)
    pos, peak = crowding.peak(counts["person"].where(~flags))
    ui.readouts([
        {"label": "Peak occupancy", "value": f"{peak:.0f}", "unit": "people", "note": f"at {counts['time'].iloc[pos]:{ctx.tf}}"},
        {"label": "Time in most crowded band", "value": f"{insights.minutes_in_worst_band(ctx):.1f}", "unit": "min",
         "note": "walking area assumed" if site["area_is_assumed"] else None},
        {"label": "Frames processed", "value": f"{len(counts)}", "note": f"{site['time_window'] or 'time window not set'}"},
    ])
    ui.chart(charts.timeline_fig(counts, "person", "People", crowding.rolling_mean(counts["person"], ctx.window),
                                 pos, flags, None, height=320))
    st.caption(ctx.t("cap_timeline"))


def _setup(ctx: Ctx) -> None:
    st.subheader("The cheapest setup that still counts reliably")
    rec = ctx.rec
    if rec is None or ctx.bench is None:
        st.info("Needs benchmark.csv from the Pi to show the affordability envelope.")
        return
    pool = ctx.bench[~ctx.bench["not_trustworthy"]] if ctx.exclude_untrusted else ctx.bench
    front = recommend.pareto_frontier(pool, "e2e_median_ms", "mae")
    hw = ctx.site["hardware"]
    board = recommend.cheapest_board(rec["peak_mem_mb"], hw["boards"], hw["os_overhead_mb"], hw["headroom_fraction"]) \
        if hw["boards"] else None
    a, b = st.columns([1.9, 1], gap="large")
    with a:
        ui.chart(charts.envelope_fig(ctx.bench, front, rec, "e2e_median_ms", adaptive=common.adaptive_result(ctx),
                                     height=400))
        st.caption(ctx.t("cap_envelope"))
    with b:
        ui.readouts([
            {"label": "Recommended setup", "value": rec["config_id"], "small": True},
            {"label": "Speed", "value": f"{rec['fps']:.1f}", "unit": "FPS", "note": f"{rec['e2e_median_ms']:.0f} ms per frame"},
            {"label": "Count error", "value": f"{rec['mae']:.2f}", "unit": "people/frame", "note": "mean absolute error"},
            {"label": "Cheapest board that fits", "value": "n/a" if board is None else f"{board['ram_gb']} GB",
             "note": None if board is None else f"{ctx.currency}{board['price']:,.0f} per node"},
        ], stack=True)


def _trust(ctx: Ctx) -> None:
    st.subheader("No video stored, and every number is checkable")
    folder = ctx.res.folder
    videos = trust.find_video_files(folder)
    chain = trust.chain_status(folder / "counts.csv")
    h = trust.camera_health(ctx.counts)
    stored = trust.privacy_key(folder)
    ui.proofs([
        {"title": {"privacy_warn": "Video files found", "stored_images": "Validation images only"}.get(stored, "Counts only"),
         "text": ("Frames are processed in memory and overwritten by the next one. Only integer counts are written."
                  if stored == "privacy_ok" else ctx.t(stored, files=", ".join(videos))),
         "tone": "bad" if videos else "good"},
        {"title": {"valid": "Hash chain verified", "broken": "Hash chain broken", "absent": "No hash chain"}[chain["state"]],
         "text": ctx.t({"valid": "chain_valid", "broken": "chain_broken",
                        "absent": trust.chain_absent_key(folder)}[chain["state"]], line=chain["first_bad_line"]),
         "tone": {"valid": "good", "broken": "bad", "absent": "warn"}[chain["state"]]},
        {"title": f"{h['frames']} frames, {h['no_data']} with no data",
         "text": "Frames the camera could not see are shown as gaps, never as zero people.",
         "tone": "good" if h["no_data"] == 0 else "warn"},
    ])
    st.write("")
    st.caption(ctx.t("privacy_text"))


def _limits(ctx: Ctx) -> None:
    st.subheader("What NEXTRAF does not claim")
    st.markdown("\n".join(f"- {line}" for line in ctx.t("limitations")))
    if ctx.is_sample:
        st.warning(ctx.t("sample_banner"))
    if ctx.site["los_bands_are_placeholders"]:
        st.warning(ctx.t("placeholder_los"))
