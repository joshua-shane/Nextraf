"""Live replay: step or play through the recorded counts, frame by frame (from the file, no camera)."""
from __future__ import annotations

import streamlit as st

from nextraf_viz import charts, insights

from . import ui
from .common import Ctx


def render(ctx: Ctx) -> None:
    ui.page_header(ctx.t("nav_replay"), ctx.t("help_replay"))
    n = len(ctx.counts)
    st.session_state.setdefault("replay_i", 0)
    c = st.columns([1, 1.4, 1.4, 3], vertical_alignment="bottom")
    playing = c[0].toggle("Play", key="replay_play")
    if c[1].button("Previous frame", icon=":material/skip_previous:", disabled=playing, width="stretch"):
        st.session_state["replay_i"] = max(0, st.session_state["replay_i"] - 1)
    if c[2].button("Next frame", icon=":material/skip_next:", disabled=playing, width="stretch"):
        st.session_state["replay_i"] = min(n - 1, st.session_state["replay_i"] + 1)
    speed = c[3].select_slider("Speed (seconds per frame)", options=[0.5, 1.0, 2.0], value=1.0)
    st.fragment(run_every=speed if playing else None)(_frame)(ctx, n, playing)


def _frame(ctx: Ctx, n: int, playing: bool) -> None:
    if playing:
        st.session_state["replay_i"] = (st.session_state["replay_i"] + 1) % n
    times = ctx.counts["time"]
    index = st.select_slider("Frame", options=range(n), format_func=lambda i: f"{times.iloc[i]:%H:%M:%S}",
                             key="replay_i", label_visibility="collapsed")
    s = insights.replay_state(ctx, index)
    left, right = st.columns(2, gap="medium")
    with left:
        ui.counter(s["time"], index, n, s["people"], s["vehicles"])
        tags = [f"Surge level: {s['level']}"]
        if s["band"]:
            tags.append(f"Band: {s['band']}")
        if s["conflict"]:
            tags.append("Conflict moment")
        if s["implausible"]:
            tags.append("Implausible count")
        if not s["camera_ok"]:
            tags.append("Camera: no data")
        ui.pills(tags)
    with right:
        cfg = ctx.site["surge"]
        if s["density"] is not None or not s["camera_ok"]:
            ui.density(s["density"], cfg["warn_density_ppm2"], cfg["critical_density_ppm2"])
            st.caption(ctx.t("cap_gauge"))
        else:
            st.info("Add a walking area in site.json to see the density gauge.")
    ui.chart(charts.replay_fig(ctx.counts, "person", index), key=f"replay_{index}")
    st.caption("Counts so far in blue, the rest faded. Replayed from the saved file, not a camera.")
