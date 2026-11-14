"""What-if and compare: try other site settings, or put two model setups side by side."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from nextraf_viz import charts, insights

from . import common, ui
from .common import Ctx


def render(ctx: Ctx) -> None:
    ui.page_header(ctx.t("nav_whatif"), ctx.t("help_whatif"))
    _site_what_if(ctx)
    _compare(ctx)


def _site_what_if(ctx: Ctx) -> None:
    st.markdown("#### What if the walking area or thresholds were different?")
    site, cfg = ctx.site, ctx.site["surge"]
    c = st.columns(4)
    area = c[0].number_input("Walking area (m²)", 1.0, 500.0, float(site["area_m2"] or 20.0), 1.0,
                             help="Assumed unless measured. Smaller area means more crowded.")
    warn = c[1].number_input("Warn density (people/m²)", 0.1, 10.0, float(cfg["warn_density_ppm2"]), 0.1)
    rise = c[2].number_input("Warn rise (people/m² per min)", 0.0, 5.0, float(cfg["warn_rise_ppm2_per_min"]), 0.01)
    crit = c[3].number_input("Critical density (people/m²)", 0.1, 20.0, float(cfg["critical_density_ppm2"]), 0.1)
    if not site["los_bands"]:
        st.info("No crowding bands in site.json, so time in the most crowded band is not shown.")
    alt = insights.with_site(ctx, area_m2=area, surge={"warn_density_ppm2": warn, "warn_rise_ppm2_per_min": rise,
                                                       "critical_density_ppm2": crit})
    rows = []
    for name, c_ in (("Current settings", ctx), ("What-if settings", alt)):
        s = common.surge(c_)
        n_warn = int(s[2]["level"].isin(["warning", "critical"]).sum()) if s else 0
        rows.append({"Setup": name, "Walking area (m²)": c_.site["area_m2"], "Verdict": insights.verdict(c_)["level"],
                     "Surge warning frames": n_warn,
                     "Minutes in most crowded band": round(insights.minutes_in_worst_band(c_), 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("All inputs here are assumptions for exploring; your files are not changed.")
    s = common.surge(alt)
    ui.chart(charts.surge_fig(alt.counts["time"], s[1], s[2], warn, crit))
    st.caption(ctx.t("cap_surge"))


def _compare(ctx: Ctx) -> None:
    st.markdown("#### Compare two model setups")
    if ctx.bench is None:
        st.info(ctx.res.notes[0])
        return
    ids = list(ctx.bench["config_id"])
    rec = ctx.rec
    a_default = ids.index(rec["config_id"]) if rec is not None and rec["config_id"] in ids else 0
    c = st.columns(2)
    a = c[0].selectbox("Setup A", ids, index=a_default)
    b = c[1].selectbox("Setup B", ids, index=min(len(ids) - 1, a_default + 1) if len(ids) > 1 else 0)
    res = insights.compare_configs(ctx.bench, ctx.res.per_frame, a, b)
    labels = {"e2e_median_ms": "Time per frame (ms)", "fps": "Throughput (FPS)", "peak_mem_mb": "Peak memory (MB)",
              "model_size_mb": "Model size (MB)", "mae": "MAE (people/frame)", "within1_pct": "Within ±1 person (%)",
              "mean_signed_error": "Mean signed error (people/frame)"}
    table = pd.DataFrame({"Measure": list(labels.values()), "A": [res["a"][k] for k in labels],
                          "B": [res["b"][k] for k in labels], "B minus A": [res["delta"][k] for k in labels]}).round(2)
    st.dataframe(table, hide_index=True, width="stretch")
    if res["frames"] is not None:
        ui.chart(charts.compare_fig(res["frames"], a, b))
        st.caption("Hand count against each setup's count, frame by frame. Closer to the grey line is more accurate.")
    elif ctx.res.per_frame is None:
        st.caption("per_frame_eval.csv not found, so the per-frame chart is hidden.")
    st.caption("Negative mean signed error means undercounting.")
