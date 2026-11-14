"""Screen 4: Validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nextraf_viz import charts

from . import ui
from .common import Ctx


def metrics(pf: pd.DataFrame) -> dict:
    """MAE, share of frames within one person, mean signed error (negative = undercount)."""
    err = pf["pred_count"] - pf["true_count"]
    return {"mae": float(err.abs().mean()), "within1": float((err.abs() <= 1).mean() * 100),
            "bias": float(err.mean())}


def render(ctx: Ctx) -> None:
    ui.page_header(ctx.t("nav_validation"), ctx.t("help_validation"))
    pf_all = ctx.res.per_frame
    if pf_all is None:
        st.info(ctx.res.notes[1] if len(ctx.res.notes) > 1 else "per_frame_eval.csv not found.")
    else:
        ids = list(dict.fromkeys(pf_all["config_id"]))
        rec = ctx.rec
        default = ids.index(rec["config_id"]) if rec is not None and rec["config_id"] in ids else 0
        cid = st.selectbox("Configuration", ids, index=default)
        pf_cfg = pf_all[pf_all["config_id"] == cid]
        classes = sorted(pf_cfg["class"].unique())
        pf = pf_cfg[pf_cfg["class"] == "person"] if "person" in classes else pf_cfg
        m = metrics(pf)
        ui.readouts([
            {"label": "Mean absolute error", "value": f"{m['mae']:.2f}", "unit": "people/frame"},
            {"label": "Frames within one person", "value": f"{m['within1']:.0f}", "unit": "%"},
            {"label": "Mean signed error", "value": f"{m['bias']:+.2f}", "unit": "people/frame",
             "note": "undercounting" if m["bias"] < 0 else "overcounting" if m["bias"] > 0 else "no bias"},
            {"label": "Frames checked by hand", "value": f"{len(pf)}"},
        ])
        a, b = st.columns(2, gap="large")
        with a:
            ui.chart(charts.pred_vs_true_fig(pf))
            st.caption(ctx.t("cap_pred_true"))
        with b:
            ui.chart(charts.error_vs_size_fig(pf))
            st.caption(ctx.t("cap_err_size"))
        if len(classes) > 1:
            per_class = {k: metrics(g)["mae"] for k, g in pf_cfg.groupby("class")}
            st.dataframe({"Class": list(per_class), "MAE (people/frame)": np.round(list(per_class.values()), 2)},
                         hide_index=True, width="stretch")
    st.markdown("#### Annotated samples")
    if not ctx.res.samples:
        st.info("No annotated sample images found in results/samples/.")
        return
    cols = st.columns(min(len(ctx.res.samples), 3))
    for i, img in enumerate(ctx.res.samples):
        with cols[i % len(cols)]:
            ui.image(img, img.name)
