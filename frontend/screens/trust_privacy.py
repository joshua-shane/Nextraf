"""Trust and privacy: is the data intact, what is stored, and how healthy was the camera."""
from __future__ import annotations

import streamlit as st

from nextraf_viz import trust

from . import ui
from .common import Ctx


def render(ctx: Ctx) -> None:
    ui.page_header(ctx.t("nav_trust"), ctx.t("help_trust"))
    has_counts = (ctx.res.folder / "counts.csv").is_file()  # else a run_nextraf.py folder (frames.csv)
    chain = trust.chain_status(ctx.res.folder / "counts.csv")
    h = trust.camera_health(ctx.counts)
    videos = trust.find_video_files(ctx.res.folder)
    ui.readouts([
        {"label": "Hash chain", "value": {"valid": "Verified", "broken": "Broken", "absent": "Not present"}[chain["state"]],
         "tone": {"valid": "good", "broken": "bad", "absent": "warn"}[chain["state"]],
         "note": f"first bad line {chain['first_bad_line']}" if chain["state"] == "broken"
         else "counts.csv" if has_counts else "frames.csv"},
        {"label": "Footage stored", **trust.footage_card(ctx.res.folder)},
        {"label": "Frames with no camera data", "value": f"{h['no_data']}", "unit": f"of {h['frames']}",
         "note": f"{h['no_data_pct']:.1f}% shown as gaps"},
    ])

    st.markdown("#### Data integrity")
    if chain["state"] == "valid":
        st.success(ctx.t("chain_valid"))
    elif chain["state"] == "broken":
        st.error(ctx.t("chain_broken", line=chain["first_bad_line"]))
    else:
        st.warning(ctx.t(trust.chain_absent_key(ctx.res.folder)))
    st.caption(ctx.t("chain_limit"))

    st.markdown("#### Privacy")
    st.write(ctx.t("privacy_text"))
    if videos:
        st.warning(ctx.t("privacy_warn", files=", ".join(videos)))

    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown("#### Files in the results folder")
        inv = trust.file_inventory(ctx.res.folder).rename(
            columns={"path": "File", "size_bytes": "Size (bytes)", "kind": "Type"})
        st.dataframe(inv, hide_index=True, width="stretch")
        if not videos:
            st.caption(ctx.t("stored_images" if (inv["Type"] == "image").any() else "stored_counts_only"))
    with b:
        st.markdown("#### Device")
        if ctx.res.device:
            st.dataframe({"Field": list(ctx.res.device), "Value": [str(v) for v in ctx.res.device.values()]},
                         hide_index=True, width="stretch")
        else:
            st.info("device.json not found, so device details are hidden.")

    if ctx.bench is not None:
        st.markdown("#### Temperature and throttling per run")
        cols = [c for c in ("config_id", "temp_before_c", "temp_after_c", "throttled", "not_trustworthy")
                if c in ctx.bench.columns]
        st.dataframe(ctx.bench[cols], hide_index=True, width="stretch")
        st.caption("Runs that throttled or ended above 80 °C are marked not trustworthy.")
