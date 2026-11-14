"""Export and report: download the data or build a one-page site report."""
from __future__ import annotations

import streamlit as st

from nextraf_viz import report

from . import ui
from .common import Ctx


def _settings_key(ctx: Ctx) -> tuple:
    return (str(ctx.res.folder), ctx.region, ctx.tolerance, ctx.stability_k, ctx.window,
            ctx.exclude_untrusted, ctx.switch_count)


def render(ctx: Ctx) -> None:
    ui.page_header(ctx.t("nav_export"), ctx.t("help_export"))
    a, b = st.columns(2, gap="large")
    with a, st.container(border=True):
        st.markdown("#### Data downloads")
        counts = ctx.counts
        if len(counts) > 1:
            lo, hi = st.select_slider("Time range to include", options=range(len(counts)),
                                      value=(0, len(counts) - 1),
                                      format_func=lambda i: f"{counts['time'].iloc[i]:%H:%M:%S}")
        else:
            lo, hi = 0, 0
        chosen = counts.iloc[lo:hi + 1].drop(columns=["minutes"])
        ui.download("Download counts CSV", chosen.to_csv(index=False).encode("utf-8"), "nextraf_counts.csv", "text/csv")
        if ctx.bench is not None:
            bench = ctx.bench[~ctx.bench["not_trustworthy"]] if ctx.exclude_untrusted else ctx.bench
            ui.download("Download benchmark CSV", bench.to_csv(index=False).encode("utf-8"), "nextraf_benchmark.csv",
                        "text/csv")
        else:
            st.info("benchmark.csv not found, so the benchmark download is hidden.")

    with b, st.container(border=True):
        st.markdown("#### One-page site report")
        st.caption("Peak occupancy, time in the worst bands, surge, conflict and encroachment events, the recommended "
                   "configuration and board, and the limitations. The HTML file works offline and includes the charts.")
        if st.button("Build site report", type="primary", icon=":material/description:"):
            with st.spinner("Building report"):
                sections = report.build_sections(ctx)
                title = f"Site report: {ctx.site['site_name']}"
                st.session_state["report"] = {"key": _settings_key(ctx), "md": report.to_markdown(title, sections),
                                              "html": report.to_html(title, sections)}
        built = st.session_state.get("report")
        if built and built["key"] == _settings_key(ctx):
            c = st.columns(2)
            with c[0]:
                ui.download("Download report (Markdown)", built["md"].encode("utf-8"), "nextraf_site_report.md",
                            "text/markdown", width="stretch")
            with c[1]:
                ui.download("Download report (HTML, offline)", built["html"].encode("utf-8"),
                            "nextraf_site_report.html", "text/html", width="stretch")
        elif built:
            st.info("Settings changed since the report was built. Build it again to include them.")
