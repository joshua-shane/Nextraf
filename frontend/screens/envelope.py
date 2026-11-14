"""Screen 3: Affordability envelope."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from nextraf_viz import charts, recommend

from . import common, ui
from .common import Ctx

TABLE_COLS = ["config_id", "model", "resolution", "precision", "threshold", "e2e_median_ms", "e2e_p95_ms",
              "fps", "peak_mem_mb", "model_size_mb", "mae", "within1_pct", "mean_signed_error", "not_trustworthy"]


def render(ctx: Ctx) -> None:
    ui.page_header(ctx.t("nav_envelope"), ctx.t("help_envelope"))
    bench = ctx.bench
    if bench is None:
        st.info(ctx.res.notes[0])
        return
    pool = bench[~bench["not_trustworthy"]] if ctx.exclude_untrusted else bench
    rec = ctx.rec
    if rec is None:
        st.info("No eligible configuration. Turn off the not-trustworthy filter in Analysis settings.")
        return
    best = pool["mae"].min()
    ui.readouts([
        {"label": "Recommended setup", "value": rec["config_id"], "small": True,
         "note": f"fastest within {ctx.tolerance:g} of the best MAE ({best:.2f})"},
        {"label": "Time per frame", "value": f"{rec['e2e_median_ms']:.0f}", "unit": "ms", "note": f"{rec['fps']:.1f} FPS"},
        {"label": "Peak memory", "value": f"{rec['peak_mem_mb']:.0f}", "unit": "MB"},
        {"label": "Count error (MAE)", "value": f"{rec['mae']:.2f}", "unit": "people/frame"},
    ])

    adaptive = common.adaptive_result(ctx)
    x_col = "fps" if st.toggle("Show throughput (FPS) instead of time per frame", value=False) else "e2e_median_ms"
    front = recommend.pareto_frontier(pool, "e2e_median_ms", "mae")
    ui.chart(charts.envelope_fig(bench, front, rec, x_col, adaptive=adaptive))
    st.caption(ctx.t("cap_envelope") + " Colour shows precision, shape shows the model, size shows peak memory. "
               "Faded dots are not-trustworthy runs.")
    ui.chart(charts.mae_memory_fig(pool, rec))
    st.caption(ctx.t("cap_memmae"))

    _adaptive_card(ctx, adaptive, rec)
    _hardware_card(ctx, rec)
    _energy_card(ctx, rec)

    st.markdown("#### All configurations")
    f1, f2 = st.columns(2)
    precisions = f1.multiselect("Precision", sorted(bench["precision"].unique()),
                                default=sorted(bench["precision"].unique()))
    resolutions = f2.multiselect("Resolution (px)", sorted(bench["resolution"].unique()),
                                 default=sorted(bench["resolution"].unique()))
    view = bench[bench["precision"].isin(precisions) & bench["resolution"].isin(resolutions)]
    view = view[[c for c in TABLE_COLS if c in view.columns]].copy()
    view["not_trustworthy"] = view["not_trustworthy"].map({True: "not trustworthy", False: ""})
    st.dataframe(view, hide_index=True, width="stretch")


def _adaptive_card(ctx: Ctx, adaptive: dict | None, rec: pd.Series) -> None:
    st.markdown("#### Adaptive model switching (simulated)")
    if adaptive is None:
        st.info("The adaptive simulation needs benchmark.csv and per_frame_eval.csv with at least two trusted configurations.")
        return
    b = ctx.bench.set_index("config_id")
    fast, acc = b.loc[adaptive["fast_id"]], b.loc[adaptive["accurate_id"]]
    st.write(f"Fast setup **{adaptive['fast_id']}**, accurate setup **{adaptive['accurate_id']}**"
             + (" (picked automatically: the fastest and the most accurate trusted runs)" if adaptive["auto_picked"] else "")
             + f". It switches to the accurate setup at {ctx.switch_count} or more people; change that in Analysis settings.")
    table = pd.DataFrame({
        "Setup": ["Adaptive (simulated)", f"Fast only ({adaptive['fast_id']})", f"Accurate only ({adaptive['accurate_id']})"],
        "MAE (people/frame)": [adaptive["mae"], fast["mae"], acc["mae"]],
        "Time per frame (ms)": [adaptive["e2e_median_ms"], fast["e2e_median_ms"], acc["e2e_median_ms"]],
        "Peak memory (MB)": [adaptive["peak_mem_mb"], fast["peak_mem_mb"], acc["peak_mem_mb"]],
    }).round(2)
    table.loc[0, "Setup"] += ", memory estimated"
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(ctx.t("cap_adaptive") + f" Accurate model used on {adaptive['accurate_share'] * 100:.0f}% of frames.")


def _hardware_card(ctx: Ctx, rec: pd.Series) -> None:
    st.markdown("#### Hardware fit and cost")
    hw = ctx.site["hardware"]
    if not hw["boards"]:
        st.info("No board prices in site.json (hardware.boards), so the cost card is hidden.")
        return
    st.caption(f"A board fits if peak memory + OS overhead ({hw['os_overhead_mb']:g} MB, assumed) ≤ "
               f"{(1 - hw['headroom_fraction']) * 100:.0f}% of its RAM. {hw['prices_note']}")
    edited = st.data_editor(pd.DataFrame(hw["boards"]), hide_index=True, width="stretch",
                            column_config={"ram_gb": st.column_config.NumberColumn("RAM (GB)", disabled=True),
                                           "price": st.column_config.NumberColumn(f"Price ({ctx.currency})", min_value=0)})
    boards = edited.to_dict("records")
    board = recommend.cheapest_board(rec["peak_mem_mb"], boards, hw["os_overhead_mb"], hw["headroom_fraction"])
    if board is None:
        st.error(f"No listed board fits {rec['config_id']} ({rec['peak_mem_mb']:.0f} MB peak).")
        return
    msg = f"Cheapest board that fits **{rec['config_id']}** ({rec['peak_mem_mb']:.0f} MB peak): " \
          f"**{board['ram_gb']} GB, {ctx.currency}{board['price']:,.0f}**."
    four = next((b for b in boards if b["ram_gb"] == 4), None)
    if four is not None:
        diff = four["price"] - board["price"]
        msg += f" Versus a 4 GB board: {ctx.currency}{abs(diff):,.0f} {'cheaper' if diff >= 0 else 'dearer'} per node."
    st.success(msg)


def _energy_card(ctx: Ctx, rec: pd.Series) -> None:
    if "power_w" not in ctx.bench.columns or pd.isna(rec.get("power_w")):
        return
    st.markdown("#### Energy (estimate)")
    e = recommend.energy(float(rec["power_w"]), float(rec["e2e_median_ms"]),
                         recommend.window_hours(ctx.site["time_window"]))
    ui.readouts([
        {"label": "Energy per frame", "value": f"{e['j_per_frame']:.2f}", "unit": "J"},
        {"label": "Per hour", "value": f"{e['wh_per_hour']:.1f}", "unit": "Wh"},
        {"label": "Over the site window", "value": "n/a" if e["wh_window"] is None else f"{e['wh_window']:.1f}",
         "unit": None if e["wh_window"] is None else "Wh"},
        {"label": "Rough battery size", "value": "n/a" if e["battery_wh"] is None else f"{e['battery_wh']:.1f}",
         "unit": None if e["battery_wh"] is None else "Wh"},
    ])
    st.caption(f"From measured power ({rec['power_w']:g} W) of {rec['config_id']}, assuming continuous running. "
               "Battery = energy needed ÷ 0.8 usable. All estimates.")
