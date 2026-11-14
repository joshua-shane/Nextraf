"""Plotly figure builders. Colour-blind-safe (Okabe-Ito) and never colour-only: markers, labels and dashes carry meaning too."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import io

# Okabe-Ito based (colour-blind safe); the main series uses the app's sign blue.
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, GREY = "#0A4DA2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#7A8799"
PRECISION_COLOURS = {"fp32": BLUE, "int8": ORANGE}
MODEL_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x"]
INK, MUTED, GRID, AXIS, NAVY = "#102A43", "#55667D", "#E9EEF4", "#C5D1DF", "#0B2545"
FONT = '"Source Sans", "Source Sans Pro", "Segoe UI", sans-serif'


def _base(fig: go.Figure, xtitle: str, ytitle: str, height: int = 380) -> go.Figure:
    is_time = "Time" in xtitle
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=14, t=44, b=8),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(family=FONT, size=13, color=MUTED),
        xaxis_title=xtitle, yaxis_title=ytitle,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12.5, color=INK)),
        hovermode="x unified" if is_time else "closest",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=13, color=INK)))
    fig.update_xaxes(showline=True, linecolor=AXIS, gridcolor=GRID, zeroline=False, ticks="outside", tickcolor=AXIS,
                     ticklen=4, title_font=dict(size=12.5, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_font=dict(size=12.5, color=MUTED))
    fig.update_annotations(font=dict(family=FONT, size=11.5, color=MUTED))
    if is_time:
        xs = [pd.Series(pd.to_datetime(list(t.x))) for t in fig.data if t.x is not None and len(t.x)]
        fig.update_xaxes(tickformat=io.time_format(pd.concat(xs)) if xs else "%H:%M")
    return fig


def timeline_fig(df: pd.DataFrame, col: str, label: str, rolling: pd.Series, peak_pos: int | None,
                 flags: pd.Series, band: tuple[pd.Series, pd.Series, str] | None, height: int = 380) -> go.Figure:
    """Count over time with a rolling mean, peak marker, stability flags, optional band and camera gaps."""
    fig = go.Figure()
    if band is not None:
        low, high, band_label = band
        fig.add_trace(go.Scatter(x=df["time"], y=high, mode="lines", line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=df["time"], y=low, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(10,77,162,0.12)", name=band_label, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines+markers", name=label, connectgaps=False,
                             line=dict(color=BLUE, width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=df["time"], y=rolling, mode="lines", name="Rolling mean", connectgaps=True,
                             line=dict(color=GREY, width=2, dash="dash")))
    if peak_pos is not None:
        fig.add_trace(go.Scatter(x=[df["time"].iloc[peak_pos]], y=[df[col].iloc[peak_pos]], mode="markers+text",
                                 name="Peak", text=["Peak"], textposition="top center",
                                 marker=dict(color=GREEN, size=13, symbol="star")))
    if flags.any():
        fig.add_trace(go.Scatter(x=df["time"][flags], y=df[col][flags], mode="markers", name="Implausible frame",
                                 marker=dict(color=RED, size=12, symbol="x", line=dict(width=2))))
    for t in df["time"][~df["camera_ok"]]:
        half = (df["time"].diff().median() or pd.Timedelta(seconds=30)) / 2
        fig.add_vrect(x0=t - half, x1=t + half, fillcolor="rgba(107,122,143,0.18)", line_width=0,
                      annotation_text="No data", annotation_position="top left")
    return _base(fig, "Time", f"{label} (people/frame)" if col == "person" else f"{label} (per frame)", height)


def space_fig(time: pd.Series, space: pd.Series, bands: list[dict]) -> go.Figure:
    """Space per person over time with LOS bands shaded behind it."""
    fig = go.Figure()
    top = float(np.nanmax(space.to_numpy(dtype=float))) * 1.15 if space.notna().any() else 10.0
    for b in bands:
        hi = b["max_m2_per_person"] if b["max_m2_per_person"] is not None else max(top, b["min_m2_per_person"] + 1)
        fig.add_hrect(y0=b["min_m2_per_person"], y1=hi, fillcolor=b.get("color", "#ddd"), opacity=0.55,
                      line_width=0, annotation_text=b["name"], annotation_position="top right",
                      annotation_font_size=10)
    fig.add_trace(go.Scatter(x=time, y=space, mode="lines+markers", name="Space per person", connectgaps=False,
                             line=dict(color=NAVY, width=2), marker=dict(size=6)))
    fig.update_yaxes(range=[0, top])
    return _base(fig, "Time", "Space per person (m²/person)")


def envelope_fig(bench: pd.DataFrame, front: pd.DataFrame, rec: pd.Series | None, x_col: str,
                 targets: bool = True, adaptive: dict | None = None, height: int = 440) -> go.Figure:
    """Accuracy-vs-cost scatter: colour = precision, symbol = model, size = peak memory."""
    fig = go.Figure()
    models = list(dict.fromkeys(bench["model"]))
    mem = bench["peak_mem_mb"]
    for model in models:
        for prec, colour in PRECISION_COLOURS.items():
            sub = bench[(bench["model"] == model) & (bench["precision"] == prec)]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub["mae"], mode="markers", name=f"{model} {prec}",
                marker=dict(color=colour, symbol=MODEL_SYMBOLS[models.index(model) % len(MODEL_SYMBOLS)],
                            size=8 + 14 * (sub["peak_mem_mb"] - mem.min()) / max(mem.max() - mem.min(), 1),
                            opacity=np.where(sub["not_trustworthy"], 0.35, 0.9), line=dict(width=1, color=NAVY)),
                customdata=sub[["config_id", "resolution", "threshold", "peak_mem_mb", "fps"]],
                hovertemplate="%{customdata[0]}<br>" + x_col + ": %{x}<br>MAE: %{y}<br>"
                              "Peak memory: %{customdata[3]} MB<br>FPS: %{customdata[4]}<extra></extra>"))
    if len(front) > 1:
        fig.add_trace(go.Scatter(x=front[x_col], y=front["mae"], mode="lines", name="Pareto frontier",
                                 line=dict(color=NAVY, width=2, dash="dot")))
    if rec is not None:
        fig.add_trace(go.Scatter(x=[rec[x_col]], y=[rec["mae"]], mode="markers", name="Recommended",
                                 marker=dict(symbol="star", size=20, color="rgba(0,0,0,0)",
                                             line=dict(color=RED, width=3))))
    if adaptive is not None:
        ax = adaptive["e2e_median_ms"] if x_col == "e2e_median_ms" else 1000.0 / adaptive["e2e_median_ms"]
        fig.add_trace(go.Scatter(x=[ax], y=[adaptive["mae"]], mode="markers+text", name="Adaptive (simulated)",
                                 text=["Adaptive"], textposition="bottom center",
                                 marker=dict(symbol="diamond-wide", size=16, color=PURPLE,
                                             line=dict(width=2, color=NAVY))))
    if targets:
        if x_col == "e2e_median_ms":
            fig.add_vline(x=200, line_dash="dash", line_color=GREY, annotation_text="Target ≤ 200 ms")
        else:
            fig.add_vline(x=5, line_dash="dash", line_color=GREY, annotation_text="Target ≥ 5 FPS")
        fig.add_hline(y=1, line_dash="dash", line_color=GREY, annotation_text="Target MAE ≤ 1")
    xtitle = "Full-pipeline median time (ms)" if x_col == "e2e_median_ms" else "Throughput (FPS)"
    fig = _base(fig, xtitle, "Count error, MAE (people/frame)", height)
    if x_col == "fps":
        fig.update_xaxes(autorange="reversed")
    return fig


def mae_memory_fig(bench: pd.DataFrame, rec: pd.Series | None) -> go.Figure:
    """MAE against peak memory for every configuration."""
    fig = go.Figure()
    for prec, colour in PRECISION_COLOURS.items():
        sub = bench[bench["precision"] == prec]
        fig.add_trace(go.Scatter(x=sub["peak_mem_mb"], y=sub["mae"], mode="markers", name=prec,
                                 marker=dict(color=colour, size=10, line=dict(width=1, color=NAVY)),
                                 text=sub["config_id"], hovertemplate="%{text}<br>%{x} MB<br>MAE %{y}<extra></extra>"))
    if rec is not None:
        fig.add_trace(go.Scatter(x=[rec["peak_mem_mb"]], y=[rec["mae"]], mode="markers", name="Recommended",
                                 marker=dict(symbol="star", size=20, color="rgba(0,0,0,0)",
                                             line=dict(color=RED, width=3))))
    fig.add_vline(x=300, line_dash="dash", line_color=GREY, annotation_text="Target ≤ 300 MB")
    return _base(fig, "Peak memory (MB)", "Count error, MAE (people/frame)")


def pred_vs_true_fig(pf: pd.DataFrame) -> go.Figure:
    """Detector count against hand count per frame."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pf["frame_idx"], y=pf["true_count"], mode="lines+markers", name="Hand count",
                             line=dict(color=GREY, width=2), marker=dict(symbol="square", size=7)))
    fig.add_trace(go.Scatter(x=pf["frame_idx"], y=pf["pred_count"], mode="lines+markers", name="Detector count",
                             line=dict(color=BLUE, width=2), marker=dict(symbol="circle", size=7)))
    return _base(fig, "Frame", "People per frame")


def error_vs_size_fig(pf: pd.DataFrame) -> go.Figure:
    """Signed error (detector minus hand count) against crowd size, with a zero line."""
    err = pf["pred_count"] - pf["true_count"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pf["true_count"], y=err, mode="markers", name="Frame",
                             marker=dict(color=BLUE, size=9, line=dict(width=1, color=NAVY))))
    fig.add_hline(y=0, line_color=GREY)
    return _base(fig, "Hand count (people/frame)", "Signed error (people/frame)")


def _half_step(time: pd.Series) -> pd.Timedelta:
    step = time.diff().median()
    return (step if pd.notna(step) else pd.Timedelta(seconds=60)) / 2


def overlay_fig(df: pd.DataFrame, conflict: pd.Series) -> go.Figure:
    """People and total vehicles per frame; conflict frames shaded."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["person"], mode="lines+markers", name="People", connectgaps=False,
                             line=dict(color=BLUE, width=2), marker=dict(symbol="circle", size=6)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["vehicles_total"], mode="lines+markers", name="Vehicles (total)",
                             connectgaps=False, line=dict(color=ORANGE, width=2, dash="dot"),
                             marker=dict(symbol="square", size=6)))
    half = _half_step(df["time"])
    for t in df["time"][conflict]:
        fig.add_vrect(x0=t - half, x1=t + half, fillcolor="rgba(213,94,0,0.18)", line_width=0)
    if conflict.any():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Conflict moment (shaded)",
                                 marker=dict(symbol="square", size=12, color="rgba(213,94,0,0.4)")))
    return _base(fig, "Time", "Count per frame")


def encroachment_fig(time: pd.Series, before: pd.Series, after: pd.Series, bands: list[dict]) -> go.Figure:
    """Space per person before and after removing area taken by vehicles."""
    fig = go.Figure()
    top = float(np.nanmax(np.r_[before.to_numpy(dtype=float), after.to_numpy(dtype=float)])) * 1.15
    for b in bands:
        hi = b["max_m2_per_person"] if b["max_m2_per_person"] is not None else max(top, b["min_m2_per_person"] + 1)
        fig.add_hrect(y0=b["min_m2_per_person"], y1=hi, fillcolor=b.get("color", "#ddd"), opacity=0.45, line_width=0)
    fig.add_trace(go.Scatter(x=time, y=before, mode="lines+markers", name="Full area (before encroachment)",
                             connectgaps=False, line=dict(color=GREY, width=2), marker=dict(symbol="circle", size=6)))
    fig.add_trace(go.Scatter(x=time, y=after, mode="lines+markers", name="Usable area (after encroachment)",
                             connectgaps=False, line=dict(color=RED, width=2, dash="dash"),
                             marker=dict(symbol="diamond", size=7)))
    fig.update_yaxes(range=[0, top])
    return _base(fig, "Time", "Space per person (m²/person)")


def surge_fig(time: pd.Series, density: pd.Series, levels: pd.DataFrame, warn: float, critical: float) -> go.Figure:
    """Density over time with warning and critical markers and threshold lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=density, mode="lines+markers", name="Density", connectgaps=False,
                             line=dict(color=NAVY, width=2), marker=dict(size=6)))
    for level, colour, symbol, name in (("warning", ORANGE, "triangle-up", "Warning (assumed threshold)"),
                                        ("critical", RED, "x", "Critical (assumed threshold)")):
        mask = (levels["level"] == level).to_numpy()
        if mask.any():
            fig.add_trace(go.Scatter(x=time[mask], y=density[mask], mode="markers", name=name,
                                     marker=dict(color=colour, symbol=symbol, size=13, line=dict(width=2))))
    fig.add_hline(y=warn, line_dash="dash", line_color=ORANGE, annotation_text="Warn density")
    fig.add_hline(y=critical, line_dash="dash", line_color=RED, annotation_text="Critical density")
    return _base(fig, "Time", "Density (people/m²)")


def compare_fig(frames: pd.DataFrame, a: str, b: str) -> go.Figure:
    """Hand count and two configurations' predictions per frame."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames["frame_idx"], y=frames["true"], mode="lines+markers", name="Hand count",
                             line=dict(color=GREY, width=2), marker=dict(symbol="square", size=7)))
    fig.add_trace(go.Scatter(x=frames["frame_idx"], y=frames["a_pred"], mode="lines+markers", name=f"A: {a}",
                             line=dict(color=BLUE, width=2), marker=dict(symbol="circle", size=7)))
    fig.add_trace(go.Scatter(x=frames["frame_idx"], y=frames["b_pred"], mode="lines+markers", name=f"B: {b}",
                             line=dict(color=ORANGE, width=2, dash="dot"), marker=dict(symbol="diamond", size=7)))
    return _base(fig, "Frame", "People per frame")


def replay_fig(df: pd.DataFrame, col: str, index: int) -> go.Figure:
    """Counts so far (solid) and still to come (faded), with a marker at the current frame."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name="All frames", connectgaps=False,
                             line=dict(color="#CBD6E3", width=2)))
    upto = df.iloc[: index + 1]
    fig.add_trace(go.Scatter(x=upto["time"], y=upto[col], mode="lines+markers", name="So far", connectgaps=False,
                             line=dict(color=BLUE, width=3), marker=dict(size=5)))
    fig.add_vline(x=df["time"].iloc[index], line_color=NAVY, line_width=1, line_dash="dot")
    if pd.notna(df[col].iloc[index]):
        fig.add_trace(go.Scatter(x=[df["time"].iloc[index]], y=[df[col].iloc[index]], mode="markers", name="Now",
                                 marker=dict(symbol="circle", size=13, color=RED, line=dict(width=2, color="#ffffff"))))
    return _base(fig, "Time", "People per frame", 300)
