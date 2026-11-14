"""Shared UI pieces: one stylesheet, the site sign plate, page headers, readouts and the chart wrapper.

Inline CSS and system/bundled fonts only, so everything works offline. Display type is Bahnschrift (the DIN road-sign
face that ships with Windows), then DIN Alternate (macOS), then Streamlit's bundled Source Sans.
"""
from __future__ import annotations

import base64
import html
import os
from pathlib import Path

import streamlit as st

from nextraf_viz import insights

from .common import Ctx

DISPLAY = 'Bahnschrift, "DIN Alternate", "Source Sans", sans-serif'
TONE = {"Calm": "calm", "Busy": "busy", "Surge risk": "surge", "Critical": "critical", "Unknown": "unknown"}
CHART_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}

_CSS = """
<style>
:root{
  --nx-blue:#0A4DA2; --nx-navy:#0B2545; --nx-ink:#102A43; --nx-muted:#55667D; --nx-line:#D6DFEA;
  --nx-mist:#F2F5F9; --nx-paper:#FFFFFF; --nx-sky:#8CC8FF;
  --nx-calm:#17795A; --nx-busy:#A86A12; --nx-surge:#C2410C; --nx-critical:#B42318; --nx-unknown:#6B7A8F;
  --nx-display:__DISPLAY__;
}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMainBlockContainer"]{padding-top:2.8rem;padding-bottom:3rem;max-width:1240px}
.st-key-findings ul{columns:2;column-gap:40px;margin-bottom:.4rem}
.st-key-findings li{break-inside:avoid}
@media (max-width: 900px){ .st-key-findings ul{columns:1} }
[data-testid="stCaptionContainer"]{max-width:82ch}
[data-testid="stMain"] h3,[data-testid="stMain"] h4{color:var(--nx-navy)}
[data-testid="stMain"] h4{margin-top:.6rem}
[data-testid="stAlertContainer"]{padding:.6rem .9rem}
[data-testid="stAlertContainer"] p{font-size:.95rem}
[data-testid="stPlotlyChart"]{background:var(--nx-paper);border:1px solid var(--nx-line);border-radius:8px;padding:6px 6px 2px}
[data-testid="stDataFrame"]{border:1px solid var(--nx-line);border-radius:8px;overflow:hidden}
[data-testid="stMain"] [data-testid="stExpander"] details{background:var(--nx-paper)}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]{background:var(--nx-paper)}
[data-testid="stMetric"]{background:var(--nx-paper);border:1px solid var(--nx-line);border-radius:8px;padding:12px 16px}
[data-testid="stMetricValue"]{font-family:var(--nx-display);color:var(--nx-navy);font-variant-numeric:tabular-nums}

/* Sidebar: brand, grouped navigation */
.nx-brand{padding:2px 2px 14px;margin-bottom:6px;border-bottom:1px solid rgba(255,255,255,.14)}
.nx-brand-name{font-family:var(--nx-display);font-size:1.55rem;font-weight:700;letter-spacing:.06em;color:#fff;line-height:1}
.nx-brand-sub{font-size:.85rem;color:rgba(255,255,255,.72);margin-top:6px}
.nx-side-label{font-size:.8rem;color:rgba(255,255,255,.6);margin:14px 0 2px 2px}
.st-key-nav [data-testid="stRadioGroup"]{gap:1px;width:100%}
.st-key-nav [data-testid="stRadioGroup"] > div{width:100%}
.st-key-nav [data-testid="stRadioOption"]{width:100%;padding:7px 10px;border-radius:6px;cursor:pointer}
.st-key-nav [data-testid="stRadioOption"] > div > div:first-child{display:none}
.st-key-nav [data-testid="stRadioOption"] p{font-size:.95rem;color:rgba(255,255,255,.86)}
.st-key-nav [data-testid="stRadioOption"]:hover{background:rgba(255,255,255,.07)}
.st-key-nav [data-testid="stRadioOption"]:focus-within{outline:2px solid var(--nx-sky);outline-offset:-2px}
.st-key-nav [data-selected="true"] > [data-testid="stRadioOption"]{background:rgba(140,200,255,.16);box-shadow:inset 3px 0 0 var(--nx-sky)}
.st-key-nav [data-selected="true"] > [data-testid="stRadioOption"] p{color:#fff;font-weight:600}
.st-key-nav [data-testid="stRadioGroup"] > div:nth-child(1)::before,
.st-key-nav [data-testid="stRadioGroup"] > div:nth-child(4)::before,
.st-key-nav [data-testid="stRadioGroup"] > div:nth-child(8)::before{display:block;font-size:.78rem;color:rgba(255,255,255,.55);padding:0 10px 4px}
.st-key-nav [data-testid="stRadioGroup"] > div:nth-child(1)::before{content:"Present"}
.st-key-nav [data-testid="stRadioGroup"] > div:nth-child(4)::before{content:"Analyse";margin-top:14px}
.st-key-nav [data-testid="stRadioGroup"] > div:nth-child(8)::before{content:"Check and share";margin-top:14px}

/* The site sign plate: a blue informatory road sign with a white inset keyline */
.nx-plate{background:var(--nx-blue);border-radius:11px;padding:5px;margin:0 0 1.2rem}
.nx-plate-in{border:2px solid rgba(255,255,255,.95);border-radius:8px;padding:14px 20px;display:flex;align-items:center;
  justify-content:space-between;gap:12px 20px;flex-wrap:wrap}
.nx-plate-site{font-family:var(--nx-display);font-size:1.75rem;font-weight:600;color:#fff;line-height:1.15}
.nx-plate-meta{display:flex;flex-wrap:wrap;gap:2px 14px;margin-top:5px;color:rgba(255,255,255,.86);font-size:.92rem}
.nx-plate-meta span + span{border-left:1px solid rgba(255,255,255,.45);padding-left:14px}
.nx-chip{display:inline-flex;align-items:center;gap:9px;background:#fff;color:var(--nx-navy);border-radius:999px;
  padding:7px 16px 7px 13px;font-family:var(--nx-display);font-weight:600;font-size:1.02rem;white-space:nowrap}
.nx-chip.on-paper{border:1px solid var(--nx-line)}
.nx-dot{width:11px;height:11px;border-radius:50%;background:var(--nx-unknown);flex:none}
.nx-calm .nx-dot{background:var(--nx-calm)} .nx-busy .nx-dot{background:var(--nx-busy)}
.nx-surge .nx-dot{background:var(--nx-surge)} .nx-critical .nx-dot{background:var(--nx-critical)}

/* Page header */
.nx-head{margin:0 0 1rem}
.nx-title{font-family:var(--nx-display);font-size:1.45rem;font-weight:600;color:var(--nx-navy);line-height:1.2}
.nx-sub{color:var(--nx-muted);font-size:.97rem;margin-top:4px;max-width:78ch;line-height:1.5}

/* Readouts: one instrument panel divided by hairlines */
.nx-readouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;background:var(--nx-line);
  border:1px solid var(--nx-line);border-radius:8px;overflow:hidden;margin:0 0 1.2rem}
.nx-readouts.stack{grid-template-columns:1fr}
.nx-ro{background:var(--nx-paper);padding:13px 18px 14px;min-width:0}
.nx-ro-label{font-size:.88rem;color:var(--nx-muted)}
.nx-ro-value{font-family:var(--nx-display);font-size:2.05rem;font-weight:600;color:var(--nx-navy);line-height:1.15;
  margin-top:3px;font-variant-numeric:tabular-nums;overflow-wrap:break-word}
.nx-ro-value.small{font-size:1.25rem;line-height:1.3;margin-top:6px}
.nx-ro-unit{font-size:.95rem;font-weight:500;color:var(--nx-muted);margin-left:6px}
.nx-ro-note{font-size:.85rem;color:var(--nx-muted);margin-top:3px}
.nx-ro.good .nx-ro-value{color:var(--nx-calm)} .nx-ro.bad .nx-ro-value{color:var(--nx-critical)}
.nx-ro.warn .nx-ro-value{color:var(--nx-busy)}

/* Spec strip: secondary key facts */
.nx-specs{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:14px 26px;background:var(--nx-paper);
  border:1px solid var(--nx-line);border-radius:8px;padding:14px 18px;margin:1rem 0 1.2rem}
.nx-spec-k{font-size:.85rem;color:var(--nx-muted)}
.nx-spec-v{font-family:var(--nx-display);font-size:1.08rem;font-weight:600;color:var(--nx-navy);margin-top:2px;overflow-wrap:break-word}
.nx-spec-n{font-size:.85rem;color:var(--nx-muted)}
.nx-spec.good .nx-spec-v{color:var(--nx-calm)} .nx-spec.bad .nx-spec-v{color:var(--nx-critical)}

/* Pills */
.nx-pill{display:inline-block;border-radius:999px;padding:3px 11px;margin:0 6px 6px 0;font-size:.85rem;
  border:1px solid var(--nx-line);background:var(--nx-paper);color:var(--nx-navy)}

/* Guided demo: stepper and slides */
.nx-steps{display:flex;gap:6px;list-style:none;padding:0;margin:0 0 1.4rem;flex-wrap:wrap}
.nx-steps li{display:flex;align-items:center;gap:8px;padding:7px 14px 7px 8px;border-radius:999px;font-size:.92rem;
  color:var(--nx-muted);background:var(--nx-paper);border:1px solid var(--nx-line);margin:0}
.nx-steps .n{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;font-family:var(--nx-display);
  font-weight:600;font-size:.85rem;background:var(--nx-mist);color:var(--nx-muted)}
.nx-steps li.done .n{background:#DCE8F7;color:var(--nx-blue)}
.nx-steps li.now{background:var(--nx-blue);border-color:var(--nx-blue);color:#fff;font-weight:600}
.nx-steps li.now .n{background:#fff;color:var(--nx-blue)}
.st-key-slide h3{font-family:var(--nx-display);font-size:2.1rem;line-height:1.15;color:var(--nx-navy);padding-top:.2rem}
.st-key-slide > [data-testid="stElementContainer"]:not(:first-child) h3{margin-top:1.6rem}
.nx-lede{font-size:1.18rem;line-height:1.55;color:var(--nx-ink);max-width:62ch;margin:0 0 .6rem}
.nx-body{font-size:1rem;line-height:1.6;color:var(--nx-muted);max-width:68ch}
.nx-stat-v{font-family:var(--nx-display);font-size:5.2rem;font-weight:700;color:var(--nx-blue);line-height:.95}
.nx-stat-l{font-size:1.1rem;color:var(--nx-ink);margin-top:10px;max-width:30ch;line-height:1.45}
.nx-stat-s{font-size:.85rem;color:var(--nx-muted);margin-top:8px}
.nx-compare{width:100%;border-collapse:separate;border-spacing:0;background:var(--nx-paper);border:1px solid var(--nx-line);
  border-radius:8px;overflow:hidden;font-size:1rem}
.nx-compare th{text-align:left;font-family:var(--nx-display);font-weight:600;font-size:1.02rem;padding:11px 16px;
  background:var(--nx-mist);color:var(--nx-muted);border-bottom:1px solid var(--nx-line)}
.nx-compare th.ours{background:var(--nx-blue);color:#fff}
.nx-compare td{padding:10px 16px;border-bottom:1px solid var(--nx-line);color:var(--nx-muted);vertical-align:top}
.nx-compare td.ours{color:var(--nx-ink);font-weight:600;background:#F5F9FE}
.nx-compare tr:last-child td{border-bottom:0}
.nx-proofs{display:grid;gap:1px;background:var(--nx-line);border:1px solid var(--nx-line);border-radius:8px;overflow:hidden}
.nx-proof{display:flex;gap:16px;background:var(--nx-paper);padding:16px 18px;align-items:flex-start}
.nx-proof-mark{flex:none;width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-weight:700;color:#fff;
  background:var(--nx-calm);font-size:1rem}
.nx-proof.warn .nx-proof-mark{background:var(--nx-busy)} .nx-proof.bad .nx-proof-mark{background:var(--nx-critical)}
.nx-proof-t{font-family:var(--nx-display);font-weight:600;font-size:1.15rem;color:var(--nx-navy)}
.nx-proof-d{color:var(--nx-muted);margin-top:2px;line-height:1.5}

/* Live replay counter */
.nx-counter{background:var(--nx-paper);border:1px solid var(--nx-line);border-radius:8px;padding:16px 20px;margin-bottom:.8rem}
.nx-counter-top{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}
.nx-counter-time{font-family:var(--nx-display);font-size:1.6rem;font-weight:600;color:var(--nx-navy);font-variant-numeric:tabular-nums}
.nx-counter-frame{color:var(--nx-muted);font-size:.9rem}
.nx-counter-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
.nx-counter-v{font-family:var(--nx-display);font-size:4.4rem;font-weight:700;line-height:1;color:var(--nx-blue);
  font-variant-numeric:tabular-nums}
.nx-counter-v.muted{color:var(--nx-unknown);font-size:2.2rem;line-height:1.9}
.nx-counter-l{color:var(--nx-muted);font-size:.95rem;margin-top:4px}
.nx-counter-title{font-family:var(--nx-display);font-size:1.15rem;font-weight:600;color:var(--nx-navy)}
.nx-density-v{font-family:var(--nx-display);font-size:3rem;font-weight:700;line-height:1;color:var(--nx-navy);
  font-variant-numeric:tabular-nums;margin-top:14px}
.nx-density-v span{font-size:1.05rem;font-weight:500;color:var(--nx-muted);margin-left:6px}
.nx-density-v.muted{color:var(--nx-unknown);font-size:2.2rem}
.nx-scale{position:relative;height:14px;border-radius:7px;margin:20px 0 6px;
  background:linear-gradient(90deg,#CFE7DC 0 var(--w),#F6DFB4 var(--w) var(--c),#F3C4BE var(--c) 100%)}
.nx-scale-mark{position:absolute;top:-6px;width:4px;height:26px;margin-left:-2px;border-radius:2px;background:var(--nx-navy);
  box-shadow:0 0 0 2px #fff}
.nx-scale-ticks{position:relative;height:1.2rem;font-size:.8rem;color:var(--nx-muted);font-variant-numeric:tabular-nums}
.nx-scale-ticks span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.nx-scale-ticks span:first-child{transform:none}

@media (max-width: 640px){
  .nx-plate-site{font-size:1.3rem}
  .nx-plate-in{padding:12px 14px}
  .nx-plate-meta{flex-direction:column}
  .nx-plate-meta span + span{border-left:0;padding-left:0}
  .nx-ro-value{font-size:1.7rem}
  .nx-stat-v{font-size:3.8rem}
  .st-key-slide h3{font-size:1.6rem}
  .nx-counter-v{font-size:3.4rem}
}
/* Downloads embedded in the page (hosted demo) and inline sample images */
.nx-download{display:inline-flex;align-items:center;gap:8px;padding:7px 14px;margin:0 0 6px;border-radius:.45rem;
  border:1px solid var(--nx-line);background:var(--nx-paper);color:var(--nx-navy) !important;text-decoration:none !important;
  font-size:.95rem;line-height:1.4}
.nx-download::before{content:"";width:16px;height:16px;background:currentColor;flex:none;
  -webkit-mask:var(--nx-dl) center/contain no-repeat;mask:var(--nx-dl) center/contain no-repeat}
.nx-download:hover{background:var(--nx-mist);border-color:var(--nx-blue)}
.nx-download:focus-visible{outline:2px solid var(--nx-blue);outline-offset:2px}
:root{--nx-dl:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'><path d='M8 2.5v8M4.8 7.6L8 10.8l3.2-3.2M3 13.5h10'/></svg>")}
.nx-figure{margin:0 0 .8rem}
.nx-figure img{width:100%;height:auto;display:block;border:1px solid var(--nx-line);border-radius:8px;background:var(--nx-paper)}
.nx-figure figcaption{font-size:.85rem;color:var(--nx-muted);margin-top:6px;text-align:center}

@media print{ [data-testid="stSidebar"]{display:none} }
</style>
""".replace("__DISPLAY__", DISPLAY)


def _esc(value: object) -> str:
    """HTML-escape any value before it goes into markup (site.json and uploads are untrusted)."""
    return html.escape(str(value), quote=True)


def _breakable(value: object) -> str:
    """Escaped text that may wrap after underscores, so config ids like yolo26n_int8_480_t0.3 never split mid-token."""
    return _esc(value).replace("_", "_<wbr>")


def _md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def inject_css() -> None:
    _md(_CSS)


def brand() -> None:
    """Product mark at the top of the sidebar."""
    st.sidebar.markdown('<div class="nx-brand"><div class="nx-brand-name">NEXTRAF</div>'
                        '<div class="nx-brand-sub">Edge crowd counting, results viewer</div></div>',
                        unsafe_allow_html=True)


def side_label(text: str) -> None:
    st.sidebar.markdown(f'<div class="nx-side-label">{_esc(text)}</div>', unsafe_allow_html=True)


def chip(level: str, on_paper: bool = False) -> str:
    """Verdict chip markup: a coloured dot plus the word, so meaning never rests on colour alone."""
    extra = " on-paper" if on_paper else ""
    return (f'<span class="nx-chip nx-{TONE.get(level, "unknown")}{extra}"><span class="nx-dot"></span>'
            f'{_esc(level)}</span>')


def masthead(ctx: Ctx) -> None:
    """The site sign plate: site name, time window and camera note, with the crowding verdict."""
    site = ctx.site
    meta = [site["time_window"] or "Time window not set", site["camera_note"]]
    spans = "".join(f"<span>{_esc(m)}</span>" for m in meta if m)
    _md(f'<div class="nx-plate" role="banner"><div class="nx-plate-in"><div>'
        f'<div class="nx-plate-site">{_esc(site["site_name"])}</div><div class="nx-plate-meta">{spans}</div></div>'
        f'{chip(insights.verdict(ctx)["level"])}</div></div>')


def page_header(title: str, subtitle: str | None = None) -> None:
    sub = f'<div class="nx-sub">{_esc(subtitle)}</div>' if subtitle else ""
    _md(f'<div class="nx-head"><div class="nx-title" role="heading" aria-level="2">{_esc(title)}</div>{sub}</div>')


def readouts(items: list[dict], stack: bool = False) -> None:
    """Key numbers in one hairline-divided panel. Item keys: label, value, unit, note, tone (good|warn|bad), small."""
    cells = []
    for it in items:
        unit = f'<span class="nx-ro-unit">{_esc(it["unit"])}</span>' if it.get("unit") else ""
        note = f'<div class="nx-ro-note">{_esc(it["note"])}</div>' if it.get("note") else ""
        size = " small" if it.get("small") else ""
        cells.append(f'<div class="nx-ro {it.get("tone") or ""}"><div class="nx-ro-label">{_esc(it["label"])}</div>'
                     f'<div class="nx-ro-value{size}">{_breakable(it["value"])}{unit}</div>{note}</div>')
    _md(f'<div class="nx-readouts{" stack" if stack else ""}">{"".join(cells)}</div>')


def specs(items: list[dict]) -> None:
    """Secondary facts in a quieter strip. Item keys: label, value, note, tone (good|bad)."""
    cells = []
    for it in items:
        note = f'<div class="nx-spec-n">{_esc(it["note"])}</div>' if it.get("note") else ""
        cells.append(f'<div class="nx-spec {it.get("tone") or ""}"><div class="nx-spec-k">{_esc(it["label"])}</div>'
                     f'<div class="nx-spec-v">{_breakable(it["value"])}</div>{note}</div>')
    _md(f'<div class="nx-specs">{"".join(cells)}</div>')


def chart(fig, key: str | None = None) -> None:
    """Every chart goes through here: our own styling (no Streamlit theme), no Plotly toolbar."""
    st.plotly_chart(fig, width="stretch", theme=None, config=CHART_CONFIG, key=key)


def stepper(steps: list[str], current: int) -> None:
    """The guided demo's steps, in order."""
    items = []
    for i, name in enumerate(steps):
        state = "now" if i == current else "done" if i < current else ""
        aria = ' aria-current="step"' if i == current else ""
        items.append(f'<li class="{state}"{aria}><span class="n">{i + 1}</span>{_esc(name)}</li>')
    _md(f'<ol class="nx-steps">{"".join(items)}</ol>')


def stat(value: str, label: str, source: str | None = None) -> None:
    src = f'<div class="nx-stat-s">{_esc(source)}</div>' if source else ""
    _md(f'<div class="nx-stat-v">{_esc(value)}</div><div class="nx-stat-l">{_esc(label)}</div>{src}')


def lede(text: str, body: str | None = None) -> None:
    extra = f'<p class="nx-body">{_esc(body)}</p>' if body else ""
    _md(f'<p class="nx-lede">{_esc(text)}</p>{extra}')


def compare(headers: tuple[str, str], rows) -> None:
    """Two-column comparison; the second column is ours."""
    body = "".join(f'<tr><td>{_esc(a)}</td><td class="ours">{_esc(b)}</td></tr>' for a, b in rows)
    _md(f'<table class="nx-compare"><thead><tr><th>{_esc(headers[0])}</th><th class="ours">{_esc(headers[1])}</th>'
        f'</tr></thead><tbody>{body}</tbody></table>')


def proofs(items: list[dict]) -> None:
    """Checkable claims. Item keys: title, text, tone (good|warn|bad)."""
    marks = {"good": "&#10003;", "warn": "!", "bad": "&#10007;"}
    rows = "".join(f'<div class="nx-proof {it["tone"]}"><div class="nx-proof-mark" aria-hidden="true">{marks[it["tone"]]}'
                   f'</div><div><div class="nx-proof-t">{_esc(it["title"])}</div>'
                   f'<div class="nx-proof-d">{_esc(it["text"])}</div></div></div>' for it in items)
    _md(f'<div class="nx-proofs">{rows}</div>')


def counter(time: str, frame: int, frames: int, people: int | None, vehicles: int | None) -> None:
    """The live replay's big frame counter."""
    def cell(v, label):
        cls = "nx-counter-v" if v is not None else "nx-counter-v muted"
        return f'<div><div class="{cls}">{"no data" if v is None else _esc(v)}</div><div class="nx-counter-l">{label}</div></div>'
    _md(f'<div class="nx-counter"><div class="nx-counter-top"><span class="nx-counter-time">{_esc(time)}</span>'
        f'<span class="nx-counter-frame">Frame {frame + 1} of {frames}</span></div>'
        f'<div class="nx-counter-grid">{cell(people, "people")}{cell(vehicles, "vehicles")}</div></div>')


def density(value: float | None, warn: float, critical: float) -> None:
    """Crowd density now, on a linear scale split at the warning and critical levels."""
    s = insights.density_scale(value, warn, critical)
    if value is None:
        big = '<div class="nx-density-v muted">no data</div>'
        mark = ""
    else:
        big = f'<div class="nx-density-v">{value:.2f}<span>people/m²</span></div>'
        mark = f'<div class="nx-scale-mark" style="left:{s["marker_pct"]:.1f}%"></div>'
    ticks = (f'<span style="left:0">0</span><span style="left:{s["warn_pct"]:.1f}%">warning {warn:g}</span>'
             f'<span style="left:{s["critical_pct"]:.1f}%">critical {critical:g}</span>')
    _md(f'<div class="nx-counter" role="img" aria-label="Crowd density {s["zone"].lower()}">'
        f'<div class="nx-counter-top"><span class="nx-counter-title">Crowd density</span>'
        f'<span class="nx-counter-frame">{_esc(s["zone"])}</span></div>{big}'
        f'<div class="nx-scale" style="--w:{s["warn_pct"]:.1f}%;--c:{s["critical_pct"]:.1f}%">{mark}</div>'
        f'<div class="nx-scale-ticks">{ticks}</div></div>')


def hosted() -> bool:
    """True on the public hosted demo (asgi.py sets it): sample data only, and files travel inside the page."""
    return os.environ.get("NEXTRAF_HOSTED") == "1"


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def download(label: str, data: bytes, file_name: str, mime: str, width: str = "content") -> None:
    """A file download. Locally a Streamlit button. On the hosted demo the file is embedded in the page, because
    st.download_button fetches it in a second request that can reach another server instance and fail."""
    if not hosted():
        st.download_button(label, data, file_name=file_name, mime=mime, icon=":material/download:", width=width)
        return
    _md(f'<a class="nx-download" href="{_data_uri(data, mime)}" download="{_esc(file_name)}">{_esc(label)}</a>')


def image(path: Path, caption: str) -> None:
    """An image inside the page (no media-file request), with its file name as the caption."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    _md(f'<figure class="nx-figure"><img src="{_data_uri(path.read_bytes(), mime)}" alt="{_esc(caption)}">'
        f'<figcaption>{_esc(caption)}</figcaption></figure>')


def pills(items: list[str]) -> None:
    """Small labelled tags, e.g. conflict or implausible."""
    _md("".join(f'<span class="nx-pill">{_esc(i)}</span>' for i in items))
