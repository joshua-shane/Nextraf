"""One-page site report as Markdown and as a self-contained HTML file (works offline).

``build_sections`` turns the run context into plain sections; ``to_markdown`` and ``to_html``
render them. Charts are embedded in the HTML with Plotly's JavaScript inlined, so no network is needed.
"""
from __future__ import annotations

import html as htmllib
from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go

from . import charts, crowding, recommend, strings, trust
from .context import Ctx, adaptive_result, conflict, has_encroachment, surge, usable_area


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    tables: list[pd.DataFrame] = field(default_factory=list)
    figures: list[go.Figure] = field(default_factory=list)


def build_sections(ctx: Ctx) -> list[Section]:
    """Collect everything the one-page report says, in reading order."""
    return [_summary(ctx), _crowding(ctx), _events(ctx), _hardware(ctx), _trust(ctx), _assumptions(ctx),
            Section("Limitations", bullets=list(strings.text("limitations")))]


def _summary(ctx: Ctx) -> Section:
    site, counts = ctx.site, ctx.counts
    people = counts["person"]
    flags = crowding.stability_flags(people, window=max(ctx.window, 3), k=ctx.stability_k)
    pos, peak = crowding.peak(people.where(~flags))
    health = trust.camera_health(counts)
    sec = Section("Summary")
    if ctx.is_sample:
        sec.paragraphs.append(ctx.t("sample_banner"))
    sec.paragraphs.append(f"Site: {site['site_name']}. Time window: {site['time_window'] or 'not set'}. "
                          f"Region profile: {ctx.region}. {site['camera_note']}")
    note = f" (excluding {int(flags.sum())} frame(s) flagged as implausible)" if flags.any() else ""
    sec.bullets += [
        f"Peak occupancy: {peak:.0f} people at {counts['time'].iloc[pos]:{ctx.tf}}{note}.",
        f"Average: {people.mean():.1f} people per frame over {health['frames']} frames; "
        f"{health['no_data']} frame(s) had no camera data.",
    ]
    return sec


def _crowding(ctx: Ctx) -> Section:
    site, counts = ctx.site, ctx.counts
    sec = Section(f"Crowding on the {ctx.t('footpath')}")
    people = counts["person"]
    flags = crowding.stability_flags(people, window=max(ctx.window, 3), k=ctx.stability_k)
    unflagged = people.where(~flags)
    rolling = crowding.rolling_mean(people, ctx.window)
    pos = crowding.peak(unflagged)[0] if unflagged.notna().any() else None
    sec.figures.append(charts.timeline_fig(counts, "person", "People", rolling, pos, flags, None))
    area, bands = site["area_m2"], site["los_bands"]
    if not area or not bands:
        sec.paragraphs.append(ctx.t("no_area"))
        return sec
    sec.paragraphs.append(ctx.t("crowding_ref"))
    space = crowding.space_per_person(people, area)
    table = crowding.time_in_band(counts["minutes"], space, bands)
    worst = crowding.worst_band(bands)
    worst_min = float(table.loc[table["band"] == worst["name"], "minutes"].sum())
    sec.bullets.append(f"Time in the worst crowding band ({worst['name']}): {worst_min:.1f} minutes.")
    if site["los_bands_are_placeholders"]:
        sec.paragraphs.append(ctx.t("placeholder_los"))
    if site["area_is_assumed"]:
        sec.paragraphs.append(ctx.t("area_assumed"))
    sec.tables.append(table.rename(columns={"band": "Crowding band", "minutes": "Minutes in band"}).round(1))
    sec.figures.append(charts.space_fig(counts["time"], space, bands))
    return sec


def _events(ctx: Ctx) -> Section:
    counts, site = ctx.counts, ctx.site
    sec = Section("Surge, conflict and encroachment events")
    s = surge(ctx)
    if s is None:
        sec.paragraphs.append("Surge warnings need a walking area in site.json, so they were not assessed.")
    else:
        usable, density, levels = s
        ev = crowding.surge_events(levels)
        if ev.empty:
            sec.bullets.append("No surge warnings in this window.")
        else:
            for row in ev.itertuples():
                t = counts["time"].iloc[row.index]
                sec.bullets.append(f"Surge {row.level} at {t:%H:%M:%S}: {density.iloc[row.index]:.2f} people/m², "
                                   f"rising {row.rise_per_min:.3f} people/m² per minute.")
        cfg = site["surge"]
        sec.paragraphs.append(f"Surge thresholds are assumptions: warn at {cfg['warn_density_ppm2']:g} people/m² "
                              f"and a rise of {cfg['warn_rise_ppm2_per_min']:g} per minute, critical at "
                              f"{cfg['critical_density_ppm2']:g} people/m². " + ctx.t("assumed_surge"))
        sec.figures.append(charts.surge_fig(counts["time"], density, levels, cfg["warn_density_ppm2"],
                                            cfg["critical_density_ppm2"]))
    cfg = site["conflict"]
    moments = conflict(ctx)
    if moments.any():
        times = ", ".join(f"{t:{ctx.tf}}" for t in counts["time"][moments])
        sec.bullets.append(f"{int(moments.sum())} conflict moment(s) (at least {cfg['min_people']} people and "
                           f"{cfg['min_vehicles']} vehicles together, assumed thresholds): {times}.")
    else:
        sec.bullets.append("No conflict moments (many people and many vehicles together).")
    if has_encroachment(ctx):
        lost = float((site["area_m2"] - usable_area(ctx)).max())
        peak_veh = int(counts["vehicles_on_footpath"].max())
        sec.bullets.append(f"Encroachment: up to {peak_veh} vehicle(s) on the {ctx.t('footpath')} at once, taking up to "
                           f"{lost:.1f} m² of {site['area_m2']:g} m². " + ctx.t("assumed_footprint"))
    else:
        sec.bullets.append(f"Encroachment on the {ctx.t('footpath')} was not assessed (needs a walking area and "
                           "vehicles_on_footpath counts).")
    sec.paragraphs.append(ctx.t("rickshaw_note"))
    return sec


def _hardware(ctx: Ctx) -> Section:
    sec = Section("Recommended configuration and hardware")
    rec = ctx.rec
    if rec is None:
        sec.paragraphs.append("No benchmark results were available, so no configuration is recommended.")
        return sec
    sec.bullets.append(f"Recommended configuration: {rec['config_id']} ({rec['e2e_median_ms']:.0f} ms per frame, "
                       f"{rec['fps']:.1f} FPS, {rec['peak_mem_mb']:.0f} MB peak, MAE {rec['mae']:.2f} people/frame). "
                       f"Rule: fastest configuration within {ctx.tolerance:g} of the best MAE.")
    sec.tables.append(recommend.targets_table(rec))
    hw = ctx.site["hardware"]
    if hw["boards"]:
        board = recommend.cheapest_board(rec["peak_mem_mb"], hw["boards"], hw["os_overhead_mb"], hw["headroom_fraction"])
        if board is None:
            sec.bullets.append("Cheapest board that fits: none of the listed boards fits this configuration.")
        else:
            sec.bullets.append(f"Cheapest board that fits: {board['ram_gb']} GB at {ctx.currency}{board['price']:,.0f} "
                               f"(OS overhead {hw['os_overhead_mb']:g} MB and {hw['headroom_fraction'] * 100:.0f}% headroom "
                               f"are assumptions). {hw['prices_note']}")
    if "power_w" in ctx.bench.columns and pd.notna(rec.get("power_w")):
        e = recommend.energy(float(rec["power_w"]), float(rec["e2e_median_ms"]),
                             recommend.window_hours(ctx.site["time_window"]))
        window = f", {e['wh_window']:.1f} Wh over the site window" if e["wh_window"] is not None else ""
        sec.bullets.append(f"Energy (estimate): {e['j_per_frame']:.2f} J per frame, {e['wh_per_hour']:.1f} Wh per hour{window}.")
    adaptive = adaptive_result(ctx)
    if adaptive is not None:
        sec.bullets.append(f"Adaptive switching (simulated estimate, switch at {ctx.switch_count} people): "
                           f"MAE {adaptive['mae']:.2f}, {adaptive['e2e_median_ms']:.0f} ms per frame, "
                           f"accurate model on {adaptive['accurate_share'] * 100:.0f}% of frames.")
    if ctx.res.benchmark is not None:
        pool = ctx.bench[~ctx.bench["not_trustworthy"]] if ctx.exclude_untrusted else ctx.bench
        front = recommend.pareto_frontier(pool, "e2e_median_ms", "mae")
        sec.figures.append(charts.envelope_fig(ctx.bench, front, rec, "e2e_median_ms", adaptive=adaptive))
    return sec


def _trust(ctx: Ctx) -> Section:
    sec = Section("Trust and privacy")
    chain = trust.chain_status(ctx.res.folder / "counts.csv")
    sec.bullets.append({"valid": ctx.t("chain_valid"), "absent": ctx.t(trust.chain_absent_key(ctx.res.folder)),
                        "broken": ctx.t("chain_broken", line=chain["first_bad_line"])}[chain["state"]])
    sec.paragraphs.append(ctx.t("chain_limit"))
    stored = ctx.t(trust.privacy_key(ctx.res.folder), files=", ".join(trust.find_video_files(ctx.res.folder)))
    sec.bullets.append(stored if stored.endswith(".") else stored + ".")
    sec.paragraphs.append(ctx.t("privacy_text"))
    return sec


def _assumptions(ctx: Ctx) -> Section:
    sec = Section("Assumed, placeholder and estimated values")
    sec.bullets += ["Walking area: " + ("assumed." if ctx.site["area_is_assumed"] else "measured."),
                    "Crowding bands: " + ("PLACEHOLDER values, not IRC:103-2012." if ctx.site["los_bands_are_placeholders"]
                                          else "taken from site.json."),
                    "Vehicle footprints, surge thresholds, conflict thresholds and OS overhead are assumptions.",
                    "Energy, battery size, adaptive memory and error bands without count_low/count_high are estimates."]
    if ctx.is_sample:
        sec.bullets.append("All numbers come from the bundled synthetic SAMPLE DATA.")
    return sec


# ---------------------------------------------------------------- rendering

def _md_table(df: pd.DataFrame) -> str:
    cells = df.astype(str).values.tolist()
    head = "| " + " | ".join(map(str, df.columns)) + " |\n| " + " | ".join("---" for _ in df.columns) + " |\n"
    return head + "\n".join("| " + " | ".join(r) + " |" for r in cells) + "\n"


def to_markdown(title: str, sections: list[Section]) -> str:
    """Markdown version of the report (charts are only in the HTML version)."""
    out = [f"# {title}\n"]
    for sec in sections:
        out.append(f"## {sec.title}\n")
        out += [p + "\n" for p in sec.paragraphs]
        if sec.bullets:
            out.append("\n".join(f"- {b}" for b in sec.bullets) + "\n")
        out += [_md_table(t) for t in sec.tables]
        if sec.figures:
            out.append("_Charts are in the HTML version of this report._\n")
    return "\n".join(out)


_CSS = ("body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:900px;margin:0 auto;padding:16px;"
        "color:#1f2937;line-height:1.5}h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:1.6rem;"
        "border-bottom:1px solid #d1d5db}table{border-collapse:collapse;width:100%;font-size:.9rem;"
        "display:block;overflow-x:auto}th,td{border:1px solid #d1d5db;padding:4px 8px;text-align:left}"
        "th{background:#f3f4f6}.js-plotly-plot{max-width:100%}")


def to_html(title: str, sections: list[Section]) -> str:
    """Self-contained HTML: inline CSS and Plotly JavaScript, charts embedded, no network needed."""
    esc = htmllib.escape
    parts, first_chart = [], True
    for sec in sections:
        parts.append(f"<h2>{esc(sec.title)}</h2>")
        parts += [f"<p>{esc(p)}</p>" for p in sec.paragraphs]
        if sec.bullets:
            parts.append("<ul>" + "".join(f"<li>{esc(b)}</li>" for b in sec.bullets) + "</ul>")
        parts += [t.to_html(index=False, escape=True, border=0) for t in sec.tables]
        for fig in sec.figures:
            parts.append(fig.to_html(full_html=False, include_plotlyjs=True if first_chart else False))
            first_chart = False
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{esc(title)}</title><style>{_CSS}</style></head><body><h1>{esc(title)}</h1>"
            + "\n".join(parts) + "</body></html>\n")
