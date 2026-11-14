import re

import pytest

from nextraf_viz import io, report
from nextraf_viz.context import Ctx
from sample_data import make_sample


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sample")
    make_sample.build(folder)
    return Ctx(res=io.load_results(folder), is_sample=True, region="IN", col="person", tolerance=0.5,
               stability_k=3.0, window=5, exclude_untrusted=True, switch_count=15)


@pytest.fixture(scope="module")
def sections(ctx):
    return report.build_sections(ctx)


def test_markdown_report_covers_the_required_content(ctx, sections):
    md = report.to_markdown("Site report", sections)
    assert md.startswith("# Site report")
    for needle in ("Example Primary School – main gate", "SAMPLE DATA", "Peak occupancy", "worst",
                   "surge", "conflict", "encroach", "yolo26n_int8_480_t0.3", "Cheapest board",
                   "Placeholder LOS bands", "Hash chain verified", "Limitations", "Dense crowds"):
        assert needle.lower() in md.lower(), needle


def test_report_peak_ignores_the_implausible_spike(sections):
    md = report.to_markdown("Site report", sections)
    assert "48" not in re.search(r"Peak occupancy[^\n]*", md).group(0)


def test_report_labels_assumptions_and_estimates(sections):
    md = report.to_markdown("Site report", sections).lower()
    assert "assumed" in md and "estimate" in md


def test_html_report_is_self_contained_and_offline(sections):
    html = report.to_html("Site report", sections)
    assert html.startswith("<!DOCTYPE html>") and "<meta name=\"viewport\"" in html
    assert "plotly" in html.lower() and html.count("<script") >= 2
    assert not re.search(r"<script[^>]+src=", html)
    assert not re.search(r"<link[^>]+href=", html)
    markup = re.sub(r"<script.*?</script>", "", html, flags=re.S)  # plotly's bundled JS only mentions URLs in strings
    assert not re.search(r"(src|href)=[\"']https?://", markup)
    assert not re.search(r"url\(", markup) and "@import" not in markup


def test_html_report_escapes_site_text(ctx):
    ctx.res.site["site_name"] = "<script>alert(1)</script>"
    html = report.to_html("T", report.build_sections(ctx))
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    ctx.res.site["site_name"] = "Example Primary School – main gate"


def test_report_for_a_run_folder_names_frames_csv_and_the_annotated_images(run_folder):
    c = Ctx(res=io.load_results(run_folder()), is_sample=False, region="IN", col="person", tolerance=0.5,
            stability_k=3.0, window=5, exclude_untrusted=True, switch_count=15)
    md = report.to_markdown("Site report", report.build_sections(c))
    assert "frames.csv from run_nextraf.py has no hash chain" in md
    assert "come from a validation run" in md and "no footage stored" not in md


def test_report_still_builds_from_a_counts_only_folder(tmp_path):
    (tmp_path / "counts.csv").write_text("time,count\n15:30:00,4\n15:30:10,6\n15:30:20,5\n")
    c = Ctx(res=io.load_results(tmp_path), is_sample=False, region="UK", col="person", tolerance=0.5,
            stability_k=3.0, window=5, exclude_untrusted=True, switch_count=15)
    md = report.to_markdown("Site report", report.build_sections(c))
    assert "Peak occupancy" in md and "pavement" in md.lower() and "No hash chain" in md
    assert "<html" in report.to_html("Site report", report.build_sections(c)).lower()
