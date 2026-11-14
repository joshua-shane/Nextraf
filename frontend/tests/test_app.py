from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SCREENS = ["nav_overview", "nav_demo", "nav_timeline", "nav_replay", "nav_envelope", "nav_whatif",
           "nav_validation", "nav_trust", "nav_export"]


def _run(screen: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.sidebar.radio[0].set_value(screen).run()
    return at


@pytest.mark.parametrize("screen", SCREENS)
def test_every_screen_renders_with_sample_data_and_banner(screen):
    at = _run(screen)
    assert not at.exception, [e.value for e in at.exception]
    assert any("SAMPLE DATA" in w.value for w in at.warning)


@pytest.mark.parametrize("col", ["person", "motorcycle", "vehicles_total"])
def test_timeline_works_for_each_class(col):
    at = _run("nav_timeline")
    at.sidebar.selectbox[0].set_value(col).run()
    assert not at.exception


def test_missing_results_folder_falls_back_to_sample():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].set_value("definitely/not/here").run()
    assert not at.exception and any("SAMPLE DATA" in w.value for w in at.warning)


def test_bad_counts_file_shows_friendly_error_not_a_crash(tmp_path, monkeypatch):
    (tmp_path / "counts.csv").write_text("time,car\n15:30:00,1\n")
    monkeypatch.setattr("sys.argv", ["app.py", "--results-dir", str(tmp_path)])
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("person" in e.value for e in at.error)


def test_folder_outside_the_cli_base_is_refused(tmp_path, monkeypatch):
    base, other = tmp_path / "results", tmp_path / "other"
    base.mkdir()
    other.mkdir()
    monkeypatch.setattr("sys.argv", ["app.py", "--results-dir", str(base)])
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.text_input[0].set_value(str(other)).run()
    assert not at.exception and any("inside" in e.value for e in at.error)


# ---- phase 2 content ----

def _texts(at):
    out = [m.value for m in at.markdown] + [m.value for m in at.success] + [m.value for m in at.error]
    out += [m.value for m in at.warning] + [m.value for m in at.caption] + [m.value for m in at.info]
    out += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(str(x) for x in out)


def test_overview_shows_verified_chain_board_and_events():
    at = _run("nav_overview")
    text = _texts(at)
    assert "Verified" in text and "Cheapest board that fits" in text
    assert "surge" in text and "conflict moment" in text


def test_trust_screen_reports_chain_and_indian_privacy_law():
    text = _texts(_run("nav_trust"))
    assert "Hash chain verified" in text and "Digital Personal Data Protection Act 2023" in text


def test_uk_region_switches_wording():
    at = _run("nav_trust")
    uk = [r for r in at.sidebar.radio if r.label == "Region profile"][0]
    uk.set_value("UK").run()
    text = _texts(at)
    assert "UK GDPR" in text and "Digital Personal Data Protection" not in text


def test_envelope_shows_adaptive_cost_and_energy_cards():
    text = _texts(_run("nav_envelope"))
    assert "Adaptive model switching" in text and "Hardware fit and cost" in text and "Energy (estimate)" in text


def test_timeline_shows_encroachment_and_surge_sections():
    text = _texts(_run("nav_timeline"))
    assert "Encroachment" in text and "Crowd-surge early warning" in text and "People and vehicles" in text


def test_tampered_counts_are_flagged_in_the_app(tmp_path, monkeypatch):
    from sample_data import make_sample
    make_sample.build(tmp_path)
    p = tmp_path / "counts.csv"
    lines = p.read_text(encoding="utf-8").split("\n")
    cells = lines[6].split(",")
    cells[1] = str(int(cells[1]) + 5)
    lines[6] = ",".join(cells)
    p.write_text("\n".join(lines), encoding="utf-8", newline="")
    monkeypatch.setattr("sys.argv", ["app.py", "--results-dir", str(tmp_path)])
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("nav_trust").run()
    assert "BROKEN at file line 7" in _texts(at)


@pytest.mark.parametrize("screen", SCREENS)
def test_counts_only_folder_never_crashes_any_screen(tmp_path, monkeypatch, screen):
    (tmp_path / "counts.csv").write_text("time,count\n15:30:00,4\n15:30:10,6\n15:30:20,5\n")
    monkeypatch.setattr("sys.argv", ["app.py", "--results-dir", str(tmp_path)])
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(screen).run()
    assert not at.exception, [e.value for e in at.exception]


def test_implausible_spike_frame_is_not_counted_as_a_conflict_moment():
    from screens import common  # noqa: PLC0415
    from nextraf_viz import io  # noqa: PLC0415
    from sample_data import make_sample  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as d:
        make_sample.build(d)
        res = io.load_results(d)
        ctx = common.Ctx(res=res, is_sample=False, region="IN", col="person", tolerance=0.5, stability_k=3.0,
                         window=5, exclude_untrusted=True, switch_count=15)
        flagged_row = 8  # the injected 48-person spike
        assert not common.conflict(ctx).iloc[flagged_row]


def test_export_builds_a_report_and_offers_both_downloads():
    at = _run("nav_export")
    assert "Download counts CSV" in [b.label for b in at.get("download_button")]
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [b.proto.label for b in at.get("download_button")]
    assert "Download report (Markdown)" in labels and "Download report (HTML, offline)" in labels


@pytest.mark.parametrize("screen", SCREENS)
def test_every_screen_renders_a_run_nextraf_output_folder(screen, run_folder, monkeypatch):
    monkeypatch.setattr("sys.argv", ["app.py", "--results-dir", str(run_folder())])
    at = _run(screen)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]  # a load failure shows an error box and stops
    assert not any("SAMPLE DATA" in w.value for w in at.warning)


def test_run_folder_with_annotated_frames_does_not_claim_counts_only(run_folder, monkeypatch):
    monkeypatch.setattr("sys.argv", ["app.py", "--results-dir", str(run_folder())])
    overview = _run("nav_overview")
    assert any("1 annotated image(s) from a validation run" in m.value for m in overview.markdown)
    assert not any(">counts only<" in m.value for m in overview.markdown)
    trust_screen = _run("nav_trust")
    assert any("--mode deployment" in c.value for c in trust_screen.caption)


# ---- hosted demo (Vercel): no uploads, files travel inside the page ----

def test_validation_samples_are_inline_images_not_media_files():
    at = _run("nav_validation")
    assert any("data:image/png;base64," in m.value for m in at.markdown)
    assert not at.get("imgs")


def test_hosted_mode_hides_data_inputs_and_embeds_downloads(monkeypatch):
    monkeypatch.setenv("NEXTRAF_HOSTED", "1")
    at = _run("nav_export")
    assert not at.exception, [e.value for e in at.exception]
    assert not at.sidebar.text_input and not at.get("file_uploader")
    assert not at.get("download_button")
    links = [m.value for m in at.markdown if "download=" in m.value]
    assert any('download="nextraf_counts.csv"' in v and 'href="data:text/csv;base64,' in v for v in links)
    at.button[0].click().run()
    assert any('download="nextraf_site_report.html"' in m.value for m in at.markdown)


def test_local_mode_keeps_the_upload_box_and_download_buttons(monkeypatch):
    monkeypatch.delenv("NEXTRAF_HOSTED", raising=False)
    at = _run("nav_export")
    assert at.sidebar.text_input and at.get("download_button")


# ---- usability features ----

def test_overview_shows_verdict_findings_help_and_checklist():
    at = _run("nav_overview")
    text = _texts(at)
    assert "Surge risk" in text and "What happened" in text and "Peak occupancy was" in text
    assert "Suggestion:" in text
    assert "Start here." in text  # the screen explains itself under its title
    assert "Which files are loaded and what each one unlocks" in [e.label for e in at.expander]


def test_demo_walks_through_all_five_steps():
    at = _run("nav_demo")
    assert at.session_state["demo_step"] == 0 and "The problem" in [h.value for h in at.subheader]
    seen = []
    for _ in range(4):
        [b for b in at.button if b.label == "Next"][0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        seen.append(at.session_state["demo_step"])
    assert seen == [1, 2, 3, 4]
    [b for b in at.button if b.label == "Previous"][0].click().run()
    assert at.session_state["demo_step"] == 3 and not at.exception


def test_demo_trust_slide_is_honest_about_a_run_folder(run_folder, monkeypatch):
    monkeypatch.setattr("sys.argv", ["app.py", "--results-dir", str(run_folder())])
    at = _run("nav_demo")
    for _ in range(3):
        [b for b in at.button if b.label == "Next"][0].click().run()
    slide = " ".join(m.value for m in at.markdown)
    assert "come from a validation run" in slide and "Only integer counts are written" not in slide
    assert "frames.csv from run_nextraf.py has no hash chain" in slide


def test_replay_steps_frame_by_frame():
    at = _run("nav_replay")
    assert at.session_state["replay_i"] == 0
    [b for b in at.button if b.label == "Next frame"][0].click().run()
    assert at.session_state["replay_i"] == 1 and not at.exception
    assert "15:32" in _texts(at) or any("15:32" in m.value for m in at.markdown)


def test_what_if_smaller_area_changes_the_verdict_table():
    at = _run("nav_whatif")
    area = [n for n in at.number_input if n.label.startswith("Walking area")][0]
    area.set_value(6.0).run()
    assert not at.exception
    table = at.dataframe[0].value
    assert table.loc[1, "Walking area (m²)"] == 6.0
    assert table.loc[1, "Minutes in most crowded band"] > table.loc[0, "Minutes in most crowded band"]


def test_compare_two_setups_shows_deltas():
    at = _run("nav_whatif")
    assert "Compare two model setups" in _texts(at)
    assert any("B minus A" in list(df.value.columns) for df in at.dataframe)


def test_html_helpers_escape_file_derived_text(monkeypatch):
    from screens import ui  # noqa: PLC0415
    captured = []
    monkeypatch.setattr(ui.st, "markdown", lambda html, **kw: captured.append(html))
    ui.pills(['<img src=x onerror=alert(1)>', "Band: <b>A</b>"])
    assert "<img" not in captured[0] and "&lt;img" in captured[0] and "<b>" not in captured[0]
