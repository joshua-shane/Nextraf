"""NEXTRAF results viewer. Run: streamlit run frontend/app.py -- --results-dir results/"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from nextraf_viz import io, strings  # noqa: E402
from screens import demo, envelope, export, overview, replay, timeline, trust_privacy, ui, validation, whatif  # noqa: E402
from screens.common import CLASS_LABELS, Ctx, available_classes  # noqa: E402

SAMPLE_DIR = HERE / "sample_data" / "results_sample"
# Order matters: the sidebar groups them as Present (3), Analyse (4), Check and share (2).
SCREENS = {"nav_overview": overview, "nav_demo": demo, "nav_replay": replay,
           "nav_timeline": timeline, "nav_envelope": envelope, "nav_whatif": whatif, "nav_validation": validation,
           "nav_trust": trust_privacy, "nav_export": export}


def _cli_results_dir() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-dir", default="results/")
    return parser.parse_known_args(sys.argv[1:])[0].results_dir


def _signature(folder: Path) -> tuple:
    """Cache key: modification times of the (few) files in the folder and its samples/ subfolder."""
    files = [*folder.glob("*"), *(folder / "samples").glob("*")]
    return tuple((str(p.relative_to(folder)), p.stat().st_mtime_ns) for p in sorted(files) if p.is_file())


@st.cache_data(show_spinner=False)
def _load(folder: str, signature: tuple) -> io.Results:
    return io.load_results(folder)


def _upload_folder() -> Path | None:
    """Sidebar upload box. Returns the temp folder holding uploaded files once counts.csv is among them."""
    with st.sidebar.expander(strings.text("sb_upload")):
        files = st.file_uploader(strings.text("sb_upload_help"), accept_multiple_files=True,
                                 type=["csv", "json", "png", "jpg", "jpeg"], key="uploads")
        if not files:
            return None
        dest = Path(st.session_state.setdefault("upload_dir", tempfile.mkdtemp(prefix="nextraf_upload_")))
        _, rejected = io.save_uploads([(f.name, f.getvalue()) for f in files], dest)
        if rejected:
            st.warning("Skipped (not a results file): " + ", ".join(rejected))
        if io.is_results_folder(dest):
            return dest
        st.info("Add counts.csv, or frames.csv and summary.json from run_nextraf.py, to start.")
        return None


def main() -> None:
    st.set_page_config(page_title="NEXTRAF", page_icon=":material/directions_walk:", layout="wide",
                       initial_sidebar_state="auto")  # open on desktop, collapsed on phones
    ui.inject_css()
    ui.brand()
    # ?screen=demo (or nav_demo) opens a screen directly, e.g. straight into the guided demo for a pitch.
    wanted = st.query_params.get("screen", "")
    wanted = wanted if wanted.startswith("nav_") else f"nav_{wanted}"
    screen_key = st.sidebar.radio("Screen", list(SCREENS), index=list(SCREENS).index(wanted) if wanted in SCREENS else 0,
                                  format_func=strings.text, key="nav", label_visibility="collapsed", width="stretch")

    ui.side_label("Data source")
    base = _cli_results_dir()
    if ui.hosted():
        # Public demo: sample data only. NEXTRAF never asks anyone to upload data to a server.
        st.sidebar.caption(strings.text("hosted_note"))
        requested, uploaded = "the bundled sample", None
    else:
        requested = st.sidebar.text_input(strings.text("sb_folder"), value=base)
        uploaded = _upload_folder()
    try:
        if ui.hosted():
            folder, is_sample = SAMPLE_DIR, True
        elif uploaded is not None:
            folder, is_sample = uploaded, False
        else:
            folder, is_sample = io.resolve_results_dir(requested, SAMPLE_DIR, base=base)
        res = _load(str(folder), _signature(folder))
    except io.DataError as exc:
        st.error(f"Could not load results from {requested}. {exc}")
        st.stop()

    with st.sidebar.expander("Analysis settings"):
        classes = available_classes(res.counts)
        col = st.selectbox(strings.text("sb_class"), classes, format_func=CLASS_LABELS.get)
        region = st.radio(strings.text("sb_region"), io.REGIONS, index=io.REGIONS.index(res.site["region"]),
                          horizontal=True)
        tolerance = st.number_input(strings.text("sb_tolerance"), 0.0, 5.0, 0.5, 0.1)
        switch = st.number_input(strings.text("sb_switch"), 1, 200, int(res.site["adaptive"]["switch_count"]), 1)
        k = st.number_input(strings.text("sb_k"), 1.0, 10.0, 3.0, 0.5)
        window = st.number_input(strings.text("sb_window"), 2, 30, 5, 1)
        exclude = st.toggle(strings.text("sb_exclude"), value=True)
        st.caption(strings.text("bench_warning"))
    st.sidebar.caption(strings.text("sidebar_footer"))

    ctx = Ctx(res=res, is_sample=is_sample, region=region, col=col, tolerance=tolerance,
              stability_k=k, window=int(window), exclude_untrusted=exclude, switch_count=int(switch))
    if is_sample:
        st.warning(strings.text("sample_banner"), icon=":material/science:")
    if screen_key != "nav_demo":
        ui.masthead(ctx)
    SCREENS[screen_key].render(ctx)


main()
