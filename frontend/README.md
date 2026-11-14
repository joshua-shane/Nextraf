# NEXTRAF results viewer

An offline Streamlit app for hackathon judges and for a ward officer reviewing one site. It opens a
`run_nextraf.py` output folder (or a `counts.csv` results folder), and shows what happened at the site. It
never runs a model and never needs the internet.

**Live demo:** https://nextraf-frontend.vercel.app (synthetic sample data). https://nextraf-frontend.vercel.app/?screen=demo
opens the guided demo for the pitch.

## Run it

The viewer has its own environment. Its pins differ from the pipeline's `requirements.txt` (numpy), and it runs on
a laptop, not on the Pi while benchmarking. From the repository root (Python 3.11 or newer):

```powershell
python -m venv .venv-viewer
.venv-viewer\Scripts\pip install -r frontend\requirements.txt
.venv-viewer\Scripts\streamlit run frontend\app.py -- --results-dir out\demo
```

* `--results-dir` is a `run_nextraf.py --out` folder (`frames.csv` + `summary.json`) or a folder with `counts.csv`.
  The folder box in the sidebar accepts that folder or any folder inside it: start with `--results-dir out` and type
  `out\demo` (or another run) into the box to switch between runs.
* With no `--results-dir`, or a folder that does not exist, the bundled synthetic sample
  (`frontend/sample_data/results_sample/`) opens with a **SAMPLE DATA** banner.
* You can also drop result files into the sidebar upload box (only the documented files and images, never video).
* Usage stats are off in `.streamlit/config.toml`; there are no CDNs, remote fonts or map tiles.
* **Do not run the viewer on the Pi while benchmarking.** It would distort timing and memory.

Tests: `.venv-viewer\Scripts\pip install pytest==9.2.0`, then `.venv-viewer\Scripts\python -m pytest` from the
repository root (175 tests, including every screen rendering a `run_nextraf.py` folder). Regenerate the sample
with `python frontend/sample_data/make_sample.py`.

## What it reads from `run_nextraf.py`

People per frame from `people_in_roi`, time from `summary.json` `started` plus `video_t_sec`, the vehicle columns,
the model and latency figures (with the `notes` caveat about where they were measured), and the annotated `*.jpg`
frames as validation images. The full mapping is in [docs/DATA_CONTRACT.md](../docs/DATA_CONTRACT.md).
`alerts.json`, `timeline.json` and `latest.json` are not read yet.

Optional files unlock more screens: `site.json` (walking area, crowding bands, surge settings), `benchmark.csv` and
`per_frame_eval.csv` from [`nextraf_pipeline/`](../nextraf_pipeline/README.md) (affordability envelope, validation),
and `device.json`. A missing file hides its section and says which file would unlock it.

## Screens

1. **Overview**: a verdict (Calm, Busy, Surge risk, Critical), a plain-English summary with clearly labelled
   suggestions, peak occupancy, busiest stretch, time in the most crowded band, recommended setup, cheapest board
   that fits, hash-chain status, what is stored, targets, events, and which files are loaded.
2. **Guided demo**: five slides for the 2-minute pitch (problem, the site, cheapest setup, trust and privacy, honest
   limits). `?screen=demo` opens it directly.
3. **Crowding timeline**: counts with rolling mean, peak, camera gaps and implausible frames; people against
   vehicles with conflict moments; space per person; encroachment; surge early warning.
4. **Live replay**: step or play through the recorded counts frame by frame, with a crowd-density scale. It replays
   the file; there is no camera and no tracking.
5. **Affordability envelope**: every configuration on accuracy against cost, the Pareto frontier, the recommended
   and adaptive-switching points, board fit, cost and energy.
6. **What-if and compare**: try another walking area or thresholds, or compare two setups side by side. Nothing
   changes your files.
7. **Validation**: predicted against hand-counted, error against crowd size, annotated images.
8. **Trust and privacy**: hash-chain check, camera health, what is stored, file list, device details, privacy text.
9. **Export and report**: CSV downloads and a one-page site report (Markdown and a self-contained HTML file that
   works offline).

## Honest labels

Anything assumed, placeholder or estimated is labelled. The sample crowding bands are **placeholders**: copy the
real values from IRC:103-2012 Section 5 into `site.json`. The walking area, vehicle footprints, surge thresholds, OS
overhead and energy are assumptions. Annotated images from a validation run are named as such, never hidden behind
"counts only". Known limits (dense crowds, auto-rickshaws not being a COCO class, untested night and rain, no
unique-person counting) are in every site report.

`site.json` `"region"` is `IN` (default) or `UK`; it changes only wording and reference values (privacy law,
currency, footpath or pavement).

## Hosted demo

The public demo is deployed from [sreevathson1530/nextraf-frontend](https://github.com/sreevathson1530/nextraf-frontend),
which holds this same viewer plus the hosting files (`asgi.py`, `vercel.json`, `pyproject.toml`). They are not in
this repository: a Vercel deploy from this root would also install the pipeline's `requirements.txt` (torch,
ExecuTorch). Hosted mode shows only the synthetic sample and has no folder or upload box.

## Layout

`app.py` is the entry point; `screens/` has one module per screen; `nextraf_viz/` holds the logic with no Streamlit
code (`io`, `crowding`, `recommend`, `trust`, `insights`, `report`, `charts`, `strings`); all UI text is in
`nextraf_viz/strings.py`. `../nextraf_common/hashlog.py` is the hash-chained CSV writer and checker shared with
`nextraf_pipeline/`.
