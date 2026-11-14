# Benchmark and parity tools

`run_nextraf.py` at the repository root is the NEXTRAF pipeline. The tools here sit beside it and measure it on
the Raspberry Pi. They record what the proposal's targets need and `run_nextraf.py` does not record yet:
ExecuTorch-vs-PyTorch parity, plus latency, peak memory, temperature, throttling and MAE for every model, input size
and confidence threshold.

| Command | Answers | Writes |
|---|---|---|
| `python -m nextraf_pipeline.parity` | Does the ExecuTorch export count the same people as the PyTorch model on the same frames? Exit code 1 if any frame differs. | a printed summary |
| `python -m nextraf_pipeline.benchmark` | For each model, size and threshold: median and p95 latency, FPS, peak memory, model size, temperature before and after, throttling, MAE against hand counts | `benchmark.csv`, `per_frame_eval.csv`, `device.json` |
| `python -m nextraf_pipeline.run` | A counting run that writes a hash-chained (tamper-evident) `counts.csv` | `counts.csv`, `inference_summary.json`, `samples/` |

The viewer in `frontend/` reads all of these files: the benchmark ones unlock the Affordability envelope and
Validation screens.

## Setup

Install into the environment that already runs `run_nextraf.py`. On the Pi that is the one set up from
`requirements-pi.txt` in the package ([HANDOVER.md](../HANDOVER.md)), never the repo's `requirements.txt`. These
tools only add pandas:

```bash
pip install -r nextraf_pipeline/requirements.txt
```

## Run on the Pi

```bash
# 1. Parity: the brief asks for this before any benchmark number is trusted. --imgsz must match the export size.
python -m nextraf_pipeline.parity --input frames/ --model-a yolo26n.pt --model-b yolo26n_executorch_model --imgsz 640

# 2. Benchmark and accuracy in one pass. An ExecuTorch export has a fixed input size, given after @.
python -m nextraf_pipeline.benchmark --input clip.mp4 --truth truth.csv \
    --model yolo26n=yolo26n_executorch_model:fp32@640 \
    --model yolo26n=yolo26n_1280_executorch_model:fp32@1280 \
    --model yolo11n=yolo11n_executorch_model:fp32@640 \
    --thresholds 0.3,0.4,0.5 --out out/bench
```

* `truth.csv` has the columns `frame_idx,true_count`: the same hand counts you make for `eval_crowd_mae.py`
  (`frame` and `gt_people` there), renamed.
* A `.pt` model without `@<size>` runs at every `--resolutions` value (default `320,416,640`).
* Close everything else and fit the cooler. Temperature comes from `/sys/class/thermal` and throttling from
  `vcgencmd get_throttled`; a value that cannot be read is left empty, never guessed. Runs that throttled or ended
  above 80 °C are marked not trustworthy and left out of the recommendation.
* Peak memory is the process-wide peak, so list models from smallest to largest.
* `power_w` and `energy_j_per_frame` need a USB power meter and are not written.

## Status

`run` and `benchmark` were tested on a laptop with stock `yolo11n.pt` (images and an MP4). The counting, benchmark
and comparison logic is unit-tested with fake detectors (`frontend/tests/test_benchmark.py`, `test_pipeline.py`).
**Not run on the Pi or on an ExecuTorch export yet.**
