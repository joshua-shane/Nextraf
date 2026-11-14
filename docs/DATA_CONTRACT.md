# NEXTRAF data contract

The viewer only **reads** files that the Pi pipeline writes into one results folder (default `results/`).
It never runs the model. Do not change this contract without telling the whole team.

* Every file is optional **except `counts.csv`**, or instead a `run_nextraf.py` output folder (see below). A missing file hides the related section and shows a short note.
* Unknown extra columns are ignored.
* If the folder does not exist, the app falls back to the bundled sample data and shows a **SAMPLE DATA** banner.
* Bad input produces a friendly message naming the file and column, never a crash.

## `run_nextraf.py` output folders (team pipeline, schema_version 1)

The viewer also opens the folder that the team pipeline's `run_nextraf.py` writes (`--out`), when it has no `counts.csv`. Its format is defined in the "Frontend data contract" section of the pipeline README (`README.md` in joshua-shane/Nextraf); the viewer maps it like this:

| Viewer field | From |
|---|---|
| people per frame | `frames.csv` `people_in_roi` (people in view in that frame, not unique people; `people_detected` if empty) |
| time | `summary.json` `started` + `frames.csv` `video_t_sec`. Still images have no video time, so frames are shown one second apart and a note says so |
| car, motorcycle, bus, truck, vehicles_total | the same `frames.csv` columns |
| model | `summary.json` `model` |
| site name, camera note, time window | `crowd_site` (else `source`), `source` + `model` + `imgsz`, first and last frame time |
| walking area | median of `people_in_roi / density_per_m2` when the crowd config set `area_m2`; otherwise unset. Still labelled assumed |
| device details | `model`, `imgsz`, `source`, `frames_processed`, `avg_latency_ms`, `steady_latency_ms`, `ram_mb`, plus `throughput_fps = 1000 / steady_latency_ms`, and `notes` as `pipeline_notes` (for example where latency was measured) |
| sample images | the annotated `*.jpg` frames in the folder (validation mode only). They can show real people, so the viewer names them instead of saying "counts only" |

A `site.json` or `device.json` placed in the folder overrides or extends these. `alerts.json`, `timeline.json` and `latest.json` are not read yet, and a run folder has no hash chain (shown as "No hash chain", naming `frames.csv`).

## `counts.csv` (required): one row per processed frame

| Column | Type | Required | Meaning |
|---|---|---|---|
| `time` | `HH:MM:SS` or ISO 8601 | yes | Frame timestamp |
| `person` or `count` | int >= 0 | yes (one) | People count. `count` is an alias for `person`. |
| `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `vehicles_total` | int >= 0 | no | Per-class counts. `vehicles_total` is derived from the classes if absent. |
| `vehicles_on_footpath`, `cars_on_footpath` | int >= 0 | no | Vehicles inside the footpath polygon (encroachment) |
| `frame_idx` | int | no | Frame number |
| `config_id` | str | no | Configuration that produced the row |
| `model_used` | str | no | Which model handled the frame (adaptive switching) |
| `mode` | `validation` or `deployment` | no | Pipeline mode |
| `camera_ok` | bool (`true`/`false`) | no | `false` = frame too dark, blurred or blocked. Shown as "no data", never as zero. |
| `count_low`, `count_high` | number | no | Uncertainty band for the person count |
| `row_hash` | 64-char hex | no | Hash chain. If present it MUST be the **last** column. |

### Hash chain (`row_hash`)

Reference implementation: `nextraf_common/hashlog.py` (`append_row`, `verify`).

* `prev_hash` for the first data row is 64 zeros.
* `payload` is the exact CSV text of the line, without the trailing `,<row_hash>` and without the newline.
* `row_hash = sha256((prev_hash + "|" + payload).encode("utf-8")).hexdigest()`.
* The header is not hashed. Verification reads the raw file text (CRLF or LF both accepted).
* `verify(path)` returns `(ok, first_bad_line)`. Lines are 1-based file lines, header = line 1.
* Write the file with LF line endings; `.gitattributes` keeps `*.csv` as LF.

**What it proves.** The chain is tamper-evident, not tamper-proof. It is unkeyed SHA-256 so anyone can verify it
without a secret. It detects edited, inserted or deleted rows in the middle of the file. It cannot detect
someone with write access who rewrites the whole chain, or who cuts rows off the end. To close that gap, record
the last `row_hash` and the row count somewhere else (a signed manifest, a printed report), or key the chain
with a device secret (HMAC). Neither is done yet.

## `benchmark.csv`: one row per configuration (expect 30)

* **Required:** `config_id, model, resolution, precision (fp32|int8), threshold, threads, model_latency_median_ms, model_latency_p95_ms, e2e_median_ms, e2e_p95_ms, fps, model_size_mb, peak_mem_mb, mae, within1_pct, mean_signed_error`
* **Optional:** `temp_before_c, temp_after_c, throttled (bool), power_w, energy_j_per_frame`
* A run is marked **not trustworthy** if `throttled` is true or `temp_after_c > 80`.

## `per_frame_eval.csv`

* **Required:** `config_id, frame_idx, pred_count, true_count`. **Optional:** `class` (default `person`).
* Must contain **every** configuration's prediction for every evaluated frame (the adaptive simulation needs this).

## `site.json`

See `frontend/sample_data/results_sample/site.example.json`. Missing keys take documented defaults.
`region` is `IN` (default) or `UK` and changes only wording and reference values.
LOS bands in the sample are **placeholders**: copy real values from IRC:103-2012 Section 5 and set
`los_bands_are_placeholders` to `false`. Surge thresholds, vehicle footprints and OS overhead are assumptions.
If `site.json` is absent, `site.example.json` is used.

## `device.json`

`board, ram_gb, os, executorch_version, cooler (bool), threads`

## `samples/*.jpg|png`

Annotated validation images. They can show real people, so the Overview and Trust screens name them; deployment mode writes none. No video may be stored in the results folder.
