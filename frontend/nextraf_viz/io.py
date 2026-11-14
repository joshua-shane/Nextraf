"""Loading and schema validation for the files in ``results/`` (see docs/DATA_CONTRACT.md)."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

VEHICLE_CLASSES = ["bicycle", "car", "motorcycle", "bus", "truck"]
BENCH_REQUIRED = [
    "config_id", "model", "resolution", "precision", "threshold", "threads",
    "model_latency_median_ms", "model_latency_p95_ms", "e2e_median_ms", "e2e_p95_ms",
    "fps", "model_size_mb", "peak_mem_mb", "mae", "within1_pct", "mean_signed_error",
]
PER_FRAME_REQUIRED = ["config_id", "frame_idx", "pred_count", "true_count"]
MAX_TRUSTED_TEMP_C = 80
REGIONS = ("IN", "UK")

DEFAULT_SITE: dict = {
    "site_name": "Unnamed site",
    "region": "IN",
    "area_m2": None,
    "area_is_assumed": True,
    "time_window": "",
    "camera_note": "",
    "los_bands": [],
    "los_bands_are_placeholders": True,
    "vehicle_footprint_m2": {"two_wheeler": 1.5, "car": 7.0},
    "surge": {"warn_density_ppm2": 2.0, "warn_rise_ppm2_per_min": 0.5,
              "critical_density_ppm2": 4.0, "window_frames": 5},
    "conflict": {"min_people": 15, "min_vehicles": 5},
    "adaptive": {"fast_config_id": None, "accurate_config_id": None, "switch_count": 15},
    "hardware": {"os_overhead_mb": 350, "headroom_fraction": 0.2, "currency": "INR",
                 "boards": [], "prices_note": ""},
}


class DataError(Exception):
    """A user-facing problem with an input file."""


@dataclass
class Results:
    folder: Path
    counts: pd.DataFrame
    site: dict
    benchmark: pd.DataFrame | None = None
    per_frame: pd.DataFrame | None = None
    device: dict | None = None
    samples: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{label} not found at {path}.")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # unreadable / malformed csv
        raise DataError(f"{label} could not be read as CSV: {exc}") from exc


def _require(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"{label} is missing required column(s): {', '.join(missing)}.")


def _to_bool(series: pd.Series, default: bool = True) -> pd.Series:
    mapped = series.str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    return mapped.fillna(default).astype(bool)


def _int_column(values: pd.Series, col: str, label: str) -> pd.Series:
    nums = pd.to_numeric(values, errors="coerce")
    bad = nums.isna() | (nums < 0) | (nums % 1 != 0)
    if bad.any():
        line = int(bad.to_numpy().argmax()) + 2
        raise DataError(f"{label}: column '{col}' must hold whole numbers >= 0 (first problem on line {line}).")
    return nums.astype(int)


def _parse_time(series: pd.Series) -> pd.Series:
    s = series.str.strip()
    hms = s.str.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?")
    s = s.where(~hms, "2000-01-01 " + s)
    try:
        return pd.to_datetime(s, errors="raise", format="mixed")
    except Exception as exc:
        raise DataError("counts.csv: column 'time' must be HH:MM:SS or ISO 8601 timestamps.") from exc


def time_format(times: pd.Series) -> str:
    """Clock format for a run: seconds matter for short clips (the PS2 test clip is 10-20 s), minutes otherwise."""
    t = pd.Series(times).dropna()
    return "%H:%M:%S" if len(t) and (t.max() - t.min()) < pd.Timedelta(minutes=10) else "%H:%M"


def load_counts(path: str | Path) -> pd.DataFrame:
    """Load and normalise counts.csv. ``person`` is the canonical people column.

    Frames with ``camera_ok = false`` get NaN counts so charts show gaps, not zeros.
    """
    return _normalise_counts(_read_csv(Path(path), "counts.csv"), "counts.csv")


def _normalise_counts(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate raw text columns (counts.csv layout) and return the typed counts table."""
    if "time" not in df.columns:
        raise DataError("counts.csv is missing required column: time.")
    if "person" not in df.columns and "count" not in df.columns:
        raise DataError("counts.csv needs a 'person' column (or its alias 'count').")
    if "person" not in df.columns:
        df["person"] = df["count"]
    out = pd.DataFrame({"time": _parse_time(df["time"])})
    out["minutes"] = (out["time"] - out["time"].iloc[0]).dt.total_seconds() / 60.0 if len(out) else []
    out["person"] = _int_column(df["person"], "person", label)
    optional_counts = VEHICLE_CLASSES + ["vehicles_total", "vehicles_on_footpath", "cars_on_footpath"]
    for col in optional_counts:
        if col in df.columns:
            out[col] = _int_column(df[col].replace("", "0"), col, label)
    if "vehicles_total" not in out.columns:
        present = [c for c in VEHICLE_CLASSES if c in out.columns]
        out["vehicles_total"] = out[present].sum(axis=1) if present else 0
    out["frame_idx"] = (pd.to_numeric(df["frame_idx"], errors="coerce")
                        if "frame_idx" in df.columns else pd.Series(range(len(df))))
    for col in ("config_id", "model_used", "mode"):
        if col in df.columns:
            out[col] = df[col]
    out["camera_ok"] = _to_bool(df["camera_ok"]) if "camera_ok" in df.columns else True
    for col in ("count_low", "count_high"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    if "row_hash" in df.columns:
        out["row_hash"] = df["row_hash"]
    for col in ["person", *optional_counts]:
        if col in out.columns:
            out[col] = out[col].astype("float").where(out["camera_ok"])
    return out


def load_benchmark(path: str | Path) -> pd.DataFrame:
    """Load benchmark.csv, converting numerics and flagging untrustworthy runs."""
    df = _read_csv(Path(path), "benchmark.csv")
    _require(df, BENCH_REQUIRED, "benchmark.csv")
    text_cols = {"config_id", "model", "precision", "throttled"}
    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["throttled"] = _to_bool(df["throttled"], default=False) if "throttled" in df.columns else False
    hot = df["temp_after_c"] > MAX_TRUSTED_TEMP_C if "temp_after_c" in df.columns else False
    df["not_trustworthy"] = (df["throttled"] | hot).astype(bool)
    return df


def load_per_frame(path: str | Path) -> pd.DataFrame:
    """Load per_frame_eval.csv; ``class`` defaults to ``person``."""
    df = _read_csv(Path(path), "per_frame_eval.csv")
    _require(df, PER_FRAME_REQUIRED, "per_frame_eval.csv")
    for col in ("frame_idx", "pred_count", "true_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["class"] = df["class"].replace("", "person") if "class" in df.columns else "person"
    return df


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_json(path: Path, label: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DataError(f"{label} must contain a JSON object.")
    return data


def load_site(path: str | Path | None) -> dict:
    """Load site.json merged over documented defaults (missing file = defaults)."""
    if path is None or not Path(path).is_file():
        return copy.deepcopy(DEFAULT_SITE)
    site = _merge(DEFAULT_SITE, _load_json(Path(path), "site.json"))
    if site["region"] not in REGIONS:
        raise DataError(f"site.json: region must be one of {', '.join(REGIONS)} (got {site['region']!r}).")
    return site


def resolve_results_dir(requested: str | Path, sample_dir: str | Path,
                        base: str | Path | None = None) -> tuple[Path, bool]:
    """Use the requested folder if it exists, else the bundled sample (is_sample=True).

    With ``base`` set, an existing folder must be ``base`` or inside it (symlinks and ``..`` resolved),
    otherwise a DataError is raised. This stops the viewer from listing arbitrary folders.
    """
    requested = Path(requested)
    if not requested.is_dir():
        return Path(sample_dir), True
    if base is None:
        return requested, False
    resolved, root = requested.resolve(), Path(base).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise DataError(f"The results folder must be inside {root}. Start the app with --results-dir to use another location.")
    return resolved, False


RUN_FILES = ("frames.csv", "summary.json")  # written by the team pipeline's run_nextraf.py (schema_version 1)


def is_run_folder(folder: str | Path) -> bool:
    """True for a run_nextraf.py output folder (frames.csv and summary.json)."""
    return all((Path(folder) / name).is_file() for name in RUN_FILES)


def is_results_folder(folder: str | Path) -> bool:
    """True when the viewer can open the folder: counts.csv, or a run_nextraf.py output folder."""
    return (Path(folder) / "counts.csv").is_file() or is_run_folder(folder)


def _load_run(folder: Path, site_file: Path) -> Results:
    """Map a run_nextraf.py output folder onto the viewer's model.

    people_in_roi is the people count (people in view in that frame, not unique people). Times are the run start
    plus video_t_sec; still images have no video time, so they are spaced one second apart. The walking area is
    people_in_roi / density_per_m2 when the crowd config set area_m2. Model, latency and RAM come from summary.json.
    """
    summary = _load_json(folder / "summary.json", "summary.json")
    raw = _read_csv(folder / "frames.csv", "frames.csv")
    _require(raw, ["frame", "people_in_roi"], "frames.csv")
    notes = []
    start = pd.to_datetime(summary.get("started"), errors="coerce")
    start = pd.Timestamp("2000-01-01") if pd.isna(start) else start
    seconds = pd.to_numeric(raw["video_t_sec"], errors="coerce") if "video_t_sec" in raw.columns else None
    if seconds is None or seconds.isna().any():
        seconds = pd.to_numeric(raw["frame"], errors="coerce").fillna(0)
        notes.append("These are still images with no video time, so frames are shown one second apart.")
    people = raw["people_in_roi"].where(raw["people_in_roi"] != "", raw.get("people_detected", ""))
    table = pd.DataFrame({"time": (start + pd.to_timedelta(seconds, unit="s")).dt.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                          "person": people, "frame_idx": raw["frame"]})
    for col in [*VEHICLE_CLASSES, "vehicles_total"]:
        if col in raw.columns:
            table[col] = raw[col]
    if summary.get("model"):
        table["model_used"] = str(summary["model"])
    counts = _normalise_counts(table, "frames.csv")

    if site_file.is_file():
        site = load_site(site_file)
    else:
        site = copy.deepcopy(DEFAULT_SITE)
        site["site_name"] = str(summary.get("crowd_site") or summary.get("source") or DEFAULT_SITE["site_name"])
        parts = [str(summary[k]) for k in ("source", "model") if summary.get(k)]
        if summary.get("imgsz"):
            parts.append(f"{summary['imgsz']} px input")
        site["camera_note"] = ", ".join(parts)
        if not notes and len(counts):
            tf = time_format(counts["time"])
            site["time_window"] = f"{counts['time'].iloc[0]:{tf}}–{counts['time'].iloc[-1]:{tf}}"
        density = pd.to_numeric(raw.get("density_per_m2"), errors="coerce") if "density_per_m2" in raw.columns else None
        if density is not None and (density > 0).any():
            ok = density > 0
            site["area_m2"] = round(float((counts["person"][ok] / density[ok]).median()), 2)

    device = {k: summary[k] for k in ("model", "imgsz", "source", "frames_processed", "avg_latency_ms",
                                      "steady_latency_ms", "ram_mb") if summary.get(k) is not None}
    if summary.get("steady_latency_ms"):
        device["throughput_fps"] = round(1000.0 / float(summary["steady_latency_ms"]), 2)
    if summary.get("notes"):  # e.g. where latency and RAM were measured: keep the caveat next to the numbers
        device["pipeline_notes"] = " ".join(str(n) for n in summary["notes"])
    samples = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return Results(folder=folder, counts=counts, site=site, device=device or None, samples=samples, notes=notes)


def load_results(folder: str | Path) -> Results:
    """Load a results folder: counts.csv, or a run_nextraf.py output folder. Other files add notes when absent."""
    folder = Path(folder)
    site_file = folder / "site.json" if (folder / "site.json").is_file() else folder / "site.example.json"
    if (folder / "counts.csv").is_file():
        res = Results(folder=folder, counts=load_counts(folder / "counts.csv"), site=load_site(site_file))
    elif is_run_folder(folder):
        res = _load_run(folder, site_file)
    else:
        raise DataError(f"No counts.csv in {folder}, and no run_nextraf.py output either (frames.csv and summary.json).")
    optional = [("benchmark.csv", "benchmark", load_benchmark, "the affordability envelope"),
                ("per_frame_eval.csv", "per_frame", load_per_frame, "validation and the adaptive simulation")]
    for name, attr, loader, section in optional:
        if (folder / name).is_file():
            setattr(res, attr, loader(folder / name))
        else:
            res.notes.append(f"{name} not found, so {section} is hidden.")
    if (folder / "device.json").is_file():
        res.device = {**(res.device or {}), **_load_json(folder / "device.json", "device.json")}
    elif res.device is None:
        res.notes.append("device.json not found, so device details are hidden.")
    samples_dir = folder / "samples"
    if samples_dir.is_dir():
        res.samples += sorted(p for p in samples_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return res


UPLOAD_NAMES = {"counts.csv", "benchmark.csv", "per_frame_eval.csv", "site.json", "device.json", *RUN_FILES}
UPLOAD_IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def save_uploads(files: list[tuple[str, bytes]], dest: str | Path, max_bytes: int = 20_000_000) -> tuple[list[str], list[str]]:
    """Write browser-uploaded files into ``dest`` and return (saved, rejected) names.

    Only the documented result files and sample images are kept. Names are reduced to their last path
    component, images go under ``samples/``, and videos, unknown names and oversized files are rejected.
    """
    dest = Path(dest)
    saved, rejected = [], []
    for raw_name, data in files:
        name = Path(raw_name.replace("\\", "/")).name
        image = Path(name).suffix.lower() in UPLOAD_IMAGE_EXT
        clean = raw_name.replace("\\", "/") == name  # a path in the upload name is only tolerated for known files
        if len(data) > max_bytes or not (name in UPLOAD_NAMES or image) or (image and not clean):
            rejected.append(raw_name)
            continue
        target = dest / "samples" / name if image else dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        saved.append(f"samples/{name}" if image else name)
    return saved, rejected
