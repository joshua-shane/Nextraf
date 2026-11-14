"""Trust and privacy helpers: hash-chain status, camera health, file inventory."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nextraf_common import hashlog

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
TABLE_EXT = {".csv"}
JSON_EXT = {".json"}


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.is_file())


def _kind(path: Path) -> str:
    ext = path.suffix.lower()
    for kind, exts in (("video", VIDEO_EXT), ("image", IMAGE_EXT), ("table", TABLE_EXT), ("config", JSON_EXT)):
        if ext in exts:
            return kind
    return "other"


def file_inventory(folder: str | Path) -> pd.DataFrame:
    """Every file under the results folder with size and a coarse type."""
    folder = Path(folder)
    rows = [{"path": p.relative_to(folder).as_posix(), "size_bytes": p.stat().st_size, "kind": _kind(p)}
            for p in _files(folder)]
    return pd.DataFrame(rows, columns=["path", "size_bytes", "kind"])


def find_video_files(folder: str | Path) -> list[str]:
    """Relative paths of any video files in the results folder."""
    folder = Path(folder)
    return [p.relative_to(folder).as_posix() for p in _files(folder) if p.suffix.lower() in VIDEO_EXT]


def find_image_files(folder: str | Path) -> list[str]:
    """Relative paths of images in the results folder: annotated frames from a validation run, or samples/."""
    folder = Path(folder)
    return [p.relative_to(folder).as_posix() for p in _files(folder) if p.suffix.lower() in IMAGE_EXT]


def chain_absent_key(folder: str | Path) -> str:
    """UI text key for a missing hash chain: counts.csv without row_hash, or a run_nextraf.py folder (frames.csv)."""
    return "chain_absent" if (Path(folder) / "counts.csv").is_file() else "chain_absent_run"


def privacy_key(folder: str | Path) -> str:
    """UI text key for what the folder stores: video (a failure), annotated images (validation run) or counts only."""
    return "privacy_warn" if find_video_files(folder) else "stored_images" if find_image_files(folder) else "privacy_ok"


def footage_card(folder: str | Path) -> dict:
    """Value, note and tone for the "Footage stored" card. Video is a privacy failure. Annotated stills come from a
    validation run (they can show real people), so the card names them instead of claiming counts only."""
    videos = find_video_files(folder)
    if videos:
        return {"value": f"{len(videos)} video file(s)", "note": "remove them", "tone": "bad"}
    images = find_image_files(folder)
    if images:
        return {"value": "No video", "note": f"{len(images)} annotated image(s) from a validation run", "tone": ""}
    return {"value": "None", "note": "counts only", "tone": "good"}


def privacy_ok(folder: str | Path) -> bool:
    """True when the results folder holds no video files."""
    return not find_video_files(folder)


def camera_health(counts: pd.DataFrame) -> dict:
    """Frames processed and frames with no data (camera_ok = false)."""
    n = len(counts)
    bad = int((~counts["camera_ok"]).sum())
    return {"frames": n, "no_data": bad, "no_data_pct": round(100.0 * bad / n, 1) if n else 0.0}


def chain_status(counts_path: str | Path) -> dict:
    """Hash-chain state of counts.csv: 'absent' (no file or no row_hash column), 'valid' or 'broken'."""
    if not Path(counts_path).is_file():  # e.g. a run_nextraf.py folder, which has frames.csv instead
        return {"state": "absent", "first_bad_line": None}
    header = Path(counts_path).read_text(encoding="utf-8").split("\n", 1)[0].rstrip("\r")
    if not header.endswith("," + hashlog.HASH_COLUMN):
        return {"state": "absent", "first_bad_line": None}
    ok, line = hashlog.verify(counts_path)
    return {"state": "valid" if ok else "broken", "first_bad_line": line}
