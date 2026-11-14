"""Tamper-evident, hash-chained CSV log (feature E).

Shared by the Pi pipeline (writer) and the viewer (verifier).

Spec: ``row_hash`` is the LAST column. For each data row,
``payload`` is the exact CSV text of the line without the trailing
``,<row_hash>`` field and without the newline, and
``row_hash = sha256((prev_hash + "|" + payload).encode("utf-8")).hexdigest()``.
``prev_hash`` is 64 zeros for the first data row. The header is not hashed.
Verification works on the raw file text, never on re-formatted values.
"""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

GENESIS = "0" * 64
HASH_COLUMN = "row_hash"


def _digest(prev_hash: str, payload: str) -> str:
    return hashlib.sha256((prev_hash + "|" + payload).encode("utf-8")).hexdigest()


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def _csv_line(fields: list[str]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(fields)
    return buf.getvalue()


def _read_lines(path: Path) -> list[str]:
    text = Path(path).read_bytes().decode("utf-8")
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def append_row(path: str | Path, values: dict, columns: list[str] | None = None) -> str:
    """Append one row to the chained CSV and return its ``row_hash``.

    ``columns`` fixes the column order and is used when the file is created;
    it defaults to the keys of ``values``. Missing keys are written empty.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        lines = _read_lines(path)
        header = next(csv.reader([lines[0]]))
        if header[-1] != HASH_COLUMN:
            raise ValueError(f"{path} has no trailing {HASH_COLUMN} column")
        cols = header[:-1]
        prev = lines[-1].rsplit(",", 1)[1] if len(lines) > 1 else GENESIS
        new_file = False
    else:
        cols = list(columns) if columns is not None else list(values)
        prev, new_file = GENESIS, True
    payload = _csv_line([_fmt(values.get(c)) for c in cols])
    row_hash = _digest(prev, payload)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        if new_file:
            fh.write(_csv_line(cols + [HASH_COLUMN]) + "\n")
        fh.write(f"{payload},{row_hash}\n")
    return row_hash


def verify(path: str | Path) -> tuple[bool, int | None]:
    """Check the chain. Returns ``(ok, first_bad_line)``.

    Lines are 1-based file lines, the header being line 1. A missing
    ``row_hash`` column is reported as a failure at line 1.
    """
    lines = _read_lines(Path(path))
    if not lines or not lines[0].endswith("," + HASH_COLUMN):
        return False, 1
    prev = GENESIS
    for n, line in enumerate(lines[1:], start=2):
        payload, sep, found = line.rpartition(",")
        if not sep or found != _digest(prev, payload):
            return False, n
        prev = found
    return True, None
