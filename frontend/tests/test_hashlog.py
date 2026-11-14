import hashlib

from nextraf_common import hashlog

COLS = ["time", "person", "camera_ok"]


def _build(tmp_path, n=4):
    path = tmp_path / "counts.csv"
    for i in range(n):
        hashlog.append_row(path, {"time": f"15:30:{i:02d}", "person": 5 + i, "camera_ok": True}, columns=COLS)
    return path


def test_valid_chain_verifies(tmp_path):
    assert hashlog.verify(_build(tmp_path)) == (True, None)


def test_row_hash_is_last_column_and_matches_spec(tmp_path):
    path = _build(tmp_path, 2)
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0] == "time,person,camera_ok,row_hash"
    payload, h = lines[1].rsplit(",", 1)
    assert h == hashlib.sha256(("0" * 64 + "|" + payload).encode()).hexdigest()
    payload2, h2 = lines[2].rsplit(",", 1)
    assert h2 == hashlib.sha256((h + "|" + payload2).encode()).hexdigest()


def test_edited_value_is_caught_at_that_line(tmp_path):
    path = _build(tmp_path)
    lines = path.read_text(encoding="utf-8").split("\n")
    lines[3] = lines[3].replace(",7,", ",70,")
    path.write_text("\n".join(lines), encoding="utf-8", newline="")
    assert hashlog.verify(path) == (False, 3 + 1)


def test_deleted_row_is_caught(tmp_path):
    path = _build(tmp_path)
    lines = path.read_text(encoding="utf-8").split("\n")
    del lines[2]
    path.write_text("\n".join(lines), encoding="utf-8", newline="")
    ok, bad = hashlog.verify(path)
    assert not ok and bad == 3


def test_crlf_line_endings_still_verify(tmp_path):
    path = _build(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert hashlog.verify(path) == (True, None)


def test_file_without_row_hash_column_is_not_ok(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("time,person\n15:30:00,3\n", encoding="utf-8")
    assert hashlog.verify(path) == (False, 1)
