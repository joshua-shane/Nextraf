import pandas as pd

from nextraf_common import hashlog
from nextraf_viz import trust


def test_file_inventory_lists_files_with_size_and_type(tmp_path):
    (tmp_path / "counts.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "x.png").write_bytes(b"1234")
    inv = trust.file_inventory(tmp_path)
    got = {r.path: (r.size_bytes, r.kind) for r in inv.itertuples()}
    assert got == {"counts.csv": (8, "table"), "samples/x.png": (4, "image")}


def test_video_files_detected_case_insensitively(tmp_path):
    (tmp_path / "clip.MP4").write_bytes(b"x")
    (tmp_path / "counts.csv").write_text("a\n")
    assert trust.find_video_files(tmp_path) == ["clip.MP4"]
    assert trust.privacy_ok(tmp_path) is False


def test_privacy_ok_when_only_counts_and_images(tmp_path):
    (tmp_path / "counts.csv").write_text("a\n")
    assert trust.privacy_ok(tmp_path) is True


def test_footage_card_flags_video_as_a_privacy_failure(tmp_path):
    (tmp_path / "counts.csv").write_text("a\n")
    (tmp_path / "clip.MP4").write_bytes(b"x")
    assert trust.footage_card(tmp_path) == {"value": "1 video file(s)", "note": "remove them", "tone": "bad"}


def test_footage_card_names_annotated_images_from_a_validation_run(tmp_path):
    # run_nextraf.py --mode validation writes annotated frames next to frames.csv; they can show real people
    (tmp_path / "frames.csv").write_text("frame\n")
    (tmp_path / "gate_f00000.jpg").write_bytes(b"x")
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "validation_01.png").write_bytes(b"x")
    assert trust.footage_card(tmp_path) == {"value": "No video", "note": "2 annotated image(s) from a validation run",
                                            "tone": ""}


def test_footage_card_is_none_when_only_counts_are_stored(tmp_path):
    (tmp_path / "counts.csv").write_text("a\n")
    assert trust.footage_card(tmp_path) == {"value": "None", "note": "counts only", "tone": "good"}


def test_camera_health_counts_no_data_frames():
    df = pd.DataFrame({"camera_ok": [True, False, True, False]})
    h = trust.camera_health(df)
    assert h == {"frames": 4, "no_data": 2, "no_data_pct": 50.0}


def test_chain_status_reports_missing_and_valid_and_broken(tmp_path):
    p = tmp_path / "counts.csv"
    p.write_text("time,person\n15:30:00,1\n")
    assert trust.chain_status(p)["state"] == "absent"
    q = tmp_path / "q.csv"
    for i in range(3):
        hashlog.append_row(q, {"time": f"15:30:0{i}", "person": i}, columns=["time", "person"])
    assert trust.chain_status(q) == {"state": "valid", "first_bad_line": None}
    q.write_text(q.read_text().replace(",1,", ",9,", 1))
    assert trust.chain_status(q) == {"state": "broken", "first_bad_line": 3}


def test_chain_status_is_absent_when_there_is_no_counts_file(tmp_path):
    assert trust.chain_status(tmp_path / "counts.csv") == {"state": "absent", "first_bad_line": None}
