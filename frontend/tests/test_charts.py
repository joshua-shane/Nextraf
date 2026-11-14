import pandas as pd

from nextraf_viz import charts


def _counts(seconds):
    t = pd.Timestamp("2026-11-14 16:20:00") + pd.to_timedelta(seconds, unit="s")
    return pd.DataFrame({"time": t, "person": range(len(seconds)), "camera_ok": True})


def test_time_axis_shows_seconds_for_a_short_clip_and_minutes_for_a_long_run():
    short, long = _counts([0, 5, 10, 15]), _counts([0, 600, 1200, 2700])
    assert charts.replay_fig(short, "person", 1).layout.xaxis.tickformat == "%H:%M:%S"
    assert charts.replay_fig(long, "person", 1).layout.xaxis.tickformat == "%H:%M"
