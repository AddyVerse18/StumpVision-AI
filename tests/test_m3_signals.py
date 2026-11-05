"""
test_m3_signals.py
==================
Unit tests for streamed motion signal extraction in StumpVision (M3).
"""

from pathlib import Path
import pytest
import numpy as np

from stumpvision.signals.motion import extract_motion_signal, save_signal_csv


def test_extract_motion_from_synthetic_video(synthetic_video: Path):
    """
    Test streaming motion extraction from synthetic video.
    Verifies that moving rectangle generates non-zero motion and exact frame counts.
    """
    res = extract_motion_signal(synthetic_video, downscale_width=160, downscale_height=90)

    assert res["total_frames"] == 30
    assert len(res["motion"]) == 30
    assert len(res["time_s"]) == 30
    assert res["fps"] == 30.0
    # First frame has 0 motion by definition
    assert res["motion"][0] == 0.0
    # Subsequent frames have motion from moving green box
    assert np.mean(res["motion"][1:]) > 0.0


def test_extract_motion_missing_file():
    """Missing video file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        extract_motion_signal(Path("nonexistent_video_file_12345.mp4"))


def test_extract_motion_save_csv(synthetic_video: Path, tmp_path: Path):
    """save_signal_csv writes valid CSV with required columns."""
    res = extract_motion_signal(synthetic_video, downscale_width=80, downscale_height=45)
    out_csv = tmp_path / "signal_out.csv"

    saved = save_signal_csv(res, out_csv)
    assert saved.exists()

    import pandas as pd
    df = pd.read_csv(saved)
    assert list(df.columns) == ["frame_index", "time_s", "motion"]
    assert len(df) == 30

