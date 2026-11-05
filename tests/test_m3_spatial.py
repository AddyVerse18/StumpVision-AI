"""
test_m3_spatial.py
==================
Tests for vertical spatial-band motion refinement (M3 Experiment).
Verifies:
1. Extraction of top, mid, bottom motion bands.
2. Bit-exact regression parity when spatial_filtering is False.
3. Soft confidence penalty application on foreground-dominant motion.
4. Zero-penalty preservation for pitch-corridor motion.
5. Numerical safety under zero corridor motion.
"""

from pathlib import Path
import numpy as np
import pytest

from stumpvision.signals.motion import extract_motion_signal
from stumpvision.detect.motion import find_candidate_peaks_from_signal, DEFAULT_CONFIG


def test_extract_motion_signal_bands_synthetic(tmp_path: Path):
    """Verify that extract_motion_signal captures band signals correctly."""
    import cv2
    video_path = tmp_path / "synthetic_bands.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 60.0, (160, 90), isColor=False)

    # 10 black frames
    for _ in range(10):
        out.write(np.zeros((90, 160), dtype=np.uint8))
    # Frame 10: motion only in top band (y: 0..30)
    top_frame = np.zeros((90, 160), dtype=np.uint8)
    top_frame[0:30, :] = 200
    out.write(top_frame)
    # Frame 11: motion only in bottom band (y: 60..90)
    bot_frame = np.zeros((90, 160), dtype=np.uint8)
    bot_frame[60:90, :] = 200
    out.write(bot_frame)
    out.release()

    res = extract_motion_signal(video_path, downscale_width=160, downscale_height=90)
    assert "motion_top" in res
    assert "motion_mid" in res
    assert "motion_bot" in res
    assert len(res["motion_top"]) == len(res["motion"])

    # At frame 10 (index 10), motion_top should be high, motion_bot should be near 0
    assert res["motion_top"][10] > 100.0
    assert res["motion_bot"][10] < 5.0

    # At frame 11 (index 11), motion_bot should be high
    assert res["motion_bot"][11] > 100.0


def test_spatial_filtering_disabled_regression():
    """Verify bit-exact predictions when spatial_filtering is False."""
    np.random.seed(42)
    n = 300
    fps = 60.0
    time_s = np.linspace(0, n / fps, n)
    motion = np.random.normal(1.0, 0.1, n)
    # Add a prominent peak at index 150
    motion[150] = 10.0

    motion_top = np.zeros(n)
    motion_mid = np.zeros(n)
    motion_bot = np.zeros(n)
    motion_bot[150] = 10.0  # Foreground-dominant

    # Without spatial filtering (default)
    cands_default = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
        config={"spatial_filtering": False},
        motion_top=motion_top,
        motion_mid=motion_mid,
        motion_bot=motion_bot,
    )

    # Calling without band arguments at all
    cands_no_bands = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
        config={"spatial_filtering": False},
    )

    assert len(cands_default) == len(cands_no_bands)
    for (t1, s1), (t2, s2) in zip(cands_default, cands_no_bands):
        assert t1 == pytest.approx(t2, abs=1e-6)
        assert s1 == pytest.approx(s2, abs=1e-6)


def test_foreground_dominant_spike_penalized():
    """Verify that a foreground-dominant peak is penalized when spatial_filtering is True."""
    n = 300
    fps = 60.0
    time_s = np.linspace(0, n / fps, n)
    motion = np.ones(n) * 0.5
    motion[150] = 6.0  # Peak

    # Bot dominates (foreground: 5.0, top+mid: 1.0 -> ratio 5.0 > 1.0)
    motion_top = np.ones(n) * 0.2
    motion_mid = np.ones(n) * 0.3
    motion_bot = np.ones(n) * 0.5
    motion_top[150] = 0.5
    motion_mid[150] = 0.5
    motion_bot[150] = 5.0

    cands_unfiltered = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
        config={"spatial_filtering": False},
        motion_top=motion_top,
        motion_mid=motion_mid,
        motion_bot=motion_bot,
    )

    cands_filtered = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
        config={
            "spatial_filtering": True,
            "foreground_ratio_thresh": 1.0,
            "spatial_penalty_alpha": 1.0,
        },
        motion_top=motion_top,
        motion_mid=motion_mid,
        motion_bot=motion_bot,
    )

    assert len(cands_unfiltered) == 1
    assert len(cands_filtered) == 1
    t_unfilt, s_unfilt = cands_unfiltered[0]
    t_filt, s_filt = cands_filtered[0]

    assert t_unfilt == pytest.approx(t_filt, abs=1e-6)
    # Filtered score must be strictly lower
    assert s_filt < s_unfilt
    # Ratio = 5.0 / (1.0 + 1e-4) ~ 4.999
    # Excess = 3.999
    # Penalty = 1 / (1 + 1.0 * 3.999) ~ 0.20
    expected_penalty = 1.0 / (1.0 + (5.0 / 1.0001 - 1.0))
    assert s_filt == pytest.approx(s_unfilt * expected_penalty, rel=1e-2)


def test_corridor_dominant_spike_unpenalized():
    """Verify that a corridor-dominant peak receives NO penalty (ratio <= threshold)."""
    n = 300
    fps = 60.0
    time_s = np.linspace(0, n / fps, n)
    motion = np.ones(n) * 0.5
    motion[150] = 6.0

    # Corridor dominates (top: 2.5, mid: 2.5, bot: 1.0 -> ratio 1.0 / 5.0 = 0.2 <= 1.0)
    motion_top = np.ones(n) * 0.2
    motion_mid = np.ones(n) * 0.3
    motion_bot = np.ones(n) * 0.5
    motion_top[150] = 2.5
    motion_mid[150] = 2.5
    motion_bot[150] = 1.0

    cands_unfiltered = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
        config={"spatial_filtering": False},
        motion_top=motion_top,
        motion_mid=motion_mid,
        motion_bot=motion_bot,
    )

    cands_filtered = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
        config={
            "spatial_filtering": True,
            "foreground_ratio_thresh": 1.0,
            "spatial_penalty_alpha": 1.0,
        },
        motion_top=motion_top,
        motion_mid=motion_mid,
        motion_bot=motion_bot,
    )

    assert len(cands_unfiltered) == 1
    assert len(cands_filtered) == 1
    assert cands_unfiltered[0][1] == pytest.approx(cands_filtered[0][1], abs=1e-6)


def test_zero_denominator_numerical_safety():
    """Verify that ratio calculation handles zero corridor motion without dividing by zero."""
    n = 100
    fps = 60.0
    time_s = np.linspace(0, n / fps, n)
    motion = np.ones(n) * 0.5
    motion[50] = 5.0

    # Top and mid are strictly 0.0
    motion_top = np.zeros(n)
    motion_mid = np.zeros(n)
    motion_bot = np.ones(n) * 0.1
    motion_bot[50] = 5.0

    # Should not raise ZeroDivisionError or produce NaN/inf; with min_score=0.01 candidate is retained at floor 0.05
    cands = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
        config={"spatial_filtering": True, "foreground_ratio_thresh": 1.0, "min_score": 0.01},
        motion_top=motion_top,
        motion_mid=motion_mid,
        motion_bot=motion_bot,
    )
    assert len(cands) == 1
    t, score = cands[0]
    assert np.isfinite(score)
    assert score == pytest.approx(0.05, abs=1e-3)
