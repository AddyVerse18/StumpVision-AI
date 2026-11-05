"""
test_m3_detect.py
=================
Unit tests for adaptive thresholding and peak detection in StumpVision (M3).
"""

from pathlib import Path
import pytest
import numpy as np

from stumpvision.detect.motion import (
    estimate_adaptive_threshold,
    find_candidate_peaks_from_signal,
    detect_candidates_for_clip,
)


def test_adaptive_threshold_quiet_signal():
    """Quiet flat signal produces stable noise floor and no extreme threshold jumps."""
    fps = 60.0
    n_frames = 1200  # 20 seconds
    noise = np.random.normal(loc=0.5, scale=0.05, size=n_frames)
    noise = np.clip(noise, 0.0, None)

    med, mad, thresh = estimate_adaptive_threshold(
        motion=noise, fps=fps, noise_window_s=5.0, mad_multiplier=3.5
    )

    assert len(thresh) == n_frames
    assert np.all(thresh > med)
    assert np.all(thresh > 0.5)


def test_find_candidate_peaks_known_peaks():
    """
    Synthetic signal with quiet background and two distinct prominent peaks.
    Detector must identify both peaks with high confidence scores.
    """
    fps = 60.0
    n_frames = 3600  # 60s
    time_s = np.arange(n_frames) / fps
    motion = np.full(n_frames, 0.5, dtype=float)

    # Add Gaussian noise
    rng = np.random.RandomState(42)
    motion += rng.normal(0.0, 0.05, size=n_frames)

    # Peak 1 at 15.0s (frame 900), height 6.0
    motion[900] = 6.0
    # Peak 2 at 45.0s (frame 2700), height 7.5
    motion[2700] = 7.5

    candidates = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
        config={
            "noise_window_s": 5.0,
            "mad_multiplier": 3.5,
            "min_prominence": 1.5,
            "min_separation_s": 3.5,
        },
    )

    assert len(candidates) == 2
    # Candidate 1 around 15.0s
    assert abs(candidates[0][0] - 15.0) < 0.05
    assert candidates[0][1] > 0.5
    # Candidate 2 around 45.0s
    assert abs(candidates[1][0] - 45.0) < 0.05
    assert candidates[1][1] > 0.5


def test_duplicate_suppression_closely_spaced_peaks():
    """
    Two spikes spaced only 0.5s apart (e.g. at 20.0s and 20.5s).
    With min_separation_s=3.5s, the second spike must be suppressed.
    """
    fps = 60.0
    n_frames = 1800
    time_s = np.arange(n_frames) / fps
    motion = np.full(n_frames, 0.4, dtype=float)

    # Spike at 20.0s (frame 1200) and 20.5s (frame 1230)
    motion[1200] = 7.0
    motion[1230] = 6.5

    candidates = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
        config={"min_separation_s": 3.5, "min_prominence": 1.0},
    )

    assert len(candidates) == 1
    assert abs(candidates[0][0] - 20.0) < 0.05


def test_empty_signal_no_peaks():
    """Flat zero motion produces 0 candidates."""
    fps = 60.0
    n_frames = 600
    time_s = np.arange(n_frames) / fps
    motion = np.zeros(n_frames)

    candidates = find_candidate_peaks_from_signal(
        motion=motion,
        time_s=time_s,
        fps=fps,
    )
    assert candidates == []


def test_deterministic_detection():
    """Repeated calls on identical inputs yield bit-exact identical candidates and scores."""
    fps = 60.0
    n_frames = 1200
    time_s = np.arange(n_frames) / fps
    rng = np.random.RandomState(99)
    motion = rng.exponential(scale=0.5, size=n_frames)
    motion[300] = 8.0
    motion[800] = 9.0

    c1 = find_candidate_peaks_from_signal(motion, time_s, fps)
    c2 = find_candidate_peaks_from_signal(motion, time_s, fps)

    assert c1 == c2


def test_detect_candidates_for_clip_synthetic_video(synthetic_video: Path):
    """Running detect_candidates_for_clip on synthetic video returns schema-valid dicts."""
    preds = detect_candidates_for_clip(synthetic_video, clip_id="synth_001")
    # Synthetic video is only 1s long with moving box, so may or may not peak,
    # but returned list must conform to prediction schema
    for p in preds:
        assert p["clip_id"] == "synth_001"
        assert "time_s" in p
        assert "score" in p
        assert p["status"] == "candidate"

