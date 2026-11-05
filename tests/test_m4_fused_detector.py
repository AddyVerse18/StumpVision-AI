"""
test_m4_fused_detector.py
=========================
Automated unit and integration tests for the M4 multimodal fusion pipeline:
Fusing M3 motion candidates with M4 batsman biomechanical pose status.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from stumpvision.schemas import PREDICTIONS_COLUMNS, validate_predictions_df
from stumpvision.signals.fused_detector import (
    fuse_motion_with_batsman_pose,
    load_poses_as_dataframe,
    FusedFilterStats,
)
from stumpvision.cli import run_detect_fused


M3_PREDS_PATH = Path("outputs/predictions/m3_motion_baseline_dev.csv")
M4_POSES_PATH = Path("labels/m4_poses/batsman_pose_dev.json")
FUSED_OUTPUT_PATH = Path("outputs/predictions/m4_fused_candidates_dev.csv")


def test_load_poses_as_dataframe():
    """Verify pose JSON correctly converts into classified DataFrame."""
    assert M4_POSES_PATH.exists(), f"Pose JSON missing: {M4_POSES_PATH}"
    df = load_poses_as_dataframe(M4_POSES_PATH)
    assert len(df) == 54
    assert "clip_id" in df.columns
    assert "is_present" in df.columns
    assert "action_type" in df.columns
    assert "wrist_distance" in df.columns
    assert "avg_knee_angle" in df.columns


def test_fuse_motion_with_batsman_pose_dev():
    """Verify fusion eliminates false alarms while preserving true deliveries."""
    fused_df, stats = fuse_motion_with_batsman_pose(
        candidates=M3_PREDS_PATH,
        poses=M4_POSES_PATH,
        max_pose_distance_s=1.0,
        method_version="m4_fused_candidates_v1",
        score_boost=1.1,
    )

    validate_predictions_df(fused_df)
    assert isinstance(stats, FusedFilterStats)
    assert stats.n_input_candidates == 180
    assert stats.n_fused_candidates == 13
    assert stats.rejection_rate_pct >= 90.0

    # Verify zero candidates in idle clips
    idle_clips = [
        "20220102063750_000131",
        "20220102063850_000132",
        "20220102063950_000133",
        "20220102064050_000134",
        "20220102064150_000135",
    ]
    for ic in idle_clips:
        assert len(fused_df[fused_df["clip_id"] == ic]) == 0, f"False alarm leaked in idle clip {ic}"


def test_fused_predictions_file_integrity():
    """Verify the saved m4_fused_candidates_dev.csv matches schema and delivery counts."""
    assert FUSED_OUTPUT_PATH.exists(), f"Fused output file missing: {FUSED_OUTPUT_PATH}"
    df = pd.read_csv(FUSED_OUTPUT_PATH, dtype=str)
    validate_predictions_df(df)
    assert len(df) == 13
    assert list(df.columns) == PREDICTIONS_COLUMNS
    assert (df["method_version"] == "m4_fused_candidates_v1").all()


def test_fuse_empty_candidates():
    """Verify empty input handling does not crash."""
    empty_cand = pd.DataFrame(columns=PREDICTIONS_COLUMNS)
    fused_df, stats = fuse_motion_with_batsman_pose(
        candidates=empty_cand,
        poses=M4_POSES_PATH,
    )
    validate_predictions_df(fused_df)
    assert len(fused_df) == 0
    assert stats.n_input_candidates == 0
    assert stats.n_fused_candidates == 0


def test_run_detect_fused_cli(tmp_path):
    """Test CLI subcommand runner with custom temporary output."""
    out_file = tmp_path / "custom_fused.csv"
    code = run_detect_fused(
        candidates_path=str(M3_PREDS_PATH),
        poses_path=str(M4_POSES_PATH),
        output_path=str(out_file),
        max_distance_s=1.0,
    )
    assert code == 0
    assert out_file.exists()
    df = pd.read_csv(out_file)
    validate_predictions_df(df)
    assert len(df) == 13

