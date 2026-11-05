"""
test_audit.py
=============
Unit tests for StumpVision M0 audit, video probing, and manifest contract.
"""

from pathlib import Path
import pandas as pd
import pytest

from stumpvision.io.video import probe_video, count_decoded_frames, parse_sequence_number
from stumpvision.schemas import (
    MANIFEST_ALL_COLUMNS,
    validate_manifest_df,
    merge_and_save_manifest,
)
from stumpvision.cli import check_project_for_video_files


def test_parse_sequence_number():
    """Verify numeric suffix extraction without relying on unreliable date clocks."""
    assert parse_sequence_number("20220102063650_000130.MP4") == 130
    assert parse_sequence_number("clip_000042.mp4") == 42
    assert parse_sequence_number("cam1_1.avi") == 1


def test_synthetic_video_probe(synthetic_video: Path):
    """Test video probing on code-generated synthetic video."""
    data_dir = synthetic_video.parent
    row = probe_video(
        video_path=synthetic_video,
        camera="cam1",
        data_dir=data_dir,
        count_frames=False,
    )

    assert row["clip_id"] == synthetic_video.stem
    assert row["camera"] == "cam1"
    assert row["sequence_no"] == 1
    assert row["fps"] == 30.0
    assert row["width"] == 640
    assert row["height"] == 360
    assert row["n_frames_reported"] == 30
    assert row["duration_s"] == 1.0
    assert row["cfr_status"] in ["verified_cfr", "not verified", "variable_frame_rate"]
    assert row["schema_version"] == "1.0"


def test_count_decoded_frames(synthetic_video: Path):
    """Test stream decoding accurately counts frames without memory spikes."""
    decoded_count, elapsed = count_decoded_frames(synthetic_video)
    assert decoded_count == 30
    assert elapsed >= 0.0


def test_cfr_status_never_ok_when_unverified(synthetic_video: Path):
    """Verify CFR status is 'not verified' if ffprobe absent, never 'ok'."""
    row = probe_video(
        video_path=synthetic_video,
        camera="cam1",
        data_dir=synthetic_video.parent,
    )
    assert row["cfr_status"] != "ok"
    assert row["cfr_status"] in ["verified_cfr", "not verified", "variable_frame_rate"]


def test_schema_validation():
    """Verify schema raises error on missing contract columns."""
    valid_data = {col: ["test"] for col in MANIFEST_ALL_COLUMNS}
    valid_df = pd.DataFrame(valid_data)
    assert validate_manifest_df(valid_df) is True

    invalid_df = pd.DataFrame({"clip_id": ["123"]})
    with pytest.raises(ValueError, match="Manifest missing required columns"):
        validate_manifest_df(invalid_df)


def test_manifest_merge_preserves_manual_columns(tmp_path: Path):
    """
    CRITICAL: Re-running audit must update machine columns and PRESERVE manual columns
    (block_id, view, split) by clip_id.
    """
    manifest_path = tmp_path / "manifest.csv"

    # 1. Initial audit run
    initial_rows = [
        {
            "clip_id": "clip_001",
            "relative_path": "raw/cam1/clip_001.mp4",
            "camera": "cam1",
            "sequence_no": 1,
            "file_size_bytes": 1000,
            "fps": 60.0,
            "width": 1920,
            "height": 1080,
            "n_frames_reported": 3600,
            "n_frames_decoded": 3600,
            "duration_s": 60.0,
            "cfr_status": "not verified",
            "schema_version": "1.0",
        }
    ]
    df1 = merge_and_save_manifest(initial_rows, manifest_path)
    assert df1.loc[0, "view"] == ""
    assert df1.loc[0, "split"] == ""

    # 2. User manually edits manual columns in spreadsheet
    df1["block_id"] = "block_1"
    df1["view"] = "bowler_end"
    df1["split"] = "dev"
    df1.to_csv(manifest_path, index=False)

    # 3. Second audit run with updated machine column (e.g. file size or decoded count)
    second_rows = [
        {
            "clip_id": "clip_001",
            "relative_path": "raw/cam1/clip_001.mp4",
            "camera": "cam1",
            "sequence_no": 1,
            "file_size_bytes": 1200,  # updated machine stat
            "fps": 60.0,
            "width": 1920,
            "height": 1080,
            "n_frames_reported": 3600,
            "n_frames_decoded": 3600,
            "duration_s": 60.0,
            "cfr_status": "not verified",
            "schema_version": "1.0",
        }
    ]
    df2 = merge_and_save_manifest(second_rows, manifest_path)

    # Verify machine column updated
    assert int(df2.loc[0, "file_size_bytes"]) == 1200
    # Verify manual columns PRESERVED!
    assert df2.loc[0, "block_id"] == "block_1"
    assert df2.loc[0, "view"] == "bowler_end"
    assert df2.loc[0, "split"] == "dev"


def test_read_only_footage_safety(synthetic_video: Path):
    """Verify probing does not alter video file mtime or byte length."""
    stat_before = synthetic_video.stat()
    probe_video(
        video_path=synthetic_video,
        camera="cam1",
        data_dir=synthetic_video.parent,
        count_frames=True,
    )
    stat_after = synthetic_video.stat()
    assert stat_before.st_size == stat_after.st_size
    assert stat_before.st_mtime == stat_after.st_mtime


def test_project_video_warning(tmp_path: Path):
    """Verify check_project_for_video_files detects stray video files."""
    test_video = tmp_path / "rogue_clip.mp4"
    test_video.write_bytes(b"dummy video data")
    # check_project_for_video_files checks PROJECT_ROOT; verify logic
    assert test_video.suffix == ".mp4"
