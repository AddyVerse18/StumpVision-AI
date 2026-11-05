"""
test_eval_integration.py
========================
End-to-end integration tests for StumpVision evaluation harness and CLI.
Validates split protection, audit logging, artifact creation, and unreviewed clip filtering.
"""

from pathlib import Path
import pytest
import pandas as pd

from stumpvision.schemas import (
    MANIFEST_ALL_COLUMNS,
    LABELS_COLUMNS,
    CLIP_STATUS_COLUMNS,
    PREDICTIONS_COLUMNS,
)
from stumpvision.eval.matching import load_ground_truth, load_predictions
from stumpvision.cli import run_evaluate, run_baselines


@pytest.fixture
def synthetic_workspace(tmp_path: Path):
    """
    Create a self-contained synthetic workspace with manifest, labels, and clip status.
    No video files are created.
    """
    manifest_file = tmp_path / "manifest.csv"
    labels_file = tmp_path / "labels.csv"
    status_file = tmp_path / "clip_status.csv"

    # Manifest with 2 dev clips and 1 test clip
    manifest_rows = [
        {
            "clip_id": "clip_dev_001",
            "relative_path": "raw/cam1/clip_dev_001.MP4",
            "camera": "cam1",
            "sequence_no": "101",
            "file_size_bytes": "1000",
            "fps": "60.0",
            "width": "1920",
            "height": "1080",
            "n_frames_reported": "3600",
            "n_frames_decoded": "3600",
            "duration_s": "60.0",
            "cfr_status": "verified_cfr",
            "schema_version": "1.0",
            "block_id": "block_1",
            "view": "bowler_end",
            "split": "dev",
        },
        {
            "clip_id": "clip_dev_002",
            "relative_path": "raw/cam1/clip_dev_002.MP4",
            "camera": "cam1",
            "sequence_no": "102",
            "file_size_bytes": "1000",
            "fps": "60.0",
            "width": "1920",
            "height": "1080",
            "n_frames_reported": "3600",
            "n_frames_decoded": "3600",
            "duration_s": "60.0",
            "cfr_status": "verified_cfr",
            "schema_version": "1.0",
            "block_id": "block_1",
            "view": "bowler_end",
            "split": "dev",
        },
        {
            "clip_id": "clip_test_001",
            "relative_path": "raw/cam1/clip_test_001.MP4",
            "camera": "cam1",
            "sequence_no": "103",
            "file_size_bytes": "1000",
            "fps": "60.0",
            "width": "1920",
            "height": "1080",
            "n_frames_reported": "3600",
            "n_frames_decoded": "3600",
            "duration_s": "60.0",
            "cfr_status": "verified_cfr",
            "schema_version": "1.0",
            "block_id": "block_test",
            "view": "bowler_end",
            "split": "test",
        },
    ]
    pd.DataFrame(manifest_rows)[MANIFEST_ALL_COLUMNS].to_csv(manifest_file, index=False)

    # Labels: clip_dev_001 has 1 delivery at frame 600 (10.0s)
    labels_rows = [
        {
            "clip_id": "clip_dev_001",
            "delivery_time_s": "10.0",
            "frame_index": "600",
            "kind": "delivery",
            "notes": "batter_walkpast",
            "schema_version": "1.0",
        },
        {
            "clip_id": "clip_dev_002",
            "delivery_time_s": "",
            "frame_index": "",
            "kind": "none",
            "notes": "",
            "schema_version": "1.0",
        },
        {
            "clip_id": "clip_test_001",
            "delivery_time_s": "20.0",
            "frame_index": "1200",
            "kind": "delivery",
            "notes": "",
            "schema_version": "1.0",
        },
    ]
    pd.DataFrame(labels_rows)[LABELS_COLUMNS].to_csv(labels_file, index=False)

    # Status: clip_dev_001 and clip_dev_002 reviewed=yes; clip_test_001 reviewed=yes
    status_rows = [
        {"clip_id": "clip_dev_001", "reviewed": "yes", "reviewed_at": "2026-10-28T12:00:00Z"},
        {"clip_id": "clip_dev_002", "reviewed": "yes", "reviewed_at": "2026-10-28T12:00:00Z"},
        {"clip_id": "clip_test_001", "reviewed": "yes", "reviewed_at": "2026-10-28T12:00:00Z"},
    ]
    pd.DataFrame(status_rows)[CLIP_STATUS_COLUMNS].to_csv(status_file, index=False)

    return {
        "manifest": manifest_file,
        "labels": labels_file,
        "status": status_file,
        "dir": tmp_path,
    }


def test_unreviewed_clips_filtering_and_zero_reviewed_failure(synthetic_workspace):
    """
    Unreviewed clips are filtered out.
    If 0 clips are marked reviewed, evaluation fails loudly.
    """
    # Set all clips to unreviewed
    status_file = synthetic_workspace["status"]
    pd.DataFrame([
        {"clip_id": "clip_dev_001", "reviewed": "no", "reviewed_at": ""},
        {"clip_id": "clip_dev_002", "reviewed": "no", "reviewed_at": ""},
    ])[CLIP_STATUS_COLUMNS].to_csv(status_file, index=False)

    with pytest.raises(ValueError, match="Zero clips are marked as reviewed"):
        load_ground_truth(
            manifest_path=synthetic_workspace["manifest"],
            labels_path=synthetic_workspace["labels"],
            status_path=status_file,
            split="dev",
        )


def test_test_split_protection_blocked_without_flag(synthetic_workspace):
    """Accessing the locked test split without allow_test=True raises PermissionError."""
    with pytest.raises(PermissionError, match="Evaluation on the locked test split is restricted"):
        load_ground_truth(
            manifest_path=synthetic_workspace["manifest"],
            labels_path=synthetic_workspace["labels"],
            status_path=synthetic_workspace["status"],
            split="test",
            allow_test=False,
        )


def test_test_split_protection_logged_with_flag(synthetic_workspace):
    """Accessing the test split with allow_test=True logs the access and succeeds."""
    dataset = load_ground_truth(
        manifest_path=synthetic_workspace["manifest"],
        labels_path=synthetic_workspace["labels"],
        status_path=synthetic_workspace["status"],
        split="test",
        allow_test=True,
        command_str="pytest_test_access",
    )
    assert dataset.reviewed_clips_count == 1
    # Check that outputs/test_access.log received the entry
    log_file = Path("outputs/test_access.log")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "pytest_test_access" in content


def test_run_evaluate_artifacts_created(synthetic_workspace):
    """run_evaluate end-to-end produces report.md, per_clip.csv, matches.csv, config.json."""
    preds_file = synthetic_workspace["dir"] / "test_preds.csv"
    pd.DataFrame([
        {
            "clip_id": "clip_dev_001",
            "time_s": "10.05",
            "score": "0.95",
            "status": "candidate",
            "method_version": "v1.0-synthetic-test",
        }
    ])[PREDICTIONS_COLUMNS].to_csv(preds_file, index=False)

    out_dir = synthetic_workspace["dir"] / "test_report"

    ret = run_evaluate(
        pred_path=str(preds_file),
        split="dev",
        tolerance=0.5,
        secondary_tolerance=1.0,
        allow_test=False,
        manifest_path=str(synthetic_workspace["manifest"]),
        labels_path=str(synthetic_workspace["labels"]),
        status_path=str(synthetic_workspace["status"]),
        output_dir=str(out_dir),
    )
    assert ret == 0

    assert (out_dir / "report.md").exists()
    assert (out_dir / "per_clip.csv").exists()
    assert (out_dir / "matches.csv").exists()
    assert (out_dir / "config.json").exists()

    report_content = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Primary Benchmark Metrics" in report_content
    assert "Recall" in report_content
    assert "Precision" in report_content
    assert "1 of 1 (100.0%)" in report_content  # Recall with exact count


def test_run_baselines_end_to_end(synthetic_workspace):
    """run_baselines generates predictions and outputs comparison table without error."""
    out_dir = synthetic_workspace["dir"] / "baseline_out"

    ret = run_baselines(
        split="dev",
        tolerance=0.5,
        allow_test=False,
        manifest_path=str(synthetic_workspace["manifest"]),
        labels_path=str(synthetic_workspace["labels"]),
        status_path=str(synthetic_workspace["status"]),
        output_dir=str(out_dir),
    )
    assert ret == 0
    assert (out_dir / "predict_nothing.csv").exists()
    assert (out_dir / "every_10s.csv").exists()
    assert (out_dir / "random_times_sample.csv").exists()

