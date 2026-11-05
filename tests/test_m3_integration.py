"""
test_m3_integration.py
======================
End-to-end integration tests for M3 candidate detection and M2 evaluation compatibility.
"""

from pathlib import Path
import pytest
import pandas as pd

from stumpvision.schemas import (
    MANIFEST_ALL_COLUMNS,
    LABELS_COLUMNS,
    CLIP_STATUS_COLUMNS,
    validate_predictions_df,
)
from stumpvision.detect.motion import run_detection_on_split
from stumpvision.cli import run_evaluate, run_detect


@pytest.fixture
def m3_synthetic_workspace(tmp_path: Path, synthetic_video: Path):
    """
    Build a minimal workspace with one synthetic clip in dev split.
    """
    raw_cam1 = tmp_path / "raw" / "cam1"
    raw_cam1.mkdir(parents=True, exist_ok=True)

    clip_file = raw_cam1 / "synthetic_001.mp4"
    import shutil
    shutil.copy(str(synthetic_video), str(clip_file))

    manifest_file = tmp_path / "manifest.csv"
    labels_file = tmp_path / "labels.csv"
    status_file = tmp_path / "clip_status.csv"

    manifest_rows = [
        {
            "clip_id": "synthetic_001",
            "relative_path": "raw/cam1/synthetic_001.mp4",
            "camera": "cam1",
            "sequence_no": "1",
            "file_size_bytes": "1000",
            "fps": "30.0",
            "width": "640",
            "height": "360",
            "n_frames_reported": "30",
            "n_frames_decoded": "30",
            "duration_s": "1.0",
            "cfr_status": "verified_cfr",
            "schema_version": "1.0",
            "block_id": "block_1",
            "view": "bowler_end",
            "split": "dev",
        }
    ]
    pd.DataFrame(manifest_rows)[MANIFEST_ALL_COLUMNS].to_csv(manifest_file, index=False)

    labels_rows = [
        {
            "clip_id": "synthetic_001",
            "delivery_time_s": "0.5",
            "frame_index": "15",
            "kind": "delivery",
            "notes": "",
            "schema_version": "1.0",
        }
    ]
    pd.DataFrame(labels_rows)[LABELS_COLUMNS].to_csv(labels_file, index=False)

    status_rows = [
        {"clip_id": "synthetic_001", "reviewed": "yes", "reviewed_at": "2026-10-28T12:00:00Z"}
    ]
    pd.DataFrame(status_rows)[CLIP_STATUS_COLUMNS].to_csv(status_file, index=False)

    return {
        "manifest": manifest_file,
        "labels": labels_file,
        "status": status_file,
        "data_dir": tmp_path,
        "tmp_path": tmp_path,
    }


def test_m3_detection_and_m2_evaluation_e2e(m3_synthetic_workspace):
    """
    Run detection on synthetic dev workspace, verify predictions schema,
    and evaluate through M2 harness.
    """
    preds_csv = m3_synthetic_workspace["tmp_path"] / "m3_preds.csv"

    preds_df, stats = run_detection_on_split(
        manifest_path=m3_synthetic_workspace["manifest"],
        status_path=m3_synthetic_workspace["status"],
        data_dir=m3_synthetic_workspace["data_dir"],
        split="dev",
    )

    assert stats["n_clips_processed"] == 1
    validate_predictions_df(preds_df)
    preds_df.to_csv(preds_csv, index=False)

    # Evaluate directly with M2 run_evaluate
    report_dir = m3_synthetic_workspace["tmp_path"] / "m3_eval_report"
    ret = run_evaluate(
        pred_path=str(preds_csv),
        split="dev",
        tolerance=0.5,
        manifest_path=str(m3_synthetic_workspace["manifest"]),
        labels_path=str(m3_synthetic_workspace["labels"]),
        status_path=str(m3_synthetic_workspace["status"]),
        output_dir=str(report_dir),
    )
    assert ret == 0
    assert (report_dir / "report.md").exists()
    assert (report_dir / "matches.csv").exists()


def test_m3_missing_video_file_handled_safely(m3_synthetic_workspace):
    """
    If a video file referenced in manifest is missing on disk,
    detector warns, emits 0 predictions for that clip, and does not crash.
    """
    # Delete the video file
    clip_file = m3_synthetic_workspace["data_dir"] / "raw" / "cam1" / "synthetic_001.mp4"
    clip_file.unlink()

    preds_df, stats = run_detection_on_split(
        manifest_path=m3_synthetic_workspace["manifest"],
        status_path=m3_synthetic_workspace["status"],
        data_dir=m3_synthetic_workspace["data_dir"],
        split="dev",
    )

    assert stats["n_clips_processed"] == 1
    assert stats["n_candidates_found"] == 0
    assert len(preds_df) == 0

