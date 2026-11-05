"""
test_labels.py
==============
Unit tests for StumpVision labeling store, split enforcement, leakage detection,
and report generation (headless, non-GUI).
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest
import yaml

from stumpvision.schemas import (
    SCHEMA_VERSION,
    LABELS_COLUMNS,
    CLIP_STATUS_COLUMNS,
    MANIFEST_ALL_COLUMNS,
)
from stumpvision.labeling.store import LabelStore
from stumpvision.labeling.splits import (
    assign_splits,
    check_leakage,
    record_test_access,
    compute_clip_set_fingerprint,
    LOCKED_TEST_FINGERPRINT,
)
from stumpvision.labeling.reports import generate_labels_report, compare_label_sets


@pytest.fixture
def temp_labeling_env(tmp_path):
    """Create temporary manifest, splits config, and labeling paths."""
    manifest_path = tmp_path / "manifest.csv"
    labels_path = tmp_path / "labels.csv"
    status_path = tmp_path / "clip_status.csv"
    config_path = tmp_path / "splits.yaml"

    # Create dummy manifest with 19 clips matching sequence 130-148
    rows = []
    for seq in range(130, 149):
        rows.append({
            "clip_id": f"clip_{seq}",
            "relative_path": f"raw/cam1/clip_{seq}.mp4",
            "camera": "cam1",
            "sequence_no": str(seq),
            "file_size_bytes": "1000",
            "fps": "60.0",
            "width": "1920",
            "height": "1080",
            "n_frames_reported": "3600",
            "n_frames_decoded": "3600",
            "duration_s": "60.0",
            "cfr_status": "verified_cfr",
            "schema_version": SCHEMA_VERSION,
            "block_id": "",
            "view": "bowler_end",
            "split": "",
        })
    m_df = pd.DataFrame(rows)[MANIFEST_ALL_COLUMNS]
    m_df.to_csv(manifest_path, index=False)

    # Create splits config matching M1 specification
    splits_dict = {
        "splits": {
            "dev": {
                "blocks": [
                    {"block_id": "block_1", "sequence_range": [130, 133]},
                    {"block_id": "block_2", "sequence_range": [134, 137]},
                    {"block_id": "block_3", "sequence_range": [138, 142]},
                ]
            },
            "buffer": {
                "blocks": [
                    {"block_id": "block_buffer", "sequence_range": [143, 143]}
                ]
            },
            "test": {
                "blocks": [
                    {"block_id": "block_test", "sequence_range": [144, 148]}
                ]
            },
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(splits_dict, f)

    return {
        "dir": tmp_path,
        "manifest": manifest_path,
        "labels": labels_path,
        "status": status_path,
        "config": config_path,
    }


def test_unreviewed_clip_not_equal_no_delivery(temp_labeling_env):
    """
    CRITICAL REQUIREMENT:
    An unreviewed clip must NOT be treated as a no-delivery clip.
    A clip with no delivery gets a kind=none row ONLY when explicitly marked reviewed.
    """
    store = LabelStore(
        labels_path=temp_labeling_env["labels"],
        clip_status_path=temp_labeling_env["status"],
        manifest_path=temp_labeling_env["manifest"],
    )

    clip_id = "clip_130"

    # Initially, clip is unreviewed
    assert store.is_clip_reviewed(clip_id) is False
    assert len(store.get_marks_for_clip(clip_id)) == 0

    # Report before review: clip_130 is listed under unreviewed clips
    rep_data, rep_str = generate_labels_report(
        manifest_path=temp_labeling_env["manifest"],
        labels_path=temp_labeling_env["labels"],
        status_path=temp_labeling_env["status"],
    )
    assert clip_id in rep_data["unreviewed_clips"]
    assert "Benchmark evaluation uses reviewed clips only" in rep_str

    # Explicitly mark clip reviewed (with no deliveries marked)
    store.set_clip_reviewed(clip_id, reviewed=True)

    assert store.is_clip_reviewed(clip_id) is True
    marks = store.get_marks_for_clip(clip_id)
    assert len(marks) == 1
    assert marks[0]["kind"] == "none"
    assert marks[0]["delivery_time_s"] == ""
    assert marks[0]["frame_index"] == ""

    # Reload store from disk to verify persistence
    reloaded_store = LabelStore(
        labels_path=temp_labeling_env["labels"],
        clip_status_path=temp_labeling_env["status"],
        manifest_path=temp_labeling_env["manifest"],
    )
    assert reloaded_store.is_clip_reviewed(clip_id) is True
    assert reloaded_store.get_marks_for_clip(clip_id)[0]["kind"] == "none"


def test_marking_delivery_frame_index_truth(temp_labeling_env):
    """
    frame_index is the stored truth; delivery_time_s = frame_index / fps.
    Adding a delivery to a previously zero-delivery clip clears the 'none' placeholder.
    """
    store = LabelStore(
        labels_path=temp_labeling_env["labels"],
        clip_status_path=temp_labeling_env["status"],
        manifest_path=temp_labeling_env["manifest"],
    )

    clip_id = "clip_131"
    store.set_clip_reviewed(clip_id, reviewed=True)
    assert store.get_marks_for_clip(clip_id)[0]["kind"] == "none"

    # Now mark an actual delivery at frame 1500 (FPS is 60.0 -> 25.0s)
    mark = store.add_mark(clip_id=clip_id, frame_index=1500, kind="delivery", notes="batter_walkpast")
    assert mark["frame_index"] == "1500"
    assert mark["delivery_time_s"] == "25.0000"
    assert mark["kind"] == "delivery"
    assert mark["notes"] == "batter_walkpast"

    # Ensure the kind=none row was removed
    all_marks = store.get_marks_for_clip(clip_id)
    assert len(all_marks) == 1
    assert all_marks[0]["kind"] == "delivery"
    assert all_marks[0]["frame_index"] == "1500"


def test_undo_delivery(temp_labeling_env):
    """Verify undo removes the most recent mark for the current clip."""
    store = LabelStore(
        labels_path=temp_labeling_env["labels"],
        clip_status_path=temp_labeling_env["status"],
        manifest_path=temp_labeling_env["manifest"],
    )

    clip_id = "clip_132"
    store.add_mark(clip_id, frame_index=600, kind="delivery")
    store.add_mark(clip_id, frame_index=1800, kind="delivery")
    assert len(store.get_marks_for_clip(clip_id)) == 2

    undone = store.undo_last_mark(clip_id)
    assert undone["frame_index"] == "1800"

    marks = store.get_marks_for_clip(clip_id)
    assert len(marks) == 1
    assert marks[0]["frame_index"] == "600"


def test_blind_relabel_mode(temp_labeling_env):
    """Verify blind mode hides existing labels and writes to dedicated file."""
    retest_path = temp_labeling_env["dir"] / "retest.csv"

    # Main store has existing mark
    main_store = LabelStore(
        labels_path=temp_labeling_env["labels"],
        clip_status_path=temp_labeling_env["status"],
        manifest_path=temp_labeling_env["manifest"],
    )
    main_store.add_mark("clip_133", frame_index=900, kind="delivery")

    # Blind retest store
    blind_store = LabelStore(
        labels_path=retest_path,
        clip_status_path=temp_labeling_env["status"],
        manifest_path=temp_labeling_env["manifest"],
        blind=True,
    )
    assert blind_store.get_marks_for_clip("clip_133") == []

    blind_store.add_mark("clip_133", frame_index=905, kind="delivery")

    # Check main labels file was not altered by blind store
    main_df = pd.read_csv(temp_labeling_env["labels"])
    retest_df = pd.read_csv(retest_path)

    assert len(main_df) == 1
    assert main_df.iloc[0]["frame_index"] == 900
    assert len(retest_df) == 1
    assert retest_df.iloc[0]["frame_index"] == 905


def test_splits_and_leakage_check(temp_labeling_env):
    """Test assigning splits and verifying zero leakage."""
    assign_splits(
        manifest_path=temp_labeling_env["manifest"],
        config_path=temp_labeling_env["config"],
    )

    test_clips = [f"clip_{i}" for i in range(144, 149)]
    fingerprint = compute_clip_set_fingerprint(test_clips)

    res = check_leakage(
        manifest_path=temp_labeling_env["manifest"],
        expected_fingerprint=fingerprint,
    )
    assert res["status"] == "PASS"
    assert res["dev_clips_count"] == 13
    assert res["buffer_clips_count"] == 1
    assert res["test_clips_count"] == 5

    # Tampering test set triggers error
    m_df = pd.read_csv(temp_labeling_env["manifest"])
    m_df.loc[m_df["clip_id"] == "clip_144", "split"] = "dev"
    m_df.to_csv(temp_labeling_env["manifest"], index=False)

    with pytest.raises(ValueError, match="Test set fingerprint mismatch"):
        check_leakage(temp_labeling_env["manifest"], expected_fingerprint=fingerprint)


def test_reports_linked_partial_counting(temp_labeling_env):
    """Verify linked boundary delivery is counted once when halves link."""
    assign_splits(temp_labeling_env["manifest"], temp_labeling_env["config"])
    store = LabelStore(
        labels_path=temp_labeling_env["labels"],
        clip_status_path=temp_labeling_env["status"],
        manifest_path=temp_labeling_env["manifest"],
    )

    # Mark clip 130 as partial_end linked to clip 131
    store.add_mark("clip_130", frame_index=3550, kind="partial_end", notes="linked_to=clip_131")
    store.set_clip_reviewed("clip_130", reviewed=True)

    # Mark clip 131 as partial_start linked to clip 130
    store.add_mark("clip_131", frame_index=50, kind="partial_start", notes="linked_to=clip_130")
    store.set_clip_reviewed("clip_131", reviewed=True)

    rep_data, rep_str = generate_labels_report(
        manifest_path=temp_labeling_env["manifest"],
        labels_path=temp_labeling_env["labels"],
        status_path=temp_labeling_env["status"],
    )

    dev_stats = rep_data["splits"]["dev"]
    assert dev_stats["linked_boundary_deliveries"] == 1
    assert dev_stats["unlinked_partial_marks"] == 0


def test_compare_labels_agreement(temp_labeling_env):
    """Test consistency comparison across two label sessions."""
    p_path = temp_labeling_env["labels"]
    r_path = temp_labeling_env["dir"] / "retest.csv"

    # Primary marks
    p_df = pd.DataFrame([
        {"clip_id": "clip_130", "delivery_time_s": "10.0", "frame_index": "600", "kind": "delivery", "notes": "", "schema_version": "1.0"},
        {"clip_id": "clip_131", "delivery_time_s": "20.0", "frame_index": "1200", "kind": "delivery", "notes": "", "schema_version": "1.0"},
    ])
    p_df.to_csv(p_path, index=False)

    # Retest marks (clip 130 within 5 frames / 0.08s, clip 131 within 10 frames / 0.16s)
    r_df = pd.DataFrame([
        {"clip_id": "clip_130", "delivery_time_s": "10.0833", "frame_index": "605", "kind": "delivery", "notes": "", "schema_version": "1.0"},
        {"clip_id": "clip_131", "delivery_time_s": "20.1667", "frame_index": "1210", "kind": "delivery", "notes": "", "schema_version": "1.0"},
    ])
    r_df.to_csv(r_path, index=False)

    stats, rep = compare_label_sets(primary_path=p_path, retest_path=r_path, tolerance_s=0.5)

    assert stats["common_clips_count"] == 2
    assert stats["total_matched_deliveries"] == 2
    assert stats["agreement_within_half_second_pct"] == 100.0
    assert stats["max_frame_diff"] == 10
    assert stats["missing_marks_in_retest"] == 0
    assert stats["extra_marks_in_retest"] == 0


def test_record_test_access(tmp_path):
    """Test audit logging for test set access."""
    log_file = tmp_path / "test_access.log"
    record_test_access("eval --split test", log_path=log_file)
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "command=eval --split test" in content
