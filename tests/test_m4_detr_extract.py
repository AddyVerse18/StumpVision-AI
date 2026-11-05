"""
test_m4_detr_extract.py
=======================
Tests for M4 RF-DETR keyframe extraction and COCO dataset schemas:
- COCO schema creation, validation, and JSON roundtrip
- Keyframe temporal index calculation (deliveries vs idle clips)
- Test-split protection (blocks test split extraction without explicit flag)
- Synthetic video end-to-end extraction and index verification
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest

from stumpvision.detr.coco_schema import (
    create_empty_coco_dataset,
    add_image_to_coco,
    add_annotation_to_coco,
    validate_coco_dataset,
    save_coco_json,
    load_coco_json,
)
from stumpvision.detr.extract_frames import (
    compute_sample_frame_indices,
    extract_keyframes_for_split,
)


def test_coco_schema_creation_and_validation(tmp_path: Path):
    """Verify COCO dataset structure, addition of images/annotations, and JSON roundtrip."""
    dataset = create_empty_coco_dataset()
    is_valid, errors = validate_coco_dataset(dataset)
    assert is_valid, f"Empty dataset should be valid: {errors}"

    img_id = add_image_to_coco(
        dataset=dataset,
        file_name="clip01_f000100.jpg",
        width=1920,
        height=1080,
        clip_id="clip01",
        frame_index=100,
        time_s=1.6667,
        sample_type="delivery_peak",
    )
    assert img_id == 1

    # Add valid annotation (category 1 = person, bbox: [x, y, w, h])
    ann_id = add_annotation_to_coco(
        dataset=dataset,
        image_id=img_id,
        category_id=1,
        bbox_xywh=(100.0, 200.0, 50.0, 150.0),
        score=0.92,
    )
    assert ann_id == 1

    is_valid, errors = validate_coco_dataset(dataset)
    assert is_valid, f"Dataset with valid annotation should be valid: {errors}"

    # Invalid category ID should be rejected
    with pytest.raises(ValueError, match="not present in dataset categories"):
        add_annotation_to_coco(
            dataset=dataset,
            image_id=img_id,
            category_id=999,
            bbox_xywh=(10.0, 10.0, 20.0, 20.0),
        )

    # Save and reload
    json_path = tmp_path / "test_coco.json"
    save_coco_json(dataset, json_path)
    loaded = load_coco_json(json_path)
    assert len(loaded["images"]) == 1
    assert len(loaded["annotations"]) == 1
    assert loaded["images"][0]["file_name"] == "clip01_f000100.jpg"


def test_compute_sample_frame_indices():
    """Verify sampling logic for delivery vs. idle clips."""
    fps = 60.0
    dur_s = 60.0

    # Clip with one delivery at t=10.0s
    samples = compute_sample_frame_indices(
        duration_s=dur_s,
        fps=fps,
        delivery_times_s=[10.0],
        delivery_offsets_s=[-0.25, 0.0, 0.25],
    )
    # Expected frame indices: 10.0 - 0.25 = 9.75s (585), 10.0s (600), 10.25s (615)
    indices = [s[0] for s in samples]
    assert indices == [585, 600, 615]
    types = [s[2] for s in samples]
    assert types == ["delivery_stride", "delivery_peak", "delivery_stride"]

    # Idle clip (no deliveries) with 20s interval
    idle_samples = compute_sample_frame_indices(
        duration_s=dur_s,
        fps=fps,
        delivery_times_s=[],
        idle_interval_s=20.0,
    )
    # Starts at 10.0s, 30.0s, 50.0s -> frames 600, 1800, 3000
    idle_indices = [s[0] for s in idle_samples]
    assert idle_indices == [600, 1800, 3000]
    assert all(s[2] == "idle_scene" for s in idle_samples)


def test_test_split_extraction_blocked_without_flag(tmp_path: Path):
    """Verify that extract_keyframes_for_split raises PermissionError if split='test' and allow_test=False."""
    with pytest.raises(PermissionError, match="locked test split requires allow_test=True"):
        extract_keyframes_for_split(
            manifest_path=tmp_path / "manifest.csv",
            labels_path=tmp_path / "labels.csv",
            status_path=tmp_path / "status.csv",
            data_dir=tmp_path,
            output_frames_dir=tmp_path / "frames",
            split="test",
            allow_test=False,
        )


def test_extract_keyframes_synthetic_video(tmp_path: Path):
    """End-to-end synthetic video frame extraction test."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw" / "cam1"
    raw_dir.mkdir(parents=True, exist_ok=True)

    video_path = raw_dir / "test_clip.mp4"
    fps = 60.0
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h), isColor=True)

    # 120 frames (2.0s duration)
    for i in range(120):
        frame = np.full((h, w, 3), (i % 256), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Manifest
    manifest_df = pd.DataFrame([
        {
            "clip_id": "test_clip",
            "relative_path": "raw/cam1/test_clip.mp4",
            "camera": "cam1",
            "sequence_no": "000100",
            "file_size_bytes": "1000",
            "fps": "60.0",
            "width": "320",
            "height": "240",
            "n_frames_reported": "120",
            "n_frames_decoded": "120",
            "duration_s": "2.0",
            "cfr_status": "ok",
            "schema_version": "1.0",
            "block_id": "block_1",
            "view": "bowler_end",
            "split": "dev",
        }
    ])
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # Status
    status_df = pd.DataFrame([
        {"clip_id": "test_clip", "reviewed": "yes", "reviewed_at": "2026-10-30", "notes": ""}
    ])
    status_path = tmp_path / "status.csv"
    status_df.to_csv(status_path, index=False)

    # Labels with 1 delivery at 1.0s
    labels_df = pd.DataFrame([
        {
            "clip_id": "test_clip",
            "delivery_time_s": "1.0",
            "frame_index": "60",
            "kind": "delivery",
            "notes": "",
            "schema_version": "1.0",
        }
    ])
    labels_path = tmp_path / "labels.csv"
    labels_df.to_csv(labels_path, index=False)

    out_frames_dir = tmp_path / "extracted_frames"
    coco_json_path = tmp_path / "coco.json"
    index_csv_path = tmp_path / "index.csv"

    index_df, stats = extract_keyframes_for_split(
        manifest_path=manifest_path,
        labels_path=labels_path,
        status_path=status_path,
        data_dir=data_dir,
        output_frames_dir=out_frames_dir,
        split="dev",
        delivery_offsets_s=[-0.25, 0.0, 0.25],
        coco_output_path=coco_json_path,
        index_csv_path=index_csv_path,
    )

    assert stats["n_clips_processed"] == 1
    assert stats["n_frames_extracted"] == 3
    assert len(index_df) == 3

    # Check files created on disk
    assert index_csv_path.exists()
    assert coco_json_path.exists()
    for _, r in index_df.iterrows():
        p = Path(r["file_path"])
        assert p.exists()
        img = cv2.imread(str(p))
        assert img is not None
        assert img.shape[:2] == (240, 320)

    # Validate generated COCO JSON
    coco_data = load_coco_json(coco_json_path)
    assert len(coco_data["images"]) == 3
