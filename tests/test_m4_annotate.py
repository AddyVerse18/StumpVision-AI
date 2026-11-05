"""
test_m4_annotate.py
===================
Automated tests for M4 RF-DETR AnnotationSession and COCO integrity validation:
- Session initialization and resume at unannotated image
- BBox coordinate bounds clamping, inversion handling, and zero-area rejection
- Duplicate bbox prevention
- Selection, deletion, and undo mechanics
- Monotonic unique annotation ID assignment
- Atomic saving and automatic timestamped backup generation
- Deep integrity validation on valid vs corrupted COCO datasets
"""

from pathlib import Path
import json
import pytest

from stumpvision.detr.coco_schema import (
    create_empty_coco_dataset,
    add_image_to_coco,
    save_coco_json,
    validate_coco_integrity,
)
from stumpvision.detr.annotate import AnnotationSession


@pytest.fixture
def sample_coco_dataset(tmp_path: Path):
    """Fixture creating a minimal valid COCO file with 3 images."""
    dataset = create_empty_coco_dataset()
    for i in range(1, 4):
        add_image_to_coco(
            dataset=dataset,
            file_name=f"frame_{i:03d}.jpg",
            width=1920,
            height=1080,
            image_id=i,
            clip_id=f"clip_{i:02d}",
            frame_index=i * 100,
            time_s=float(i * 1.6667),
            sample_type="delivery_peak",
        )
    coco_path = tmp_path / "annotations" / "test_coco.json"
    save_coco_json(dataset, coco_path)

    # Create dummy images on disk in frames_dir
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        (frames_dir / f"frame_{i:03d}.jpg").write_text("fake_image_bytes")

    return coco_path, frames_dir


def test_session_initialization_and_resume(sample_coco_dataset):
    coco_path, frames_dir = sample_coco_dataset
    session = AnnotationSession(coco_path=coco_path, frames_dir=frames_dir, auto_resume=True)

    assert len(session.images) == 3
    assert session.active_category_id == 1  # person by default
    assert session.image_idx == 0  # image 1 is first unannotated
    assert session.current_image()["id"] == 1


def test_add_bounding_box_clamping_and_validation(sample_coco_dataset):
    coco_path, frames_dir = sample_coco_dataset
    session = AnnotationSession(coco_path=coco_path, frames_dir=frames_dir)

    # 1. Normal valid bbox
    ann1 = session.add_bounding_box((100.0, 200.0, 150.0, 300.0), category_id=1)
    assert ann1 is not None
    assert ann1["bbox"] == [100.0, 200.0, 150.0, 300.0]
    assert ann1["area"] == 45000.0
    assert len(session.current_annotations()) == 1

    # 2. Inverted coordinates (dragged bottom-right to top-left)
    ann2 = session.add_bounding_box((500.0, 500.0, -100.0, -80.0), category_id=2)
    assert ann2 is not None
    assert ann2["bbox"] == [400.0, 420.0, 100.0, 80.0]

    # 3. Out-of-bounds coordinates (should clamp to 1920x1080)
    ann3 = session.add_bounding_box((1800.0, 1000.0, 300.0, 200.0), category_id=3)
    assert ann3 is not None
    assert ann3["bbox"][0] == 1800.0
    assert ann3["bbox"][1] == 1000.0
    assert ann3["bbox"][2] == 120.0  # 1920 - 1800
    assert ann3["bbox"][3] == 80.0   # 1080 - 1000

    # 4. Tiny/zero-area box rejected (< 2px)
    ann_tiny = session.add_bounding_box((50.0, 50.0, 1.0, 1.0), category_id=1)
    assert ann_tiny is None

    # 5. Duplicate box rejected
    ann_dup = session.add_bounding_box((100.0, 200.0, 150.0, 300.0), category_id=1)
    assert ann_dup is None


def test_selection_deletion_and_undo(sample_coco_dataset):
    coco_path, frames_dir = sample_coco_dataset
    session = AnnotationSession(coco_path=coco_path, frames_dir=frames_dir)

    ann1 = session.add_bounding_box((50.0, 50.0, 60.0, 60.0), category_id=1)
    ann2 = session.add_bounding_box((200.0, 200.0, 80.0, 80.0), category_id=2)
    assert len(session.current_annotations()) == 2

    # Select and delete ann2 (selected_bbox_idx was set to 1 upon adding ann2)
    deleted = session.delete_selected_bounding_box()
    assert deleted is not None
    assert deleted["id"] == ann2["id"]
    assert len(session.current_annotations()) == 1

    # Undo delete restores ann2
    assert session.undo() is True
    assert len(session.current_annotations()) == 2
    assert session.current_annotations()[-1]["id"] == ann2["id"]

    # Undo add removes ann2
    assert session.undo() is True
    assert len(session.current_annotations()) == 1
    assert session.current_annotations()[0]["id"] == ann1["id"]


def test_annotation_id_monotonicity_and_resume(sample_coco_dataset):
    coco_path, frames_dir = sample_coco_dataset
    session = AnnotationSession(coco_path=coco_path, frames_dir=frames_dir)

    # Annotate image 1
    session.add_bounding_box((10, 10, 50, 50), category_id=1)  # ID 1
    session.add_bounding_box((60, 60, 50, 50), category_id=2)  # ID 2

    # Move to image 2 and annotate
    session.next_image()
    assert session.image_idx == 1
    ann3 = session.add_bounding_box((100, 100, 40, 40), category_id=3)
    assert ann3["id"] == 3

    # Save session
    session.save()

    # Re-open session in new instance
    session2 = AnnotationSession(coco_path=coco_path, frames_dir=frames_dir, auto_resume=True)
    # Image 1 and 2 are annotated, image 3 is unannotated -> auto_resume should place us at index 2
    assert session2.image_idx == 2
    assert session2.current_image()["id"] == 3

    # Add box to image 3 -> ID must be 4 (monotonic without collision)
    ann4 = session2.add_bounding_box((200, 200, 50, 50), category_id=4)
    assert ann4["id"] == 4


def test_atomic_save_and_timestamped_backup(sample_coco_dataset):
    coco_path, frames_dir = sample_coco_dataset
    backup_dir = coco_path.parent / "backups"

    session = AnnotationSession(
        coco_path=coco_path,
        frames_dir=frames_dir,
        backup_dir=backup_dir,
    )
    session.add_bounding_box((20, 20, 40, 40), category_id=1)

    saved_p, backup_p = session.save()
    assert saved_p.exists()
    assert backup_p is not None
    assert backup_p.exists()
    assert "backup" in backup_p.name
    assert backup_p.parent == backup_dir

    # Check saved contents
    with open(saved_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["bbox"] == [20.0, 20.0, 40.0, 40.0]


def test_validate_coco_integrity(sample_coco_dataset):
    coco_path, frames_dir = sample_coco_dataset

    # 1. Valid dataset
    is_valid, errors, stats = validate_coco_integrity(coco_path, frames_dir)
    assert is_valid is True
    assert len(errors) == 0
    assert stats["total_images"] == 3
    assert stats["total_annotations"] == 0

    # 2. Add an annotation and validate again
    session = AnnotationSession(coco_path=coco_path, frames_dir=frames_dir)
    session.add_bounding_box((50, 50, 100, 100), category_id=1)
    session.save()

    is_valid, errors, stats = validate_coco_integrity(coco_path, frames_dir)
    assert is_valid is True
    assert stats["total_annotations"] == 1
    assert stats["annotated_images"] == 1
    assert stats["unannotated_images"] == 2
    assert stats["category_counts"]["person"] == 1

    # 3. Corrupt coordinates directly in JSON and verify caught
    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Corrupt bbox to be out-of-bounds
    data["annotations"][0]["bbox"] = [1900.0, 1000.0, 500.0, 200.0]
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    is_valid, errors, stats = validate_coco_integrity(coco_path, frames_dir)
    assert is_valid is False
    assert any("exceeds image boundaries" in err for err in errors)

