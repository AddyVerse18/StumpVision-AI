"""
test_m4_pretrain_validation.py
==============================
Tests for M4 RF-DETR pre-training validation, category normalization,
provenance tracking, and dry-run execution.
"""

from pathlib import Path
import json
import pytest

from stumpvision.detr.coco_schema import (
    create_empty_coco_dataset,
    add_image_to_coco,
    add_annotation_to_coco,
    save_coco_json,
)
from stumpvision.detr.dataset_prep import (
    normalize_dataset_categories,
    validate_pretrain_coco,
    update_dataset_provenance,
    execute_pretraining_dry_run,
    CANONICAL_NAME_TO_ID,
)


@pytest.fixture
def valid_coco_with_annotations(tmp_path: Path):
    """Fixture creating a valid COCO dataset with 2 images and 2 annotations."""
    dataset = create_empty_coco_dataset()
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 3):
        fname = f"frame_{i:02d}.jpg"
        (img_dir / fname).write_text("img_bytes")
        iid = add_image_to_coco(dataset, fname, 1920, 1080, image_id=i)
        add_annotation_to_coco(dataset, iid, category_id=1, bbox_xywh=(100, 100, 50, 150))

    return dataset, img_dir


def test_validate_pretrain_coco_zero_annotations_rejected(tmp_path: Path):
    """Verify that unannotated local frames are explicitly rejected from training."""
    dataset = create_empty_coco_dataset()
    add_image_to_coco(dataset, "frame_01.jpg", 1920, 1080, image_id=1)

    is_valid, errors, summary = validate_pretrain_coco(
        coco_data=dataset,
        check_zero_annotations=True,
    )
    assert is_valid is False
    assert any("0 bounding-box annotations" in e for e in errors)


def test_validate_pretrain_coco_out_of_bounds_and_invalid_bboxes(valid_coco_with_annotations):
    dataset, img_dir = valid_coco_with_annotations

    # 1. Non-positive width/height
    dataset["annotations"].append({
        "id": 99,
        "image_id": 1,
        "category_id": 1,
        "bbox": [10.0, 10.0, 0.0, 50.0],
        "area": 0.0,
        "iscrowd": 0,
    })
    is_valid, errors, _ = validate_pretrain_coco(dataset, images_dir=img_dir)
    assert is_valid is False
    assert any("non-positive width/height" in e for e in errors)

    # 2. Out-of-bounds coordinate (x + w > 1920)
    dataset["annotations"][-1]["bbox"] = [1900.0, 500.0, 50.0, 50.0]
    is_valid, errors, _ = validate_pretrain_coco(dataset, images_dir=img_dir)
    assert is_valid is False
    assert any("exceeds image bounds" in e for e in errors)


def test_validate_pretrain_coco_duplicate_and_missing_images(valid_coco_with_annotations, tmp_path: Path):
    dataset, img_dir = valid_coco_with_annotations

    # Duplicate image ID
    dataset["images"].append({
        "id": 1,  # already exists
        "file_name": "unique_name.jpg",
        "width": 1920,
        "height": 1080,
    })
    is_valid, errors, _ = validate_pretrain_coco(dataset, images_dir=img_dir)
    assert is_valid is False
    assert any("Duplicate image ID" in e for e in errors)

    # Missing image on disk
    dataset["images"][-1]["id"] = 999
    is_valid, errors, _ = validate_pretrain_coco(dataset, images_dir=img_dir)
    assert is_valid is False
    assert any("not found on disk" in e for e in errors)


def test_normalize_dataset_categories_explicit_and_ambiguous():
    raw_coco = {
        "categories": [
            {"id": 10, "name": "batsman"},
            {"id": 20, "name": "wicket"},
            {"id": 30, "name": "cheerleader"},  # Ambiguous/unmapped
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [10, 10, 50, 50]},
            {"id": 2, "image_id": 1, "category_id": 20, "bbox": [60, 60, 20, 20]},
            {"id": 3, "image_id": 1, "category_id": 30, "bbox": [90, 90, 30, 30]},
        ],
    }

    # Should raise ValueError because 'cheerleader' is unmapped and not in allowed_unmapped_to_drop
    with pytest.raises(ValueError, match="Unresolved / ambiguous categories"):
        normalize_dataset_categories(raw_coco)

    # Allowed unmapped drop
    norm_coco, report = normalize_dataset_categories(
        raw_coco,
        allowed_unmapped_to_drop={"cheerleader"},
    )
    assert len(norm_coco["annotations"]) == 2
    assert report["dropped_annotations_count"] == 1
    assert "cheerleader" in report["dropped_categories"]

    # Check mapped IDs conform to canonical definitions
    cats_in_anns = {a["category_id"] for a in norm_coco["annotations"]}
    assert cats_in_anns == {CANONICAL_NAME_TO_ID["person"], CANONICAL_NAME_TO_ID["stumps"]}


def test_update_dataset_provenance():
    base = create_empty_coco_dataset()
    updated = update_dataset_provenance(
        coco_data=base,
        source_name="Roboflow Universe - Cricket Detection",
        source_version="v2.1",
        source_license="CC BY 4.0",
        source_split="train",
    )
    assert updated["info"]["source_name"] == "Roboflow Universe - Cricket Detection"
    assert updated["info"]["source_version"] == "v2.1"
    assert any(l["name"] == "CC BY 4.0" for l in updated["licenses"])


def test_execute_pretraining_dry_run_success_and_failure(tmp_path: Path):
    dataset_dir = tmp_path / "ds"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 1. Empty setup fails
    is_ready, report = execute_pretraining_dry_run(dataset_dir)
    assert is_ready is False
    assert len(report["errors"]) > 0

    # 2. Add valid train and valid annotations with images
    for split_dir in [train_dir, val_dir]:
        img_name = "test_img.jpg"
        (split_dir / img_name).write_text("data")
        coco = create_empty_coco_dataset()
        iid = add_image_to_coco(coco, img_name, 1920, 1080, image_id=1)
        add_annotation_to_coco(coco, iid, category_id=1, bbox_xywh=(100, 100, 50, 50))
        save_coco_json(coco, split_dir / "_annotations.coco.json")

    # Dry run should now pass
    is_ready, report = execute_pretraining_dry_run(dataset_dir)
    assert is_ready is True
    assert "train" in report["splits"]
    assert "valid" in report["splits"]
    assert report["splits"]["train"]["images_count"] == 1
    assert report["splits"]["valid"]["images_count"] == 1

    # Check resume checkpoint verification
    fake_ckpt = tmp_path / "checkpoint_last.pth"
    fake_ckpt.write_text("fake_weights")
    is_ready_resume, report_resume = execute_pretraining_dry_run(
        dataset_dir,
        resume_checkpoint=fake_ckpt,
    )
    assert is_ready_resume is True
    assert "resume_checkpoint" in report_resume

