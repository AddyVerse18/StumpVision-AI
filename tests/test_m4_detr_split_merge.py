"""
test_m4_detr_split_merge.py
===========================
Tests for RF-DETR dataset splitting, Roboflow ingestion, category remapping,
and safe merger routines.
"""

from pathlib import Path
import json
import pytest

from stumpvision.detr.coco_schema import (
    create_empty_coco_dataset,
    add_image_to_coco,
    add_annotation_to_coco,
    save_coco_json,
    load_coco_json,
    validate_coco_integrity,
)
from stumpvision.detr.split_dataset import (
    split_coco_dataset_by_blocks,
    export_rf_detr_dataset,
)
from stumpvision.detr.import_roboflow import (
    remap_roboflow_coco_data,
    merge_coco_datasets,
    import_and_merge_roboflow_split,
)


@pytest.fixture
def mock_dataset_with_blocks(tmp_path: Path):
    """Create a mock dataset containing images across block_1, block_2, and block_3."""
    coco = create_empty_coco_dataset()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 4 images: 2 in block_1, 1 in block_2, 1 in block_3
    blocks = ["block_1", "block_1", "block_2", "block_3"]
    for i, b in enumerate(blocks, 1):
        fname = f"clip_{i:02d}_f000100.jpg"
        (frames_dir / fname).write_text("image_data")
        img_id = add_image_to_coco(
            dataset=coco,
            file_name=fname,
            width=1920,
            height=1080,
            image_id=i,
            clip_id=f"clip_{i:02d}",
        )
        coco["images"][-1]["block_id"] = b

        # Add 1 annotation per image
        add_annotation_to_coco(
            dataset=coco,
            image_id=img_id,
            category_id=1,  # person
            bbox_xywh=(100.0, 100.0, 50.0, 120.0),
        )

    coco_path = tmp_path / "annotations" / "dev_coco.json"
    save_coco_json(coco, coco_path)
    return coco_path, frames_dir


def test_split_coco_dataset_by_blocks(mock_dataset_with_blocks):
    coco_path, _ = mock_dataset_with_blocks
    coco_data = load_coco_json(coco_path)

    train_coco, val_coco = split_coco_dataset_by_blocks(
        coco_data=coco_data,
        train_blocks=["block_1", "block_2"],
        val_blocks=["block_3"],
    )

    # Block 1 (2 images) + Block 2 (1 image) = 3 train images
    assert len(train_coco["images"]) == 3
    assert len(train_coco["annotations"]) == 3

    # Block 3 = 1 val image
    assert len(val_coco["images"]) == 1
    assert len(val_coco["annotations"]) == 1
    assert val_coco["images"][0]["block_id"] == "block_3"

    # Verify zero overlap between train and val image IDs
    train_ids = {img["id"] for img in train_coco["images"]}
    val_ids = {img["id"] for img in val_coco["images"]}
    assert train_ids.isdisjoint(val_ids)


def test_export_rf_detr_dataset(mock_dataset_with_blocks, tmp_path: Path):
    coco_path, frames_dir = mock_dataset_with_blocks
    out_dir = tmp_path / "rf_detr_export"

    stats = export_rf_detr_dataset(
        coco_path=coco_path,
        frames_dir=frames_dir,
        output_dataset_dir=out_dir,
        train_blocks=["block_1", "block_2"],
        val_blocks=["block_3"],
        copy_images=True,
    )

    train_json = out_dir / "train" / "_annotations.coco.json"
    val_json = out_dir / "valid" / "_annotations.coco.json"
    assert train_json.exists()
    assert val_json.exists()

    train_coco = load_coco_json(train_json)
    val_coco = load_coco_json(val_json)
    assert len(train_coco["images"]) == 3
    assert len(val_coco["images"]) == 1

    # Verify copied image files
    assert stats["train_images_copied"] == 3
    assert stats["val_images_copied"] == 1
    for img in train_coco["images"]:
        assert (out_dir / "train" / img["file_name"]).exists()
    for img in val_coco["images"]:
        assert (out_dir / "valid" / img["file_name"]).exists()


def test_remap_roboflow_coco_data():
    raw_rf_coco = {
        "info": {"description": "Roboflow Raw"},
        "licenses": [],
        "categories": [
            {"id": 0, "name": "batsman"},
            {"id": 1, "name": "wicket"},
            {"id": 2, "name": "cricket-ball"},
            {"id": 3, "name": "spectator"},  # unmapped class
        ],
        "images": [
            {"id": 1, "file_name": "match01.jpg", "width": 640, "height": 640},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 0, "bbox": [50, 50, 60, 150]},  # batsman -> person (1)
            {"id": 11, "image_id": 1, "category_id": 1, "bbox": [200, 200, 20, 40]}, # wicket -> stumps (2)
            {"id": 12, "image_id": 1, "category_id": 2, "bbox": [300, 300, 10, 10]}, # cricket-ball -> ball (6)
            {"id": 13, "image_id": 1, "category_id": 3, "bbox": [10, 10, 30, 30]},   # spectator -> dropped
        ],
    }

    remapped, stats = remap_roboflow_coco_data(
        rf_coco_data=raw_rf_coco,
        image_id_offset=10000,
        ann_id_offset=100000,
        filename_prefix="rf_",
    )

    # 1 image remapped with new id 10001 and prefix rf_
    assert len(remapped["images"]) == 1
    assert remapped["images"][0]["id"] == 10001
    assert remapped["images"][0]["file_name"] == "rf_match01.jpg"

    # 3 annotations kept, 1 dropped (spectator)
    assert len(remapped["annotations"]) == 3
    assert stats["dropped_annotations"] == 1
    assert "spectator" in stats["dropped_categories"]

    cat_ids = [a["category_id"] for a in remapped["annotations"]]
    assert cat_ids == [1, 2, 6]  # person, stumps, ball
    assert remapped["annotations"][0]["id"] == 100010
    assert remapped["annotations"][0]["image_id"] == 10001


def test_merge_coco_datasets_and_collision_guard():
    base_coco = create_empty_coco_dataset()
    add_image_to_coco(base_coco, "img1.jpg", 1920, 1080, image_id=1)
    add_annotation_to_coco(base_coco, image_id=1, category_id=1, bbox_xywh=(10, 10, 20, 20), annotation_id=1)

    # Clean secondary dataset with distinct IDs and filenames
    sec_coco = create_empty_coco_dataset()
    add_image_to_coco(sec_coco, "rf_img2.jpg", 640, 640, image_id=10001)
    add_annotation_to_coco(sec_coco, image_id=10001, category_id=2, bbox_xywh=(30, 30, 20, 20), annotation_id=100001)

    merged = merge_coco_datasets(base_coco, sec_coco)
    assert len(merged["images"]) == 2
    assert len(merged["annotations"]) == 2

    # Collision test: conflicting image ID
    colliding_coco = create_empty_coco_dataset()
    add_image_to_coco(colliding_coco, "different_name.jpg", 640, 640, image_id=1)  # collision on ID 1
    with pytest.raises(ValueError, match="collision detected"):
        merge_coco_datasets(base_coco, colliding_coco)


def test_import_and_merge_roboflow_split(tmp_path: Path):
    # Target split dir with 1 existing image
    target_split = tmp_path / "dataset" / "train"
    target_split.mkdir(parents=True, exist_ok=True)
    base_coco = create_empty_coco_dataset()
    add_image_to_coco(base_coco, "local_01.jpg", 1920, 1080, image_id=1)
    save_coco_json(base_coco, target_split / "_annotations.coco.json")
    (target_split / "local_01.jpg").write_text("local")

    # Mock Roboflow export dir
    rf_dir = tmp_path / "rf_download" / "train"
    rf_dir.mkdir(parents=True, exist_ok=True)
    rf_coco = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "batsman"}, {"id": 2, "name": "stumps"}],
        "images": [{"id": 5, "file_name": "rf_source.jpg", "width": 640, "height": 640}],
        "annotations": [{"id": 15, "image_id": 5, "category_id": 1, "bbox": [10, 10, 50, 100]}],
    }
    save_coco_json(rf_coco, rf_dir / "_annotations.coco.json")
    (rf_dir / "rf_source.jpg").write_text("rf_bytes")

    stats = import_and_merge_roboflow_split(
        rf_split_dir=rf_dir,
        target_rf_detr_split_dir=target_split,
        image_id_offset=10000,
        ann_id_offset=100000,
        filename_prefix="rf_",
        copy_images=True,
    )

    assert stats["total_images"] == 2
    assert stats["roboflow_images_added"] == 1
    assert stats["roboflow_images_copied"] == 1

    # Verify merged file exists and is valid
    final_coco = load_coco_json(target_split / "_annotations.coco.json")
    assert len(final_coco["images"]) == 2
    assert (target_split / "rf_rf_source.jpg").exists() or (target_split / "rf_source.jpg").exists()

