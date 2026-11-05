"""
split_dataset.py
================
Prepares RF-DETR formatted dataset splits (train and valid).
RF-DETR expects:
    dataset_dir/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    └── valid/
        ├── _annotations.coco.json
        └── *.jpg

Features:
- Enforces zero-leakage split at the contiguous block or clip level.
- Preserves all canonical category definitions and image dimensions.
- Copies or links image files into train/ and valid/ subfolders.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Set, Tuple, Any

from stumpvision.detr.coco_schema import (
    load_coco_json,
    save_coco_json,
    validate_coco_dataset,
    DEFAULT_CATEGORIES,
)


def split_coco_dataset_by_blocks(
    coco_data: Dict[str, Any],
    train_blocks: Optional[List[str]] = None,
    val_blocks: Optional[List[str]] = None,
    default_val_ratio: float = 0.25,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split a COCO dataset dictionary into train and valid sub-datasets.
    Uses clip block_id if present, or clip_id grouping to prevent temporal frame leakage.

    Returns:
        (train_coco_dict, val_coco_dict)
    """
    is_valid, errors = validate_coco_dataset(coco_data)
    if not is_valid:
        raise ValueError(f"Input COCO dataset is invalid: {errors}")

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", list(DEFAULT_CATEGORIES))

    if train_blocks is None:
        train_blocks = ["block_1", "block_2"]
    if val_blocks is None:
        val_blocks = ["block_3"]

    train_blocks_set = set(train_blocks)
    val_blocks_set = set(val_blocks)

    # Group images by clip_id
    clip_to_images: Dict[str, List[Dict[str, Any]]] = {}
    for img in images:
        cid = img.get("clip_id") or Path(img["file_name"]).stem.split("_f")[0]
        clip_to_images.setdefault(cid, []).append(img)

    train_images: List[Dict[str, Any]] = []
    val_images: List[Dict[str, Any]] = []

    # If images have block_id metadata, use block assignment
    has_blocks = any("block_id" in img for img in images)
    if has_blocks:
        for img in images:
            b_id = img.get("block_id", "")
            if b_id in val_blocks_set:
                val_images.append(deepcopy(img))
            else:
                train_images.append(deepcopy(img))
    else:
        # Sort clips deterministically and partition
        unique_clips = sorted(clip_to_images.keys())
        n_val_clips = max(1, int(round(len(unique_clips) * default_val_ratio)))
        val_clips_set = set(unique_clips[-n_val_clips:])

        for cid, img_list in clip_to_images.items():
            if cid in val_clips_set:
                val_images.extend([deepcopy(im) for im in img_list])
            else:
                train_images.extend([deepcopy(im) for im in img_list])

    train_img_ids = {img["id"] for img in train_images}
    val_img_ids = {img["id"] for img in val_images}

    train_anns = [deepcopy(a) for a in annotations if a["image_id"] in train_img_ids]
    val_anns = [deepcopy(a) for a in annotations if a["image_id"] in val_img_ids]

    train_coco = {
        "info": deepcopy(coco_data.get("info", {})),
        "licenses": deepcopy(coco_data.get("licenses", [])),
        "images": train_images,
        "annotations": train_anns,
        "categories": deepcopy(categories),
    }

    val_coco = {
        "info": deepcopy(coco_data.get("info", {})),
        "licenses": deepcopy(coco_data.get("licenses", [])),
        "images": val_images,
        "annotations": val_anns,
        "categories": deepcopy(categories),
    }

    return train_coco, val_coco


def export_rf_detr_dataset(
    coco_path: Path,
    frames_dir: Path,
    output_dataset_dir: Path,
    train_blocks: Optional[List[str]] = None,
    val_blocks: Optional[List[str]] = None,
    copy_images: bool = True,
) -> Dict[str, Any]:
    """
    Export the dataset in the standard RF-DETR layout:
        output_dataset_dir/train/_annotations.coco.json
        output_dataset_dir/valid/_annotations.coco.json
    """
    coco_path = Path(coco_path)
    frames_dir = Path(frames_dir)
    output_dataset_dir = Path(output_dataset_dir)

    coco_data = load_coco_json(coco_path)
    train_coco, val_coco = split_coco_dataset_by_blocks(
        coco_data=coco_data,
        train_blocks=train_blocks,
        val_blocks=val_blocks,
    )

    train_dir = output_dataset_dir / "train"
    val_dir = output_dataset_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Save COCO annotation files named _annotations.coco.json (RF-DETR convention)
    save_coco_json(train_coco, train_dir / "_annotations.coco.json")
    save_coco_json(val_coco, val_dir / "_annotations.coco.json")

    # Copy / transfer image files
    train_copied = 0
    val_copied = 0

    if copy_images:
        for img in train_coco["images"]:
            src_file = frames_dir / img["file_name"]
            dst_file = train_dir / img["file_name"]
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                train_copied += 1

        for img in val_coco["images"]:
            src_file = frames_dir / img["file_name"]
            dst_file = val_dir / img["file_name"]
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                val_copied += 1

    stats = {
        "output_dir": str(output_dataset_dir),
        "train_images_count": len(train_coco["images"]),
        "train_annotations_count": len(train_coco["annotations"]),
        "train_images_copied": train_copied,
        "val_images_count": len(val_coco["images"]),
        "val_annotations_count": len(val_coco["annotations"]),
        "val_images_copied": val_copied,
    }
    return stats

