"""
import_roboflow.py
===================
Utilities to import, remap, and merge external Roboflow Universe COCO datasets
with StumpVision AI M4 RF-DETR datasets.

Guarantees:
- Safe category remapping to canonical 6 classes (person, stumps, pitch, crease, bat, ball).
- Zero ID collision: offsets external image IDs and annotation IDs deterministically.
- Zero filename collision: prefixes external image files with dataset namespace (e.g. rf_).
- Produces valid COCO JSON compatible with RF-DETR data loaders.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple, Any

from stumpvision.detr.coco_schema import (
    DEFAULT_CATEGORIES,
    load_coco_json,
    save_coco_json,
    validate_coco_dataset,
)

DEFAULT_ROBOFLOW_CATEGORY_MAP: Dict[str, str] = {
    # Person variants
    "batsman": "person",
    "batter": "person",
    "bowler": "person",
    "wicket-keeper": "person",
    "wicketkeeper": "person",
    "keeper": "person",
    "umpire": "person",
    "non-striker": "person",
    "player": "person",
    "person": "person",
    # Stumps variants
    "stump": "stumps",
    "stumps": "stumps",
    "wicket": "stumps",
    "wickets": "stumps",
    # Pitch / Crease
    "pitch": "pitch",
    "pitch-crease": "pitch",
    "crease": "crease",
    # Equipment
    "bat": "bat",
    "cricket-bat": "bat",
    "ball": "ball",
    "cricket-ball": "ball",
}


def remap_roboflow_coco_data(
    rf_coco_data: Dict[str, Any],
    category_map: Optional[Dict[str, str]] = None,
    image_id_offset: int = 10000,
    ann_id_offset: int = 100000,
    filename_prefix: str = "rf_",
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Remap Roboflow COCO dataset dictionary into canonical 6-class StumpVision format.

    Args:
        rf_coco_data: Raw parsed Roboflow COCO dict
        category_map: Mapping of source category names to target canonical category names
        image_id_offset: Base offset to add to image IDs to prevent collisions
        ann_id_offset: Base offset to add to annotation IDs to prevent collisions
        filename_prefix: Prefix prepended to image file_name

    Returns:
        (remapped_coco_dict, remapping_stats)
    """
    cat_map = category_map or DEFAULT_ROBOFLOW_CATEGORY_MAP
    canonical_categories = list(DEFAULT_CATEGORIES)
    canonical_name_to_id = {c["name"]: c["id"] for c in canonical_categories}

    # Build mapping from source category ID to canonical category ID
    src_cats = rf_coco_data.get("categories", [])
    src_id_to_canonical_id: Dict[int, int] = {}
    dropped_classes: Dict[str, int] = {}

    for sc in src_cats:
        src_id = sc["id"]
        src_name = sc.get("name", "").strip().lower()
        if src_name in cat_map:
            target_name = cat_map[src_name]
            if target_name in canonical_name_to_id:
                src_id_to_canonical_id[src_id] = canonical_name_to_id[target_name]
        else:
            dropped_classes[src_name] = 0

    # Remap images with offset and filename prefix
    old_to_new_img_id: Dict[int, int] = {}
    new_images: List[Dict[str, Any]] = []

    for img in rf_coco_data.get("images", []):
        old_id = img["id"]
        new_id = old_id + image_id_offset
        old_to_new_img_id[old_id] = new_id

        orig_fname = img.get("file_name", "")
        new_fname = f"{filename_prefix}{orig_fname}" if not orig_fname.startswith(filename_prefix) else orig_fname

        new_img_entry = {
            "id": new_id,
            "file_name": new_fname,
            "width": img.get("width", 640),
            "height": img.get("height", 640),
            "source": "roboflow",
        }
        new_images.append(new_img_entry)

    # Remap annotations
    new_annotations: List[Dict[str, Any]] = []
    dropped_count = 0
    kept_count = 0

    for idx, ann in enumerate(rf_coco_data.get("annotations", []), 1):
        src_cat_id = ann["category_id"]
        if src_cat_id not in src_id_to_canonical_id:
            # Category was unmapped / dropped
            dropped_count += 1
            continue

        orig_img_id = ann["image_id"]
        if orig_img_id not in old_to_new_img_id:
            continue

        canonical_cat_id = src_id_to_canonical_id[src_cat_id]
        new_ann_id = ann["id"] + ann_id_offset if "id" in ann else (ann_id_offset + idx)

        bbox = [round(float(v), 2) for v in ann.get("bbox", [])]
        w = bbox[2] if len(bbox) > 2 else 0.0
        h = bbox[3] if len(bbox) > 3 else 0.0
        area = ann.get("area", round(w * h, 2))

        new_ann_entry = {
            "id": new_ann_id,
            "image_id": old_to_new_img_id[orig_img_id],
            "category_id": canonical_cat_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": int(ann.get("iscrowd", 0)),
        }
        new_annotations.append(new_ann_entry)
        kept_count += 1

    remapped_coco = {
        "info": {
            "year": 2026,
            "version": "1.0",
            "description": "Merged Roboflow Universe Cricket Dataset",
            "contributor": "StumpVision AI",
        },
        "licenses": rf_coco_data.get("licenses", [{"id": 1, "name": "Open Source", "url": ""}]),
        "images": new_images,
        "annotations": new_annotations,
        "categories": canonical_categories,
    }

    stats = {
        "source_images_count": len(new_images),
        "retained_annotations": kept_count,
        "dropped_annotations": dropped_count,
        "dropped_categories": list(dropped_classes.keys()),
    }
    return remapped_coco, stats


def merge_coco_datasets(
    primary_coco: Dict[str, Any],
    secondary_coco: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two canonical COCO datasets into one unified dataset without ID collisions.
    """
    merged_images = list(primary_coco.get("images", []))
    merged_annotations = list(primary_coco.get("annotations", []))

    prim_img_ids = {img["id"] for img in merged_images}
    prim_fnames = {img["file_name"] for img in merged_images}
    prim_ann_ids = {a["id"] for a in merged_annotations}

    for img in secondary_coco.get("images", []):
        if img["id"] in prim_img_ids or img["file_name"] in prim_fnames:
            raise ValueError(f"Image ID or filename collision detected during merge: {img}")
        merged_images.append(deepcopy(img))

    for ann in secondary_coco.get("annotations", []):
        if ann["id"] in prim_ann_ids:
            raise ValueError(f"Annotation ID collision detected during merge: {ann}")
        merged_annotations.append(deepcopy(ann))

    merged = {
        "info": deepcopy(primary_coco.get("info", {})),
        "licenses": deepcopy(primary_coco.get("licenses", [])),
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": deepcopy(primary_coco.get("categories", list(DEFAULT_CATEGORIES))),
    }
    return merged


def import_and_merge_roboflow_split(
    rf_split_dir: Path,
    target_rf_detr_split_dir: Path,
    category_map: Optional[Dict[str, str]] = None,
    image_id_offset: int = 10000,
    ann_id_offset: int = 100000,
    filename_prefix: str = "rf_",
    copy_images: bool = True,
) -> Dict[str, Any]:
    """
    Import a Roboflow split (containing _annotations.coco.json) and merge with target split.
    """
    rf_split_dir = Path(rf_split_dir)
    target_rf_detr_split_dir = Path(target_rf_detr_split_dir)
    target_rf_detr_split_dir.mkdir(parents=True, exist_ok=True)

    rf_json = rf_split_dir / "_annotations.coco.json"
    if not rf_json.exists():
        # Fallback to any .json
        jsons = list(rf_split_dir.glob("*.json"))
        if not jsons:
            raise FileNotFoundError(f"No COCO JSON annotations found in {rf_split_dir}")
        rf_json = jsons[0]

    rf_raw = load_coco_json(rf_json)
    remapped_rf, remap_stats = remap_roboflow_coco_data(
        rf_coco_data=rf_raw,
        category_map=category_map,
        image_id_offset=image_id_offset,
        ann_id_offset=ann_id_offset,
        filename_prefix=filename_prefix,
    )

    target_json = target_rf_detr_split_dir / "_annotations.coco.json"
    if target_json.exists():
        target_raw = load_coco_json(target_json)
        final_coco = merge_coco_datasets(target_raw, remapped_rf)
    else:
        final_coco = remapped_rf

    save_coco_json(final_coco, target_json)

    # Copy images with prefix
    copied_count = 0
    if copy_images:
        for orig_img in rf_raw.get("images", []):
            orig_name = orig_img["file_name"]
            src_p = rf_split_dir / orig_name
            new_name = f"{filename_prefix}{orig_name}" if not orig_name.startswith(filename_prefix) else orig_name
            dst_p = target_rf_detr_split_dir / new_name

            if src_p.exists():
                shutil.copy2(src_p, dst_p)
                copied_count += 1

    stats = {
        "split_dir": str(target_rf_detr_split_dir),
        "total_images": len(final_coco["images"]),
        "total_annotations": len(final_coco["annotations"]),
        "roboflow_images_added": len(remapped_rf["images"]),
        "roboflow_images_copied": copied_count,
        "roboflow_annotations_added": len(remapped_rf["annotations"]),
        "dropped_annotations": remap_stats["dropped_annotations"],
    }
    return stats

