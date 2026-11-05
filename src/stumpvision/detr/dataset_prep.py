"""
dataset_prep.py
==============
Pre-training dataset validation, category normalization, provenance tracking,
and dry-run verification for M4 RF-DETR cricket object detection.

Requirements:
- Strict validation of bounding boxes, dimensions, duplicates, and image files.
- Explicit category normalization report (zero silent guesswork).
- Detection and rejection of unannotated local frames.
- Dataset provenance and license tracking.
- Pre-training dry-run verification before model initialization.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from stumpvision.detr.coco_schema import DEFAULT_CATEGORIES

CANONICAL_CLASSES = {c["id"]: c["name"] for c in DEFAULT_CATEGORIES}
CANONICAL_NAME_TO_ID = {c["name"]: c["id"] for c in DEFAULT_CATEGORIES}

# Documented mappings for cricket entities
EXPLICIT_CATEGORY_MAPPINGS: Dict[str, str] = {
    # Person
    "batsman": "person",
    "batter": "person",
    "bowler": "person",
    "wicket-keeper": "person",
    "wicketkeeper": "person",
    "wicket keeper": "person",
    "keeper": "person",
    "umpire": "person",
    "leg umpire": "person",
    "main umpire": "person",
    "fielder": "person",
    "fielders": "person",
    "non-striker": "person",
    "player": "person",
    "person": "person",
    # Stumps
    "stump": "stumps",
    "stumps": "stumps",
    "wicket": "stumps",
    "wickets": "stumps",
    "bails": "stumps",
    "bail": "stumps",
    # Pitch & Crease
    "pitch": "pitch",
    "pitch-crease": "pitch",
    "crease": "crease",
    # Equipment
    "bat": "bat",
    "cricket-bat": "bat",
    "ball": "ball",
    "cricket-ball": "ball",
}


def normalize_dataset_categories(
    coco_data: Dict[str, Any],
    custom_mapping: Optional[Dict[str, str]] = None,
    allowed_unmapped_to_drop: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Normalize category names to canonical 6 classes with an explicit mapping report.
    Raises ValueError if unmapped categories are found that are not in allowed_unmapped_to_drop.

    Returns:
        (normalized_coco_dict, mapping_report)
    """
    mapping = {**EXPLICIT_CATEGORY_MAPPINGS, **(custom_mapping or {})}
    allowed_drop = set(allowed_unmapped_to_drop or set())

    src_categories = coco_data.get("categories", [])
    src_annotations = coco_data.get("annotations", [])

    src_id_to_cat = {c["id"]: c.get("name", "").strip().lower() for c in src_categories}

    src_id_to_canonical_id: Dict[int, int] = {}
    unresolved_categories: Set[str] = set()
    mapped_summary: Dict[str, Dict[str, Any]] = {}
    dropped_summary: Dict[str, int] = {}

    for src_id, src_name in src_id_to_cat.items():
        if src_name in mapping:
            target_name = mapping[src_name]
            if target_name in CANONICAL_NAME_TO_ID:
                target_id = CANONICAL_NAME_TO_ID[target_name]
                src_id_to_canonical_id[src_id] = target_id
                mapped_summary[src_name] = {
                    "source_id": src_id,
                    "target_name": target_name,
                    "target_id": target_id,
                    "count": 0,
                }
            else:
                unresolved_categories.add(src_name)
        elif src_name in allowed_drop:
            dropped_summary[src_name] = 0
        else:
            unresolved_categories.add(src_name)

    if unresolved_categories:
        raise ValueError(
            f"Unresolved / ambiguous categories found in dataset: {sorted(unresolved_categories)}. "
            f"Define an explicit mapping or add to allowed_unmapped_to_drop. "
            f"Known canonical classes are: {list(CANONICAL_NAME_TO_ID.keys())}."
        )

    # Remap annotations
    new_annotations: List[Dict[str, Any]] = []
    for ann in src_annotations:
        sid = ann.get("category_id")
        src_name = src_id_to_cat.get(sid, "unknown")
        if sid in src_id_to_canonical_id:
            new_ann = deepcopy(ann)
            new_ann["category_id"] = src_id_to_canonical_id[sid]
            new_annotations.append(new_ann)
            if src_name in mapped_summary:
                mapped_summary[src_name]["count"] += 1
        elif src_name in dropped_summary:
            dropped_summary[src_name] += 1

    normalized_data = deepcopy(coco_data)
    normalized_data["categories"] = deepcopy(list(DEFAULT_CATEGORIES))
    normalized_data["annotations"] = new_annotations

    report = {
        "mapped_categories": mapped_summary,
        "dropped_categories": dropped_summary,
        "retained_annotations_count": len(new_annotations),
        "dropped_annotations_count": sum(dropped_summary.values()),
    }
    return normalized_data, report


def validate_pretrain_coco(
    coco_data: Dict[str, Any],
    images_dir: Optional[Path] = None,
    check_zero_annotations: bool = True,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Comprehensive pre-training COCO validation:
    - Verifies top-level structure.
    - Rejects empty datasets or zero-annotation sets when check_zero_annotations is True.
    - Verifies image dimensions and existence on disk.
    - Validates bounding boxes: non-negative, non-zero area, and inside bounds.
    - Validates category IDs and duplicate IDs.

    Returns:
        (is_valid, error_list, validation_summary)
    """
    errors: List[str] = []

    # 1. Top-level structure
    for key in ["info", "licenses", "images", "annotations", "categories"]:
        if key not in coco_data:
            errors.append(f"Missing required COCO top-level key: '{key}'")

    if errors:
        return False, errors, {}

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    # 2. Check images presence
    if not images:
        errors.append("Dataset has zero registered images.")

    # 3. Check zero annotations condition
    if check_zero_annotations and len(annotations) == 0:
        errors.append(
            f"Dataset has {len(images)} images but 0 bounding-box annotations. "
            "Cannot use unannotated frames for object detection training."
        )

    # 4. Categories validation
    cat_ids = set()
    for cat in categories:
        cid = cat.get("id")
        cname = cat.get("name")
        if cid is None or not cname:
            errors.append(f"Malformed category entry: {cat}")
        elif cid in cat_ids:
            errors.append(f"Duplicate category ID: {cid}")
        else:
            cat_ids.add(cid)

    # 5. Image entries and disk existence
    img_map: Dict[int, Dict[str, Any]] = {}
    img_ids: Set[int] = set()
    img_filenames: Set[str] = set()

    for img in images:
        iid = img.get("id")
        fname = img.get("file_name")
        w = img.get("width")
        h = img.get("height")

        if iid is None or not fname or w is None or h is None:
            errors.append(f"Malformed image entry: {img}")
            continue

        if iid in img_ids:
            errors.append(f"Duplicate image ID: {iid}")
        img_ids.add(iid)

        if fname in img_filenames:
            errors.append(f"Duplicate image filename in dataset: '{fname}'")
        img_filenames.add(fname)

        if w <= 0 or h <= 0:
            errors.append(f"Invalid non-positive dimensions ({w}x{h}) for image {iid} ('{fname}')")

        img_map[iid] = img

        if images_dir is not None:
            img_path = Path(images_dir) / fname
            if not img_path.exists():
                errors.append(f"Image file not found on disk: '{img_path}'")

    # 6. Annotations validation
    ann_ids: Set[int] = set()
    per_class_counts: Dict[int, int] = {cid: 0 for cid in cat_ids}

    for ann in annotations:
        aid = ann.get("id")
        iid = ann.get("image_id")
        cid = ann.get("category_id")
        bbox = ann.get("bbox")

        if aid is None or iid is None or cid is None or bbox is None:
            errors.append(f"Malformed annotation entry: {ann}")
            continue

        if aid in ann_ids:
            errors.append(f"Duplicate annotation ID: {aid}")
        ann_ids.add(aid)

        if iid not in img_ids:
            errors.append(f"Annotation {aid} references nonexistent image ID: {iid}")
            continue

        if cid not in cat_ids:
            errors.append(f"Annotation {aid} references unknown category ID: {cid}")
        else:
            per_class_counts[cid] += 1

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            errors.append(f"Annotation {aid} has invalid bbox format (expected [x, y, w, h]): {bbox}")
            continue

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            errors.append(f"Annotation {aid} has non-positive width/height: {bbox}")
        if x < 0 or y < 0:
            errors.append(f"Annotation {aid} has negative coordinates: {bbox}")

        img_info = img_map[iid]
        img_w = img_info.get("width", 1920)
        img_h = img_info.get("height", 1080)

        # Allow 1.0px tolerance for float rounding
        if (x + w) > (img_w + 1.0) or (y + h) > (img_h + 1.0):
            errors.append(
                f"Annotation {aid} exceeds image bounds ({img_w}x{img_h}): bbox={bbox}, right={x+w}, bottom={y+h}"
            )

    cat_name_counts = {
        CANONICAL_CLASSES.get(cid, str(cid)): count
        for cid, count in per_class_counts.items()
    }

    summary = {
        "images_count": len(images),
        "annotations_count": len(annotations),
        "categories_count": len(categories),
        "per_category_counts": cat_name_counts,
        "is_valid": len(errors) == 0,
    }

    return len(errors) == 0, errors, summary


def update_dataset_provenance(
    coco_data: Dict[str, Any],
    source_name: str,
    source_version: str = "1.0",
    source_license: str = "Proprietary",
    source_split: str = "train",
) -> Dict[str, Any]:
    """Attach dataset provenance, version, and license metadata to COCO info block."""
    updated = deepcopy(coco_data)
    info = updated.get("info", {})
    info["source_name"] = str(source_name)
    info["source_version"] = str(source_version)
    info["source_split"] = str(source_split)
    info["provenance"] = f"Imported via StumpVision M4 from {source_name} (v{source_version})"
    updated["info"] = info

    licenses = updated.get("licenses", [])
    if not any(l.get("name") == source_license for l in licenses):
        licenses.append({
            "id": len(licenses) + 1,
            "name": str(source_license),
            "url": "",
        })
    updated["licenses"] = licenses
    return updated


def execute_pretraining_dry_run(
    dataset_dir: Path,
    check_images_exist: bool = True,
    resume_checkpoint: Optional[Path] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Perform a complete pre-training dry run of an RF-DETR dataset directory:
    - Inspects train/ and valid/ subfolders.
    - Validates both _annotations.coco.json files.
    - Verifies image files exist.
    - Checks resume checkpoint if specified.

    Returns:
        (is_ready, dry_run_report)
    """
    dataset_dir = Path(dataset_dir)
    report: Dict[str, Any] = {"dataset_dir": str(dataset_dir), "splits": {}, "errors": []}

    if not dataset_dir.exists():
        report["errors"].append(f"Dataset directory does not exist: {dataset_dir}")
        return False, report

    all_valid = True

    for split in ["train", "valid"]:
        s_dir = dataset_dir / split
        ann_file = s_dir / "_annotations.coco.json"

        if not s_dir.exists():
            report["errors"].append(f"Missing required split directory: '{s_dir}'")
            all_valid = False
            continue

        if not ann_file.exists():
            report["errors"].append(f"Missing annotations file: '{ann_file}'")
            all_valid = False
            continue

        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            report["errors"].append(f"Failed to parse JSON in '{ann_file}': {e}")
            all_valid = False
            continue

        img_check_dir = s_dir if check_images_exist else None
        is_val, errs, summary = validate_pretrain_coco(
            coco_data=data,
            images_dir=img_check_dir,
            check_zero_annotations=True,
        )

        if not is_val:
            all_valid = False
            report["errors"].extend([f"[{split.upper()}] {e}" for e in errs])

        report["splits"][split] = summary

    # Check resume checkpoint if provided
    if resume_checkpoint:
        resume_p = Path(resume_checkpoint)
        if not resume_p.exists():
            report["errors"].append(f"Resume checkpoint file not found: '{resume_p}'")
            all_valid = False
        else:
            report["resume_checkpoint"] = str(resume_p)
            report["resume_size_bytes"] = resume_p.stat().st_size

    report["is_ready"] = all_valid
    return all_valid, report

