"""
coco_schema.py
==============
COCO Detection JSON format builder and validator for StumpVision M4 (RF-DETR).
Ensures standard compatibility with RF-DETR / RT-DETR datasets:
- info
- licenses
- images: [ {id, file_name, width, height, ...} ]
- annotations: [ {id, image_id, category_id, bbox: [x, y, w, h], area, iscrowd, ...} ]
- categories: [ {id, name, supercategory} ]
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


DEFAULT_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "cricket"},
    {"id": 2, "name": "stumps", "supercategory": "cricket"},
    {"id": 3, "name": "pitch", "supercategory": "cricket"},
    {"id": 4, "name": "crease", "supercategory": "cricket"},
    {"id": 5, "name": "bat", "supercategory": "cricket"},
    {"id": 6, "name": "ball", "supercategory": "cricket"},
]


def create_empty_coco_dataset(
    categories: Optional[List[Dict[str, Any]]] = None,
    description: str = "StumpVision M4 Cricket Dataset",
) -> Dict[str, Any]:
    """Create a minimal valid COCO detection dataset structure."""
    return {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": description,
            "contributor": "StumpVision AI",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Proprietary / Confidential",
                "url": "",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories if categories is not None else list(DEFAULT_CATEGORIES),
    }


def add_image_to_coco(
    dataset: Dict[str, Any],
    file_name: str,
    width: int,
    height: int,
    image_id: Optional[int] = None,
    clip_id: Optional[str] = None,
    frame_index: Optional[int] = None,
    time_s: Optional[float] = None,
    sample_type: Optional[str] = None,
) -> int:
    """Add an image entry to the COCO dataset dict, returning its assigned integer ID."""
    if image_id is None:
        image_id = len(dataset["images"]) + 1

    entry: Dict[str, Any] = {
        "id": int(image_id),
        "file_name": str(file_name),
        "width": int(width),
        "height": int(height),
    }
    if clip_id is not None:
        entry["clip_id"] = str(clip_id)
    if frame_index is not None:
        entry["frame_index"] = int(frame_index)
    if time_s is not None:
        entry["time_s"] = float(np_round_time(time_s))
    if sample_type is not None:
        entry["sample_type"] = str(sample_type)

    dataset["images"].append(entry)
    return image_id


def add_annotation_to_coco(
    dataset: Dict[str, Any],
    image_id: int,
    category_id: int,
    bbox_xywh: Tuple[float, float, float, float],
    annotation_id: Optional[int] = None,
    score: Optional[float] = None,
    iscrowd: int = 0,
) -> int:
    """Add a bounding box annotation [x, y, width, height] to the COCO dataset."""
    if annotation_id is None:
        annotation_id = len(dataset["annotations"]) + 1

    x, y, w, h = bbox_xywh
    area = float(w * h)

    valid_cat_ids = {c["id"] for c in dataset.get("categories", [])}
    if category_id not in valid_cat_ids:
        raise ValueError(f"Category ID {category_id} not present in dataset categories {valid_cat_ids}")

    entry: Dict[str, Any] = {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": [round(float(x), 2), round(float(y), 2), round(float(w), 2), round(float(h), 2)],
        "area": round(area, 2),
        "iscrowd": int(iscrowd),
    }
    if score is not None:
        entry["score"] = round(float(score), 4)

    dataset["annotations"].append(entry)
    return annotation_id


def validate_coco_dataset(dataset: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a dictionary conforms to COCO Detection specification.
    Returns (is_valid, error_list).
    """
    errors: List[str] = []
    required_keys = ["info", "licenses", "images", "annotations", "categories"]
    for k in required_keys:
        if k not in dataset:
            errors.append(f"Missing required top-level key: '{k}'")

    if errors:
        return False, errors

    # Check categories
    cat_ids = set()
    for cat in dataset["categories"]:
        if "id" not in cat or "name" not in cat:
            errors.append(f"Malformed category entry: {cat}")
        elif cat["id"] in cat_ids:
            errors.append(f"Duplicate category id: {cat['id']}")
        else:
            cat_ids.add(cat["id"])

    # Check images
    image_ids = set()
    for img in dataset["images"]:
        if "id" not in img or "file_name" not in img or "width" not in img or "height" not in img:
            errors.append(f"Malformed image entry: {img}")
        elif img["id"] in image_ids:
            errors.append(f"Duplicate image id: {img['id']}")
        else:
            image_ids.add(img["id"])

    # Check annotations
    ann_ids = set()
    for ann in dataset["annotations"]:
        if (
            "id" not in ann
            or "image_id" not in ann
            or "category_id" not in ann
            or "bbox" not in ann
        ):
            errors.append(f"Malformed annotation entry: {ann}")
            continue

        if ann["id"] in ann_ids:
            errors.append(f"Duplicate annotation id: {ann['id']}")
        else:
            ann_ids.add(ann["id"])

        if ann["image_id"] not in image_ids:
            errors.append(f"Annotation {ann['id']} references nonexistent image_id: {ann['image_id']}")

        if ann["category_id"] not in cat_ids:
            errors.append(f"Annotation {ann['id']} references nonexistent category_id: {ann['category_id']}")

        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or any(v < 0 for v in bbox[2:]):
            errors.append(f"Invalid bbox dimensions in annotation {ann['id']}: {bbox}")

    return len(errors) == 0, errors


def save_coco_json(dataset: Dict[str, Any], output_path: Path) -> Path:
    """Validate and write COCO JSON to disk."""
    is_valid, errors = validate_coco_dataset(dataset)
    if not is_valid:
        raise ValueError(f"Cannot save invalid COCO dataset: {errors}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    temp_path.replace(output_path)
    return output_path


def save_coco_with_backup(
    dataset: Dict[str, Any],
    output_path: Path,
    backup_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Validate and save COCO JSON atomically, creating a timestamped backup if target exists.
    Returns (saved_path, backup_path_or_none).
    """
    output_path = Path(output_path)
    backup_path = None

    if output_path.exists():
        b_dir = Path(backup_dir) if backup_dir else output_path.parent / "backups"
        b_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = b_dir / f"{output_path.stem}_backup_{ts}{output_path.suffix}"
        import shutil
        shutil.copy2(output_path, backup_path)

    saved_path = save_coco_json(dataset, output_path)
    return saved_path, backup_path


def validate_coco_integrity(
    coco_path: Path,
    frames_dir: Optional[Path] = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Deep integrity validation of a COCO dataset and associated image files on disk.

    Returns:
        (is_valid, error_list, stats_summary)
    """
    coco_path = Path(coco_path)
    if not coco_path.exists():
        return False, [f"COCO file does not exist: {coco_path}"], {}

    try:
        with open(coco_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Failed to parse JSON in '{coco_path}': {e}"], {}

    is_valid, errors = validate_coco_dataset(data)

    img_map = {img["id"]: img for img in data.get("images", [])}
    cat_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    # Frame file checks
    missing_images = []
    if frames_dir is not None:
        frames_dir = Path(frames_dir)
        for img in data.get("images", []):
            fname = img.get("file_name", "")
            fpath = frames_dir / fname
            if not fpath.exists():
                missing_images.append(fname)

    if missing_images:
        errors.append(f"{len(missing_images)} image files not found on disk in '{frames_dir}' (e.g. {missing_images[:3]})")

    # BBox coordinate bounds checks
    seen_ann_signatures = set()
    category_counts: Dict[str, int] = {name: 0 for name in cat_map.values()}
    annotated_image_ids = set()

    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        img_info = img_map.get(img_id, {})
        w_img = img_info.get("width", 1920)
        h_img = img_info.get("height", 1080)

        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                errors.append(f"Annotation {ann.get('id')} has non-positive width/height: {bbox}")
            if x < 0 or y < 0:
                errors.append(f"Annotation {ann.get('id')} has negative coordinates: {bbox}")
            if x + w > w_img + 1.0 or y + h > h_img + 1.0:
                errors.append(f"Annotation {ann.get('id')} exceeds image boundaries ({w_img}x{h_img}): {bbox}")

        # Check duplicate
        sig = (img_id, ann.get("category_id"), tuple(round(v, 1) for v in bbox))
        if sig in seen_ann_signatures:
            errors.append(f"Duplicate annotation found for image {img_id}: {ann}")
        seen_ann_signatures.add(sig)

        cat_id = ann.get("category_id")
        if cat_id in cat_map:
            category_counts[cat_map[cat_id]] += 1
            annotated_image_ids.add(img_id)

    stats = {
        "total_images": len(data.get("images", [])),
        "annotated_images": len(annotated_image_ids),
        "unannotated_images": len(data.get("images", [])) - len(annotated_image_ids),
        "total_annotations": len(data.get("annotations", [])),
        "category_counts": category_counts,
        "categories": list(cat_map.values()),
    }

    return len(errors) == 0, errors, stats


def load_coco_json(json_path: Path) -> Dict[str, Any]:
    """Read and validate COCO JSON from disk."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    is_valid, errors = validate_coco_dataset(data)
    if not is_valid:
        raise ValueError(f"Invalid COCO JSON file '{json_path}': {errors}")
    return data


def np_round_time(val: float) -> float:
    return round(float(val), 4)

