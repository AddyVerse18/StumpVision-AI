"""
extract_frames.py
=================
Keyframe extraction from reviewed clips for M4 RF-DETR object detection.
Deterministically samples:
1. Delivery stride frames (peak approach and temporal offsets) for clips with deliveries.
2. Idle/distractor frames for clips with no deliveries.

Security & Integrity:
- By default extracts strictly from the DEV split.
- Test split is blocked unless allow_test=True is explicitly passed.
- Output frames are written to an external data directory (default: STUMPVISION_DATA_DIR/m4_frames)
  to ensure zero video/image media enters the project workspace.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import pandas as pd

from stumpvision.schemas import (
    validate_manifest_df,
    validate_labels_df,
    validate_clip_status_df,
)
from stumpvision.detr.coco_schema import (
    create_empty_coco_dataset,
    add_image_to_coco,
    save_coco_json,
)


def compute_sample_frame_indices(
    duration_s: float,
    fps: float,
    delivery_times_s: List[float],
    delivery_offsets_s: List[float] = [-0.25, 0.0, 0.25],
    idle_interval_s: float = 20.0,
) -> List[Tuple[int, float, str]]:
    """
    Compute frame indices and metadata to sample for a single clip.

    Returns:
        List of (frame_index, timestamp_s, sample_type) tuples, sorted by frame_index.
    """
    total_frames = max(1, int(round(duration_s * fps)))
    samples_dict: Dict[int, Tuple[float, str]] = {}

    if delivery_times_s:
        # Sample delivery strides
        for t_deliv in delivery_times_s:
            for offset in delivery_offsets_s:
                t_target = max(0.0, min(duration_s, t_deliv + offset))
                f_idx = min(total_frames - 1, max(0, int(round(t_target * fps))))
                stype = "delivery_peak" if abs(offset) < 1e-4 else "delivery_stride"
                # Keep most descriptive tag if duplicate
                if f_idx not in samples_dict or stype == "delivery_peak":
                    samples_dict[f_idx] = (round(f_idx / fps, 4), stype)
    else:
        # Sample idle interval points
        t_curr = idle_interval_s / 2.0
        while t_curr < duration_s:
            f_idx = min(total_frames - 1, max(0, int(round(t_curr * fps))))
            samples_dict[f_idx] = (round(f_idx / fps, 4), "idle_scene")
            t_curr += idle_interval_s

    sorted_samples = sorted(
        [(f_idx, t_s, stype) for f_idx, (t_s, stype) in samples_dict.items()],
        key=lambda x: x[0],
    )
    return sorted_samples


def extract_keyframes_for_split(
    manifest_path: Path,
    labels_path: Path,
    status_path: Path,
    data_dir: Path,
    output_frames_dir: Path,
    split: str = "dev",
    allow_test: bool = False,
    delivery_offsets_s: List[float] = [-0.25, 0.0, 0.25],
    idle_interval_s: float = 20.0,
    jpeg_quality: int = 95,
    coco_output_path: Optional[Path] = None,
    index_csv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract keyframes for all reviewed clips in the requested split.

    Args:
        manifest_path: Path to labels/manifest.csv
        labels_path: Path to labels/labels.csv
        status_path: Path to labels/clip_status.csv
        data_dir: Path to raw footage root directory
        output_frames_dir: Directory where extracted JPEG frames will be stored
        split: Target split ("dev", "test", etc.)
        allow_test: Explicit safety gate to allow extraction from test clips
        delivery_offsets_s: Offsets in seconds around each delivery point
        idle_interval_s: Interval in seconds for sampling clips without deliveries
        jpeg_quality: JPEG compression quality (1-100)
        coco_output_path: Optional destination to write initialized COCO JSON
        index_csv_path: Optional destination to write frame index CSV

    Returns:
        (index_df, summary_stats)
    """
    if split == "test" and not allow_test:
        raise PermissionError(
            "Access to the locked test split requires allow_test=True. "
            "Model development and frame extraction must be performed on the dev split."
        )

    manifest_df = pd.read_csv(manifest_path, dtype=str)
    validate_manifest_df(manifest_df)

    status_df = pd.read_csv(status_path, dtype=str)
    validate_clip_status_df(status_df)

    labels_df = pd.read_csv(labels_path, dtype=str)
    validate_labels_df(labels_df)

    # Filter reviewed clips for split
    reviewed_ids = set(
        status_df[status_df["reviewed"].str.lower() == "yes"]["clip_id"].astype(str).str.strip()
    )

    split_manifest = manifest_df[
        (manifest_df["split"] == split) & (manifest_df["clip_id"].isin(reviewed_ids))
    ].copy()

    if split_manifest.empty:
        raise ValueError(f"No reviewed clips found for split='{split}'")

    split_manifest = split_manifest.sort_values(by=["camera", "sequence_no"])

    output_frames_dir = Path(output_frames_dir)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    coco_dataset = create_empty_coco_dataset(
        description=f"StumpVision M4 Cricket Dataset [{split.upper()} split]"
    )

    records: List[Dict[str, Any]] = []
    image_counter = 1

    for idx, (_, row) in enumerate(split_manifest.iterrows(), 1):
        cid = str(row["clip_id"]).strip()
        rel_p = str(row["relative_path"]).strip()
        video_p = data_dir / rel_p

        fps = float(row.get("fps", 60.0))
        dur_s = float(row.get("duration_s", 60.0))

        # Get delivery times for this clip
        clip_labels = labels_df[
            (labels_df["clip_id"] == cid) & (labels_df["kind"] == "delivery")
        ]
        deliv_times = []
        for _, lrow in clip_labels.iterrows():
            t_val = lrow.get("delivery_time_s")
            if pd.notna(t_val) and str(t_val).strip():
                deliv_times.append(float(t_val))

        # Determine target frames
        targets = compute_sample_frame_indices(
            duration_s=dur_s,
            fps=fps,
            delivery_times_s=deliv_times,
            delivery_offsets_s=delivery_offsets_s,
            idle_interval_s=idle_interval_s,
        )

        if not video_p.exists():
            print(f"[WARNING] Video file missing: {video_p}. Skipping extraction for {cid}.")
            continue

        cap = cv2.VideoCapture(str(video_p))
        if not cap.isOpened():
            print(f"[WARNING] Could not open video: {video_p}. Skipping.")
            continue

        for f_idx, t_s, stype in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            frame_filename = f"{cid}_f{f_idx:06d}.jpg"
            dest_path = output_frames_dir / frame_filename

            cv2.imwrite(
                str(dest_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )

            records.append({
                "image_id": image_counter,
                "clip_id": cid,
                "frame_index": f_idx,
                "time_s": t_s,
                "sample_type": stype,
                "file_name": frame_filename,
                "file_path": str(dest_path),
                "width": w,
                "height": h,
                "split": split,
            })

            add_image_to_coco(
                dataset=coco_dataset,
                file_name=frame_filename,
                width=w,
                height=h,
                image_id=image_counter,
                clip_id=cid,
                frame_index=f_idx,
                time_s=t_s,
                sample_type=stype,
            )
            image_counter += 1

        cap.release()
        print(f"[{idx:02d}/{len(split_manifest):02d}] {cid}: extracted {len(targets)} keyframes.")

    index_df = pd.DataFrame(records)

    if index_csv_path:
        index_csv_path = Path(index_csv_path)
        index_csv_path.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(index_csv_path, index=False)

    if coco_output_path:
        save_coco_json(coco_dataset, coco_output_path)

    stats = {
        "split": split,
        "n_clips_processed": len(split_manifest),
        "n_frames_extracted": len(index_df),
        "output_dir": str(output_frames_dir),
        "delivery_frames_count": int((index_df["sample_type"] == "delivery_peak").sum()) if not index_df.empty else 0,
        "stride_frames_count": int((index_df["sample_type"] == "delivery_stride").sum()) if not index_df.empty else 0,
        "idle_frames_count": int((index_df["sample_type"] == "idle_scene").sum()) if not index_df.empty else 0,
    }
    return index_df, stats

