"""
schemas.py
==========
Data contracts and schema definitions for StumpVision AI Delivery Finder.
Every output artifact enforces schema_version = "1.0".
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path


SCHEMA_VERSION = "1.0"

# Ordered columns for labels/manifest.csv
MANIFEST_MACHINE_COLUMNS = [
    "clip_id",
    "relative_path",
    "camera",
    "sequence_no",
    "file_size_bytes",
    "fps",
    "width",
    "height",
    "n_frames_reported",
    "n_frames_decoded",
    "duration_s",
    "cfr_status",
    "schema_version",
]

MANIFEST_MANUAL_COLUMNS = [
    "block_id",
    "view",
    "split",
]

MANIFEST_ALL_COLUMNS = MANIFEST_MACHINE_COLUMNS + MANIFEST_MANUAL_COLUMNS


# Ordered columns for labels/labels.csv
LABELS_COLUMNS = [
    "clip_id",
    "delivery_time_s",
    "frame_index",
    "kind",
    "notes",
    "schema_version",
]

# Ordered columns for labels/clip_status.csv
CLIP_STATUS_COLUMNS = [
    "clip_id",
    "reviewed",
    "reviewed_at",
]

VALID_KINDS = ["delivery", "partial_start", "partial_end", "none"]
VALID_VIEWS = ["bowler_end", "batter_end", "mixed", "unclear"]
VALID_SPLITS = ["dev", "buffer", "test"]


def validate_manifest_df(df: pd.DataFrame) -> bool:
    """Validate that dataframe has all required columns and valid types."""
    missing = [c for c in MANIFEST_ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return True


def validate_labels_df(df: pd.DataFrame) -> bool:
    """Validate that labels dataframe matches schema."""
    missing = [c for c in LABELS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Labels CSV missing required columns: {missing}")
    return True


def validate_clip_status_df(df: pd.DataFrame) -> bool:
    """Validate that clip status dataframe matches schema."""
    missing = [c for c in CLIP_STATUS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Clip status CSV missing required columns: {missing}")
    return True


PREDICTIONS_COLUMNS = [
    "clip_id",
    "time_s",
    "score",
    "status",
    "method_version",
]


def validate_predictions_df(df: pd.DataFrame) -> bool:
    """Validate that predictions dataframe matches schema."""
    missing = [c for c in PREDICTIONS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions CSV missing required columns: {missing}")
    return True


def merge_and_save_manifest(
    new_rows: List[Dict[str, Any]],
    manifest_path: Path,
) -> pd.DataFrame:
    """
    Merge newly audited machine rows with existing manifest, preserving manual columns
    (block_id, view, split) for existing clip_ids.
    """
    new_df = pd.DataFrame(new_rows)
    for col in MANIFEST_MANUAL_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = ""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        existing_df = pd.read_csv(manifest_path, dtype=str)
        # Preserve manual columns if present
        manual_records = {}
        for _, row in existing_df.iterrows():
            cid = str(row.get("clip_id", "")).strip()
            if cid:
                manual_records[cid] = {
                    m_col: row.get(m_col, "") for m_col in MANIFEST_MANUAL_COLUMNS
                }

        # Apply existing manual values to matching new rows
        for idx, row in new_df.iterrows():
            cid = str(row["clip_id"]).strip()
            if cid in manual_records:
                for m_col, val in manual_records[cid].items():
                    if pd.notna(val) and val != "":
                        new_df.at[idx, m_col] = val

    # Enforce column order and validation
    validate_manifest_df(new_df)
    new_df = new_df[MANIFEST_ALL_COLUMNS]
    new_df = new_df.sort_values(by=["camera", "sequence_no"])
    new_df.to_csv(manifest_path, index=False)
    return new_df

