"""
splits.py
=========
Dataset splitting and leakage prevention rules for StumpVision AI.
Reads configs/splits.yaml and partitions clips strictly by sequence number.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import pandas as pd
import yaml

from stumpvision.schemas import VALID_SPLITS

LOCKED_TEST_FINGERPRINT = "5f083e0a59514ee69c31e4d550bf8a881912fe480f7c9c08e14acd12e3ecde24"


def assign_splits(manifest_path: Path, config_path: Path) -> pd.DataFrame:
    """
    Read splits configuration and assign block_id and split to manifest.csv.
    """
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Splits config not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    splits_conf = config.get("splits", {})
    manifest_df = pd.read_csv(manifest_path, dtype=str)

    # Initialize columns if missing
    for col in ["block_id", "split"]:
        if col not in manifest_df.columns:
            manifest_df[col] = ""

    # Map sequence_no to block_id and split
    for split_name, split_info in splits_conf.items():
        if split_name not in VALID_SPLITS:
            raise ValueError(f"Unknown split name '{split_name}' in {config_path}")

        blocks = split_info.get("blocks", [])
        for block in blocks:
            b_id = str(block.get("block_id", "")).strip()
            seq_range = block.get("sequence_range", [])
            if len(seq_range) != 2:
                raise ValueError(f"Invalid sequence_range {seq_range} in block {b_id}")
            start_seq, end_seq = int(seq_range[0]), int(seq_range[1])

            for idx, row in manifest_df.iterrows():
                try:
                    s_no = int(row["sequence_no"])
                except (ValueError, TypeError):
                    continue
                if start_seq <= s_no <= end_seq:
                    manifest_df.at[idx, "block_id"] = b_id
                    manifest_df.at[idx, "split"] = split_name

    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df


def compute_clip_set_fingerprint(clip_ids: List[str]) -> str:
    """Compute deterministic SHA-256 hash of sorted clip IDs."""
    sorted_ids = sorted([str(cid).strip() for cid in clip_ids if str(cid).strip()])
    content = "\n".join(sorted_ids).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def check_leakage(
    manifest_path: Path,
    expected_fingerprint: str = LOCKED_TEST_FINGERPRINT,
) -> Dict[str, Any]:
    """
    Enforce zero leakage between splits, buffer isolation, and locked test fingerprint.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    df = pd.read_csv(manifest_path, dtype=str)

    # 1. Check all clips have split and block_id
    missing_split = df[df["split"].isna() | (df["split"] == "")]["clip_id"].tolist()
    missing_block = df[df["block_id"].isna() | (df["block_id"] == "")]["clip_id"].tolist()
    if missing_split:
        raise ValueError(f"Clips missing split assignment: {missing_split}")
    if missing_block:
        raise ValueError(f"Clips missing block_id assignment: {missing_block}")

    # 2. Check disjoint clip IDs across splits
    dev_clips = set(df[df["split"] == "dev"]["clip_id"])
    buffer_clips = set(df[df["split"] == "buffer"]["clip_id"])
    test_clips = set(df[df["split"] == "test"]["clip_id"])

    dev_buffer_overlap = dev_clips.intersection(buffer_clips)
    dev_test_overlap = dev_clips.intersection(test_clips)
    buffer_test_overlap = buffer_clips.intersection(test_clips)

    if dev_buffer_overlap:
        raise ValueError(f"Leakage detected between dev and buffer: {dev_buffer_overlap}")
    if dev_test_overlap:
        raise ValueError(f"Leakage detected between dev and test: {dev_test_overlap}")
    if buffer_test_overlap:
        raise ValueError(f"Leakage detected between buffer and test: {buffer_test_overlap}")

    # 3. Buffer isolation check
    if not buffer_clips:
        raise ValueError("Buffer split is empty! Buffer isolation is required.")

    # 4. Test split fingerprint verification
    test_ids_list = list(test_clips)
    actual_fingerprint = compute_clip_set_fingerprint(test_ids_list)
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Test set fingerprint mismatch! Expected: {expected_fingerprint}, "
            f"Found: {actual_fingerprint}. Test set clip list has changed."
        )

    return {
        "status": "PASS",
        "dev_clips_count": len(dev_clips),
        "buffer_clips_count": len(buffer_clips),
        "test_clips_count": len(test_clips),
        "total_clips": len(df),
        "test_fingerprint": actual_fingerprint,
    }


def record_test_access(
    command_name: str,
    log_path: Path = Path("outputs/test_access.log"),
) -> None:
    """Log an access event to test-split labels."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{ts}, command={command_name}\n")
