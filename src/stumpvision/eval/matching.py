"""
matching.py
===========
Optimal one-to-one bipartite matching and ground truth loading for StumpVision.
Uses scipy.optimize.linear_sum_assignment with tolerance gating.
Boundary condition: |pred - truth| <= tolerance counts as a hit.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from stumpvision.schemas import (
    validate_manifest_df,
    validate_labels_df,
    validate_clip_status_df,
    validate_predictions_df,
)
from stumpvision.labeling.splits import record_test_access


@dataclass
class ClipMatchResult:
    clip_id: str
    block_id: str
    view: str
    split: str
    notes: List[str]
    n_truths: int
    n_preds: int
    tp_pairs: List[Tuple[float, float, float]]  # (pred_time, truth_time, error_s)
    fn_truths: List[float]                      # missed ground truth times
    fp_preds: List[Tuple[float, float]]         # (pred_time, score) false positives
    partial_tp_pairs: List[Tuple[float, float, float]] = field(default_factory=list)
    partial_fn_truths: List[float] = field(default_factory=list)
    partial_fp_preds: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class EvalDataset:
    split: str
    reviewed_clips_count: int
    unreviewed_clips_count: int
    clips_data: Dict[str, Dict[str, Any]]  # clip_id -> metadata and truth times


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file if it exists."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_ground_truth(
    manifest_path: Path,
    labels_path: Path,
    status_path: Path,
    split: str = "dev",
    allow_test: bool = False,
    command_str: str = "stumpvision evaluate",
    pred_path: Optional[Path] = None,
) -> EvalDataset:
    """
    Load ground truth for reviewed clips in the requested split.
    Fails loudly if zero clips are reviewed.
    Enforces test split locking via record_test_access.
    """
    manifest_path = Path(manifest_path)
    labels_path = Path(labels_path)
    status_path = Path(status_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")
    if not status_path.exists():
        raise FileNotFoundError(f"Clip status CSV not found: {status_path}")

    manifest_df = pd.read_csv(manifest_path, dtype=str)
    validate_manifest_df(manifest_df)

    labels_df = pd.read_csv(labels_path, dtype=str)
    validate_labels_df(labels_df)

    status_df = pd.read_csv(status_path, dtype=str)
    validate_clip_status_df(status_df)

    # Check test split safeguard
    if split == "test":
        if not allow_test:
            raise PermissionError(
                "Evaluation on the locked test split is restricted. "
                "You must pass --allow-test to evaluate the locked test set."
            )
        pred_hash = compute_file_sha256(pred_path) if pred_path else "no_pred_file"
        record_test_access(command_name=f"{command_str} pred_hash={pred_hash}")

    # Filter to requested split
    split_manifest = manifest_df[manifest_df["split"] == split].copy()
    if split_manifest.empty:
        raise ValueError(f"No clips found in manifest for split='{split}'")

    # Map review status
    reviewed_map = {}
    for _, r in status_df.iterrows():
        cid = str(r["clip_id"]).strip()
        is_rev = str(r.get("reviewed", "")).strip().lower() == "yes"
        reviewed_map[cid] = is_rev

    # Split into reviewed vs unreviewed
    reviewed_clips = []
    unreviewed_clips = []
    for _, r in split_manifest.iterrows():
        cid = str(r["clip_id"]).strip()
        if reviewed_map.get(cid, False):
            reviewed_clips.append(r)
        else:
            unreviewed_clips.append(r)

    unreviewed_count = len(unreviewed_clips)
    reviewed_count = len(reviewed_clips)

    if reviewed_count == 0:
        raise ValueError(
            f"Zero clips are marked as reviewed in split='{split}'. "
            f"Excluded {unreviewed_count} unreviewed clips. "
            "Please review and mark clips using 'stumpvision label' before evaluating."
        )

    # Build dataset for reviewed clips
    clips_data = {}
    for r in reviewed_clips:
        cid = str(r["clip_id"]).strip()
        fps = float(r["fps"]) if pd.notna(r.get("fps")) and r["fps"] != "" else 60.0
        dur_s = float(r["duration_s"]) if pd.notna(r.get("duration_s")) and r["duration_s"] != "" else 60.0
        block_id = str(r.get("block_id", "")).strip()
        view = str(r.get("view", "")).strip() or "unclear"

        # Find labels for this clip
        c_labels = labels_df[labels_df["clip_id"] == cid]
        deliveries = []
        partials = []
        notes = []
        is_empty = False

        for _, l_row in c_labels.iterrows():
            kind = str(l_row.get("kind", "")).strip()
            note = str(l_row.get("notes", "")).strip()
            if note:
                notes.append(note)

            if kind == "none":
                is_empty = True
                continue

            # Ground truth time = frame_index / fps from manifest
            f_idx_str = str(l_row.get("frame_index", "")).strip()
            deliv_time_str = str(l_row.get("delivery_time_s", "")).strip()

            if f_idx_str and f_idx_str.lower() != "nan":
                t_s = float(f_idx_str) / fps
            elif deliv_time_str and deliv_time_str.lower() != "nan":
                t_s = float(deliv_time_str)
            else:
                continue

            if kind == "delivery":
                deliveries.append(t_s)
            elif kind in ("partial_start", "partial_end"):
                partials.append((t_s, kind))

        if not deliveries and not partials and is_empty:
            # Explicit empty clip
            pass
        elif not deliveries and not partials:
            # No deliveries marked
            is_empty = True

        clips_data[cid] = {
            "fps": fps,
            "duration_s": dur_s,
            "block_id": block_id,
            "view": view,
            "deliveries": sorted(deliveries),
            "partials": sorted(partials, key=lambda x: x[0]),
            "notes": notes,
            "is_empty": is_empty,
        }

    return EvalDataset(
        split=split,
        reviewed_clips_count=reviewed_count,
        unreviewed_clips_count=unreviewed_count,
        clips_data=clips_data,
    )


def load_predictions(pred_path: Path) -> pd.DataFrame:
    """Load and validate predictions CSV."""
    pred_path = Path(pred_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    df = pd.read_csv(pred_path)
    validate_predictions_df(df)

    # Standardize types
    df["clip_id"] = df["clip_id"].astype(str).str.strip()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(1.0)
    df["status"] = df["status"].astype(str).str.strip()
    df["method_version"] = df["method_version"].astype(str).str.strip()

    # Drop invalid rows
    df = df.dropna(subset=["clip_id", "time_s"])
    return df


def match_single_clip(
    truths: List[float],
    preds: List[Tuple[float, float]],  # (time_s, score)
    tolerance: float = 0.5,
) -> Tuple[List[Tuple[float, float, float]], List[float], List[Tuple[float, float]]]:
    """
    Optimal 1-to-1 matching for a single clip within tolerance.
    Boundary rule: |pred - truth| <= tolerance is a match (hit).

    Returns:
        (tp_pairs, fn_truths, fp_preds)
        where tp_pairs is list of (pred_time, truth_time, error_s)
    """
    n_preds = len(preds)
    n_truths = len(truths)

    if n_truths == 0 and n_preds == 0:
        return [], [], []

    if n_truths == 0:
        return [], [], list(preds)

    if n_preds == 0:
        return [], list(truths), []

    # Build cost matrix
    # Cost is absolute time difference. Pairs > tolerance get large penalty.
    cost_matrix = np.zeros((n_preds, n_truths), dtype=np.float64)
    PENALTY = 1e9

    for i, (p_t, _) in enumerate(preds):
        for j, t_t in enumerate(truths):
            diff = abs(p_t - t_t)
            if diff <= tolerance:
                cost_matrix[i, j] = diff
            else:
                cost_matrix[i, j] = PENALTY + diff

    pred_ind, truth_ind = linear_sum_assignment(cost_matrix)

    tp_pairs = []
    matched_preds = set()
    matched_truths = set()

    for p_idx, t_idx in zip(pred_ind, truth_ind):
        p_t, p_score = preds[p_idx]
        t_t = truths[t_idx]
        diff = abs(p_t - t_t)
        # Boundary condition: diff <= tolerance counts as a hit
        if diff <= tolerance:
            tp_pairs.append((p_t, t_t, diff))
            matched_preds.add(p_idx)
            matched_truths.add(t_idx)

    fn_truths = [t_t for j, t_t in enumerate(truths) if j not in matched_truths]
    fp_preds = [preds[i] for i in range(n_preds) if i not in matched_preds]

    return tp_pairs, fn_truths, fp_preds


def match_eval_dataset(
    dataset: EvalDataset,
    preds_df: pd.DataFrame,
    tolerance: float = 0.5,
    min_score: float = -np.inf,
) -> List[ClipMatchResult]:
    """
    Run matching across all reviewed clips in dataset at given tolerance and score threshold.
    """
    results: List[ClipMatchResult] = []

    # Group predictions by clip_id with score threshold applied
    preds_by_clip: Dict[str, List[Tuple[float, float]]] = {}
    for cid in dataset.clips_data:
        preds_by_clip[cid] = []

    if not preds_df.empty and "score" in preds_df.columns:
        numeric_scores = pd.to_numeric(preds_df["score"], errors="coerce").fillna(1.0)
        filtered_preds = preds_df[numeric_scores >= min_score] if min_score > -np.inf else preds_df
    else:
        filtered_preds = preds_df

    for _, row in filtered_preds.iterrows():
        cid = str(row["clip_id"]).strip()
        if cid in preds_by_clip:
            t_s = float(row["time_s"])
            score = float(row["score"])
            preds_by_clip[cid].append((t_s, score))

    # Match each clip
    for cid, c_info in dataset.clips_data.items():
        c_preds = sorted(preds_by_clip.get(cid, []), key=lambda x: x[0])
        c_truths = c_info["deliveries"]

        tp_pairs, fn_truths, fp_preds = match_single_clip(
            truths=c_truths,
            preds=c_preds,
            tolerance=tolerance,
        )

        # Partials matching: separate from main score
        c_partials = [p[0] for p in c_info["partials"]]
        part_tp, part_fn, part_fp = match_single_clip(
            truths=c_partials,
            preds=c_preds,
            tolerance=tolerance,
        )

        res = ClipMatchResult(
            clip_id=cid,
            block_id=c_info["block_id"],
            view=c_info["view"],
            split=dataset.split,
            notes=c_info["notes"],
            n_truths=len(c_truths),
            n_preds=len(c_preds),
            tp_pairs=tp_pairs,
            fn_truths=fn_truths,
            fp_preds=fp_preds,
            partial_tp_pairs=part_tp,
            partial_fn_truths=part_fn,
            partial_fp_preds=part_fp,
        )
        results.append(res)

    return results
