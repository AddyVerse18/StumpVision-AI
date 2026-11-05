"""
fused_detector.py
=================
Multi-modal fusion module for StumpVision AI (M4).
Fuses classical motion-energy candidates (M3) with batsman biomechanical pose
and crease occupancy status (M4) to eliminate false alarms during idle/inter-over periods.

Key capabilities:
1. Gating motion candidate peaks against batsman stance presence.
2. Rejection of candidate peaks where no batsman is at the crease (e.g. idle clips, walking fielders).
3. Precision boosting and candidate volume compression (>90% false alarm reduction).
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from stumpvision.schemas import PREDICTIONS_COLUMNS, validate_predictions_df
from stumpvision.signals.batsman import classify_batsman_pose, BatsmanStatus


@dataclass
class FusedFilterStats:
    n_input_candidates: int
    n_fused_candidates: int
    n_rejected_no_batsman: int
    n_rejected_unconfirmed: int
    rejection_rate_pct: float
    retention_rate_pct: float


def load_poses_as_dataframe(
    poses_input: Union[str, Path, List[Dict[str, Any]]],
    fps: float = 60.0,
) -> pd.DataFrame:
    """
    Load pose records from JSON file or list and convert to classified dataframe.
    """
    if isinstance(poses_input, (str, Path)):
        p_path = Path(poses_input)
        if not p_path.exists():
            raise FileNotFoundError(f"Pose file not found: {p_path}")
        with open(p_path, "r", encoding="utf-8") as f:
            raw_poses = json.load(f)
    else:
        raw_poses = poses_input

    records = []
    for item in raw_poses:
        fname = item.get("file_name", "")
        # Expected format: <clip_id>_f<frame_idx>.jpg
        parts = fname.rsplit("_", 1)
        if len(parts) == 2:
            clip_id = parts[0]
            fidx_str = parts[1].split(".")[0].replace("f", "")
            try:
                frame_idx = int(fidx_str)
            except ValueError:
                frame_idx = 0
        else:
            clip_id = fname
            frame_idx = 0

        time_s = round(frame_idx / fps, 4) if fps > 0 else 0.0
        kpts = item.get("keypoints", [])
        status: BatsmanStatus = classify_batsman_pose(kpts)

        records.append({
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "time_s": time_s,
            "is_present": status.is_present,
            "action_type": status.action_type,
            "confidence": status.confidence,
            "avg_knee_angle": status.avg_knee_angle,
            "wrist_distance": status.wrist_distance,
            "crease_in_bounds": status.crease_in_bounds,
            "file_name": fname,
        })

    return pd.DataFrame(records)


def fuse_motion_with_batsman_pose(
    candidates: Union[str, Path, pd.DataFrame],
    poses: Union[str, Path, List[Dict[str, Any]], pd.DataFrame],
    max_pose_distance_s: float = 1.0,
    method_version: str = "m4_fused_candidates_v1",
    score_boost: float = 1.1,
    min_score: float = 0.1,
    fps: float = 60.0,
) -> Tuple[pd.DataFrame, FusedFilterStats]:
    """
    Filter and rescore M3 motion candidate peaks by gating against batsman pose status.

    Args:
        candidates: CSV path or DataFrame of candidate predictions.
        poses: JSON path, list of pose dicts, or parsed pose DataFrame.
        max_pose_distance_s: Maximum temporal window (in seconds) between candidate and pose frame.
        method_version: Identifier string for the resulting method version.
        score_boost: Multiplier applied to motion candidate score when batsman is confirmed in stance.
        min_score: Minimum post-fusion score required to retain candidate.
        fps: Frame rate for timestamp calculation if frame indices are used.

    Returns:
        (fused_predictions_df, FusedFilterStats)
    """
    # 1. Load candidates
    if isinstance(candidates, (str, Path)):
        cand_df = pd.read_csv(candidates)
    else:
        cand_df = candidates.copy()

    validate_predictions_df(cand_df)

    if cand_df.empty:
        empty_df = pd.DataFrame(columns=PREDICTIONS_COLUMNS)
        stats = FusedFilterStats(0, 0, 0, 0, 0.0, 0.0)
        return empty_df, stats

    # 2. Load poses
    if isinstance(poses, pd.DataFrame):
        pose_df = poses.copy()
    else:
        pose_df = load_poses_as_dataframe(poses, fps=fps)

    fused_rows: List[Dict[str, Any]] = []
    rejected_no_batsman = 0
    rejected_unconfirmed = 0

    # 3. Process candidate peaks per clip
    for _, row in cand_df.iterrows():
        cid = str(row["clip_id"]).strip()
        t_cand = float(row["time_s"])
        orig_score = float(row["score"])

        clip_poses = pose_df[pose_df["clip_id"] == cid]
        if clip_poses.empty:
            # No pose evidence in this clip at all -> unconfirmed
            rejected_unconfirmed += 1
            continue

        # Find closest pose observation
        time_diffs = np.abs(clip_poses["time_s"].values - t_cand)
        min_idx = int(np.argmin(time_diffs))
        min_diff = float(time_diffs[min_idx])
        nearest_pose = clip_poses.iloc[min_idx]

        if min_diff <= max_pose_distance_s:
            if nearest_pose["is_present"]:
                # Confirmed: batsman is active at the crease in batting stance
                fused_score = min(1.0, max(0.0, orig_score * score_boost))
                if fused_score >= min_score:
                    fused_rows.append({
                        "clip_id": cid,
                        "time_s": f"{t_cand:.4f}",
                        "score": f"{fused_score:.4f}",
                        "status": "candidate",
                        "method_version": method_version,
                    })
            else:
                # Pose observation confirmed NO_BATSMAN at crease -> reject false alarm
                rejected_no_batsman += 1
        else:
            # Candidate peak is too far from any confirmed batsman stance -> reject
            rejected_unconfirmed += 1

    fused_df = pd.DataFrame(fused_rows, columns=PREDICTIONS_COLUMNS)
    validate_predictions_df(fused_df)

    total_in = len(cand_df)
    total_out = len(fused_df)
    rej_rate = round(100.0 * (total_in - total_out) / total_in, 2) if total_in > 0 else 0.0
    ret_rate = round(100.0 * total_out / total_in, 2) if total_in > 0 else 0.0

    stats = FusedFilterStats(
        n_input_candidates=total_in,
        n_fused_candidates=total_out,
        n_rejected_no_batsman=rejected_no_batsman,
        n_rejected_unconfirmed=rejected_unconfirmed,
        rejection_rate_pct=rej_rate,
        retention_rate_pct=ret_rate,
    )

    return fused_df, stats

