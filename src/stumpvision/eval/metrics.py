"""
metrics.py
==========
Metric calculations, breakdowns, threshold sweeps, and cluster bootstrap for StumpVision.
All rates report exact counts (e.g. 17 of 21). Recall is reported first.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from stumpvision.eval.matching import ClipMatchResult, EvalDataset, match_eval_dataset


def compute_aggregate_counts(clip_results: List[ClipMatchResult]) -> Dict[str, Any]:
    """Aggregate TP, FP, FN, and count totals."""
    total_tp = sum(len(r.tp_pairs) for r in clip_results)
    total_fn = sum(len(r.fn_truths) for r in clip_results)
    total_fp = sum(len(r.fp_preds) for r in clip_results)
    total_truths = sum(r.n_truths for r in clip_results)
    total_preds = sum(r.n_preds for r in clip_results)

    # Core rates
    # Recall = TP / (TP + FN) = TP / total_truths
    recall = (total_tp / total_truths) if total_truths > 0 else 0.0
    recall_str = f"{total_tp} of {total_truths} ({recall * 100:.1f}%)" if total_truths > 0 else "0 of 0 (N/A)"

    # Precision = TP / (TP + FP) = TP / total_preds
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    prec_denom = total_tp + total_fp
    prec_str = f"{total_tp} of {prec_denom} ({precision * 100:.1f}%)" if prec_denom > 0 else "0 of 0 (N/A)"

    # F1 score
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": total_tp,
        "fn": total_fn,
        "fp": total_fp,
        "total_truths": total_truths,
        "total_preds": total_preds,
        "recall": recall,
        "recall_str": recall_str,
        "precision": precision,
        "precision_str": prec_str,
        "f1": f1,
    }


def compute_timestamp_errors(
    clip_results: List[ClipMatchResult],
    dataset: EvalDataset,
) -> Dict[str, Any]:
    """Compute mean and max absolute timestamp errors in seconds and frames."""
    errors_s = []
    errors_frames = []

    for r in clip_results:
        fps = dataset.clips_data.get(r.clip_id, {}).get("fps", 60.0)
        for _, _, err in r.tp_pairs:
            errors_s.append(err)
            errors_frames.append(err * fps)

    if not errors_s:
        return {
            "count": 0,
            "mean_error_s": 0.0,
            "max_error_s": 0.0,
            "mean_error_frames": 0.0,
            "max_error_frames": 0.0,
        }

    return {
        "count": len(errors_s),
        "mean_error_s": float(np.mean(errors_s)),
        "max_error_s": float(np.max(errors_s)),
        "mean_error_frames": float(np.mean(errors_frames)),
        "max_error_frames": float(np.max(errors_frames)),
    }


def compute_review_burden(
    clip_results: List[ClipMatchResult],
    dataset: EvalDataset,
) -> Dict[str, Any]:
    """Compute predictions per clip and per minute of reviewed footage."""
    total_preds = sum(r.n_preds for r in clip_results)
    total_clips = len(clip_results)

    total_dur_s = sum(
        dataset.clips_data.get(r.clip_id, {}).get("duration_s", 60.0)
        for r in clip_results
    )
    total_min = total_dur_s / 60.0 if total_dur_s > 0 else 0.0

    preds_per_clip = (total_preds / total_clips) if total_clips > 0 else 0.0
    preds_per_min = (total_preds / total_min) if total_min > 0 else 0.0
    max_preds_in_single_clip = max((r.n_preds for r in clip_results), default=0)

    return {
        "total_preds": total_preds,
        "total_clips": total_clips,
        "total_duration_s": total_dur_s,
        "preds_per_clip": preds_per_clip,
        "max_preds_clip": max_preds_in_single_clip,
        "preds_per_min": preds_per_min,
    }


def compute_clip_level_metrics(clip_results: List[ClipMatchResult]) -> Dict[str, Any]:
    """
    Compute clip-level hit rates:
    - fraction of delivery-containing clips with >= 1 hit (TP)
    - fraction of empty clips with 0 predictions
    """
    delivery_clips = [r for r in clip_results if r.n_truths > 0]
    empty_clips = [r for r in clip_results if r.n_truths == 0]

    # Delivery clips with >= 1 TP
    n_deliv = len(delivery_clips)
    deliv_with_hit = sum(1 for r in delivery_clips if len(r.tp_pairs) > 0)
    deliv_hit_rate = (deliv_with_hit / n_deliv) if n_deliv > 0 else 0.0
    deliv_str = f"{deliv_with_hit} of {n_deliv} ({deliv_hit_rate * 100:.1f}%)" if n_deliv > 0 else "0 of 0 (N/A)"

    # Empty clips with 0 predictions
    n_empty = len(empty_clips)
    empty_clean = sum(1 for r in empty_clips if r.n_preds == 0)
    empty_rate = (empty_clean / n_empty) if n_empty > 0 else 0.0
    empty_str = f"{empty_clean} of {n_empty} ({empty_rate * 100:.1f}%)" if n_empty > 0 else "0 of 0 (N/A)"

    return {
        "n_delivery_clips": n_deliv,
        "delivery_clips_with_hit": deliv_with_hit,
        "delivery_hit_rate": deliv_hit_rate,
        "delivery_hit_str": deliv_str,
        "n_empty_clips": n_empty,
        "empty_clips_clean": empty_clean,
        "empty_clean_rate": empty_rate,
        "empty_clean_str": empty_str,
    }


def compute_breakdowns(
    clip_results: List[ClipMatchResult],
) -> Dict[str, Dict[str, Any]]:
    """Compute breakdowns by block, view, and distractor tag."""
    # 1. By Block
    blocks: Dict[str, List[ClipMatchResult]] = {}
    for r in clip_results:
        b = r.block_id or "unassigned"
        blocks.setdefault(b, []).append(r)

    block_stats = {}
    for b_name, b_res in sorted(blocks.items()):
        block_stats[b_name] = compute_aggregate_counts(b_res)

    # 2. By View
    views: Dict[str, List[ClipMatchResult]] = {}
    for r in clip_results:
        v = r.view or "unclear"
        views.setdefault(v, []).append(r)

    view_stats = {}
    for v_name, v_res in sorted(views.items()):
        view_stats[v_name] = compute_aggregate_counts(v_res)

    # 3. By Distractor Tag
    # Common tags: batter_walkpast, fielder, camera_moved, other
    distractor_tags = ["batter_walkpast", "fielder", "camera_moved", "other"]
    distractor_stats = {}

    for tag in distractor_tags:
        tagged_clips = [r for r in clip_results if any(tag in n for n in r.notes)]
        if tagged_clips:
            t_counts = compute_aggregate_counts(tagged_clips)
            distractor_stats[tag] = {
                "clip_count": len(tagged_clips),
                "fn_misses": t_counts["fn"],
                "fp_alarms": t_counts["fp"],
                "tp_hits": t_counts["tp"],
            }
        else:
            distractor_stats[tag] = {
                "clip_count": 0,
                "fn_misses": 0,
                "fp_alarms": 0,
                "tp_hits": 0,
            }

    return {
        "by_block": block_stats,
        "by_view": view_stats,
        "by_distractor": distractor_stats,
    }


def compute_partials_summary(clip_results: List[ClipMatchResult]) -> Dict[str, Any]:
    """Report partial deliveries found vs missed separately from main score."""
    total_partial_truths = sum(len(r.partial_tp_pairs) + len(r.partial_fn_truths) for r in clip_results)
    total_partial_found = sum(len(r.partial_tp_pairs) for r in clip_results)
    total_partial_missed = sum(len(r.partial_fn_truths) for r in clip_results)

    found_rate = (total_partial_found / total_partial_truths) if total_partial_truths > 0 else 0.0
    summary_str = (
        f"{total_partial_found} of {total_partial_truths} found ({found_rate * 100:.1f}%), "
        f"{total_partial_missed} missed"
        if total_partial_truths > 0
        else "0 partials present in reviewed clips"
    )

    return {
        "total_partials": total_partial_truths,
        "found": total_partial_found,
        "missed": total_partial_missed,
        "found_rate": found_rate,
        "summary_str": summary_str,
    }


def compute_threshold_sweep(
    dataset: EvalDataset,
    preds_df: pd.DataFrame,
    tolerance: float = 0.5,
    thresholds: Optional[List[float]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Sweep confidence thresholds on predictions.
    NOTE: Operating point selection is STRICTLY permitted on DEV split only.
    If called on test split, returns empty operating point recommendation.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    sweep_rows = []
    best_f1 = -1.0
    best_op = None

    for th in thresholds:
        res = match_eval_dataset(dataset, preds_df, tolerance=tolerance, min_score=th)
        counts = compute_aggregate_counts(res)
        row = {
            "threshold": th,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "recall": counts["recall"],
            "recall_str": counts["recall_str"],
            "precision": counts["precision"],
            "precision_str": counts["precision_str"],
            "f1": counts["f1"],
        }
        sweep_rows.append(row)

        if dataset.split == "dev" and counts["f1"] > best_f1 and (counts["tp"] + counts["fp"]) > 0:
            best_f1 = counts["f1"]
            best_op = row

    # Safeguard: if test split, do NOT provide an operating point selection
    if dataset.split == "test":
        best_op = None

    return sweep_rows, best_op


def compute_bootstrap_ci(
    clip_results: List[ClipMatchResult],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Cluster bootstrap resampling of CLIPS (not individual deliveries).
    Produces 95% confidence intervals on recall and precision.
    Includes warning when reviewed clips < 20 or deliveries < 30.
    """
    rng = np.random.RandomState(seed)
    n_clips = len(clip_results)
    total_deliveries = sum(r.n_truths for r in clip_results)

    warning = ""
    if n_clips < 20 or total_deliveries < 30:
        warning = (
            f"WARNING: Dataset contains only {n_clips} reviewed clips and {total_deliveries} "
            "deliveries (<20 clips or <30 deliveries threshold). Bootstrap confidence intervals "
            "are expected to be wide and results must be treated as PRELIMINARY ONLY."
        )

    if n_clips == 0:
        return {
            "warning": warning,
            "recall_ci": (0.0, 0.0),
            "precision_ci": (0.0, 0.0),
            "f1_ci": (0.0, 0.0),
        }

    boot_recalls = []
    boot_precisions = []
    boot_f1s = []

    for _ in range(n_boot):
        sample_indices = rng.choice(n_clips, size=n_clips, replace=True)
        sample_results = [clip_results[idx] for idx in sample_indices]
        c = compute_aggregate_counts(sample_results)
        boot_recalls.append(c["recall"])
        boot_precisions.append(c["precision"])
        boot_f1s.append(c["f1"])

    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100

    rec_ci = (float(np.percentile(boot_recalls, lower_p)), float(np.percentile(boot_recalls, upper_p)))
    prec_ci = (float(np.percentile(boot_precisions, lower_p)), float(np.percentile(boot_precisions, upper_p)))
    f1_ci = (float(np.percentile(boot_f1s, lower_p)), float(np.percentile(boot_f1s, upper_p)))

    return {
        "warning": warning,
        "n_clips": n_clips,
        "total_deliveries": total_deliveries,
        "n_boot": n_boot,
        "seed": seed,
        "recall_ci": rec_ci,
        "precision_ci": prec_ci,
        "f1_ci": f1_ci,
    }
