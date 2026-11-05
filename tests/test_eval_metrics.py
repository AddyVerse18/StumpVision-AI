"""
test_eval_metrics.py
====================
Unit tests for metric calculations, bootstrap confidence intervals, and threshold sweep (M2).
"""

import pytest
import numpy as np
import pandas as pd

from stumpvision.eval.matching import ClipMatchResult, EvalDataset
from stumpvision.eval.metrics import (
    compute_aggregate_counts,
    compute_bootstrap_ci,
    compute_threshold_sweep,
    compute_timestamp_errors,
    compute_review_burden,
    compute_clip_level_metrics,
    compute_breakdowns,
)


def make_mock_results(n_clips=5):
    """Helper to generate mock ClipMatchResults."""
    results = []
    for i in range(n_clips):
        cid = f"clip_{i:03d}"
        res = ClipMatchResult(
            clip_id=cid,
            block_id=f"block_{(i % 2) + 1}",
            view="bowler_end" if i % 2 == 0 else "batter_end",
            split="dev",
            notes=["batter_walkpast"] if i == 0 else [],
            n_truths=2,
            n_preds=2,
            tp_pairs=[(10.1, 10.0, 0.1), (25.05, 25.0, 0.05)],
            fn_truths=[],
            fp_preds=[],
        )
        results.append(res)
    return results


def test_deterministic_bootstrap():
    """Bootstrap CI is perfectly reproducible with fixed seed."""
    results = make_mock_results(n_clips=10)
    ci_1 = compute_bootstrap_ci(results, n_boot=200, seed=12345)
    ci_2 = compute_bootstrap_ci(results, n_boot=200, seed=12345)

    assert ci_1["recall_ci"] == ci_2["recall_ci"]
    assert ci_1["precision_ci"] == ci_2["precision_ci"]
    assert ci_1["f1_ci"] == ci_2["f1_ci"]


def test_bootstrap_warning_on_small_dataset():
    """A clear warning is emitted when dataset has <20 clips or <30 deliveries."""
    results = make_mock_results(n_clips=5)  # 5 clips, 10 deliveries (<20 clips, <30 deliveries)
    ci = compute_bootstrap_ci(results, n_boot=100, seed=42)

    assert "WARNING" in ci["warning"]
    assert "<20 clips or <30 deliveries" in ci["warning"]


def test_threshold_sweep_dev_vs_test():
    """
    Threshold sweep evaluates multiple confidence cutoffs.
    Best operating point is selected on DEV, but blocked on TEST split.
    """
    dataset_dev = EvalDataset(
        split="dev",
        reviewed_clips_count=1,
        unreviewed_clips_count=0,
        clips_data={
            "clip_001": {
                "fps": 60.0,
                "duration_s": 60.0,
                "block_id": "block_1",
                "view": "bowler_end",
                "deliveries": [10.0, 30.0],
                "partials": [],
                "notes": [],
                "is_empty": False,
            }
        },
    )
    # 2 predictions: one high confidence (0.95), one lower confidence (0.45)
    preds_df = pd.DataFrame([
        {"clip_id": "clip_001", "time_s": "10.0", "score": "0.95", "status": "candidate", "method_version": "v1"},
        {"clip_id": "clip_001", "time_s": "30.0", "score": "0.45", "status": "candidate", "method_version": "v1"},
    ])

    sweep_dev, best_op_dev = compute_threshold_sweep(
        dataset_dev, preds_df, tolerance=0.5, thresholds=[0.3, 0.5, 0.9]
    )
    assert len(sweep_dev) == 3
    # At threshold 0.3: both preds survive -> recall = 2/2 = 1.0
    assert sweep_dev[0]["tp"] == 2
    # At threshold 0.9: only high-confidence survives -> recall = 1/2 = 0.5
    assert sweep_dev[2]["tp"] == 1
    # On DEV, best operating point is returned
    assert best_op_dev is not None

    # On TEST split, best operating point MUST be None (tuning strictly forbidden)
    dataset_test = EvalDataset(
        split="test",
        reviewed_clips_count=1,
        unreviewed_clips_count=0,
        clips_data=dataset_dev.clips_data,
    )
    sweep_test, best_op_test = compute_threshold_sweep(
        dataset_test, preds_df, tolerance=0.5, thresholds=[0.3, 0.5, 0.9]
    )
    assert best_op_test is None


def test_review_burden_and_clip_level():
    """Review burden and clip-level metrics are computed accurately."""
    dataset = EvalDataset(
        split="dev",
        reviewed_clips_count=2,
        unreviewed_clips_count=0,
        clips_data={
            "c1": {"fps": 60.0, "duration_s": 60.0, "deliveries": [10.0]},
            "c2": {"fps": 60.0, "duration_s": 60.0, "deliveries": []},  # empty clip
        },
    )
    r1 = ClipMatchResult(
        clip_id="c1", block_id="b1", view="v1", split="dev", notes=[],
        n_truths=1, n_preds=2, tp_pairs=[(10.0, 10.0, 0.0)], fn_truths=[], fp_preds=[(20.0, 0.8)]
    )
    r2 = ClipMatchResult(
        clip_id="c2", block_id="b1", view="v1", split="dev", notes=[],
        n_truths=0, n_preds=0, tp_pairs=[], fn_truths=[], fp_preds=[]
    )
    results = [r1, r2]

    burden = compute_review_burden(results, dataset)
    assert burden["total_preds"] == 2
    assert burden["total_clips"] == 2
    assert burden["preds_per_clip"] == 1.0
    assert burden["preds_per_min"] == 1.0  # 2 preds / 2 minutes

    clip_stats = compute_clip_level_metrics(results)
    # c1 had a hit
    assert clip_stats["delivery_clips_with_hit"] == 1
    # c2 had 0 predictions (clean empty)
    assert clip_stats["empty_clips_clean"] == 1

