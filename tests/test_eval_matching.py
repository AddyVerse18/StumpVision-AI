"""
test_eval_matching.py
=====================
Unit tests for one-to-one bipartite matching logic in StumpVision (M2).
Uses hand-made synthetic timestamps with zero video decoding.
"""

import pytest
from pathlib import Path
import pandas as pd

from stumpvision.eval.matching import (
    match_single_clip,
    match_eval_dataset,
    load_ground_truth,
    load_predictions,
    EvalDataset,
)
from stumpvision.eval.metrics import compute_aggregate_counts, compute_partials_summary


def test_perfect_predictions():
    """All predictions match ground truth within tolerance."""
    truths = [10.0, 25.0, 42.0]
    preds = [(10.1, 0.95), (24.95, 0.90), (42.05, 0.99)]
    tp, fn, fp = match_single_clip(truths, preds, tolerance=0.5)

    assert len(tp) == 3
    assert len(fn) == 0
    assert len(fp) == 0


def test_one_miss():
    """One truth is missed (no prediction near 25.0s)."""
    truths = [10.0, 25.0, 42.0]
    preds = [(10.0, 0.9), (42.0, 0.9)]
    tp, fn, fp = match_single_clip(truths, preds, tolerance=0.5)

    assert len(tp) == 2
    assert len(fn) == 1
    assert fn[0] == 25.0
    assert len(fp) == 0


def test_one_false_positive():
    """One spurious prediction at 15.0s."""
    truths = [10.0, 40.0]
    preds = [(10.0, 0.9), (15.0, 0.85), (40.0, 0.95)]
    tp, fn, fp = match_single_clip(truths, preds, tolerance=0.5)

    assert len(tp) == 2
    assert len(fn) == 0
    assert len(fp) == 1
    assert fp[0][0] == 15.0


def test_duplicate_predictions_near_one_truth():
    """
    Two predictions near one ground truth.
    One-to-one optimal matching must assign one TP and one FP.
    """
    truths = [10.0]
    preds = [(10.1, 0.9), (10.3, 0.8)]
    tp, fn, fp = match_single_clip(truths, preds, tolerance=0.5)

    assert len(tp) == 1
    assert tp[0][0] == 10.1  # Closer prediction chosen
    assert len(fn) == 0
    assert len(fp) == 1
    assert fp[0][0] == 10.3


def test_two_truths_within_tolerance_of_one_prediction():
    """
    Two ground truths near one prediction.
    One-to-one matching assigns one TP and leaves the other as a miss (FN).
    """
    truths = [10.0, 10.4]
    preds = [(10.2, 0.9)]  # Equidistant from both (diff=0.2s <= 0.5s)
    tp, fn, fp = match_single_clip(truths, preds, tolerance=0.5)

    assert len(tp) == 1
    assert len(fn) == 1
    assert len(fp) == 0


def test_tolerance_boundary():
    """
    Boundary condition:
    |pred - truth| == 0.5s is a match (hit).
    |pred - truth| == 0.51s is NOT a match (miss + FP).
    """
    # Exact boundary (diff = 0.5000)
    truths = [10.0]
    preds_hit = [(10.5, 0.9)]
    tp, fn, fp = match_single_clip(truths, preds_hit, tolerance=0.5)
    assert len(tp) == 1
    assert len(fn) == 0
    assert len(fp) == 0

    # Just outside boundary (diff = 0.5100)
    preds_miss = [(10.51, 0.9)]
    tp, fn, fp = match_single_clip(truths, preds_miss, tolerance=0.5)
    assert len(tp) == 0
    assert len(fn) == 1
    assert len(fp) == 1


def test_empty_clip_with_predictions():
    """
    Clip with kind=none (no deliveries) where detector fired 2 candidate predictions.
    Both must be counted as false positives (FPs).
    """
    truths = []
    preds = [(5.0, 0.8), (35.0, 0.75)]
    tp, fn, fp = match_single_clip(truths, preds, tolerance=0.5)

    assert len(tp) == 0
    assert len(fn) == 0
    assert len(fp) == 2


def test_empty_predictions_file():
    """Empty predictions on clips with truths results in 0 TP, all truths FN, recall 0."""
    dataset = EvalDataset(
        split="dev",
        reviewed_clips_count=1,
        unreviewed_clips_count=0,
        clips_data={
            "clip_001": {
                "fps": 60.0,
                "duration_s": 60.0,
                "block_id": "block_1",
                "view": "bowler_end",
                "deliveries": [10.0, 20.0],
                "partials": [],
                "notes": [],
                "is_empty": False,
            }
        },
    )
    empty_df = pd.DataFrame(columns=["clip_id", "time_s", "score", "status", "method_version"])
    results = match_eval_dataset(dataset, empty_df, tolerance=0.5)
    counts = compute_aggregate_counts(results)

    assert counts["tp"] == 0
    assert counts["fn"] == 2
    assert counts["fp"] == 0
    assert counts["recall"] == 0.0


def test_partials_excluded_from_main_score():
    """
    Partial deliveries (partial_start/partial_end) are excluded from primary score
    and tracked in separate partial fields.
    """
    dataset = EvalDataset(
        split="dev",
        reviewed_clips_count=1,
        unreviewed_clips_count=0,
        clips_data={
            "clip_001": {
                "fps": 60.0,
                "duration_s": 60.0,
                "block_id": "block_1",
                "view": "bowler_end",
                "deliveries": [15.0],
                "partials": [(2.0, "partial_start")],
                "notes": [],
                "is_empty": False,
            }
        },
    )
    # Prediction matches the partial (at 2.0s) and full delivery (at 15.0s)
    preds_df = pd.DataFrame([
        {"clip_id": "clip_001", "time_s": "2.0", "score": "0.9", "status": "candidate", "method_version": "test"},
        {"clip_id": "clip_001", "time_s": "15.0", "score": "0.95", "status": "candidate", "method_version": "test"},
    ])
    results = match_eval_dataset(dataset, preds_df, tolerance=0.5)
    counts = compute_aggregate_counts(results)
    partials_summary = compute_partials_summary(results)

    # Main score only counts the 1 real delivery
    assert counts["total_truths"] == 1
    assert counts["tp"] == 1
    assert counts["recall"] == 1.0

    # Partials summary tracks the partial
    assert partials_summary["total_partials"] == 1
    assert partials_summary["found"] == 1
    assert partials_summary["missed"] == 0

