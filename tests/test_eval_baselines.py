"""
test_eval_baselines.py
======================
Unit tests for the three reference baselines:
predict_nothing, every_n_seconds, and random_times (M2).
"""

import pytest
import pandas as pd
import numpy as np

from stumpvision.eval.matching import EvalDataset, match_eval_dataset
from stumpvision.eval.metrics import compute_aggregate_counts
from stumpvision.eval.baselines import (
    generate_predict_nothing,
    generate_every_n_seconds,
    generate_random_predictions_single_seed,
    evaluate_random_baseline_multi_seed,
)


@pytest.fixture
def mock_dataset():
    """Create a mock 2-clip dataset with 3 total deliveries."""
    return EvalDataset(
        split="dev",
        reviewed_clips_count=2,
        unreviewed_clips_count=0,
        clips_data={
            "clip_001": {
                "fps": 60.0,
                "duration_s": 60.0,
                "block_id": "block_1",
                "view": "bowler_end",
                "deliveries": [12.0, 35.0],
                "partials": [],
                "notes": [],
                "is_empty": False,
            },
            "clip_002": {
                "fps": 60.0,
                "duration_s": 60.0,
                "block_id": "block_1",
                "view": "bowler_end",
                "deliveries": [20.0],
                "partials": [],
                "notes": [],
                "is_empty": False,
            },
        },
    )


def test_predict_nothing(mock_dataset):
    """predict_nothing emits 0 predictions; recall=0.0."""
    df = generate_predict_nothing(mock_dataset)
    assert len(df) == 0

    results = match_eval_dataset(mock_dataset, df, tolerance=0.5)
    counts = compute_aggregate_counts(results)
    assert counts["tp"] == 0
    assert counts["fn"] == 3
    assert counts["fp"] == 0
    assert counts["recall"] == 0.0


def test_every_n_seconds(mock_dataset):
    """every_n_seconds generates predictions at fixed grid (0, 10, 20... up to duration)."""
    df = generate_every_n_seconds(mock_dataset, interval_s=10.0)
    # For a 60s clip with interval 10s: 0, 10, 20, 30, 40, 50, 60 -> 7 preds per clip = 14 total
    assert len(df) == 14

    results = match_eval_dataset(mock_dataset, df, tolerance=0.5)
    counts = compute_aggregate_counts(results)
    # clip_002 has a delivery at 20.0s, matching the grid point at 20.0s exactly
    assert counts["tp"] >= 1


def test_random_times_deterministic_and_reproducible(mock_dataset):
    """
    random_times sampling is deterministic given a seed.
    Integer count follows Poisson with rate matching ground truth delivery rate.
    """
    df_s1 = generate_random_predictions_single_seed(mock_dataset, seed=42, lam_deliveries_per_clip=1.5)
    df_s2 = generate_random_predictions_single_seed(mock_dataset, seed=42, lam_deliveries_per_clip=1.5)

    pd.testing.assert_frame_equal(df_s1, df_s2)


def test_random_times_multi_seed_evaluation(mock_dataset):
    """Multi-seed evaluation runs over seeds and computes mean and spread."""
    summary = evaluate_random_baseline_multi_seed(mock_dataset, n_seeds=20, tolerance=0.5)
    assert summary["n_seeds"] == 20
    assert 0.0 <= summary["mean_recall"] <= 1.0
    assert 0.0 <= summary["mean_precision"] <= 1.0
    assert summary["std_recall"] >= 0.0
    assert summary["representative_df"] is not None

