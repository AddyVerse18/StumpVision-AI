"""
baselines.py
============
Trivial reference baselines for StumpVision AI Delivery Finder (M2):
1. predict_nothing: emits zero candidate detections.
2. random_times: uniform random timestamps over clip duration with integer count
   sampled via Poisson distribution matching ground-truth delivery rate across 200 seeds.
3. every_n_seconds: fixed temporal grid (default 10s intervals).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from stumpvision.schemas import PREDICTIONS_COLUMNS, validate_predictions_df
from stumpvision.eval.matching import (
    EvalDataset,
    match_eval_dataset,
)
from stumpvision.eval.metrics import compute_aggregate_counts


def generate_predict_nothing(dataset: EvalDataset) -> pd.DataFrame:
    """Generate empty predictions dataframe."""
    return pd.DataFrame(columns=PREDICTIONS_COLUMNS)


def generate_every_n_seconds(
    dataset: EvalDataset,
    interval_s: float = 10.0,
) -> pd.DataFrame:
    """
    Generate predictions on a fixed temporal grid at interval_s.
    E.g., for a 60s clip with interval 10s: 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0.
    """
    rows = []
    for cid, c_data in dataset.clips_data.items():
        dur_s = float(c_data.get("duration_s", 60.0))
        t = 0.0
        while t <= dur_s + 1e-5:
            rows.append({
                "clip_id": cid,
                "time_s": f"{t:.4f}",
                "score": 1.0,
                "status": "candidate",
                "method_version": f"baseline_every_{int(interval_s)}s",
            })
            t += interval_s

    df = pd.DataFrame(rows, columns=PREDICTIONS_COLUMNS)
    validate_predictions_df(df)
    return df


def generate_random_predictions_single_seed(
    dataset: EvalDataset,
    seed: int,
    lam_deliveries_per_clip: float,
) -> pd.DataFrame:
    """
    Generate random predictions for one seed.
    Integer count per clip sampled via Poisson(lambda) to match expected ground truth rate.
    Timestamps sampled uniformly on [0, duration_s].
    """
    rng = np.random.RandomState(seed)
    rows = []

    for cid, c_data in dataset.clips_data.items():
        dur_s = float(c_data.get("duration_s", 60.0))
        if lam_deliveries_per_clip > 0:
            k = int(rng.poisson(lam_deliveries_per_clip))
        else:
            k = 0

        if k > 0:
            times = rng.uniform(0.0, dur_s, size=k)
            scores = rng.uniform(0.1, 1.0, size=k)
            for t_s, sc in zip(times, scores):
                rows.append({
                    "clip_id": cid,
                    "time_s": f"{t_s:.4f}",
                    "score": f"{sc:.4f}",
                    "status": "candidate",
                    "method_version": "baseline_random_times",
                })

    df = pd.DataFrame(rows, columns=PREDICTIONS_COLUMNS)
    validate_predictions_df(df)
    return df


def evaluate_random_baseline_multi_seed(
    dataset: EvalDataset,
    n_seeds: int = 200,
    tolerance: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate random baseline across multiple seeds (default 200).
    Computes mean and spread (standard deviation and 95% range) of recall, precision, and F1.
    """
    total_reviewed_clips = dataset.reviewed_clips_count
    total_deliveries = sum(len(c["deliveries"]) for c in dataset.clips_data.values())

    lam = (total_deliveries / total_reviewed_clips) if total_reviewed_clips > 0 else 0.0

    recalls = []
    precisions = []
    f1s = []

    representative_df = None

    for s in range(n_seeds):
        preds_df = generate_random_predictions_single_seed(
            dataset=dataset,
            seed=s,
            lam_deliveries_per_clip=lam,
        )
        if s == 0:
            representative_df = preds_df

        match_res = match_eval_dataset(dataset, preds_df, tolerance=tolerance)
        c = compute_aggregate_counts(match_res)
        recalls.append(c["recall"])
        precisions.append(c["precision"])
        f1s.append(c["f1"])

    return {
        "n_seeds": n_seeds,
        "lam_per_clip": lam,
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "mean_precision": float(np.mean(precisions)),
        "std_precision": float(np.std(precisions)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "representative_df": representative_df,
    }

