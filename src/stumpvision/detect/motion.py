"""
motion.py
=========
Classical motion-signal peak detection and candidate generation for StumpVision (M3).
Uses streamed frame differences, rolling median noise estimation, and SciPy peak finding.
Deterministic and configurable.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from stumpvision.schemas import (
    PREDICTIONS_COLUMNS,
    validate_predictions_df,
    validate_manifest_df,
    validate_clip_status_df,
)
from stumpvision.signals.motion import extract_motion_signal


DEFAULT_CONFIG: Dict[str, Any] = {
    "method_version": "m3_motion_baseline_v1",
    "downscale_width": 160,
    "downscale_height": 90,
    "noise_window_s": 5.0,
    "mad_multiplier": 3.5,
    "min_prominence": 1.5,
    "min_separation_s": 3.5,
    "min_score": 0.1,
    "spatial_filtering": False,
    "foreground_ratio_thresh": 1.0,
    "spatial_penalty_alpha": 1.0,
}


def estimate_adaptive_threshold(
    motion: np.ndarray,
    fps: float,
    noise_window_s: float = 5.0,
    mad_multiplier: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rolling median, rolling MAD, and dynamic detection threshold.

    Args:
        motion: 1D array of motion values.
        fps: Video frame rate.
        noise_window_s: Rolling window length in seconds.
        mad_multiplier: Scaling factor for MAD above median noise floor.

    Returns:
        (rolling_median, rolling_mad, dynamic_threshold)
    """
    n = len(motion)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    window_size = max(5, int(round(noise_window_s * fps)))
    if window_size % 2 == 0:
        window_size += 1

    # Cap window_size to length of motion if array is smaller than window
    if window_size > n:
        window_size = n if n % 2 != 0 else max(1, n - 1)

    rolling_med = median_filter(motion, size=window_size, mode="reflect")
    residual = np.abs(motion - rolling_med)
    rolling_mad = median_filter(residual, size=window_size, mode="reflect")

    # Guard against zero MAD in static scenes
    rolling_mad = np.maximum(rolling_mad, 0.05)

    threshold = rolling_med + mad_multiplier * rolling_mad
    return rolling_med, rolling_mad, threshold


def find_candidate_peaks_from_signal(
    motion: np.ndarray,
    time_s: np.ndarray,
    fps: float,
    config: Optional[Dict[str, Any]] = None,
    motion_top: Optional[np.ndarray] = None,
    motion_mid: Optional[np.ndarray] = None,
    motion_bot: Optional[np.ndarray] = None,
) -> List[Tuple[float, float]]:
    """
    Detect candidate delivery peaks from motion signal.

    Args:
        motion: 1D array of frame differences.
        time_s: 1D array of corresponding timestamps in seconds.
        fps: Frame rate.
        config: Dictionary of detector hyperparameters.
        motion_top: Optional 1D array of top-band motion values.
        motion_mid: Optional 1D array of middle-band motion values.
        motion_bot: Optional 1D array of bottom-band motion values.

    Returns:
        List of (candidate_timestamp_s, confidence_score) tuples, sorted by timestamp.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    n = len(motion)
    if n < 5:
        return []

    noise_window_s = float(cfg.get("noise_window_s", 5.0))
    mad_mult = float(cfg.get("mad_multiplier", 3.5))
    min_prom = float(cfg.get("min_prominence", 1.5))
    min_sep_s = float(cfg.get("min_separation_s", 3.5))
    min_score = float(cfg.get("min_score", 0.1))

    rolling_med, rolling_mad, threshold = estimate_adaptive_threshold(
        motion=motion,
        fps=fps,
        noise_window_s=noise_window_s,
        mad_multiplier=mad_mult,
    )

    min_dist_frames = max(1, int(round(min_sep_s * fps)))

    peaks, properties = find_peaks(
        motion,
        height=threshold,
        distance=min_dist_frames,
        prominence=min_prom,
    )

    candidates = []
    for p_idx in peaks:
        t_cand = float(time_s[p_idx])
        # Compute monotonic SNR-based confidence score
        peak_val = float(motion[p_idx])
        med_val = float(rolling_med[p_idx])
        mad_val = float(rolling_mad[p_idx])
        snr = max(0.0, (peak_val - med_val) / mad_val)

        # Map SNR to [0.1, 1.0]
        score = float(np.clip(snr / (snr + 4.0), 0.1, 1.0))

        # Soft spatial penalty for foreground-dominant motion
        if (
            cfg.get("spatial_filtering", False)
            and motion_bot is not None
            and motion_top is not None
            and motion_mid is not None
            and p_idx < len(motion_bot)
            and p_idx < len(motion_top)
            and p_idx < len(motion_mid)
        ):
            corridor = float(motion_top[p_idx] + motion_mid[p_idx])
            foreground = float(motion_bot[p_idx])
            ratio = foreground / (corridor + 1e-4)
            fg_thresh = float(cfg.get("foreground_ratio_thresh", 1.0))
            alpha = float(cfg.get("spatial_penalty_alpha", 1.0))
            if ratio > fg_thresh:
                penalty = 1.0 / (1.0 + alpha * (ratio - fg_thresh))
                score = float(np.clip(score * penalty, 0.05, 1.0))

        if score >= min_score:
            candidates.append((t_cand, score))

    candidates.sort(key=lambda x: x[0])
    return candidates


def detect_candidates_for_clip(
    video_path: Path,
    clip_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run candidate detection on a single video file.

    Returns:
        List of candidate prediction dicts conforming to PREDICTIONS_COLUMNS.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    method_ver = str(cfg.get("method_version", "m3_motion_baseline_v1"))

    downscale_w = int(cfg.get("downscale_width", 160))
    downscale_h = int(cfg.get("downscale_height", 90))

    signal_data = extract_motion_signal(
        video_path=video_path,
        downscale_width=downscale_w,
        downscale_height=downscale_h,
    )

    if signal_data["total_frames"] == 0:
        return []

    candidates = find_candidate_peaks_from_signal(
        motion=signal_data["motion"],
        time_s=signal_data["time_s"],
        fps=signal_data["fps"],
        config=cfg,
        motion_top=signal_data.get("motion_top"),
        motion_mid=signal_data.get("motion_mid"),
        motion_bot=signal_data.get("motion_bot"),
    )

    preds = []
    for t_s, sc in candidates:
        preds.append({
            "clip_id": clip_id,
            "time_s": f"{t_s:.4f}",
            "score": f"{sc:.4f}",
            "status": "candidate",
            "method_version": method_ver,
        })

    return preds


def run_detection_on_split(
    manifest_path: Path,
    status_path: Path,
    data_dir: Path,
    split: str = "dev",
    config: Optional[Dict[str, Any]] = None,
    save_signals_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute detection over all reviewed clips in the requested split.

    Args:
        manifest_path: Path to labels/manifest.csv.
        status_path: Path to labels/clip_status.csv.
        data_dir: Path to raw footage root directory.
        split: Target split ("dev", "test", etc.).
        config: Detector hyperparameters.
        save_signals_dir: Optional folder to dump per-clip motion signal CSVs.

    Returns:
        (predictions_df, execution_stats)
    """
    manifest_path = Path(manifest_path)
    status_path = Path(status_path)
    data_dir = Path(data_dir)

    manifest_df = pd.read_csv(manifest_path, dtype=str)
    validate_manifest_df(manifest_df)

    status_df = pd.read_csv(status_path, dtype=str)
    validate_clip_status_df(status_df)

    # Filter reviewed clips
    reviewed_ids = set(
        status_df[status_df["reviewed"].str.lower() == "yes"]["clip_id"].astype(str).str.strip()
    )

    split_manifest = manifest_df[
        (manifest_df["split"] == split) & (manifest_df["clip_id"].isin(reviewed_ids))
    ].copy()

    if split_manifest.empty:
        raise ValueError(f"No reviewed clips found for split='{split}'")

    split_manifest = split_manifest.sort_values(by=["camera", "sequence_no"])

    all_preds: List[Dict[str, Any]] = []
    clip_timings: List[float] = []
    total_t0 = time.time()

    for idx, (_, row) in enumerate(split_manifest.iterrows(), 1):
        cid = str(row["clip_id"]).strip()
        rel_p = str(row["relative_path"]).strip()
        video_full_p = data_dir / rel_p

        t_start = time.time()
        if not video_full_p.exists():
            print(f"[WARNING] Video file missing: {video_full_p}. Generating 0 predictions for {cid}.")
            elapsed = time.time() - t_start
            clip_timings.append(elapsed)
            continue

        preds = detect_candidates_for_clip(
            video_path=video_full_p,
            clip_id=cid,
            config=config,
        )
        elapsed = time.time() - t_start
        clip_timings.append(elapsed)
        all_preds.extend(preds)
        print(f"[{idx:02d}/{len(split_manifest):02d}] Processed {cid} in {elapsed:.2f}s ({len(preds)} candidates)", flush=True)

        # Optional signal saving
        if save_signals_dir:
            from stumpvision.signals.motion import save_signal_csv
            save_signals_dir = Path(save_signals_dir)
            sig_data = extract_motion_signal(
                video_full_p,
                downscale_width=config.get("downscale_width", 160) if config else 160,
                downscale_height=config.get("downscale_height", 90) if config else 90,
            )
            save_signal_csv(sig_data, save_signals_dir / f"{cid}_signal.csv")

    total_elapsed = time.time() - total_t0

    if all_preds:
        preds_df = pd.DataFrame(all_preds, columns=PREDICTIONS_COLUMNS)
    else:
        preds_df = pd.DataFrame(columns=PREDICTIONS_COLUMNS)

    validate_predictions_df(preds_df)

    stats = {
        "split": split,
        "n_clips_processed": len(split_manifest),
        "n_candidates_found": len(preds_df),
        "total_elapsed_s": total_elapsed,
        "mean_time_per_clip_s": float(np.mean(clip_timings)) if clip_timings else 0.0,
        "max_time_per_clip_s": float(np.max(clip_timings)) if clip_timings else 0.0,
        "clip_timings": clip_timings,
    }

    return preds_df, stats

