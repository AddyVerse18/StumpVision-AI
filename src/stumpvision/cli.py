"""
cli.py
======
Command-line interface for StumpVision AI Delivery Finder.
Commands:
  stumpvision audit [--data-dir PATH] [--count-frames] [--config PATH]
  stumpvision assign-splits [--config PATH] [--manifest PATH]
  stumpvision check-leakage [--manifest PATH]
  stumpvision label [--data-dir PATH] [--labels-file PATH] [--status-file PATH] [--blind]
  stumpvision labels-report [--manifest PATH] [--labels-file PATH] [--status-file PATH]
  stumpvision compare-labels --primary PATH --retest PATH [--tolerance FLOAT]
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml
import pandas as pd

from stumpvision.schemas import (
    SCHEMA_VERSION,
    merge_and_save_manifest,
)
from stumpvision.io.video import probe_video, count_decoded_frames
from stumpvision.labeling.splits import assign_splits, check_leakage, record_test_access
from stumpvision.labeling.reports import generate_labels_report, compare_label_sets
from stumpvision.eval.matching import (
    load_ground_truth,
    load_predictions,
    match_eval_dataset,
)
from stumpvision.eval.metrics import (
    compute_aggregate_counts,
    compute_timestamp_errors,
    compute_review_burden,
    compute_clip_level_metrics,
    compute_breakdowns,
    compute_partials_summary,
    compute_threshold_sweep,
    compute_bootstrap_ci,
)
from stumpvision.eval.report import write_eval_report
from stumpvision.eval.baselines import (
    generate_predict_nothing,
    generate_every_n_seconds,
    evaluate_random_baseline_multi_seed,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stumpvision.cli")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def check_project_for_video_files() -> List[Path]:
    """Scan project folder for any media files. Returns any found."""
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    found = []
    for p in PROJECT_ROOT.rglob("*"):
        if ".venv" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() in video_exts:
            found.append(p)
    return found


def check_disk_space(path: Path) -> float:
    """Return free disk space in GB for given path."""
    try:
        total, used, free = shutil.disk_usage(str(path))
        return free / (1024 ** 3)
    except Exception:
        return -1.0


def load_config(config_path: Optional[Path] = None) -> dict:
    default_cfg = PROJECT_ROOT / "configs" / "default.yaml"
    cfg_file = config_path if config_path and config_path.exists() else default_cfg
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_audit(
    data_dir_override: Optional[str] = None,
    config_path: Optional[str] = None,
    count_frames: bool = False,
    manifest_override: Optional[str] = None,
) -> int:
    """Execute M0 audit on footage directory."""
    print("=" * 70)
    print("StumpVision - M0 Footage Audit")
    print("=" * 70)

    # 1. Check for forbidden video files inside project
    project_videos = check_project_for_video_files()
    if project_videos:
        print("[WARNING] SAFETY ALERT: Video files detected inside project directory:")
        for pv in project_videos:
            print(f"  -> {pv}")
    else:
        print("[OK] Workspace Clean: No video files stored inside project directory.")

    # 2. Free disk space check
    proj_free = check_disk_space(PROJECT_ROOT)
    if proj_free >= 0:
        print(f"[INFO] Free disk space on project drive: {proj_free:.2f} GB")
        if proj_free < 5.0:
            print("[WARNING] Low disk space warning: less than 5.0 GB available!")

    # 3. Resolve data dir
    cfg = load_config(Path(config_path) if config_path else None)
    data_dir_env = os.environ.get("STUMPVISION_DATA_DIR")
    data_dir_str = data_dir_override or data_dir_env or cfg.get("data_dir", "D:/Computer Vision Projects/StumpEye_Footage")
    data_dir = Path(data_dir_str)

    if not data_dir.exists():
        print(f"[ERROR] Footage data directory does not exist: {data_dir}")
        print("Please set STUMPVISION_DATA_DIR or pass --data-dir.")
        return 1

    print(f"[INFO] Scanning footage in: {data_dir}")

    # 4. Find video files
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"}
    raw_dir = data_dir / "raw" if (data_dir / "raw").exists() else data_dir
    found_clips = []

    for camera_dir in sorted(raw_dir.glob("cam*")):
        if camera_dir.is_dir():
            cam_name = camera_dir.name
            for clip in sorted(camera_dir.iterdir()):
                if clip.is_file() and clip.suffix in video_exts:
                    found_clips.append((clip, cam_name))

    if not found_clips:
        for clip in sorted(raw_dir.rglob("*")):
            if clip.is_file() and clip.suffix in video_exts:
                cam_name = clip.parent.name
                found_clips.append((clip, cam_name))

    print(f"[INFO] Found {len(found_clips)} clips across camera folders.")
    if not found_clips:
        print("[WARNING] No video clips found to audit.")
        return 0

    # 5. Audit each clip
    manifest_rows = []
    t_start = time.time()

    print("-" * 70)
    for idx, (clip_path, cam) in enumerate(found_clips, 1):
        print(f"[{idx:02d}/{len(found_clips):02d}] Probing {clip_path.name} (camera: {cam})...")
        t0 = time.time()
        row = probe_video(
            video_path=clip_path,
            camera=cam,
            data_dir=data_dir,
            count_frames=count_frames,
            schema_version=SCHEMA_VERSION,
        )
        elapsed = time.time() - t0

        detail_str = f"{row['width']}x{row['height']} | {row['fps']} fps | reported: {row['n_frames_reported']} frames | {row['duration_s']}s"
        if count_frames and row["n_frames_decoded"] != "":
            decoded_cnt = row["n_frames_decoded"]
            match_status = "MATCH" if decoded_cnt == row["n_frames_reported"] else "MISMATCH!"
            detail_str += f" | decoded: {decoded_cnt} ({match_status})"
        detail_str += f" | CFR: {row['cfr_status']}"
        print(f"       -> {detail_str} [{elapsed:.2f}s]")

        manifest_rows.append(row)

    total_time = time.time() - t_start

    # 6. Save/Merge into labels/manifest.csv
    manifest_file = Path(manifest_override) if manifest_override else PROJECT_ROOT / cfg.get("manifest_path", "labels/manifest.csv")
    manifest_df = merge_and_save_manifest(manifest_rows, manifest_file)

    print("-" * 70)
    print(f"[SUCCESS] Audit completed in {total_time:.2f}s.")
    print(f"[SAVED] Manifest written to: {manifest_file} ({len(manifest_df)} rows)")
    print("=" * 70)
    return 0


def run_assign_splits(config_path: Optional[str] = None, manifest_path: Optional[str] = None) -> int:
    cfg_file = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "splits.yaml"
    m_file = Path(manifest_path) if manifest_path else PROJECT_ROOT / "labels" / "manifest.csv"
    try:
        df = assign_splits(manifest_path=m_file, config_path=cfg_file)
        print(f"[SUCCESS] Assigned splits to manifest: {m_file}")
        counts = df["split"].value_counts().to_dict()
        print(f"[INFO] Split counts: {counts}")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to assign splits: {e}")
        return 1


def run_check_leakage(manifest_path: Optional[str] = None) -> int:
    m_file = Path(manifest_path) if manifest_path else PROJECT_ROOT / "labels" / "manifest.csv"
    try:
        res = check_leakage(manifest_path=m_file)
        print("=" * 64)
        print(" [PASS] SPLIT LEAKAGE & INTEGRITY CHECK")
        print("=" * 64)
        print(f"Total Clips in Manifest:   {res['total_clips']}")
        print(f"Dev Split Clips:           {res['dev_clips_count']}")
        print(f"Buffer Split Clips:        {res['buffer_clips_count']} (isolated)")
        print(f"Test Split Clips:          {res['test_clips_count']} (locked)")
        print(f"Test Set SHA-256:          {res['test_fingerprint']}")
        print("=" * 64)
        return 0
    except Exception as e:
        print(f"[ERROR] Leakage check failed: {e}")
        return 1


def run_labels_report(
    manifest_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    status_path: Optional[str] = None,
) -> int:
    m_file = Path(manifest_path) if manifest_path else PROJECT_ROOT / "labels" / "manifest.csv"
    l_file = Path(labels_path) if labels_path else PROJECT_ROOT / "labels" / "labels.csv"
    s_file = Path(status_path) if status_path else PROJECT_ROOT / "labels" / "clip_status.csv"

    try:
        data, formatted_report = generate_labels_report(
            manifest_path=m_file,
            labels_path=l_file,
            status_path=s_file,
        )
        print(formatted_report)
        return 0
    except Exception as e:
        print(f"[ERROR] Could not generate labels report: {e}")
        return 1


def run_compare_labels(
    primary_path: Optional[str] = None,
    retest_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    tolerance: float = 0.5,
) -> int:
    p_file = Path(primary_path) if primary_path else PROJECT_ROOT / "labels" / "labels.csv"
    r_file = Path(retest_path) if retest_path else PROJECT_ROOT / "labels" / "retest.csv"
    m_file = Path(manifest_path) if manifest_path else PROJECT_ROOT / "labels" / "manifest.csv"

    try:
        stats, report_str = compare_label_sets(
            primary_path=p_file,
            retest_path=r_file,
            manifest_path=m_file if m_file.exists() else None,
            tolerance_s=tolerance,
        )
        print(report_str)
        return 0
    except Exception as e:
        print(f"[ERROR] Could not compare label sets: {e}")
        return 1


def run_label_ui(
    data_dir_override: Optional[str] = None,
    manifest_override: Optional[str] = None,
    labels_file_override: Optional[str] = None,
    status_file_override: Optional[str] = None,
    blind: bool = False,
) -> int:
    cfg = load_config()
    data_dir_env = os.environ.get("STUMPVISION_DATA_DIR")
    data_dir_str = data_dir_override or data_dir_env or cfg.get("data_dir", "D:/Computer Vision Projects/StumpEye_Footage")
    data_dir = Path(data_dir_str)

    m_file = Path(manifest_override) if manifest_override else PROJECT_ROOT / "labels" / "manifest.csv"
    l_file = Path(labels_file_override) if labels_file_override else PROJECT_ROOT / "labels" / "labels.csv"
    s_file = Path(status_file_override) if status_file_override else PROJECT_ROOT / "labels" / "clip_status.csv"

    from stumpvision.labeling.ui import run_labeling_gui
    try:
        run_labeling_gui(
            data_dir=data_dir,
            manifest_path=m_file,
            labels_path=l_file,
            status_path=s_file,
            blind=blind,
        )
        return 0
    except Exception as e:
        print(f"[ERROR] Labeling tool error: {e}")
        return 1


def run_evaluate(
    pred_path: str,
    split: str = "dev",
    tolerance: float = 0.5,
    secondary_tolerance: float = 1.0,
    allow_test: bool = False,
    manifest_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    status_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> int:
    m_file = Path(manifest_path) if manifest_path else PROJECT_ROOT / "labels" / "manifest.csv"
    l_file = Path(labels_path) if labels_path else PROJECT_ROOT / "labels" / "labels.csv"
    s_file = Path(status_path) if status_path else PROJECT_ROOT / "labels" / "clip_status.csv"
    p_file = Path(pred_path)

    try:
        dataset = load_ground_truth(
            manifest_path=m_file,
            labels_path=l_file,
            status_path=s_file,
            split=split,
            allow_test=allow_test,
            command_str=f"stumpvision evaluate --pred {pred_path} --split {split} --tolerance {tolerance}",
            pred_path=p_file,
        )
    except Exception as e:
        print(f"[ERROR] Could not load ground truth: {e}")
        return 1

    try:
        preds_df = load_predictions(p_file)
    except Exception as e:
        print(f"[ERROR] Could not load predictions: {e}")
        return 1

    method_version = str(preds_df["method_version"].iloc[0]) if not preds_df.empty and "method_version" in preds_df.columns else "unknown"

    # Match at primary and secondary tolerance
    results_primary = match_eval_dataset(dataset, preds_df, tolerance=tolerance)
    results_secondary = match_eval_dataset(dataset, preds_df, tolerance=secondary_tolerance)

    primary_metrics = compute_aggregate_counts(results_primary)
    secondary_metrics = compute_aggregate_counts(results_secondary)
    error_stats = compute_timestamp_errors(results_primary, dataset)
    burden_stats = compute_review_burden(results_primary, dataset)
    clip_level_stats = compute_clip_level_metrics(results_primary)
    breakdown_stats = compute_breakdowns(results_primary)
    partials_stats = compute_partials_summary(results_primary)
    sweep_rows, best_op = compute_threshold_sweep(dataset, preds_df, tolerance=tolerance)
    bootstrap_stats = compute_bootstrap_ci(results_primary)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    safe_method = method_version.replace(" ", "_").replace("/", "_")
    run_id = f"{safe_method}_{split}_{timestamp_str}"
    out_path = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / "reports" / run_id

    write_eval_report(
        dataset=dataset,
        clip_results_primary=results_primary,
        clip_results_secondary=results_secondary,
        primary_metrics=primary_metrics,
        secondary_metrics=secondary_metrics,
        error_stats=error_stats,
        burden_stats=burden_stats,
        clip_level_stats=clip_level_stats,
        breakdown_stats=breakdown_stats,
        partials_stats=partials_stats,
        sweep_rows=sweep_rows,
        best_operating_point=best_op,
        bootstrap_stats=bootstrap_stats,
        manifest_path=m_file,
        labels_path=l_file,
        status_path=s_file,
        pred_path=p_file,
        output_dir=out_path,
        primary_tolerance=tolerance,
        secondary_tolerance=secondary_tolerance,
        method_version=method_version,
    )

    print("=" * 70)
    print(f"StumpVision AI Delivery Finder — Evaluation Summary [{split.upper()}]")
    print("=" * 70)
    print(f"Method Version:           {method_version}")
    print(f"Reviewed Clips:           {dataset.reviewed_clips_count} (Excluded Unreviewed: {dataset.unreviewed_clips_count})")
    print(f"Recall (±{tolerance}s):          {primary_metrics['recall_str']}")
    print(f"Precision (±{tolerance}s):       {primary_metrics['precision_str']}")
    print(f"F1 Score:                 {primary_metrics['f1']:.3f}")
    print(f"Recall (±{secondary_tolerance}s):         {secondary_metrics['recall_str']}")
    print(f"Precision (±{secondary_tolerance}s):      {secondary_metrics['precision_str']}")
    print(f"Mean Abs Error (s):       {error_stats['mean_error_s']:.3f}s ({error_stats['mean_error_frames']:.1f} frames)")
    print(f"Review Burden:            {burden_stats['preds_per_min']:.2f} preds/min ({burden_stats['preds_per_clip']:.2f} preds/clip)")
    print(f"Partials:                 {partials_stats['summary_str']}")
    if bootstrap_stats.get("warning"):
        print(f"[NOTE] {bootstrap_stats['warning']}")
    print("-" * 70)
    print(f"[SAVED] Full report and artifacts written to: {out_path}")
    print("=" * 70)
    return 0


def run_baselines(
    split: str = "dev",
    tolerance: float = 0.5,
    allow_test: bool = False,
    manifest_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    status_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> int:
    m_file = Path(manifest_path) if manifest_path else PROJECT_ROOT / "labels" / "manifest.csv"
    l_file = Path(labels_path) if labels_path else PROJECT_ROOT / "labels" / "labels.csv"
    s_file = Path(status_path) if status_path else PROJECT_ROOT / "labels" / "clip_status.csv"

    try:
        dataset = load_ground_truth(
            manifest_path=m_file,
            labels_path=l_file,
            status_path=s_file,
            split=split,
            allow_test=allow_test,
            command_str=f"stumpvision baselines --split {split}",
        )
    except Exception as e:
        print(f"[ERROR] Could not load ground truth: {e}")
        return 1

    baselines_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predict Nothing
    df_nothing = generate_predict_nothing(dataset)
    nothing_csv = baselines_dir / "predict_nothing.csv"
    df_nothing.to_csv(nothing_csv, index=False)
    res_nothing = match_eval_dataset(dataset, df_nothing, tolerance=tolerance)
    m_nothing = compute_aggregate_counts(res_nothing)

    # 2. Every 10 Seconds
    df_grid = generate_every_n_seconds(dataset, interval_s=10.0)
    grid_csv = baselines_dir / "every_10s.csv"
    df_grid.to_csv(grid_csv, index=False)
    res_grid = match_eval_dataset(dataset, df_grid, tolerance=tolerance)
    m_grid = compute_aggregate_counts(res_grid)

    # 3. Random Times (200 seeds)
    rand_summary = evaluate_random_baseline_multi_seed(dataset, n_seeds=200, tolerance=tolerance)
    if rand_summary["representative_df"] is not None:
        rand_csv = baselines_dir / "random_times_sample.csv"
        rand_summary["representative_df"].to_csv(rand_csv, index=False)

    print("=" * 76)
    print(f"StumpVision AI Delivery Finder — Baseline Benchmark Comparison [{split.upper()}]")
    print("=" * 76)
    print(f"Reviewed Clips Evaluated: {dataset.reviewed_clips_count} (Excluded Unreviewed: {dataset.unreviewed_clips_count})")
    print(f"Tolerance: ±{tolerance}s")
    print("-" * 76)
    print(f"{'Baseline Strategy':<25} | {'Recall':<18} | {'Precision':<18} | {'F1':<6}")
    print("-" * 76)
    print(f"{'predict_nothing':<25} | {m_nothing['recall_str']:<18} | {m_nothing['precision_str']:<18} | {m_nothing['f1']:.3f}")
    rand_rec_str = f"{rand_summary['mean_recall']*100:.1f}% (±{rand_summary['std_recall']*100:.1f}%)"
    rand_prec_str = f"{rand_summary['mean_precision']*100:.1f}% (±{rand_summary['std_precision']*100:.1f}%)"
    print(f"{'random_times (200 seeds)':<25} | {rand_rec_str:<18} | {rand_prec_str:<18} | {rand_summary['mean_f1']:.3f}")
    print(f"{'every_10s':<25} | {m_grid['recall_str']:<18} | {m_grid['precision_str']:<18} | {m_grid['f1']:.3f}")
    print("-" * 76)
    print(f"[INFO] Baseline predictions saved under: {baselines_dir}")
    print("=" * 76)
    return 0


def run_detect(
    split: str = "dev",
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    data_dir_override: Optional[str] = None,
    manifest_path: Optional[str] = None,
    status_path: Optional[str] = None,
    save_signals: bool = False,
) -> int:
    cfg = load_config(Path(config_path) if config_path else None)
    m3_cfg = cfg.get("m3_detector", {})

    data_dir_env = os.environ.get("STUMPVISION_DATA_DIR")
    data_dir_str = data_dir_override or data_dir_env or cfg.get("data_dir", "D:/Computer Vision Projects/StumpEye_Footage")
    data_dir = Path(data_dir_str)

    m_file = Path(manifest_path) if manifest_path else PROJECT_ROOT / "labels" / "manifest.csv"
    s_file = Path(status_path) if status_path else PROJECT_ROOT / "labels" / "clip_status.csv"

    default_out = PROJECT_ROOT / "outputs" / "predictions" / f"m3_motion_baseline_{split}.csv"
    out_file = Path(output_path) if output_path else default_out
    out_file.parent.mkdir(parents=True, exist_ok=True)

    signals_dir = PROJECT_ROOT / "outputs" / "signals" if save_signals else None

    from stumpvision.detect.motion import run_detection_on_split
    try:
        preds_df, stats = run_detection_on_split(
            manifest_path=m_file,
            status_path=s_file,
            data_dir=data_dir,
            split=split,
            config=m3_cfg,
            save_signals_dir=signals_dir,
        )
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return 1

    preds_df.to_csv(out_file, index=False)

    print("=" * 72)
    print(f"StumpVision AI Delivery Finder — M3 Candidate Detection [{split.upper()}]")
    print("=" * 72)
    print(f"Clips Processed:          {stats['n_clips_processed']}")
    print(f"Candidates Found:         {stats['n_candidates_found']}")
    total_min = stats['n_clips_processed'] * 1.0
    candidates_per_min = (stats['n_candidates_found'] / total_min) if total_min > 0 else 0.0
    candidates_per_clip = (stats['n_candidates_found'] / stats['n_clips_processed']) if stats['n_clips_processed'] > 0 else 0.0
    print(f"Review Burden:            {candidates_per_min:.2f} candidates/min ({candidates_per_clip:.2f} per clip)")
    print(f"Total Processing Time:    {stats['total_elapsed_s']:.2f}s ({stats['mean_time_per_clip_s']:.2f}s/clip average)")
    t_speed = (stats['n_clips_processed'] * 60.0 / stats['total_elapsed_s']) if stats['total_elapsed_s'] > 0 else 0.0
    print(f"Throughput:               {t_speed:.1f}x real-time speed")
    print("-" * 72)
    print(f"[SAVED] Predictions written to: {out_file}")
    print("=" * 72)
    return 0


def run_m4_extract(
    config_path: Optional[str] = None,
    split: str = "dev",
    allow_test: bool = False,
    output_frames_override: Optional[str] = None,
) -> int:
    """Execute deterministic keyframe extraction for M4 RF-DETR dataset."""
    default_cfg_p = PROJECT_ROOT / "configs" / "m4_rf_detr.yaml"
    cfg = load_config(Path(config_path) if config_path else (default_cfg_p if default_cfg_p.exists() else None))

    data_dir_env = os.environ.get("STUMPVISION_DATA_DIR")
    data_dir_str = data_dir_env or cfg.get("data_dir", "D:/Computer Vision Projects/StumpEye_Footage")
    data_dir = Path(data_dir_str)

    m_file = PROJECT_ROOT / cfg.get("manifest_path", "labels/manifest.csv")
    l_file = PROJECT_ROOT / cfg.get("labels_path", "labels/labels.csv")
    s_file = PROJECT_ROOT / cfg.get("status_path", "labels/clip_status.csv")

    f_ext_cfg = cfg.get("frame_extraction", {})
    frames_dirname = f_ext_cfg.get("frames_dirname", "m4_frames")
    out_dir = Path(output_frames_override) if output_frames_override else (data_dir / frames_dirname)

    idx_csv = PROJECT_ROOT / f_ext_cfg.get("index_manifest", "labels/m4_extracted_frames.csv")
    coco_json = PROJECT_ROOT / f_ext_cfg.get("coco_annotations", "labels/m4_annotations/cricket_dev_coco.json")

    offsets = f_ext_cfg.get("delivery_offsets_s", [-0.25, 0.0, 0.25])
    idle_int = float(f_ext_cfg.get("idle_interval_s", 20.0))
    quality = int(f_ext_cfg.get("jpeg_quality", 95))

    from stumpvision.detr.extract_frames import extract_keyframes_for_split
    try:
        df, stats = extract_keyframes_for_split(
            manifest_path=m_file,
            labels_path=l_file,
            status_path=s_file,
            data_dir=data_dir,
            output_frames_dir=out_dir,
            split=split,
            allow_test=allow_test,
            delivery_offsets_s=offsets,
            idle_interval_s=idle_int,
            jpeg_quality=quality,
            coco_output_path=coco_json,
            index_csv_path=idx_csv,
        )
    except Exception as e:
        print(f"[ERROR] M4 Keyframe extraction failed: {e}")
        return 1

    print("=" * 72)
    print(f"StumpVision AI Delivery Finder — M4 Keyframe Dataset Extraction [{split.upper()}]")
    print("=" * 72)
    print(f"Clips Processed:          {stats['n_clips_processed']}")
    print(f"Total Frames Extracted:   {stats['n_frames_extracted']}")
    print(f"Delivery Peak Frames:     {stats['delivery_frames_count']}")
    print(f"Delivery Stride Frames:   {stats['stride_frames_count']}")
    print(f"Idle / Distractor Frames: {stats['idle_frames_count']}")
    print(f"Frames Destination:       {stats['output_dir']}")
    print(f"Frame Index Manifest:     {idx_csv}")
    print(f"Initial COCO JSON:        {coco_json}")
    print("=" * 72)
    return 0


def run_m4_annotate(
    config_path: Optional[str] = None,
    coco_path: Optional[str] = None,
    frames_dir: Optional[str] = None,
) -> int:
    """Launch the OpenCV bounding-box annotation GUI."""
    default_cfg_p = PROJECT_ROOT / "configs" / "m4_rf_detr.yaml"
    cfg = load_config(Path(config_path) if config_path else (default_cfg_p if default_cfg_p.exists() else None))

    data_dir_env = os.environ.get("STUMPVISION_DATA_DIR")
    data_dir_str = data_dir_env or cfg.get("data_dir", "D:/Computer Vision Projects/StumpEye_Footage")
    data_dir = Path(data_dir_str)

    f_ext_cfg = cfg.get("frame_extraction", {})
    frames_dirname = f_ext_cfg.get("frames_dirname", "m4_frames")
    f_dir = Path(frames_dir) if frames_dir else (data_dir / frames_dirname)

    c_json = Path(coco_path) if coco_path else (PROJECT_ROOT / f_ext_cfg.get("coco_annotations", "labels/m4_annotations/cricket_dev_coco.json"))

    if not c_json.exists():
        print(f"[ERROR] COCO JSON file not found: {c_json}. Run `stumpvision m4-extract` first.")
        return 1

    if not f_dir.exists():
        print(f"[ERROR] Extracted frames directory not found: {f_dir}. Run `stumpvision m4-extract` first.")
        return 1

    from stumpvision.detr.annotate import run_annotation_gui
    try:
        run_annotation_gui(coco_path=c_json, frames_dir=f_dir)
    except Exception as e:
        print(f"[ERROR] Annotation tool failed: {e}")
        return 1
    return 0


def run_m4_validate(
    config_path: Optional[str] = None,
    coco_path: Optional[str] = None,
    frames_dir: Optional[str] = None,
) -> int:
    """Perform integrity and bounds validation on COCO annotations and frame assets."""
    default_cfg_p = PROJECT_ROOT / "configs" / "m4_rf_detr.yaml"
    cfg = load_config(Path(config_path) if config_path else (default_cfg_p if default_cfg_p.exists() else None))

    data_dir_env = os.environ.get("STUMPVISION_DATA_DIR")
    data_dir_str = data_dir_env or cfg.get("data_dir", "D:/Computer Vision Projects/StumpEye_Footage")
    data_dir = Path(data_dir_str)

    f_ext_cfg = cfg.get("frame_extraction", {})
    frames_dirname = f_ext_cfg.get("frames_dirname", "m4_frames")
    f_dir = Path(frames_dir) if frames_dir else (data_dir / frames_dirname)

    c_json = Path(coco_path) if coco_path else (PROJECT_ROOT / f_ext_cfg.get("coco_annotations", "labels/m4_annotations/cricket_dev_coco.json"))

    from stumpvision.detr.coco_schema import validate_coco_integrity
    is_valid, errors, stats = validate_coco_integrity(coco_path=c_json, frames_dir=f_dir)

    print("=" * 72)
    print("StumpVision AI — M4 COCO Dataset Integrity Validation")
    print("=" * 72)
    print(f"COCO Dataset Path:        {c_json}")
    print(f"Frames Directory:         {f_dir}")
    print(f"Total Registered Images:  {stats.get('total_images', 0)}")
    print(f"Annotated Images:         {stats.get('annotated_images', 0)}")
    print(f"Unannotated Images:       {stats.get('unannotated_images', 0)}")
    print(f"Total Bounding Boxes:     {stats.get('total_annotations', 0)}")
    print("-" * 72)
    print("Bounding Boxes Per Category:")
    cat_counts = stats.get("category_counts", {})
    for cat_name, cnt in cat_counts.items():
        print(f"  - {cat_name:<15}: {cnt}")
    print("-" * 72)
    if is_valid:
        print("[OK] Integrity Check PASSED: Valid COCO detection dataset format.")
        print("=" * 72)
        return 0
    else:
        print("[FAIL] Integrity Check FAILED:")
        for err in errors:
            print(f"  [ERROR] {err}")
        print("=" * 72)
        return 1


def run_m4_split(
    config_path: Optional[str] = None,
    coco_path: Optional[str] = None,
    frames_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    copy_images: bool = True,
) -> int:
    """Split local COCO dataset into RF-DETR train/valid layout by block."""
    default_cfg_p = PROJECT_ROOT / "configs" / "m4_rf_detr.yaml"
    cfg = load_config(Path(config_path) if config_path else (default_cfg_p if default_cfg_p.exists() else None))

    data_dir_env = os.environ.get("STUMPVISION_DATA_DIR")
    data_dir_str = data_dir_env or cfg.get("data_dir", "D:/Computer Vision Projects/StumpEye_Footage")
    data_dir = Path(data_dir_str)

    f_ext_cfg = cfg.get("frame_extraction", {})
    frames_dirname = f_ext_cfg.get("frames_dirname", "m4_frames")
    f_dir = Path(frames_dir) if frames_dir else (data_dir / frames_dirname)

    c_json = Path(coco_path) if coco_path else (PROJECT_ROOT / f_ext_cfg.get("coco_annotations", "labels/m4_annotations/cricket_dev_coco.json"))
    out_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "outputs" / "m4_rf_detr_dataset")

    from stumpvision.detr.split_dataset import export_rf_detr_dataset
    try:
        stats = export_rf_detr_dataset(
            coco_path=c_json,
            frames_dir=f_dir,
            output_dataset_dir=out_dir,
            copy_images=copy_images,
        )
    except Exception as e:
        print(f"[ERROR] Failed to split RF-DETR dataset: {e}")
        return 1

    print("=" * 72)
    print("StumpVision AI — M4 RF-DETR Dataset Split Export")
    print("=" * 72)
    print(f"Output Dataset Directory: {stats['output_dir']}")
    print(f"Train Images:             {stats['train_images_count']} (Copied: {stats['train_images_copied']})")
    print(f"Train Bounding Boxes:     {stats['train_annotations_count']}")
    print(f"Valid Images:             {stats['val_images_count']} (Copied: {stats['val_images_copied']})")
    print(f"Valid Bounding Boxes:     {stats['val_annotations_count']}")
    print("=" * 72)
    return 0


def run_m4_merge(
    rf_dir: str,
    target_dataset_dir: Optional[str] = None,
    split_name: str = "train",
) -> int:
    """Import and remap a Roboflow COCO split into the target RF-DETR dataset split."""
    rf_path = Path(rf_dir)
    target_dir = Path(target_dataset_dir) if target_dataset_dir else (PROJECT_ROOT / "outputs" / "m4_rf_detr_dataset")
    target_split_dir = target_dir / split_name

    if not rf_path.exists():
        print(f"[ERROR] Roboflow source directory not found: {rf_path}")
        return 1

    from stumpvision.detr.import_roboflow import import_and_merge_roboflow_split
    try:
        stats = import_and_merge_roboflow_split(
            rf_split_dir=rf_path,
            target_rf_detr_split_dir=target_split_dir,
        )
    except Exception as e:
        print(f"[ERROR] Failed to merge Roboflow dataset: {e}")
        return 1

    print("=" * 72)
    print(f"StumpVision AI — M4 Roboflow Dataset Merge [{split_name.upper()}]")
    print("=" * 72)
    print(f"Merged Split Directory:     {stats['split_dir']}")
    print(f"Roboflow Images Added:      {stats['roboflow_images_added']}")
    print(f"Roboflow Boxes Added:       {stats['roboflow_annotations_added']}")
    print(f"Dropped Unmapped Boxes:     {stats['dropped_annotations']}")
    print(f"Total Images in Split:      {stats['total_images']}")
    print(f"Total BBoxes in Split:      {stats['total_annotations']}")
    print("=" * 72)
    return 0


def run_detect_fused(
    candidates_path: Optional[str] = None,
    poses_path: Optional[str] = None,
    output_path: Optional[str] = None,
    max_distance_s: float = 1.0,
    method_version: str = "m4_fused_candidates_v1",
    score_boost: float = 1.1,
    min_score: float = 0.1,
) -> int:
    """Fuse M3 motion candidates with M4 batsman pose status to filter false alarms."""
    c_file = Path(candidates_path) if candidates_path else PROJECT_ROOT / "outputs" / "predictions" / "m3_motion_baseline_dev.csv"
    p_file = Path(poses_path) if poses_path else PROJECT_ROOT / "labels" / "m4_poses" / "batsman_pose_dev.json"
    o_file = Path(output_path) if output_path else PROJECT_ROOT / "outputs" / "predictions" / "m4_fused_candidates_dev.csv"

    if not c_file.exists():
        print(f"[ERROR] Motion candidates file not found: {c_file}")
        return 1
    if not p_file.exists():
        print(f"[ERROR] Batsman pose file not found: {p_file}")
        return 1

    from stumpvision.signals.fused_detector import fuse_motion_with_batsman_pose
    try:
        fused_df, stats = fuse_motion_with_batsman_pose(
            candidates=c_file,
            poses=p_file,
            max_pose_distance_s=max_distance_s,
            method_version=method_version,
            score_boost=score_boost,
            min_score=min_score,
        )
    except Exception as e:
        print(f"[ERROR] Fusion failed: {e}")
        return 1

    o_file.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_csv(o_file, index=False)

    print("=" * 72)
    print("StumpVision AI — M4 Multimodal Motion & Pose Fusion")
    print("=" * 72)
    print(f"Input Candidates:           {stats.n_input_candidates} (from {c_file.name})")
    print(f"Fused Candidates Output:    {stats.n_fused_candidates} (written to {o_file.name})")
    print(f"Rejected (NO_BATSMAN):      {stats.n_rejected_no_batsman}")
    print(f"Rejected (Unconfirmed):     {stats.n_rejected_unconfirmed}")
    print(f"False Alarm Rejection Rate: {stats.rejection_rate_pct:.1f}%")
    print(f"Candidate Retention Rate:   {stats.retention_rate_pct:.1f}%")
    print("=" * 72)
    return 0


def run_track_trajectory(
    candidates_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    data_dir_override: Optional[str] = None,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    search_window_s: float = 0.5,
) -> int:
    """Track ball trajectories and pitch bounce points for confirmed deliveries (M5)."""
    cfg = load_config(Path(config_path) if config_path else None)
    data_dir_env = os.environ.get("STUMPVISION_DATA_DIR")
    data_dir_str = data_dir_override or data_dir_env or cfg.get("data_dir", "D:/Computer Vision Projects/StumpEye_Footage")
    data_dir = Path(data_dir_str)

    c_file = Path(candidates_path) if candidates_path else PROJECT_ROOT / "outputs" / "predictions" / "m4_fused_candidates_dev.csv"
    m_file = Path(manifest_path) if manifest_path else PROJECT_ROOT / "labels" / "manifest.csv"
    out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / "trajectories"

    if not c_file.exists():
        print(f"[ERROR] Candidates file not found: {c_file}")
        return 1
    if not m_file.exists():
        print(f"[ERROR] Manifest file not found: {m_file}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "dev_trajectories.json"
    out_csv = out_dir / "dev_trajectories_summary.csv"

    from stumpvision.trajectory.tracker import track_all_deliveries
    try:
        trajectories, summary_df = track_all_deliveries(
            candidates_csv_path=c_file,
            manifest_csv_path=m_file,
            data_dir=data_dir,
            output_json_path=out_json,
            output_summary_csv_path=out_csv,
            search_window_s=search_window_s,
        )
    except Exception as e:
        print(f"[ERROR] Trajectory tracking failed: {e}")
        return 1

    valid_count = sum(1 for t in trajectories if t.is_valid)

    print("=" * 72)
    print("StumpVision AI — M5 Ball Pitch Trajectory Tracking")
    print("=" * 72)
    print(f"Total Deliveries Processed: {len(trajectories)}")
    print(f"Deliveries with Valid Arc:  {valid_count} ({100.0 * valid_count / max(1, len(trajectories)):.1f}%)")
    print(f"Trajectory Details JSON:    {out_json}")
    print(f"Trajectory Summary CSV:     {out_csv}")
    print("-" * 72)
    print("DELIVERY TRAJECTORY SUMMARY:")
    for _, row in summary_df.iterrows():
        status = "TRACKED" if row["is_valid"] else "NO_TRACK"
        flight_val = row.get("flight_duration_s")
        flight_str = f"{flight_val:.2f}s" if pd.notnull(flight_val) else "N/A"
        print(
            f"  Delivery #{int(row['delivery_idx']):02d} ({row['clip_id'][-6:]} @ {row['candidate_time_s']:.2f}s): "
            f"[{status}] Pts={int(row['point_count']):02d} | Length={str(row['length']).upper():11s} | "
            f"Line={str(row['line']).upper():6s} | Flight={flight_str}"
        )
    print("=" * 72)
    return 0


def run_pipeline(
    input_video: str,
    output_dir: Optional[str] = None,
    candidate_time_s: Optional[float] = None,
) -> int:
    """Run end-to-end StumpEye AI Pipeline on any cricket footage."""
    import cv2
    video_path = Path(input_video).resolve()
    if not video_path.exists():
        print(f"[ERROR] Input video not found: {video_path}")
        return 1

    out_base = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / "trajectories"
    out_base.mkdir(parents=True, exist_ok=True)
    clips_dir = out_base / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("StumpEye AI — End-to-End Match Video Analysis & Trajectory Pipeline")
    print("=" * 72)
    print(f"Input Video:  {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    duration_s = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Resolution:   {width}x{height} @ {fps:.1f} FPS ({duration_s:.1f}s)")

    delivery_time = candidate_time_s if candidate_time_s is not None else min(1.2, duration_s / 2.0)
    print(f"Analyzing Delivery Window around t={delivery_time:.2f}s...")

    from stumpvision.trajectory.tracker import BallTrajectoryTracker
    from stumpvision.trajectory.viz import render_delivery_video_clip
    from stumpvision.trajectory.drs import evaluate_drs

    tracker = BallTrajectoryTracker(fps=fps)
    traj = tracker.track_clip_delivery(
        video_path=video_path,
        clip_id=video_path.stem,
        delivery_idx=1,
        candidate_time_s=delivery_time,
        search_window_s=0.6,
    )

    drs = evaluate_drs(traj)
    print("-" * 72)
    print(f"Tracking Status:       {'SUCCESS (Tracked)' if traj.is_valid else 'DETECTED (Initial Arc)'}")
    print(f"Detected Pitch Length: {str(traj.length_classification).upper()}")
    print(f"Pitch Line:            {traj.stump_line_display}")
    print(f"Speed Gun:             {traj.speed_release_kmh:.1f} km/h ({traj.speed_release_mph:.1f} mph)")
    print(f"DRS LBW Verdict:       {drs.decision} (Wickets: {drs.wickets_display})")

    out_clip_name = f"{video_path.stem}_annotated.mp4"
    out_clip_path = clips_dir / out_clip_name
    print(f"Rendering StumpEye 60 FPS Replay with Broadcast HUD -> {out_clip_path}...")
    success = render_delivery_video_clip(
        video_path=video_path,
        trajectory=traj,
        output_video_path=out_clip_path,
        fps=fps,
    )
    if success:
        print(f"[OK] Replay clip ready: {out_clip_path}")
    print("=" * 72)
    print("SUCCESS: Pipeline run completed. Video and metrics ready for StumpEye Web App.")
    print("=" * 72)
    return 0


def run_app(port: int = 8501) -> int:
    """Launch the StumpVision AI production Streamlit web dashboard."""
    app_script = PROJECT_ROOT / "app.py"
    if not app_script.exists():
        print(f"[ERROR] App script not found: {app_script}")
        return 1

    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_script), "--server.port", str(port)]
    print("=" * 72)
    print("StumpVision AI — Launching Streamlit Web Dashboard")
    print("=" * 72)
    print(f"URL: http://localhost:{port}")
    print("=" * 72)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped Streamlit dashboard.")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="stumpvision", description="StumpVision AI Delivery Finder CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # audit subcommand
    audit_parser = subparsers.add_parser("audit", help="Audit raw stump camera clips and produce labels/manifest.csv")
    audit_parser.add_argument("--data-dir", type=str, default=None, help="Path to footage root directory")
    audit_parser.add_argument("--config", type=str, default=None, help="Path to custom config yaml")
    audit_parser.add_argument("--count-frames", action="store_true", help="Decode every frame to verify actual count vs header")
    audit_parser.add_argument("--manifest", type=str, default=None, help="Custom manifest output path")

    # assign-splits subcommand
    assign_parser = subparsers.add_parser("assign-splits", help="Assign blocks and splits to manifest from config")
    assign_parser.add_argument("--config", type=str, default=None, help="Path to configs/splits.yaml")
    assign_parser.add_argument("--manifest", type=str, default=None, help="Path to labels/manifest.csv")

    # check-leakage subcommand
    leakage_parser = subparsers.add_parser("check-leakage", help="Verify zero leakage between splits and locked test set")
    leakage_parser.add_argument("--manifest", type=str, default=None, help="Path to labels/manifest.csv")

    # label subcommand
    label_parser = subparsers.add_parser("label", help="Launch local OpenCV interactive delivery labeling HUD")
    label_parser.add_argument("--data-dir", type=str, default=None, help="Footage root folder")
    label_parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path")
    label_parser.add_argument("--labels-file", type=str, default=None, help="Labels CSV destination (default: labels/labels.csv)")
    label_parser.add_argument("--status-file", type=str, default=None, help="Clip reviewed status CSV (default: labels/clip_status.csv)")
    label_parser.add_argument("--blind", action="store_true", help="Blind relabeling mode (hides existing labels, protects main labels.csv)")

    # labels-report subcommand
    report_parser = subparsers.add_parser("labels-report", help="Report counts of clips, deliveries, partials, and views per split")
    report_parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path")
    report_parser.add_argument("--labels-file", type=str, default=None, help="Labels CSV path")
    report_parser.add_argument("--status-file", type=str, default=None, help="Clip status CSV path")

    # compare-labels subcommand
    compare_parser = subparsers.add_parser("compare-labels", help="Compare primary vs retest labels for consistency audit")
    compare_parser.add_argument("--primary", type=str, default=None, help="Primary labels CSV path (default: labels/labels.csv)")
    compare_parser.add_argument("--retest", type=str, default=None, help="Retest labels CSV path (default: labels/retest.csv)")
    compare_parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path")
    compare_parser.add_argument("--tolerance", type=float, default=0.5, help="Temporal agreement tolerance in seconds (default: 0.5)")

    # evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions CSV against reviewed ground truth")
    eval_parser.add_argument("--pred", type=str, required=True, help="Path to predictions CSV")
    eval_parser.add_argument("--split", type=str, default="dev", help="Split to evaluate on (default: dev)")
    eval_parser.add_argument("--tolerance", type=float, default=0.5, help="Primary tolerance in seconds (default: 0.5)")
    eval_parser.add_argument("--secondary-tolerance", type=float, default=1.0, help="Secondary tolerance in seconds (default: 1.0)")
    eval_parser.add_argument("--allow-test", action="store_true", help="Explicit confirmation required to evaluate test split")
    eval_parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path")
    eval_parser.add_argument("--labels-file", type=str, default=None, help="Labels CSV path")
    eval_parser.add_argument("--status-file", type=str, default=None, help="Clip status CSV path")
    eval_parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for reports")

    # baselines subcommand
    base_parser = subparsers.add_parser("baselines", help="Run reference baseline predictions on reviewed split clips")
    base_parser.add_argument("--split", type=str, default="dev", help="Split to evaluate baselines on (default: dev)")
    base_parser.add_argument("--tolerance", type=float, default=0.5, help="Matching tolerance in seconds (default: 0.5)")
    base_parser.add_argument("--allow-test", action="store_true", help="Explicit confirmation required to run on test split")
    base_parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path")
    base_parser.add_argument("--labels-file", type=str, default=None, help="Labels CSV path")
    base_parser.add_argument("--status-file", type=str, default=None, help="Clip status CSV path")
    base_parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for baseline files")

    # detect subcommand
    detect_parser = subparsers.add_parser("detect", help="Run M3 classical motion candidate detection on clips")
    detect_parser.add_argument("--split", type=str, default="dev", help="Split to run detection on (default: dev)")
    detect_parser.add_argument("--config", type=str, default=None, help="Path to custom config yaml")
    detect_parser.add_argument("--output", type=str, default=None, help="Path for output predictions CSV")
    detect_parser.add_argument("--data-dir", type=str, default=None, help="Path to footage root directory")
    detect_parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path")
    detect_parser.add_argument("--status-file", type=str, default=None, help="Clip status CSV path")
    detect_parser.add_argument("--save-signals", action="store_true", help="Dump per-clip motion signal CSVs to outputs/signals/")

    # detect-fused subcommand
    fused_parser = subparsers.add_parser("detect-fused", help="Fuse M3 motion candidates with M4 batsman pose status to filter false alarms")
    fused_parser.add_argument("--candidates", type=str, default=None, help="Path to input candidates CSV (default: outputs/predictions/m3_motion_baseline_dev.csv)")
    fused_parser.add_argument("--poses", type=str, default=None, help="Path to batsman poses JSON (default: labels/m4_poses/batsman_pose_dev.json)")
    fused_parser.add_argument("--output", type=str, default=None, help="Path for output predictions CSV (default: outputs/predictions/m4_fused_candidates_dev.csv)")
    fused_parser.add_argument("--max-distance", type=float, default=1.0, help="Max temporal distance in seconds between candidate and pose observation (default: 1.0)")
    fused_parser.add_argument("--method-version", type=str, default="m4_fused_candidates_v1", help="Method version string")
    fused_parser.add_argument("--score-boost", type=float, default=1.1, help="Score boost multiplier for confirmed batsman (default: 1.1)")
    fused_parser.add_argument("--min-score", type=float, default=0.1, help="Minimum score threshold (default: 0.1)")

    # track-trajectory subcommand (M5)
    traj_parser = subparsers.add_parser("track-trajectory", help="Track ball trajectory down pitch and classify bounce (M5)")
    traj_parser.add_argument("--candidates", type=str, default=None, help="Path to input candidates CSV (default: outputs/predictions/m4_fused_candidates_dev.csv)")
    traj_parser.add_argument("--manifest", type=str, default=None, help="Path to labels/manifest.csv")
    traj_parser.add_argument("--data-dir", type=str, default=None, help="Path to footage root directory")
    traj_parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for trajectories")
    traj_parser.add_argument("--search-window", type=float, default=0.5, help="Temporal window in seconds around delivery point (default: 0.5)")

    # app subcommand
    app_parser = subparsers.add_parser("app", help="Launch interactive Streamlit web dashboard and pitch map")
    app_parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on (default: 8501)")

    # m4-extract subcommand
    m4_parser = subparsers.add_parser("m4-extract", help="Extract keyframes from clips for M4 RF-DETR object detection dataset")
    m4_parser.add_argument("--split", type=str, default="dev", help="Split to extract keyframes from (default: dev)")
    m4_parser.add_argument("--config", type=str, default=None, help="Path to custom config yaml")
    m4_parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for JPEG frames")
    m4_parser.add_argument("--allow-test", action="store_true", help="Explicit confirmation required to extract from test split")

    # m4-annotate subcommand
    ann_parser = subparsers.add_parser("m4-annotate", help="Launch interactive OpenCV bounding-box annotation HUD")
    ann_parser.add_argument("--config", type=str, default=None, help="Path to custom config yaml")
    ann_parser.add_argument("--coco", type=str, default=None, help="Path to COCO JSON annotations file")
    ann_parser.add_argument("--frames-dir", type=str, default=None, help="Path to extracted frames folder")

    # m4-validate subcommand
    val_parser = subparsers.add_parser("m4-validate", help="Validate COCO annotations integrity and bounds")
    val_parser.add_argument("--config", type=str, default=None, help="Path to custom config yaml")
    val_parser.add_argument("--coco", type=str, default=None, help="Path to COCO JSON annotations file")
    val_parser.add_argument("--frames-dir", type=str, default=None, help="Path to extracted frames folder")

    # m4-split subcommand
    split_parser = subparsers.add_parser("m4-split", help="Split local COCO dataset into RF-DETR train/valid format")
    split_parser.add_argument("--config", type=str, default=None, help="Path to custom config yaml")
    split_parser.add_argument("--coco", type=str, default=None, help="Path to COCO JSON annotations file")
    split_parser.add_argument("--frames-dir", type=str, default=None, help="Path to extracted frames folder")
    split_parser.add_argument("--output-dir", type=str, default=None, help="Output destination folder for RF-DETR dataset")

    # m4-merge subcommand
    merge_parser = subparsers.add_parser("m4-merge", help="Merge external Roboflow COCO dataset into target split")
    merge_parser.add_argument("--rf-dir", type=str, required=True, help="Path to Roboflow split directory containing COCO JSON")
    merge_parser.add_argument("--target-dir", type=str, default=None, help="Target RF-DETR dataset directory")
    merge_parser.add_argument("--split-name", type=str, default="train", help="Target split to merge into (train or valid)")

    # run-pipeline subcommand (End-to-End StumpEye Match Processing)
    pipeline_parser = subparsers.add_parser("run-pipeline", help="Run end-to-end StumpEye AI analysis and trajectory rendering on any cricket video")
    pipeline_parser.add_argument("--input", "-i", type=str, required=True, help="Path to cricket match footage or clip MP4")
    pipeline_parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output destination folder for trajectory and replay")
    pipeline_parser.add_argument("--time", "-t", type=float, default=None, help="Optional timestamp of delivery release in seconds")

    args = parser.parse_args()

    if args.command == "audit":
        sys.exit(run_audit(
            data_dir_override=args.data_dir,
            config_path=args.config,
            count_frames=args.count_frames,
            manifest_override=args.manifest,
        ))
    elif args.command == "assign-splits":
        sys.exit(run_assign_splits(config_path=args.config, manifest_path=args.manifest))
    elif args.command == "check-leakage":
        sys.exit(run_check_leakage(manifest_path=args.manifest))
    elif args.command == "label":
        sys.exit(run_label_ui(
            data_dir_override=args.data_dir,
            manifest_override=args.manifest,
            labels_file_override=args.labels_file,
            status_file_override=args.status_file,
            blind=args.blind,
        ))
    elif args.command == "labels-report":
        sys.exit(run_labels_report(
            manifest_path=args.manifest,
            labels_path=args.labels_file,
            status_path=args.status_file,
        ))
    elif args.command == "compare-labels":
        sys.exit(run_compare_labels(
            primary_path=args.primary,
            retest_path=args.retest,
            manifest_path=args.manifest,
            tolerance=args.tolerance,
        ))
    elif args.command == "evaluate":
        sys.exit(run_evaluate(
            pred_path=args.pred,
            split=args.split,
            tolerance=args.tolerance,
            secondary_tolerance=args.secondary_tolerance,
            allow_test=args.allow_test,
            manifest_path=args.manifest,
            labels_path=args.labels_file,
            status_path=args.status_file,
            output_dir=args.output_dir,
        ))
    elif args.command == "baselines":
        sys.exit(run_baselines(
            split=args.split,
            tolerance=args.tolerance,
            allow_test=args.allow_test,
            manifest_path=args.manifest,
            labels_path=args.labels_file,
            status_path=args.status_file,
            output_dir=args.output_dir,
        ))
    elif args.command == "detect":
        sys.exit(run_detect(
            split=args.split,
            config_path=args.config,
            output_path=args.output,
            data_dir_override=args.data_dir,
            manifest_path=args.manifest,
            status_path=args.status_file,
            save_signals=args.save_signals,
        ))
    elif args.command == "detect-fused":
        sys.exit(run_detect_fused(
            candidates_path=args.candidates,
            poses_path=args.poses,
            output_path=args.output,
            max_distance_s=args.max_distance,
            method_version=args.method_version,
            score_boost=args.score_boost,
            min_score=args.min_score,
        ))
    elif args.command == "track-trajectory":
        sys.exit(run_track_trajectory(
            candidates_path=args.candidates,
            manifest_path=args.manifest,
            data_dir_override=args.data_dir,
            output_dir=args.output_dir,
            search_window_s=args.search_window,
        ))
    elif args.command == "app":
        sys.exit(run_app(port=args.port))
    elif args.command == "m4-extract":
        sys.exit(run_m4_extract(
            config_path=args.config,
            split=args.split,
            allow_test=args.allow_test,
            output_frames_override=args.output_dir,
        ))
    elif args.command == "m4-annotate":
        sys.exit(run_m4_annotate(
            config_path=args.config,
            coco_path=args.coco,
            frames_dir=args.frames_dir,
        ))
    elif args.command == "m4-validate":
        sys.exit(run_m4_validate(
            config_path=args.config,
            coco_path=args.coco,
            frames_dir=args.frames_dir,
        ))
    elif args.command == "m4-split":
        sys.exit(run_m4_split(
            config_path=args.config,
            coco_path=args.coco,
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
        ))
    elif args.command == "m4-merge":
        sys.exit(run_m4_merge(
            rf_dir=args.rf_dir,
            target_dataset_dir=args.target_dir,
            split_name=args.split_name,
        ))
    elif args.command == "run-pipeline":
        sys.exit(run_pipeline(
            input_video=args.input,
            output_dir=args.output_dir,
            candidate_time_s=args.time,
        ))


if __name__ == "__main__":
    main()
