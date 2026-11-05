"""
report.py
=========
Report generation for StumpVision AI Delivery Finder.
Produces human-readable report.md, per_clip.csv, matches.csv, and config.json.
All file hashes and system package versions are captured for reproducibility without git.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from stumpvision.eval.matching import (
    ClipMatchResult,
    EvalDataset,
    compute_file_sha256,
)


def get_environment_info() -> Dict[str, str]:
    """Capture environment and package versions without external network."""
    import numpy as np
    import scipy
    import pandas as pd
    import cv2
    import yaml

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "pyyaml": yaml.__version__,
    }


def write_eval_report(
    dataset: EvalDataset,
    clip_results_primary: List[ClipMatchResult],    # at primary tolerance (e.g. 0.5s)
    clip_results_secondary: List[ClipMatchResult],  # at secondary tolerance (e.g. 1.0s)
    primary_metrics: Dict[str, Any],
    secondary_metrics: Dict[str, Any],
    error_stats: Dict[str, Any],
    burden_stats: Dict[str, Any],
    clip_level_stats: Dict[str, Any],
    breakdown_stats: Dict[str, Any],
    partials_stats: Dict[str, Any],
    sweep_rows: List[Dict[str, Any]],
    best_operating_point: Optional[Dict[str, Any]],
    bootstrap_stats: Dict[str, Any],
    manifest_path: Path,
    labels_path: Path,
    status_path: Path,
    pred_path: Optional[Path],
    output_dir: Path,
    primary_tolerance: float = 0.5,
    secondary_tolerance: float = 1.0,
    method_version: str = "unknown",
) -> Path:
    """
    Generate all evaluation artifacts under output_dir/
    Returns the path to the written report.md.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Config & Hashes Snapshot
    env_info = get_environment_info()
    config_snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method_version": method_version,
        "split": dataset.split,
        "primary_tolerance_s": primary_tolerance,
        "secondary_tolerance_s": secondary_tolerance,
        "reviewed_clips_count": dataset.reviewed_clips_count,
        "unreviewed_clips_count": dataset.unreviewed_clips_count,
        "file_hashes": {
            "manifest_sha256": compute_file_sha256(manifest_path),
            "labels_sha256": compute_file_sha256(labels_path),
            "clip_status_sha256": compute_file_sha256(status_path),
            "predictions_sha256": compute_file_sha256(pred_path) if pred_path else "none",
        },
        "bootstrap_seed": bootstrap_stats.get("seed", 42),
        "environment": env_info,
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    # 2. per_clip.csv
    per_clip_rows = []
    for r in clip_results_primary:
        tp = len(r.tp_pairs)
        fn = len(r.fn_truths)
        fp = len(r.fp_preds)
        rec = (tp / r.n_truths) if r.n_truths > 0 else (1.0 if r.n_preds == 0 else 0.0)
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if r.n_truths == 0 else 0.0)

        per_clip_rows.append({
            "clip_id": r.clip_id,
            "block_id": r.block_id,
            "view": r.view,
            "split": r.split,
            "n_truths": r.n_truths,
            "n_preds": r.n_preds,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": f"{rec:.3f}",
            "precision": f"{prec:.3f}",
            "notes": ";".join(r.notes),
        })

    per_clip_df = pd.DataFrame(per_clip_rows)
    per_clip_df.to_csv(output_dir / "per_clip.csv", index=False)

    # 3. matches.csv
    match_rows = []
    for r in clip_results_primary:
        fps = dataset.clips_data.get(r.clip_id, {}).get("fps", 60.0)
        # Matched TPs
        for p_t, t_t, err in r.tp_pairs:
            match_rows.append({
                "clip_id": r.clip_id,
                "pred_time_s": f"{p_t:.4f}",
                "truth_time_s": f"{t_t:.4f}",
                "error_s": f"{err:.4f}",
                "error_frames": f"{err * fps:.1f}",
                "kind": "delivery",
                "matched": True,
            })
        # Unmatched FNs (misses)
        for t_t in r.fn_truths:
            match_rows.append({
                "clip_id": r.clip_id,
                "pred_time_s": "",
                "truth_time_s": f"{t_t:.4f}",
                "error_s": "",
                "error_frames": "",
                "kind": "delivery",
                "matched": False,
            })
        # Unmatched FPs (false alarms)
        for p_t, _ in r.fp_preds:
            match_rows.append({
                "clip_id": r.clip_id,
                "pred_time_s": f"{p_t:.4f}",
                "truth_time_s": "",
                "error_s": "",
                "error_frames": "",
                "kind": "delivery",
                "matched": False,
            })

    match_df = pd.DataFrame(match_rows)
    match_df.to_csv(output_dir / "matches.csv", index=False)

    # 4. report.md
    md_lines = []
    md_lines.append(f"# StumpVision AI Delivery Finder — Evaluation Report")
    md_lines.append("")
    md_lines.append(f"**Split Evaluated:** `{dataset.split.upper()}`  ")
    md_lines.append(f"**Method Version:** `{method_version}`  ")
    md_lines.append(f"**Report Generated:** `{config_snapshot['timestamp_utc']}`  ")
    md_lines.append(f"**Reviewed Clips Included:** {dataset.reviewed_clips_count}  ")
    md_lines.append(f"**Unreviewed Clips Excluded:** {dataset.unreviewed_clips_count}  ")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Benchmark Summary Table
    md_lines.append("## 1. Primary Benchmark Metrics")
    md_lines.append("")
    md_lines.append("| Metric | Tolerance ±0.5s (Primary) | Tolerance ±1.0s (Secondary) |")
    md_lines.append("| :--- | :--- | :--- |")
    md_lines.append(f"| **Recall** | **{primary_metrics['recall_str']}** | {secondary_metrics['recall_str']} |")
    md_lines.append(f"| **Precision** | **{primary_metrics['precision_str']}** | {secondary_metrics['precision_str']} |")
    md_lines.append(f"| **F1 Score** | **{primary_metrics['f1']:.3f}** | {secondary_metrics['f1']:.3f} |")
    md_lines.append(f"| True Positives (Hits) | {primary_metrics['tp']} | {secondary_metrics['tp']} |")
    md_lines.append(f"| False Negatives (Misses) | {primary_metrics['fn']} | {secondary_metrics['fn']} |")
    md_lines.append(f"| False Positives (False Alarms) | {primary_metrics['fp']} | {secondary_metrics['fp']} |")
    md_lines.append("")

    # Timestamp Errors
    md_lines.append("## 2. Temporal Precision (Matched Hits)")
    md_lines.append("")
    if error_stats["count"] > 0:
        md_lines.append(f"- **Mean Absolute Error:** {error_stats['mean_error_s']:.3f} s ({error_stats['mean_error_frames']:.1f} frames)")
        md_lines.append(f"- **Max Absolute Error:** {error_stats['max_error_s']:.3f} s ({error_stats['max_error_frames']:.1f} frames)")
    else:
        md_lines.append("- *No true positive matches to compute timestamp errors.*")
    md_lines.append("")

    # Review Burden & Clip-Level Performance
    md_lines.append("## 3. Review Burden & Clip-Level Performance")
    md_lines.append("")
    md_lines.append(f"- **Predictions per Clip:** {burden_stats['preds_per_clip']:.2f} (max in a single clip: {burden_stats['max_preds_clip']})")
    md_lines.append(f"- **Predictions per Minute of Footage:** {burden_stats['preds_per_min']:.2f} preds/min")
    md_lines.append(f"- **Clips with ≥1 Hit:** {clip_level_stats['delivery_hit_str']}")
    md_lines.append(f"- **Empty Clips with 0 Predictions:** {clip_level_stats['empty_clean_str']}")
    md_lines.append("")

    # Partial Deliveries
    md_lines.append("## 4. Partial Deliveries (Excluded from Primary Score)")
    md_lines.append("")
    md_lines.append(f"{partials_stats['summary_str']}")
    md_lines.append("")

    # Breakdowns
    md_lines.append("## 5. Performance Breakdowns")
    md_lines.append("")
    md_lines.append("### By Contiguous Block")
    md_lines.append("")
    md_lines.append("| Block | Recall (Count) | Precision (Count) | F1 | Total Truths | Total Preds |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for b_name, b_data in breakdown_stats["by_block"].items():
        md_lines.append(
            f"| `{b_name}` | {b_data['recall_str']} | {b_data['precision_str']} | {b_data['f1']:.3f} | "
            f"{b_data['total_truths']} | {b_data['total_preds']} |"
        )
    md_lines.append("")

    md_lines.append("### By Camera View")
    md_lines.append("")
    md_lines.append("| View | Recall (Count) | Precision (Count) | F1 | Total Truths | Total Preds |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for v_name, v_data in breakdown_stats["by_view"].items():
        md_lines.append(
            f"| `{v_name}` | {v_data['recall_str']} | {v_data['precision_str']} | {v_data['f1']:.3f} | "
            f"{v_data['total_truths']} | {v_data['total_preds']} |"
        )
    md_lines.append("")

    md_lines.append("### By Distractor Tags")
    md_lines.append("")
    md_lines.append("| Distractor Tag | Clips with Tag | Misses (FN) | False Alarms (FP) | Hits (TP) |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for tag_name, t_data in breakdown_stats["by_distractor"].items():
        md_lines.append(
            f"| `{tag_name}` | {t_data['clip_count']} | {t_data['fn_misses']} | {t_data['fp_alarms']} | {t_data['tp_hits']} |"
        )
    md_lines.append("")

    # Confidence Interval & Uncertainty
    md_lines.append("## 6. Uncertainty & Confidence Intervals (Clip-Level Bootstrap)")
    md_lines.append("")
    if bootstrap_stats.get("warning"):
        md_lines.append(f"> ⚠️ **PRELIMINARY DATA WARNING:**  \n> {bootstrap_stats['warning']}")
        md_lines.append("")
    rec_ci = bootstrap_stats["recall_ci"]
    prec_ci = bootstrap_stats["precision_ci"]
    md_lines.append(f"- **Recall 95% CI:** [{rec_ci[0] * 100:.1f}%, {rec_ci[1] * 100:.1f}%]")
    md_lines.append(f"- **Precision 95% CI:** [{prec_ci[0] * 100:.1f}%, {prec_ci[1] * 100:.1f}%]")
    md_lines.append("")

    # Threshold Sweep
    md_lines.append("## 7. Score-Threshold Operating Points")
    md_lines.append("")
    if dataset.split == "dev":
        if best_operating_point:
            md_lines.append(
                f"**Optimal Operating Point on Dev:** Threshold `{best_operating_point['threshold']}` "
                f"→ Recall {best_operating_point['recall_str']}, Precision {best_operating_point['precision_str']}, "
                f"F1 `{best_operating_point['f1']:.3f}`"
            )
        md_lines.append("")
        md_lines.append("| Threshold | Recall | Precision | F1 | Hits (TP) | False Alarms (FP) |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for s in sweep_rows:
            md_lines.append(
                f"| {s['threshold']:.2f} | {s['recall_str']} | {s['precision_str']} | {s['f1']:.3f} | {s['tp']} | {s['fp']} |"
            )
    else:
        md_lines.append("> 🔒 **SPLIT SAFEGUARD:** Threshold sweep and operating point selection are restricted to the DEV split. "
                        "Test split must only be evaluated at a fixed, predetermined operating point.")
    md_lines.append("")

    # Reproducibility & Config Snapshot
    md_lines.append("## 8. Reproducibility & File Hashes")
    md_lines.append("")
    md_lines.append("| File | SHA-256 Digest |")
    md_lines.append("| :--- | :--- |")
    for f_label, f_hash in config_snapshot["file_hashes"].items():
        md_lines.append(f"| `{f_label}` | `{f_hash}` |")
    md_lines.append("")
    md_lines.append(f"**Packages:** Python {env_info['python_version']}, numpy {env_info['numpy']}, scipy {env_info['scipy']}, pandas {env_info['pandas']}")
    md_lines.append("")

    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return report_path

