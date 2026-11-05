"""
reports.py
==========
Reporting and consistency audit tools for StumpVision AI labels.
Provides split-level dataset reports and cross-session relabeling comparison.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from stumpvision.schemas import VALID_SPLITS, VALID_VIEWS


def generate_labels_report(
    manifest_path: Path,
    labels_path: Path,
    status_path: Path,
) -> Tuple[Dict[str, Any], str]:
    """
    Generate counts of clips, full deliveries, partial deliveries, and views per split.
    Lists unreviewed clips explicitly and states evaluation policy.
    Counts a delivery cut between two consecutive clips once when both halves are linked.
    """
    manifest_path = Path(manifest_path)
    labels_path = Path(labels_path)
    status_path = Path(status_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    m_df = pd.read_csv(manifest_path, dtype=str)
    l_df = pd.read_csv(labels_path, dtype=str) if labels_path.exists() else pd.DataFrame()
    s_df = pd.read_csv(status_path, dtype=str) if status_path.exists() else pd.DataFrame()

    # Determine reviewed status for each clip
    reviewed_clips: set = set()
    if not s_df.empty and "reviewed" in s_df.columns:
        reviewed_clips = set(
            s_df[s_df["reviewed"].str.lower() == "yes"]["clip_id"].astype(str).str.strip()
        )

    all_clips = set(m_df["clip_id"].astype(str).str.strip())
    unreviewed_clips = sorted(list(all_clips - reviewed_clips))

    report_data: Dict[str, Any] = {
        "splits": {},
        "unreviewed_clips": unreviewed_clips,
        "total_clips": len(all_clips),
        "total_reviewed": len(reviewed_clips),
    }

    lines = []
    lines.append("=" * 72)
    lines.append(" STUMPVISION AI -- DATASET & LABELS REPORT (M1)")
    lines.append("=" * 72)
    lines.append(f"Total Clips in Manifest: {len(all_clips)}")
    lines.append(f"Reviewed Clips:          {len(reviewed_clips)}")
    lines.append(f"Unreviewed Clips:        {len(unreviewed_clips)}")
    lines.append("")

    if unreviewed_clips:
        lines.append(" [!] UNREVIEWED CLIPS (Excluded from evaluation):")
        for u_cid in unreviewed_clips:
            lines.append(f"     - {u_cid}")
        lines.append("     Note: Benchmark evaluation uses reviewed clips only.")
        lines.append("")

    lines.append("-" * 72)
    lines.append(f"{'Split':<10} | {'Clips (Rev/Tot)':<15} | {'Full Deliv':<10} | {'Linked Part':<11} | {'Unlinked Part':<13} | {'Zero-Deliv':<10}")
    lines.append("-" * 72)

    for split in VALID_SPLITS:
        split_m = m_df[m_df["split"] == split]
        split_cids = set(split_m["clip_id"].astype(str).str.strip())
        split_rev_cids = split_cids.intersection(reviewed_clips)

        # Labels belonging to reviewed clips in this split
        split_labels = pd.DataFrame()
        if not l_df.empty and "clip_id" in l_df.columns:
            split_labels = l_df[l_df["clip_id"].astype(str).str.strip().isin(split_rev_cids)]

        full_deliv = 0
        part_start_df = pd.DataFrame()
        part_end_df = pd.DataFrame()
        zero_deliv = 0

        if not split_labels.empty and "kind" in split_labels.columns:
            full_deliv = len(split_labels[split_labels["kind"] == "delivery"])
            part_start_df = split_labels[split_labels["kind"] == "partial_start"]
            part_end_df = split_labels[split_labels["kind"] == "partial_end"]
            zero_deliv = len(split_labels[split_labels["kind"] == "none"])

        # Linked partial delivery accounting
        # A linked boundary delivery is counted once when partial_end links to partial_start
        linked_pairs = 0
        part_end_notes = []
        if not part_end_df.empty and "notes" in part_end_df.columns:
            part_end_notes = [str(n) for n in part_end_df["notes"].dropna()]

        part_start_cids = set()
        if not part_start_df.empty and "clip_id" in part_start_df.columns:
            part_start_cids = set(part_start_df["clip_id"].astype(str).str.strip())

        for note in part_end_notes:
            if "linked_to=" in note:
                target = note.split("linked_to=")[1].strip().split()[0]
                if target in part_start_cids:
                    linked_pairs += 1

        total_partial_marks = len(part_start_df) + len(part_end_df)
        unlinked_partials = total_partial_marks - (linked_pairs * 2)

        # Views breakdown
        view_counts = {v: 0 for v in VALID_VIEWS}
        view_counts["unassigned"] = 0
        for _, r in split_m.iterrows():
            v = str(r.get("view", "")).strip()
            if v in view_counts:
                view_counts[v] += 1
            else:
                view_counts["unassigned"] += 1

        report_data["splits"][split] = {
            "total_clips": len(split_cids),
            "reviewed_clips": len(split_rev_cids),
            "full_deliveries": full_deliv,
            "linked_boundary_deliveries": linked_pairs,
            "unlinked_partial_marks": unlinked_partials,
            "zero_delivery_clips": zero_deliv,
            "views": view_counts,
        }

        rev_tot_str = f"{len(split_rev_cids)} / {len(split_cids)}"
        lines.append(
            f"{split:<10} | {rev_tot_str:<15} | {full_deliv:<10} | {linked_pairs:<11} | {unlinked_partials:<13} | {zero_deliv:<10}"
        )

    lines.append("-" * 72)
    lines.append("")
    lines.append("CAMERA VIEW DISTRIBUTION PER SPLIT:")
    for split in VALID_SPLITS:
        vc = report_data["splits"][split]["views"]
        v_str = ", ".join([f"{k}: {val}" for k, val in vc.items() if val > 0])
        lines.append(f" - {split:<8}: {v_str if v_str else 'none assigned'}")
    lines.append("=" * 72)

    output_str = "\n".join(lines)
    return report_data, output_str


def compare_label_sets(
    primary_path: Path,
    retest_path: Path,
    manifest_path: Optional[Path] = None,
    tolerance_s: float = 0.5,
) -> Tuple[Dict[str, Any], str]:
    """
    Compare primary vs retest labels on common clips.
    Reports mean/max difference in frames and seconds, missing/extra marks, and agreement within 0.5s.
    """
    primary_path = Path(primary_path)
    retest_path = Path(retest_path)

    if not primary_path.exists():
        raise FileNotFoundError(f"Primary labels not found: {primary_path}")
    if not retest_path.exists():
        raise FileNotFoundError(f"Retest labels not found: {retest_path}")

    p_df = pd.read_csv(primary_path, dtype=str)
    r_df = pd.read_csv(retest_path, dtype=str)

    # Filter to deliveries/partials (ignore none)
    p_df = p_df[p_df["kind"].isin(["delivery", "partial_start", "partial_end"])].copy()
    r_df = r_df[r_df["kind"].isin(["delivery", "partial_start", "partial_end"])].copy()

    # Find common clips
    p_clips = set(p_df["clip_id"].astype(str).str.strip())
    r_clips = set(r_df["clip_id"].astype(str).str.strip())
    common_clips = sorted(list(p_clips.intersection(r_clips)))

    if not common_clips:
        return {"status": "NO_OVERLAP", "common_clips": 0}, "No overlapping clips found between primary and retest sets."

    matched_pairs = []
    missing_in_retest = 0
    extra_in_retest = 0

    for cid in common_clips:
        p_c = p_df[p_df["clip_id"] == cid]
        r_c = r_df[r_df["clip_id"] == cid]

        p_frames = []
        for _, row in p_c.iterrows():
            try:
                p_frames.append(int(float(row["frame_index"])))
            except (ValueError, TypeError):
                pass

        r_frames = []
        for _, row in r_c.iterrows():
            try:
                r_frames.append(int(float(row["frame_index"])))
            except (ValueError, TypeError):
                pass

        # Greedy closest match
        unmatched_r = list(r_frames)
        for pf in p_frames:
            if not unmatched_r:
                missing_in_retest += 1
                continue
            closest_rf = min(unmatched_r, key=lambda rf: abs(rf - pf))
            diff_frames = abs(closest_rf - pf)
            # Default to 60 fps if manifest not provided
            fps = 60.0
            diff_s = diff_frames / fps

            matched_pairs.append({
                "clip_id": cid,
                "primary_frame": pf,
                "retest_frame": closest_rf,
                "diff_frames": diff_frames,
                "diff_s": diff_s,
                "agrees": diff_s <= tolerance_s,
            })
            unmatched_r.remove(closest_rf)

        extra_in_retest += len(unmatched_r)

    total_matched = len(matched_pairs)
    frame_diffs = [p["diff_frames"] for p in matched_pairs]
    time_diffs = [p["diff_s"] for p in matched_pairs]
    agreements = [p for p in matched_pairs if p["agrees"]]

    mean_frame_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0
    max_frame_diff = int(np.max(frame_diffs)) if frame_diffs else 0
    mean_s_diff = float(np.mean(time_diffs)) if time_diffs else 0.0
    max_s_diff = float(np.max(time_diffs)) if time_diffs else 0.0
    agreement_pct = (len(agreements) / total_matched * 100.0) if total_matched > 0 else 0.0

    stats = {
        "common_clips_count": len(common_clips),
        "total_matched_deliveries": total_matched,
        "agreement_within_half_second_pct": agreement_pct,
        "mean_frame_diff": mean_frame_diff,
        "max_frame_diff": max_frame_diff,
        "mean_seconds_diff": mean_s_diff,
        "max_seconds_diff": max_s_diff,
        "missing_marks_in_retest": missing_in_retest,
        "extra_marks_in_retest": extra_in_retest,
    }

    lines = []
    lines.append("=" * 64)
    lines.append(" STUMPVISION AI -- LABEL CONSISTENCY AUDIT REPORT")
    lines.append("=" * 64)
    lines.append(f"Overlapping Clips Compared:      {len(common_clips)}")
    lines.append(f"Matched Delivery Pairs:         {total_matched}")
    lines.append(f"Agreement within ±0.5s:         {agreement_pct:.1f}% ({len(agreements)}/{total_matched})")
    lines.append(f"Mean Temporal Difference:       {mean_frame_diff:.2f} frames ({mean_s_diff:.3f} s)")
    lines.append(f"Maximum Temporal Difference:    {max_frame_diff} frames ({max_s_diff:.3f} s)")
    lines.append(f"Missing Marks in Retest:        {missing_in_retest}")
    lines.append(f"Extra Marks in Retest:          {extra_in_retest}")
    lines.append("=" * 64)

    return stats, "\n".join(lines)
