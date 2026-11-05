"""
rebuild_from_annotations.py
============================
Reads manual 3-click annotations (manual_annotations.json), rebuilds
the trajectory data (dev_trajectories.json), and re-renders all 13
annotated replay clips with accurate ball arcs + batsman skeleton.

Usage:
  .venv\\Scripts\\python tools\\rebuild_from_annotations.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stumpvision.trajectory.tracker import DeliveryTrajectory, TrajectoryPoint
from stumpvision.trajectory.viz import (
    render_delivery_video_clip,
    compute_smooth_trajectory_points,
)
from stumpvision.trajectory.stumps import classify_stump_line

DATA_DIR = Path("D:/Computer Vision Projects/StumpEye_Footage")
MANUAL_JSON = PROJECT_ROOT / "outputs" / "trajectories" / "manual_annotations.json"
TRAJ_JSON = PROJECT_ROOT / "outputs" / "trajectories" / "dev_trajectories.json"
SUMMARY_CSV = PROJECT_ROOT / "outputs" / "trajectories" / "dev_trajectories_summary.csv"
CLIPS_DIR = PROJECT_ROOT / "outputs" / "trajectories" / "clips"
POSES_JSON = PROJECT_ROOT / "labels" / "m4_poses" / "batsman_pose_dev.json"

FPS = 60.0


def classify_length(bounce_y):
    """Pitching length classification from Camera 1 perspective bounce y-coordinate."""
    if bounce_y >= 485.0:
        return "yorker"
    elif bounce_y >= 460.0:
        return "full"
    elif bounce_y >= 415.0:
        return "good_length"
    else:
        return "short"


def classify_line(bounce_x):
    """Pitching line classification from Camera 1 perspective bounce x-coordinate."""
    if bounce_x < 815.0:
        return "off"
    elif bounce_x > 860.0:
        return "leg"
    else:
        return "middle"


def build_trajectory_from_annotation(ann):
    """Build a DeliveryTrajectory from a manual 3-click annotation dict."""
    d_idx = ann["delivery_idx"]
    clip_id = ann["clip_id"]
    cand_t = ann["candidate_time_s"]

    rel = ann["release"]
    bnc = ann["bounce"]
    bat = ann["bat_contact"]

    # Build trajectory points: Release -> Bounce -> Bat Contact
    # Phase 1: Release to Bounce (2-frame stride)
    rel_f = rel["frame_idx"]
    bnc_f = bnc["frame_idx"]
    bat_f = bat["frame_idx"]

    all_pts = []

    # Release point
    all_pts.append(TrajectoryPoint(
        frame_idx=rel_f,
        time_s=round(rel_f / FPS, 4),
        x=float(rel["x"]),
        y=float(rel["y"]),
        radius=5.0,
        area=25.0,
        confidence=1.0,
    ))

    # Interpolated points from Release to Bounce
    n_phase1 = max(1, bnc_f - rel_f)
    stride1 = max(1, n_phase1 // 6)  # ~6 intermediate points
    for f in range(rel_f + stride1, bnc_f, stride1):
        alpha = float(f - rel_f) / float(n_phase1)
        px = (1.0 - alpha) * rel["x"] + alpha * bnc["x"]
        # Use parabolic y interpolation for gravity-like descent
        # y = y_rel + alpha * (y_bnc - y_rel) + 4 * sag * alpha * (1 - alpha)
        # sag gives slight downward curve
        linear_y = (1.0 - alpha) * rel["y"] + alpha * bnc["y"]
        # Natural projectile flight: straight lateral travel, subtle parabolic gravity descent
        sag = 0.04 * abs(bnc["y"] - rel["y"])
        py = linear_y + sag * 4.0 * alpha * (1.0 - alpha)

        all_pts.append(TrajectoryPoint(
            frame_idx=f,
            time_s=round(f / FPS, 4),
            x=round(float(px), 1),
            y=round(float(py), 1),
            radius=5.0,
            area=25.0,
            confidence=0.95,
        ))

    # Bounce point
    all_pts.append(TrajectoryPoint(
        frame_idx=bnc_f,
        time_s=round(bnc_f / FPS, 4),
        x=float(bnc["x"]),
        y=float(bnc["y"]),
        radius=5.0,
        area=25.0,
        confidence=1.0,
    ))

    # Phase 2: Bounce to Bat Contact
    n_phase2 = max(1, bat_f - bnc_f)
    stride2 = max(1, n_phase2 // 4)  # ~4 intermediate points
    for f in range(bnc_f + stride2, bat_f, stride2):
        alpha = float(f - bnc_f) / float(n_phase2)
        px = (1.0 - alpha) * bnc["x"] + alpha * bat["x"]
        py = (1.0 - alpha) * bnc["y"] + alpha * bat["y"]
        all_pts.append(TrajectoryPoint(
            frame_idx=f,
            time_s=round(f / FPS, 4),
            x=round(float(px), 1),
            y=round(float(py), 1),
            radius=5.0,
            area=25.0,
            confidence=0.95,
        ))

    # Bat contact point
    all_pts.append(TrajectoryPoint(
        frame_idx=bat_f,
        time_s=round(bat_f / FPS, 4),
        x=float(bat["x"]),
        y=float(bat["y"]),
        radius=5.0,
        area=25.0,
        confidence=1.0,
    ))

    # Remove duplicate frame_idx entries (keep latest)
    seen_frames = {}
    for pt in all_pts:
        seen_frames[pt.frame_idx] = pt
    all_pts = sorted(seen_frames.values(), key=lambda p: p.frame_idx)

    bounce_x = float(bnc["x"])
    bounce_y = float(bnc["y"])
    length_cat = classify_length(bounce_y)

    # Calculate ball coordinate at popping crease / bat line (y = 540)
    last_pt = all_pts[-1]
    if last_pt.y >= 530.0:
        x_crease = last_pt.x
    elif len(all_pts) >= 2:
        dy = max(1.0, last_pt.y - bounce_y)
        dx = last_pt.x - bounce_x
        x_crease = last_pt.x + (dx / dy) * (540.0 - last_pt.y)
    else:
        x_crease = bounce_x

    stump_line_id, stump_line_display, line_cat = classify_stump_line(x_crease)

    t_release = all_pts[0].time_s
    t_bounce = round(bnc_f / FPS, 4)
    t_impact = all_pts[-1].time_s

    flight_dur = max(0.01, t_impact - t_release)
    total_dist = sum(
        np.hypot(all_pts[i].x - all_pts[i - 1].x, all_pts[i].y - all_pts[i - 1].y)
        for i in range(1, len(all_pts))
    )

    from stumpvision.trajectory.drs import evaluate_drs, calculate_speed_gun_metrics
    # Pre-calculate speed gun metrics
    dummy_traj = {"points": all_pts, "delivery_idx": d_idx}
    s_rel_kmh, s_rel_mph, s_post_kmh, s_post_mph, drop_pct, pace_cat = calculate_speed_gun_metrics(dummy_traj)

    return DeliveryTrajectory(
        clip_id=clip_id,
        delivery_idx=d_idx,
        candidate_time_s=cand_t,
        points=all_pts,
        t_release=t_release,
        t_bounce=t_bounce,
        t_impact=t_impact,
        bounce_x=bounce_x,
        bounce_y=bounce_y,
        length_classification=length_cat,
        line_classification=line_cat,
        stump_line=stump_line_id,
        stump_line_display=stump_line_display,
        mean_speed_px_per_s=round(float(total_dist / flight_dur), 1),
        estimated_speed_kmh=s_rel_kmh,
        speed_release_kmh=s_rel_kmh,
        speed_release_mph=s_rel_mph,
        is_valid=True,
    )


def main():
    print("=" * 70)
    print("  StumpVision AI — Rebuild Trajectories from Manual Annotations")
    print("=" * 70)
    print()

    if not MANUAL_JSON.exists():
        print("ERROR: Manual annotations not found at", MANUAL_JSON)
        print("Run 'python tools/annotate_ball_3click.py' first.")
        sys.exit(1)

    with open(MANUAL_JSON, "r") as f:
        annotations = json.load(f)
    print("Loaded %d manual annotations." % len(annotations))

    # Load existing trajectory JSON to preserve non-annotated deliveries
    existing_trajs = {}
    if TRAJ_JSON.exists():
        with open(TRAJ_JSON, "r") as f:
            for t in json.load(f):
                existing_trajs[t["delivery_idx"]] = t

    # Load poses
    poses_list = None
    if POSES_JSON.exists():
        with open(POSES_JSON, "r") as f:
            poses_list = json.load(f)
        print("Loaded %d batsman pose entries." % len(poses_list))
    else:
        print("WARNING: Pose data not found at", POSES_JSON)

    # Build trajectories from annotations
    new_trajs = {}
    for ann in annotations:
        d_idx = ann["delivery_idx"]
        traj = build_trajectory_from_annotation(ann)
        new_trajs[d_idx] = traj
        print("  Built trajectory for Delivery #%02d: %s %s (%d pts)" % (
            d_idx, traj.length_classification, traj.line_classification,
            len(traj.points)))

    # Merge: use manual annotation where available, keep existing otherwise
    all_delivery_indices = sorted(set(list(existing_trajs.keys()) + list(new_trajs.keys())))
    final_trajs = []
    for d_idx in all_delivery_indices:
        if d_idx in new_trajs:
            final_trajs.append(new_trajs[d_idx])
        elif d_idx in existing_trajs:
            t = existing_trajs[d_idx]
            pts = [TrajectoryPoint(**p) for p in t["points"]]
            final_trajs.append(DeliveryTrajectory(
                clip_id=t["clip_id"],
                delivery_idx=t["delivery_idx"],
                candidate_time_s=t["candidate_time_s"],
                points=pts,
                t_release=t.get("t_release"),
                t_bounce=t.get("t_bounce"),
                t_impact=t.get("t_impact"),
                bounce_x=t.get("bounce_x"),
                bounce_y=t.get("bounce_y"),
                length_classification=t.get("length_classification", "good_length"),
                line_classification=t.get("line_classification", "middle"),
                estimated_speed_kmh=t.get("estimated_speed_kmh", 0.0),
                is_valid=t.get("is_valid", True),
            ))

    # Save updated trajectory JSON
    print()
    print("Saving %d trajectories to %s ..." % (len(final_trajs), TRAJ_JSON))
    TRAJ_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAJ_JSON, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in final_trajs], f, indent=2)

    # Save summary CSV
    import pandas as pd
    from stumpvision.trajectory.drs import evaluate_drs
    rows = []
    for t in final_trajs:
        flight_dur = (
            round(t.t_impact - t.t_release, 3)
            if (t.t_impact is not None and t.t_release is not None)
            else None
        )
        drs = evaluate_drs(t)
        rows.append({
            "clip_id": t.clip_id,
            "delivery_idx": t.delivery_idx,
            "candidate_time_s": t.candidate_time_s,
            "is_valid": t.is_valid,
            "point_count": len(t.points),
            "flight_duration_s": flight_dur,
            "t_release": t.t_release,
            "t_bounce": t.t_bounce,
            "t_impact": t.t_impact,
            "bounce_x": round(t.bounce_x, 1) if t.bounce_x else None,
            "bounce_y": round(t.bounce_y, 1) if t.bounce_y else None,
            "length": t.length_classification,
            "line": t.line_classification,
            "stump_line": getattr(t, "stump_line", "middle_stump"),
            "stump_line_display": getattr(t, "stump_line_display", "Middle Stump"),
            "speed_kmh": drs.speed_release_kmh,
            "speed_mph": drs.speed_release_mph,
            "speed_post_kmh": drs.speed_post_bounce_kmh,
            "pace_category": drs.pace_category,
            "drs_pitching": drs.pitching_display,
            "drs_impact": drs.impact_display,
            "drs_wickets": drs.wickets_display,
            "drs_decision": drs.decision,
            "drs_reason": drs.decision_reason,
            "arrival_height_cm": drs.arrival_height_cm,
        })
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print("Saved summary CSV to %s" % SUMMARY_CSV)

    # Re-render replay clips
    print()
    print("Re-rendering %d replay clips..." % len(final_trajs))
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    for traj in final_trajs:
        d_idx = traj.delivery_idx
        clip_id = traj.clip_id

        video_path = DATA_DIR / "raw" / "cam1" / ("%s.mp4" % clip_id)
        if not video_path.exists():
            video_path = DATA_DIR / "raw" / "cam1" / ("%s.MP4" % clip_id)

        out_clip = CLIPS_DIR / ("delivery_%02d_annotated.mp4" % d_idx)

        success = render_delivery_video_clip(
            video_path=video_path,
            trajectory=traj,
            output_video_path=out_clip,
            poses_list=poses_list,
            draw_pitch_map=False,
        )
        print("  Delivery #%02d: %s" % (d_idx, "OK" if success else "FAILED"))

    print()
    print("=" * 70)
    print("  REBUILD COMPLETE")
    print("  Trajectories: %s" % TRAJ_JSON)
    print("  Summary CSV:  %s" % SUMMARY_CSV)
    print("  Replay Clips: %s" % CLIPS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
