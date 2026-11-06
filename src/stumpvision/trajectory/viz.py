"""
viz.py
======
Visualization tools for cricket ball delivery pitch trajectories, 2D bounce heatmaps,
and broadcast-quality animated replay video generation with dynamic batsman pose tracking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import shutil
import subprocess
import cv2
import numpy as np
import pandas as pd

from stumpvision.trajectory.tracker import DeliveryTrajectory, TrajectoryPoint


# ---------------------------------------------------------------------------
# Camera 1 Broadcast Pitch Map & Length Zones Calibration
# ---------------------------------------------------------------------------

PITCH_ZONE_CONFIG = [
    # (Zone Name, y_start, y_end, BGR color, description)
    ("FULL TOSS", 540, 575, (80, 80, 225), "Above waist height"),
    ("FULL LENGTH", 575, 630, (40, 205, 245), "Around waist to knee"),
    ("GOOD LENGTH", 630, 715, (60, 210, 85), "Ideal bounce area"),
    ("SHORT OF GOOD", 715, 815, (235, 145, 45), "Rises towards chest"),
    ("SHORT LENGTH", 815, 940, (215, 80, 160), "Bounces shoulder/head"),
]


def get_camera1_pitch_bounds(y: float) -> Tuple[float, float]:
    """
    Compute perspective left and right x boundaries of the cricket pitch turf
    in Camera 1 (Bowler-end stump camera view) for any vertical y coordinate.
    """
    alpha = (y - 540.0) / (1000.0 - 540.0)
    xl = 835.0 + alpha * (270.0 - 835.0)
    xr = 965.0 + alpha * (1630.0 - 965.0)
    return xl, xr


def render_pitch_zones_overlay(
    frame: np.ndarray,
    alpha: float = 0.35,
    draw_labels: bool = True,
    draw_hud_cards: bool = True,
    active_zone_name: Optional[str] = None,
) -> np.ndarray:
    """
    Render broadcast-quality international cricket pitch mapping onto Camera 1 footage.
    Overlays calibrated perspective trapezoids for the 5 official length zones:
      1. Full Toss (Coral Red)
      2. Full Length (Golden Yellow)
      3. Good Length (Emerald Green)
      4. Short of Good (Sky Blue)
      5. Short Length (Regal Purple)
    Follows perspective geometry of the stump camera without writing meter distances.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    canvas = frame.copy()
    overlay = frame.copy()

    scale_x = w / 1920.0
    scale_y = h / 1080.0

    def _proj(x: float, y: float) -> Tuple[int, int]:
        return int(round(x * scale_x)), int(round(y * scale_y))

    # 1. Fill translucent colored perspective polygons
    for name, y1, y2, color, _ in PITCH_ZONE_CONFIG:
        xl1, xr1 = get_camera1_pitch_bounds(y1)
        xl2, xr2 = get_camera1_pitch_bounds(y2)

        pts = np.array([
            _proj(xl1, y1),
            _proj(xr1, y1),
            _proj(xr2, y2),
            _proj(xl2, y2),
        ], dtype=np.int32)

        # Highlight active zone if specified (e.g. current ball bounce zone)
        zone_alpha = 0.52 if (active_zone_name and active_zone_name.upper().replace('_', ' ') in name) else alpha
        cv2.fillPoly(overlay, [pts], color)

    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # 2. Draw crisp white crease & boundary lines
    for name, y1, y2, _, _ in PITCH_ZONE_CONFIG:
        xl1, xr1 = get_camera1_pitch_bounds(y1)
        xl2, xr2 = get_camera1_pitch_bounds(y2)

        pt_tl = _proj(xl1, y1)
        pt_tr = _proj(xr1, y1)
        pt_bl = _proj(xl2, y2)
        pt_br = _proj(xr2, y2)

        # Horizontal divider lines
        cv2.line(canvas, pt_tl, pt_tr, (255, 255, 255), max(1, int(round(2 * scale_y))), cv2.LINE_AA)
        cv2.line(canvas, pt_bl, pt_br, (255, 255, 255), max(1, int(round(2 * scale_y))), cv2.LINE_AA)

        # Left and right boundary lines
        cv2.line(canvas, pt_tl, pt_bl, (255, 255, 255), max(1, int(round(2 * scale_x))), cv2.LINE_AA)
        cv2.line(canvas, pt_tr, pt_br, (255, 255, 255), max(1, int(round(2 * scale_x))), cv2.LINE_AA)

        # Centered zone labels on pitch turf
        if draw_labels:
            mid_y = int((y1 + y2) / 2.0)
            mid_xl, mid_xr = get_camera1_pitch_bounds(mid_y)
            cx = int((mid_xl + mid_xr) / 2.0)

            f_scale = (0.40 if (y2 - y1) < 45 else (0.50 if mid_y < 700 else 0.65)) * min(scale_x, scale_y)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, f_scale, 2)
            pt_center = _proj(cx, mid_y)
            tx = pt_center[0] - tw // 2
            ty = pt_center[1] + th // 2

            # Drop-shadow + white text
            cv2.putText(canvas, name, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Top-left Broadcast Title Card
    if draw_hud_cards and w >= 800:
        card_x1, card_y1 = int(round(30 * scale_x)), int(round(30 * scale_y))
        card_x2, card_y2 = int(round(280 * scale_x)), int(round(105 * scale_y))

        bg_card = canvas.copy()
        cv2.rectangle(bg_card, (card_x1, card_y1), (card_x2, card_y2), (15, 18, 24), -1)
        cv2.addWeighted(bg_card, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (60, 70, 85), 1)

        cv2.putText(canvas, "Pitch Map", (card_x1 + 18, card_y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70 * min(scale_x, scale_y), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "(Bowler's View)", (card_x1 + 18, card_y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50 * min(scale_x, scale_y), (180, 195, 210), 1, cv2.LINE_AA)

    # 4. Right Broadcast Legend Card
    if draw_hud_cards and w >= 1200:
        leg_x1 = w - int(round(330 * scale_x))
        leg_x2 = w - int(round(30 * scale_x))
        leg_y1 = int(round(160 * scale_y))
        leg_y2 = int(round(480 * scale_y))

        bg_leg = canvas.copy()
        cv2.rectangle(bg_leg, (leg_x1, leg_y1), (leg_x2, leg_y2), (15, 18, 24), -1)
        cv2.addWeighted(bg_leg, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (leg_x1, leg_y1), (leg_x2, leg_y2), (60, 70, 85), 1)

        cv2.putText(canvas, "Length Zones", (leg_x1 + 22, leg_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70 * min(scale_x, scale_y), (255, 255, 255), 2, cv2.LINE_AA)

        start_y = leg_y1 + int(round(70 * scale_y))
        step_y = int(round(50 * scale_y))
        chip_w = int(round(28 * scale_x))
        chip_h = int(round(24 * scale_y))

        for name, _, _, color, desc in PITCH_ZONE_CONFIG:
            # Color Chip
            c_y1 = start_y - chip_h // 2
            c_y2 = start_y + chip_h // 2
            c_x1 = leg_x1 + 22
            c_x2 = c_x1 + chip_w
            cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), color, -1)
            cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), (255, 255, 255), 1)

            # Title & Description
            title_txt = name.title()
            cv2.putText(canvas, title_txt, (c_x2 + 12, start_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * min(scale_x, scale_y), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, desc, (c_x2 + 12, start_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * min(scale_x, scale_y), (160, 175, 190), 1, cv2.LINE_AA)

            start_y += step_y

    return canvas




# ---------------------------------------------------------------------------
# Biomechanical Batsman Pose Helpers
# ---------------------------------------------------------------------------

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # head / face
    (5, 6),                                       # shoulders
    (5, 7), (7, 9),                               # left arm
    (6, 8), (8, 10),                              # right arm
    (5, 11), (6, 12), (11, 12),                   # torso
    (11, 13), (13, 15),                           # left leg
    (12, 14), (14, 16),                           # right leg
]


def build_pose_timeline(
    clip_id: str,
    candidate_time_s: float,
    poses_list: Optional[List[dict]] = None,
    fps: float = 60.0,
):
    """
    Build an interpolated batsman pose timeline for the delivery event.
    The skeleton dynamically moves with the batsman from Stance -> Backlift -> Shot Contact,
    and disappears immediately after the shot is played (f > f_shot + 6).
    """
    if not poses_list:
        return lambda f: None

    cand_f = int(round(candidate_time_s * fps))
    c_poses = []
    for p in poses_list:
        fname = p.get("file_name", "")
        if fname.startswith(clip_id) and "_f" in fname:
            try:
                fidx = int(fname.split("_f")[1].split(".")[0])
                if abs(fidx - cand_f) < 180 and "keypoints" in p:
                    c_poses.append((fidx, p["keypoints"]))
            except (ValueError, IndexError):
                pass

    c_poses.sort(key=lambda x: x[0])
    if not c_poses:
        return lambda f: None

    first_f = c_poses[0][0]
    last_f = c_poses[-1][0]  # Shot contact frame

    def get_pose(f_idx: int):
        # 1. Before batsman takes active stance: no skeleton
        if f_idx < first_f - 30:
            return None

        # 2. After shot is played: skeleton disappears!
        if f_idx > last_f + 6:
            return None

        # 3. Leading into stance
        if f_idx <= first_f:
            return c_poses[0][1]

        # 4. Interpolate smoothly between Stance -> Backlift -> Shot
        for i in range(len(c_poses) - 1):
            f1, k1 = c_poses[i]
            f2, k2 = c_poses[i + 1]
            if f1 <= f_idx <= f2:
                span = max(1, f2 - f1)
                alpha = (f_idx - f1) / float(span)
                interp_k = []
                for kp1, kp2 in zip(k1, k2):
                    x = (1.0 - alpha) * kp1[0] + alpha * kp2[0]
                    y = (1.0 - alpha) * kp1[1] + alpha * kp2[1]
                    conf = min(kp1[2], kp2[2])
                    interp_k.append([x, y, conf])
                return interp_k

        return c_poses[-1][1]

    return get_pose


def draw_batsman_skeleton(
    canvas: np.ndarray,
    keypoints: List[List[float]],
    line_thickness: int = 1,
    joint_radius: int = 2,
):
    """
    Draw ultra-thin, delicate, broadcast-grade skeleton.
    Focuses on athletic posture (shoulders, arms, torso, legs) without facial clutter
    so the batsman's head, jersey, and pads remain completely visible.
    """
    if not keypoints or len(keypoints) < 17:
        return

    # Athletic posture pairs (shoulders, arms, torso, legs)
    athletic_pairs = [
        (5, 6),                                       # shoulders
        (5, 7), (7, 9),                               # left arm
        (6, 8), (8, 10),                              # right arm
        (5, 11), (6, 12), (11, 12),                   # torso
        (11, 13), (13, 15),                           # left leg
        (12, 14), (14, 16),                           # right leg
    ]

    # Draw thin bones
    for p1, p2 in athletic_pairs:
        if p1 < len(keypoints) and p2 < len(keypoints):
            k1, k2 = keypoints[p1], keypoints[p2]
            if len(k1) >= 3 and len(k2) >= 3 and k1[2] > 0.2 and k2[2] > 0.2:
                pt1 = (int(round(k1[0])), int(round(k1[1])))
                pt2 = (int(round(k2[0])), int(round(k2[1])))

                # Arm / Torso / Leg specific styling with delicate luminous colors
                if p1 in (5, 6, 7, 8, 9, 10) or p2 in (7, 8, 9, 10):
                    bone_col = (0, 255, 255)       # Yellow for arms/wrists
                elif p1 in (13, 14, 15, 16) or p2 in (13, 14, 15, 16):
                    bone_col = (0, 185, 255)       # Amber for legs/ankles
                else:
                    bone_col = (255, 230, 0)       # Cyan for torso/shoulders

                cv2.line(canvas, pt1, pt2, bone_col, line_thickness, cv2.LINE_AA)

    # Draw tiny elegant joint nodes only on major anatomical joints (5 to 16)
    for idx in range(5, min(17, len(keypoints))):
        pt = keypoints[idx]
        if len(pt) >= 3 and pt[2] > 0.2:
            cx, cy = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(canvas, (cx, cy), joint_radius, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), max(1, joint_radius - 1), (255, 255, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Smooth Ball Trajectory Helpers
# ---------------------------------------------------------------------------

def compute_smooth_trajectory_points(
    points: List[TrajectoryPoint],
    bounce_x: Optional[float] = None,
    bounce_y: Optional[float] = None,
    samples_per_segment: int = 4,
) -> List[Tuple[float, float, float]]:
    """
    Interpolate dense, sub-pixel points along the ball delivery arc (x, y, frame_idx).
    Supports two physical flight phases:
      Phase 1: Bowler Hand Release -> Pitch Turf Bounce (parabolic descent under gravity)
      Phase 2: Pitch Turf Bounce -> Batsman Bat Contact (rebound rise into the bat)
    Provides smooth Hawk-Eye trajectory visualization without angular jumps.
    """
    if not points:
        return []
    if len(points) < 3:
        return [(p.x, p.y, float(p.frame_idx)) for p in points]

    def _interp_pts(pts_sub: List[TrajectoryPoint], use_poly: bool = True) -> List[Tuple[float, float, float]]:
        if len(pts_sub) < 2:
            return [(p.x, p.y, float(p.frame_idx)) for p in pts_sub]
        sub_f = np.array([p.frame_idx for p in pts_sub], dtype=np.float64)
        sub_x = np.array([p.x for p in pts_sub], dtype=np.float64)
        sub_y = np.array([p.y for p in pts_sub], dtype=np.float64)

        dense_f = np.linspace(sub_f[0], sub_f[-1], (len(pts_sub) - 1) * samples_per_segment + 1)
        # Cricket ball in mid-air cannot turn laterally: straight, monotonic drift in x
        dense_x = np.linspace(sub_x[0], sub_x[-1], len(dense_f))

        if use_poly and len(pts_sub) >= 3:
            try:
                poly_y = np.polyfit(sub_f, sub_y, deg=2)
                dense_y = np.polyval(poly_y, dense_f)
            except Exception:
                dense_y = np.linspace(sub_y[0], sub_y[-1], len(dense_f))
        else:
            dense_y = np.linspace(sub_y[0], sub_y[-1], len(dense_f))

        return list(zip(dense_x, dense_y, dense_f))

    # Detect if trajectory spans across turf bounce to bat contact
    bnc_idx = -1
    if bounce_x is not None and bounce_y is not None:
        bnc_idx = int(np.argmin([np.hypot(p.x - bounce_x, p.y - bounce_y) for p in points]))

    if 1 <= bnc_idx < len(points) - 1:
        # Two-phase trajectory: Release -> Bounce -> Bat Contact
        seg1 = points[: bnc_idx + 1]
        seg2 = points[bnc_idx:]

        # Physical momentum safeguard: post-bounce ball cannot reverse lateral direction mid-air
        if len(seg1) >= 2 and len(seg2) >= 2:
            dx_pre = seg1[-1].x - seg1[0].x
            dx_post = seg2[-1].x - seg2[0].x
            if dx_pre * dx_post < 0:
                dy_post = max(1.0, seg2[-1].y - seg2[0].y)
                natural_dx = (dx_pre / max(1.0, seg1[-1].y - seg1[0].y)) * dy_post * 0.7
                target_x = seg2[0].x + natural_dx
                seg2 = [
                    TrajectoryPoint(
                        frame_idx=p.frame_idx,
                        time_s=p.time_s,
                        x=float(seg2[0].x + (target_x - seg2[0].x) * (i / max(1, len(seg2) - 1))),
                        y=p.y,
                        radius=p.radius,
                        area=p.area,
                        confidence=p.confidence,
                    )
                    for i, p in enumerate(seg2)
                ]

        dense1 = _interp_pts(seg1, use_poly=True)
        dense2 = _interp_pts(seg2, use_poly=False)
        return dense1 + dense2[1:]

    # Single-phase interpolation fallback
    return _interp_pts(points, use_poly=True)


# ---------------------------------------------------------------------------
# Frame & Replay Renderers
# ---------------------------------------------------------------------------

def render_trajectory_on_frame(
    frame: np.ndarray,
    trajectory: DeliveryTrajectory,
    draw_hud: bool = True,
    draw_pitch_map: bool = False,
) -> np.ndarray:
    """
    Draw broadcast-quality 2D pitch trajectory arc and telemetry HUD on a frame.
    Tracks ball flight: Bowler Release -> Pitch Turf Bounce -> Batsman Bat Contact.
    """
    canvas = frame.copy()

    if draw_pitch_map:
        canvas = render_pitch_zones_overlay(
            canvas,
            alpha=0.32,
            draw_labels=True,
            draw_hud_cards=False,
            active_zone_name=trajectory.length_classification,
        )

    if not trajectory.points or len(trajectory.points) < 2:
        if draw_hud:
            cv2.putText(
                canvas,
                f"Delivery #{trajectory.delivery_idx}: No Trajectory Tracked",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas



    dense_pts = compute_smooth_trajectory_points(
        trajectory.points,
        bounce_x=trajectory.bounce_x,
        bounce_y=trajectory.bounce_y,
        samples_per_segment=5,
    )
    n_dense = len(dense_pts)

    # 1. Dual-layer glowing trajectory ribbon (outer halo + inner laser core)
    overlay = canvas.copy()
    for i in range(1, n_dense):
        pt1 = (int(round(dense_pts[i - 1][0])), int(round(dense_pts[i - 1][1])))
        pt2 = (int(round(dense_pts[i][0])), int(round(dense_pts[i][1])))

        frac = i / float(n_dense)
        if frac < 0.5:
            # Red to Yellow
            sub = frac * 2.0
            core_col = (0, int(round(255 * sub)), 255)
            halo_col = (0, int(round(160 * sub)), 200)
        else:
            # Yellow to Cyan
            sub = (frac - 0.5) * 2.0
            core_col = (int(round(255 * sub)), 255, int(round(255 * (1.0 - sub))))
            halo_col = (int(round(180 * sub)), 200, int(round(180 * (1.0 - sub))))

        # Outer glowing halo
        cv2.line(overlay, pt1, pt2, halo_col, 8, cv2.LINE_AA)
        # Inner bright core
        cv2.line(canvas, pt1, pt2, core_col, 3, cv2.LINE_AA)

    # Blend outer halo
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    # 2. Glowing Ball Head at bat contact position (end of flight)
    lead_pt = (int(round(dense_pts[-1][0])), int(round(dense_pts[-1][1])))
    cv2.circle(canvas, lead_pt, 8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, lead_pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    # 2.5 Broadcast Badge for Stump Line (no artificial poles drawn over batsman body)
    from stumpvision.trajectory.stumps import draw_batter_stumps
    stump_line_str = getattr(trajectory, "stump_line_display", None)
    if not stump_line_str:
        stump_line_str = f"{trajectory.line_classification.title()} Stump"
    canvas = draw_batter_stumps(
        canvas,
        draw_guidelines=False,
        draw_stump_poles=False,
        draw_badge=draw_hud,
        stump_line_label=stump_line_str,
    )

    # 2.6 Hawk-Eye Virtual Predictor Extrapolation Path
    from stumpvision.trajectory.drs import evaluate_drs
    drs = evaluate_drs(trajectory)

    # Draw extrapolated path to wickets if trajectory points exist
    if len(dense_pts) >= 2:
        impact_pt = (int(round(dense_pts[-1][0])), int(round(dense_pts[-1][1])))
        stump_pt = (int(round(drs.projected_stump_x)), int(round(drs.projected_stump_y)))

        # Translucent dashed Hawk-Eye projection beam
        overlay_proj = canvas.copy()
        proj_col = (0, 220, 255) if drs.wickets == "hitting" else ((0, 160, 240) if drs.wickets == "umpires_call" else (180, 120, 255))
        # Draw dotted line from impact to projected stumps
        num_dots = 12
        for k in range(num_dots):
            t_k1 = k / float(num_dots)
            t_k2 = (k + 0.6) / float(num_dots)
            xk1 = int(round(impact_pt[0] + (stump_pt[0] - impact_pt[0]) * t_k1))
            yk1 = int(round(impact_pt[1] + (stump_pt[1] - impact_pt[1]) * t_k1))
            xk2 = int(round(impact_pt[0] + (stump_pt[0] - impact_pt[0]) * t_k2))
            yk2 = int(round(impact_pt[1] + (stump_pt[1] - impact_pt[1]) * t_k2))
            cv2.line(overlay_proj, (xk1, yk1), (xk2, yk2), proj_col, 2, cv2.LINE_AA)

        # Projected target circle at stumps line
        cv2.circle(overlay_proj, stump_pt, 6, proj_col, 1, cv2.LINE_AA)
        cv2.circle(overlay_proj, stump_pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay_proj, 0.70, canvas, 0.30, 0, canvas)

    # 3. Telemetry HUD box
    if draw_hud:
        hud_bg_x1, hud_bg_y1 = 25, 25
        hud_bg_x2, hud_bg_y2 = 450, 185
        overlay = canvas.copy()
        cv2.rectangle(overlay, (hud_bg_x1, hud_bg_y1), (hud_bg_x2, hud_bg_y2), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (hud_bg_x1, hud_bg_y1), (hud_bg_x2, hud_bg_y2), (0, 200, 100), 2)

        flight_time_s = (
            max(0.0, trajectory.t_impact - trajectory.t_release)
            if (trajectory.t_impact is not None and trajectory.t_release is not None)
            else 0.0
        )

        title = f"DELIVERY #{trajectory.delivery_idx:02d}  [Clip {trajectory.clip_id[-6:]}]"
        info1 = f"Pitch Length:  {trajectory.length_classification.replace('_', ' ').title()}"
        info2 = f"Stump Line:    {stump_line_str}"
        info3 = f"Radar Speed:   {drs.speed_release_kmh:.1f} km/h ({drs.speed_release_mph:.1f} mph)"
        info4 = f"StumpEye LBW:  {drs.decision} ({drs.wickets_display})"

        cv2.putText(canvas, title, (42, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, info1, (42, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, info2, (42, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, info3, (42, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, info4, (42, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 200, 100), 1, cv2.LINE_AA)

    return canvas




def draw_2d_pitch_map(
    summary_df: Optional[Any] = None,
    selected_delivery_idx: Optional[int] = None,
    width: int = 600,
    height: int = 860,
) -> np.ndarray:
    """
    Render a 2D aerial cricket pitch diagram with delivery bounce point heatmaps.
    """
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 35  # Dark charcoal theme

    # Pitch turf rectangle (proportional to 22 yards)
    px1, py1 = 110, 80
    px2, py2 = width - 110, height - 90
    turf_w = px2 - px1
    turf_h = py2 - py1

    # Turf grass
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (185, 215, 180), -1)
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (110, 150, 100), 2)

    # Creases
    bowling_crease_y = py1 + 60
    popping_crease_y = py2 - 80
    cv2.line(canvas, (px1 - 20, bowling_crease_y), (px2 + 20, bowling_crease_y), (255, 255, 255), 2)
    cv2.line(canvas, (px1 - 20, popping_crease_y), (px2 + 20, popping_crease_y), (255, 255, 255), 3)

    # Stumps
    cx = (px1 + px2) // 2
    # Bowler end stumps
    for offset in [-14, 0, 14]:
        cv2.circle(canvas, (cx + offset, bowling_crease_y - 12), 4, (40, 60, 200), -1)

    # Striker Batter End Stumps (Off, Middle, Leg)
    # Off: cx - 14, Mid: cx, Leg: cx + 14
    stump_offsets = [(-14, "OFF", (100, 200, 255)), (0, "MID", (80, 240, 120)), (14, "LEG", (255, 180, 80))]
    for offset, label, col in stump_offsets:
        cv2.circle(canvas, (cx + offset, popping_crease_y + 12), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx + offset, popping_crease_y + 12), 4, col, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (cx + offset - 8, popping_crease_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)

    # Wide Guidelines at Striker End
    wide_off_px = int(round(cx - (turf_w * 0.32)))
    wide_leg_px = int(round(cx + (turf_w * 0.22)))
    cv2.line(canvas, (wide_off_px, popping_crease_y - 25), (wide_off_px, popping_crease_y + 15), (70, 180, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WIDE (OFF)", (wide_off_px - 45, popping_crease_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 180, 245), 1, cv2.LINE_AA)
    cv2.line(canvas, (wide_leg_px, popping_crease_y - 25), (wide_leg_px, popping_crease_y + 15), (70, 180, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WIDE (LEG)", (wide_leg_px + 6, popping_crease_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 180, 245), 1, cv2.LINE_AA)

    # Zone dividers
    y_full = popping_crease_y - 120
    y_good = popping_crease_y - 280
    y_short = popping_crease_y - 450

    cv2.line(canvas, (px1, y_full), (px2, y_full), (140, 170, 135), 1, cv2.LINE_AA)
    cv2.line(canvas, (px1, y_good), (px2, y_good), (140, 170, 135), 1, cv2.LINE_AA)
    cv2.line(canvas, (px1, y_short), (px2, y_short), (140, 170, 135), 1, cv2.LINE_AA)

    # Labels
    cv2.putText(canvas, "STUMPEYE - 2D PITCH MAP", (180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "YORKER / FULL", (px1 + 8, popping_crease_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 90, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GOOD LENGTH", (px1 + 8, y_full - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 90, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "SHORT PITCH", (px1 + 8, y_good - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 90, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "BATSMAN STRIKER CREASE", (px1 + 70, popping_crease_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Plot delivery bounce points if summary_df provided
    color_map = {
        "yorker": (0, 0, 220),       # Red
        "full": (0, 140, 255),       # Orange
        "good_length": (30, 180, 40),# Green
        "short": (220, 100, 30),     # Blue
    }

    if summary_df is not None and not summary_df.empty:
        for _, row in summary_df.iterrows():
            if not row.get("is_valid", False):
                continue
            bx = row.get("bounce_x")
            by = row.get("bounce_y")
            if bx is None or by is None or np.isnan(bx) or np.isnan(by):
                continue

            d_idx = int(row.get("delivery_idx", 1))
            length_cat = str(row.get("length", "good_length")).lower()
            dot_color = color_map.get(length_cat, (0, 255, 255))

            # Centerline at vertical pitch position by (running from bowler 735 to middle stump 901.5)
            center_x_at_by = 901.5 - 0.4269 * (540.0 - by)
            offset_from_center = bx - center_x_at_by
            # Scale: ~43 px in Camera 1 corresponds to wide guideline (32% of turf width)
            pitch_x = int(round(cx + (offset_from_center / 42.8) * (turf_w * 0.32)))
            pitch_x = max(px1 + 10, min(px2 - 10, pitch_x))

            norm_y = (560.0 - by) / (560.0 - 240.0)
            pitch_y = int(round(popping_crease_y - norm_y * (popping_crease_y - y_short - 30)))
            pitch_y = max(py1 + 80, min(popping_crease_y, pitch_y))

            is_sel = (selected_delivery_idx is not None and d_idx == selected_delivery_idx)

            if is_sel:
                cv2.circle(canvas, (pitch_x, pitch_y), 16, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(canvas, (pitch_x, pitch_y), 9, dot_color, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (pitch_x, pitch_y), 10, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(canvas, (pitch_x, pitch_y), 8, dot_color, -1, cv2.LINE_AA)

            num_str = f"{d_idx}"
            cv2.putText(canvas, num_str, (pitch_x - 4, pitch_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend at bottom
    leg_y = height - 35
    leg_items = [
        ("Yorker", (0, 0, 220)),
        ("Full", (0, 140, 255)),
        ("Good Length", (30, 180, 40)),
        ("Short", (220, 100, 30)),
    ]
    cur_x = 50
    for label, col in leg_items:
        cv2.circle(canvas, (cur_x, leg_y), 6, col, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (cur_x + 12, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cur_x += 135

    return canvas




def render_delivery_video_clip(
    video_path: Union[str, Path],
    trajectory: DeliveryTrajectory,
    output_video_path: Union[str, Path],
    poses_list: Optional[List[dict]] = None,
    pre_buffer_s: float = 1.0,
    post_buffer_s: float = 1.2,
    fps: float = 60.0,
    draw_pitch_map: bool = False,
) -> bool:
    """
    Render an enhanced broadcast MP4 replay:
    1. Continuous smooth glowing ball trajectory arc tracking from bowler release -> turf bounce -> batsman bat contact.
    2. Dynamic batsman skeleton that tracks bodily motion until shot execution, then disappears cleanly.
    3. Complete trajectory arc remains visible throughout replay after bat contact.
    4. Clean broadcast HUD telemetry box.
    """
    video_path = Path(video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    start_f = max(0, int(round((trajectory.candidate_time_s - pre_buffer_s) * fps)))
    end_f = int(round((trajectory.candidate_time_s + post_buffer_s) * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    # 1. Build dynamic batsman pose interpolator
    get_dynamic_pose = build_pose_timeline(
        clip_id=trajectory.clip_id,
        candidate_time_s=trajectory.candidate_time_s,
        poses_list=poses_list,
        fps=fps,
    )

    # 2. Build dense smooth trajectory points (Release -> Bounce -> Bat Contact)
    pts = trajectory.points or []
    dense_pts = compute_smooth_trajectory_points(
        pts,
        bounce_x=trajectory.bounce_x,
        bounce_y=trajectory.bounce_y,
        samples_per_segment=4,
    )

    bounce_f = int(round(trajectory.t_bounce * fps)) if trajectory.t_bounce else (pts[-1].frame_idx - 6 if pts else int(round(trajectory.candidate_time_s * fps)))
    impact_f = int(round(trajectory.t_impact * fps)) if trajectory.t_impact else (pts[-1].frame_idx if pts else int(round(trajectory.candidate_time_s * fps)))

    has_ffmpeg = shutil.which("ffmpeg") is not None
    proc = None
    writer = None

    if has_ffmpeg:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(int(fps)),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            str(output_video_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    f_idx = start_f
    while f_idx <= end_f:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional Pitch Zones (disabled by default to avoid intersecting with batsman)
        if draw_pitch_map:
            frame = render_pitch_zones_overlay(
                frame,
                alpha=0.30,
                draw_labels=True,
                draw_hud_cards=False,
                active_zone_name=trajectory.length_classification,
            )

        # A. Dynamic Batsman Skeleton: Moves with body, disappears after shot!
        # Rendered with thin, razor-sharp styling so batsman body/pads are clearly visible.
        curr_pose = get_dynamic_pose(f_idx)
        if curr_pose:
            draw_batsman_skeleton(frame, curr_pose, line_thickness=1, joint_radius=2)

        # B. Smooth Progressive Trajectory Arc (Release -> Pitch Bounce -> Bat Contact)
        # When ball reaches bat contact (impact_f), keep full trajectory visible on screen for rest of replay
        if f_idx < impact_f:
            active_dense = [p for p in dense_pts if p[2] <= float(f_idx)]
        else:
            active_dense = dense_pts

        n_act = len(active_dense)
        if n_act >= 2:
            overlay_arc = frame.copy()
            for i in range(1, n_act):
                pt1 = (int(round(active_dense[i - 1][0])), int(round(active_dense[i - 1][1])))
                pt2 = (int(round(active_dense[i][0])), int(round(active_dense[i][1])))

                frac = i / max(1.0, float(len(dense_pts)))
                if frac < 0.5:
                    sub = frac * 2.0
                    core_col = (0, int(round(255 * sub)), 255)
                    halo_col = (0, int(round(160 * sub)), 200)
                else:
                    sub = (frac - 0.5) * 2.0
                    core_col = (int(round(255 * sub)), 255, int(round(255 * (1.0 - sub))))
                    halo_col = (int(round(180 * sub)), 200, int(round(180 * (1.0 - sub))))

                # Outer soft glow
                cv2.line(overlay_arc, pt1, pt2, halo_col, 8, cv2.LINE_AA)
                # Inner bright beam
                cv2.line(frame, pt1, pt2, core_col, 3, cv2.LINE_AA)

            cv2.addWeighted(overlay_arc, 0.40, frame, 0.60, 0, frame)

            # Leading glowing ball head (travels with ball flight, rests at bat contact)
            lead_x, lead_y, _ = active_dense[-1]
            lead_pt = (int(round(lead_x)), int(round(lead_y)))
            cv2.circle(frame, lead_pt, 8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, lead_pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # B.2 Hawk-Eye Virtual Predictor Extrapolation Path (after impact)
        from stumpvision.trajectory.drs import evaluate_drs
        drs = evaluate_drs(trajectory)

        if f_idx >= impact_f and len(dense_pts) >= 2:
            impact_pt = (int(round(dense_pts[-1][0])), int(round(dense_pts[-1][1])))
            stump_pt = (int(round(drs.projected_stump_x)), int(round(drs.projected_stump_y)))

            overlay_proj = frame.copy()
            proj_col = (0, 220, 255) if drs.wickets == "hitting" else ((0, 160, 240) if drs.wickets == "umpires_call" else (180, 120, 255))
            num_dots = 12
            for k in range(num_dots):
                t_k1 = k / float(num_dots)
                t_k2 = (k + 0.6) / float(num_dots)
                xk1 = int(round(impact_pt[0] + (stump_pt[0] - impact_pt[0]) * t_k1))
                yk1 = int(round(impact_pt[1] + (stump_pt[1] - impact_pt[1]) * t_k1))
                xk2 = int(round(impact_pt[0] + (stump_pt[0] - impact_pt[0]) * t_k2))
                yk2 = int(round(impact_pt[1] + (stump_pt[1] - impact_pt[1]) * t_k2))
                cv2.line(overlay_proj, (xk1, yk1), (xk2, yk2), proj_col, 2, cv2.LINE_AA)

            cv2.circle(overlay_proj, stump_pt, 6, proj_col, 1, cv2.LINE_AA)
            cv2.circle(overlay_proj, stump_pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay_proj, 0.70, frame, 0.30, 0, frame)

        # C. Broadcast Telemetry Badge for Stump Line (no artificial poles drawn over batsman body)
        from stumpvision.trajectory.stumps import draw_batter_stumps
        stump_line_display = getattr(trajectory, "stump_line_display", f"{trajectory.line_classification.title()} Stump")
        frame = draw_batter_stumps(
            frame,
            draw_guidelines=False,
            draw_stump_poles=False,
            draw_badge=True,
            stump_line_label=stump_line_display,
        )

        # D. Broadcast Telemetry HUD box
        overlay_hud = frame.copy()
        cv2.rectangle(overlay_hud, (30, 30), (460, 180), (15, 18, 24), -1)
        cv2.addWeighted(overlay_hud, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (30, 30), (460, 180), (0, 220, 120), 2)

        flight_time_s = (
            max(0.0, trajectory.t_impact - trajectory.t_release)
            if (trajectory.t_impact is not None and trajectory.t_release is not None)
            else 0.0
        )
        title = f"DELIVERY #{trajectory.delivery_idx:02d}  [Clip {trajectory.clip_id[-6:]}]"
        info1 = f"Pitch Length:  {trajectory.length_classification.replace('_', ' ').title()}"
        info2 = f"Stump Line:    {stump_line_display}"
        info3 = f"Radar Speed:   {drs.speed_release_kmh:.1f} km/h ({drs.speed_release_mph:.1f} mph)"
        info4 = f"StumpEye LBW:  {drs.decision} ({drs.wickets_display})"

        cv2.putText(frame, title, (45, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, info1, (45, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, info2, (45, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, info3, (45, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, info4, (45, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 200, 100), 1, cv2.LINE_AA)

        if proc and proc.stdin:
            proc.stdin.write(frame.tobytes())
        elif writer:
            writer.write(frame)

        f_idx += 1

    cap.release()
    if proc and proc.stdin:
        proc.stdin.close()
        proc.wait()
    elif writer:
        writer.release()

    return output_video_path.exists()


def draw_pitch_heatmap_diagram(
    summary_df: Optional[Any] = None,
    selected_delivery_idx: Optional[int] = None,
    width: int = 680,
    height: int = 240,
) -> np.ndarray:
    """
    Render a modern dark-theme horizontal pitch diagram with kernel density
    bounce heatmap and highlighted delivery bounce dots.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 23, 36)  # Deep dark navy #141724

    px1, py1 = 70, 40
    px2, py2 = width - 80, height - 40
    pitch_h = py2 - py1
    pitch_cy = (py1 + py2) // 2

    # Subtle grid lines
    for gx in range(px1, px2 + 1, 35):
        cv2.line(canvas, (gx, py1), (gx, py2), (28, 34, 52), 1)
    for gy in range(py1, py2 + 1, 35):
        cv2.line(canvas, (px1, gy), (px2, gy), (28, 34, 52), 1)

    # Pitch boundary
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (60, 75, 105), 1)

    # Creases
    bowler_crease_x = px1 + 35
    cv2.line(canvas, (bowler_crease_x, py1 - 8), (bowler_crease_x, py2 + 8), (140, 160, 190), 1)
    striker_crease_x = px2 - 35
    cv2.line(canvas, (striker_crease_x, py1 - 8), (striker_crease_x, py2 + 8), (140, 160, 190), 2)

    # Stumps
    cv2.line(canvas, (px1 + 12, pitch_cy - 10), (px1 + 12, pitch_cy + 10), (220, 190, 130), 2)
    cv2.line(canvas, (px2 - 12, pitch_cy - 10), (px2 - 12, pitch_cy + 10), (220, 190, 130), 2)

    bounce_pts = []
    density = np.zeros((height, width), dtype=np.float32)

    if summary_df is not None and not summary_df.empty:
        for _, row in summary_df.iterrows():
            bx = row.get("bounce_x")
            by = row.get("bounce_y")
            d_idx = int(row.get("delivery_idx", 1))
            if bx is not None and by is not None and not np.isnan(float(bx)) and not np.isnan(float(by)):
                frac_len = max(0.0, min(1.0, (by - 220.0) / (540.0 - 220.0)))
                map_x = int(round(bowler_crease_x + frac_len * (striker_crease_x - bowler_crease_x)))

                center_x_at_by = 901.5 - 0.4269 * (540.0 - by)
                offset_x = (bx - center_x_at_by)
                map_y = int(round(pitch_cy + (offset_x / 45.0) * (pitch_h * 0.35)))
                map_y = max(py1 + 8, min(py2 - 8, map_y))

                bounce_pts.append((map_x, map_y, d_idx))

                sigma = 32.0
                y_grid, x_grid = np.ogrid[:height, :width]
                dist_sq = (x_grid - map_x) ** 2 + (y_grid - map_y) ** 2
                gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
                density += gaussian

    # Heatmap blending
    if density.max() > 0:
        norm_density = (density / density.max() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_density, cv2.COLORMAP_MAGMA)

        pitch_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(pitch_mask, (px1, py1), (px2, py2), 255, -1)

        blend_mask = (norm_density > 14) & (pitch_mask == 255)
        for c in range(3):
            canvas[:, :, c] = np.where(
                blend_mask,
                (canvas[:, :, c] * 0.32 + heatmap[:, :, c] * 0.68).astype(np.uint8),
                canvas[:, :, c],
            )

    # Plot dots
    sel_dot = None
    for map_x, map_y, d_idx in bounce_pts:
        is_sel = (selected_delivery_idx is not None and d_idx == selected_delivery_idx)
        if is_sel:
            sel_dot = (map_x, map_y, d_idx)
        else:
            cv2.circle(canvas, (map_x, map_y), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (map_x, map_y), 6, (15, 18, 25), 1, cv2.LINE_AA)

    # Highlight selected delivery
    if sel_dot:
        sx, sy, sd = sel_dot
        cv2.circle(canvas, (sx, sy), 14, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 5, (40, 50, 235), -1, cv2.LINE_AA)

    # Colorbar on right
    cb_x = width - 40
    cb_y1 = py1 + 15
    cb_y2 = py2 - 15
    cb_h = cb_y2 - cb_y1
    cb_w = 10

    if cb_h > 10:
        cb_gradient = np.linspace(255, 0, cb_h, dtype=np.uint8).reshape(-1, 1)
        cb_gradient = np.repeat(cb_gradient, cb_w, axis=1)
        cb_colored = cv2.applyColorMap(cb_gradient, cv2.COLORMAP_MAGMA)
        canvas[cb_y1:cb_y2, cb_x:cb_x + cb_w] = cb_colored
        cv2.rectangle(canvas, (cb_x, cb_y1), (cb_x + cb_w, cb_y2), (80, 95, 125), 1)
        cv2.putText(canvas, "High", (cb_x - 4, cb_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Low", (cb_x - 2, cb_y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 150, 170), 1, cv2.LINE_AA)

    return canvas

def draw_perspective_pitch_zone_diagram(
    del_info=None,
    width: int = 360,
    height: int = 270,
) -> np.ndarray:
    """
    Renders an isometric perspective pitch card matching broadcast studio graphics.
    Features 5 color-coded length bands (Full Toss, Full Length, Good Length, Short of Good, Short Length),
    perspective pitch narrowing towards bowler end, top and bottom wickets, and a glowing white
    bounce indicator in the active zone.
    """
    # Background #141724 in BGR
    canvas = np.full((height, width, 3), (36, 23, 20), dtype=np.uint8)

    length_name = "good_length"
    stump_line = "middle"
    if del_info is not None:
        if isinstance(del_info, dict):
            length_name = str(del_info.get("length", "good_length"))
            stump_line = str(del_info.get("stump_line_display", del_info.get("stump_line", "middle")))
        elif hasattr(del_info, "get"):
            length_name = str(del_info.get("length", "good_length"))
            stump_line = str(del_info.get("stump_line_display", del_info.get("stump_line", "middle")))

    top_w = 96
    bot_w = 160
    center_x = int(width * 0.40)
    y_top = 34
    y_bot = height - 34
    total_h = y_bot - y_top

    # 5 zones matching the broadcast studio mockup
    zones = [
        ("Full Toss", (90, 80, 240), 0.0, 0.18),       # Red/Pink
        ("Full Length", (40, 150, 245), 0.18, 0.38),   # Amber/Orange
        ("Good Length", (80, 200, 40), 0.38, 0.62),    # Emerald Green
        ("Short of Good", (220, 130, 60), 0.62, 0.82), # Blue
        ("Short Length", (200, 70, 160), 0.82, 1.0),   # Violet
    ]

    tl_x = int(center_x - top_w / 2)
    tr_x = int(center_x + top_w / 2)
    bl_x = int(center_x - bot_w / 2)
    br_x = int(center_x + bot_w / 2)

    # Render colored trapezoid bands
    for name, bgr_color, f_start, f_end in zones:
        y1 = int(y_top + f_start * total_h)
        y2 = int(y_top + f_end * total_h)

        w1 = top_w + (bot_w - top_w) * f_start
        w2 = top_w + (bot_w - top_w) * f_end

        x1_l = int(center_x - w1 / 2)
        x1_r = int(center_x + w1 / 2)
        x2_l = int(center_x - w2 / 2)
        x2_r = int(center_x + w2 / 2)

        pts = np.array([[x1_l, y1], [x1_r, y1], [x2_r, y2], [x2_l, y2]], np.int32)

        is_active = False
        n_clean = name.lower().replace(" ", "_")
        l_clean = length_name.lower().replace(" ", "_")
        if n_clean in l_clean or l_clean in n_clean:
            is_active = True
        elif "good" in l_clean and "good_length" in n_clean:
            is_active = True

        sub = canvas.copy()
        cv2.fillPoly(sub, [pts], bgr_color)
        alpha = 0.85 if is_active else 0.40
        cv2.addWeighted(sub, alpha, canvas, 1 - alpha, 0, canvas)

        # Boundary divider line
        cv2.line(canvas, (x1_l, y1), (x1_r, y1), (90, 95, 120), 1, cv2.LINE_AA)

        # Zone text label on right
        label_y = int((y1 + y2) / 2) + 4
        text_color = (255, 255, 255) if is_active else (150, 160, 180)
        cv2.putText(canvas, name, (br_x + 14, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, text_color, 1, cv2.LINE_AA)

    # Pitch boundary
    pitch_poly = np.array([[tl_x, y_top], [tr_x, y_top], [br_x, y_bot], [bl_x, y_bot]], np.int32)
    cv2.polylines(canvas, [pitch_poly], True, (180, 190, 215), 1, cv2.LINE_AA)

    # Creases extending sideways
    cv2.line(canvas, (tl_x - 16, y_top), (tr_x + 16, y_top), (210, 220, 235), 2, cv2.LINE_AA)
    cv2.line(canvas, (bl_x - 20, y_bot), (br_x + 20, y_bot), (210, 220, 235), 2, cv2.LINE_AA)

    # Stumps at top (Bowler's end)
    stump_w = 16
    s_top_y = y_top - 16
    for i in range(3):
        sx = int(center_x - stump_w / 2 + i * (stump_w / 2))
        cv2.line(canvas, (sx, y_top), (sx, s_top_y), (230, 235, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (int(center_x - stump_w / 2 - 2), s_top_y), (int(center_x + stump_w / 2 + 2), s_top_y), (230, 235, 245), 2, cv2.LINE_AA)

    # Stumps at bottom (Batter's end)
    s_bot_y = y_bot + 16
    for i in range(3):
        sx = int(center_x - stump_w / 2 + i * (stump_w / 2))
        cv2.line(canvas, (sx, y_bot), (sx, s_bot_y), (230, 235, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (int(center_x - stump_w / 2 - 2), s_bot_y), (int(center_x + stump_w / 2 + 2), s_bot_y), (230, 235, 245), 2, cv2.LINE_AA)

    # Glowing bounce indicator
    f_mid = 0.50
    l_lower = length_name.lower()
    if "toss" in l_lower or "yorker" in l_lower:
        f_mid = 0.10
    elif "full" in l_lower:
        f_mid = 0.28
    elif "good" in l_lower:
        f_mid = 0.50
    elif "short of" in l_lower or "short_of" in l_lower:
        f_mid = 0.72
    elif "short" in l_lower:
        f_mid = 0.91

    by = int(y_top + f_mid * total_h)
    bw = top_w + (bot_w - top_w) * f_mid
    bx = center_x
    s_lower = stump_line.lower()
    if "off" in s_lower:
        bx = int(center_x - bw * 0.16)
    elif "leg" in s_lower:
        bx = int(center_x + bw * 0.16)

    # Glow rings
    for r in range(11, 4, -2):
        cv2.circle(canvas, (bx, by), r, (120, 255, 120), 1, cv2.LINE_AA)
    cv2.circle(canvas, (bx, by), 5, (255, 255, 255), -1, cv2.LINE_AA)

    return canvas


def draw_perspective_pitch_zone_diagram(
    del_info=None,
    width: int = 360,
    height: int = 270,
) -> np.ndarray:
    """
    Renders an isometric perspective pitch card matching broadcast studio graphics.
    Features 5 color-coded length bands (Full Toss, Full Length, Good Length, Short of Good, Short Length),
    perspective pitch narrowing towards bowler end, top and bottom wickets, and a glowing white
    bounce indicator in the active zone.
    """
    # Background #141724 in BGR
    canvas = np.full((height, width, 3), (36, 23, 20), dtype=np.uint8)

    length_name = "good_length"
    stump_line = "middle"
    if del_info is not None:
        if isinstance(del_info, dict):
            length_name = str(del_info.get("length", "good_length"))
            stump_line = str(del_info.get("stump_line_display", del_info.get("stump_line", "middle")))
        elif hasattr(del_info, "get"):
            length_name = str(del_info.get("length", "good_length"))
            stump_line = str(del_info.get("stump_line_display", del_info.get("stump_line", "middle")))

    top_w = 96
    bot_w = 160
    center_x = int(width * 0.40)
    y_top = 34
    y_bot = height - 34
    total_h = y_bot - y_top

    # 5 zones matching the broadcast studio mockup
    zones = [
        ("Full Toss", (90, 80, 240), 0.0, 0.18),       # Red/Pink
        ("Full Length", (40, 150, 245), 0.18, 0.38),   # Amber/Orange
        ("Good Length", (80, 200, 40), 0.38, 0.62),    # Emerald Green
        ("Short of Good", (220, 130, 60), 0.62, 0.82), # Blue
        ("Short Length", (200, 70, 160), 0.82, 1.0),   # Violet
    ]

    tl_x = int(center_x - top_w / 2)
    tr_x = int(center_x + top_w / 2)
    bl_x = int(center_x - bot_w / 2)
    br_x = int(center_x + bot_w / 2)

    # Render colored trapezoid bands
    for name, bgr_color, f_start, f_end in zones:
        y1 = int(y_top + f_start * total_h)
        y2 = int(y_top + f_end * total_h)

        w1 = top_w + (bot_w - top_w) * f_start
        w2 = top_w + (bot_w - top_w) * f_end

        x1_l = int(center_x - w1 / 2)
        x1_r = int(center_x + w1 / 2)
        x2_l = int(center_x - w2 / 2)
        x2_r = int(center_x + w2 / 2)

        pts = np.array([[x1_l, y1], [x1_r, y1], [x2_r, y2], [x2_l, y2]], np.int32)

        is_active = False
        n_clean = name.lower().replace(" ", "_")
        l_clean = length_name.lower().replace(" ", "_")
        if n_clean in l_clean or l_clean in n_clean:
            is_active = True
        elif "good" in l_clean and "good_length" in n_clean:
            is_active = True

        sub = canvas.copy()
        cv2.fillPoly(sub, [pts], bgr_color)
        alpha = 0.85 if is_active else 0.40
        cv2.addWeighted(sub, alpha, canvas, 1 - alpha, 0, canvas)

        # Boundary divider line
        cv2.line(canvas, (x1_l, y1), (x1_r, y1), (90, 95, 120), 1, cv2.LINE_AA)

        # Zone text label on right
        label_y = int((y1 + y2) / 2) + 4
        text_color = (255, 255, 255) if is_active else (150, 160, 180)
        cv2.putText(canvas, name, (br_x + 14, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, text_color, 1, cv2.LINE_AA)

    # Pitch boundary
    pitch_poly = np.array([[tl_x, y_top], [tr_x, y_top], [br_x, y_bot], [bl_x, y_bot]], np.int32)
    cv2.polylines(canvas, [pitch_poly], True, (180, 190, 215), 1, cv2.LINE_AA)

    # Creases extending sideways
    cv2.line(canvas, (tl_x - 16, y_top), (tr_x + 16, y_top), (210, 220, 235), 2, cv2.LINE_AA)
    cv2.line(canvas, (bl_x - 20, y_bot), (br_x + 20, y_bot), (210, 220, 235), 2, cv2.LINE_AA)

    # Stumps at top (Bowler's end)
    stump_w = 16
    s_top_y = y_top - 16
    for i in range(3):
        sx = int(center_x - stump_w / 2 + i * (stump_w / 2))
        cv2.line(canvas, (sx, y_top), (sx, s_top_y), (230, 235, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (int(center_x - stump_w / 2 - 2), s_top_y), (int(center_x + stump_w / 2 + 2), s_top_y), (230, 235, 245), 2, cv2.LINE_AA)

    # Stumps at bottom (Batter's end)
    s_bot_y = y_bot + 16
    for i in range(3):
        sx = int(center_x - stump_w / 2 + i * (stump_w / 2))
        cv2.line(canvas, (sx, y_bot), (sx, s_bot_y), (230, 235, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (int(center_x - stump_w / 2 - 2), s_bot_y), (int(center_x + stump_w / 2 + 2), s_bot_y), (230, 235, 245), 2, cv2.LINE_AA)

    # Glowing bounce indicator
    f_mid = 0.50
    l_lower = length_name.lower()
    if "toss" in l_lower or "yorker" in l_lower:
        f_mid = 0.10
    elif "full" in l_lower:
        f_mid = 0.28
    elif "good" in l_lower:
        f_mid = 0.50
    elif "short of" in l_lower or "short_of" in l_lower:
        f_mid = 0.72
    elif "short" in l_lower:
        f_mid = 0.91

    by = int(y_top + f_mid * total_h)
    bw = top_w + (bot_w - top_w) * f_mid
    bx = center_x
    s_lower = stump_line.lower()
    if "off" in s_lower:
        bx = int(center_x - bw * 0.16)
    elif "leg" in s_lower:
        bx = int(center_x + bw * 0.16)

    # Glow rings
    for r in range(11, 4, -2):
        cv2.circle(canvas, (bx, by), r, (120, 255, 120), 1, cv2.LINE_AA)
    cv2.circle(canvas, (bx, by), 5, (255, 255, 255), -1, cv2.LINE_AA)

    return canvas
