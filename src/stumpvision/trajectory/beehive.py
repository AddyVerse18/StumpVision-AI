"""
beehive.py
==========
Broadcast Front-On Stumps "Beehive" Visualizer for StumpVision AI.

Renders high-definition front-on arrival grid at the batter's popping crease:
1. Proportional 3 Stumps (Off, Middle, Leg) and Bails.
2. Batter vertical anatomy reference lines (Turf, Knee Roll, Stumps Top, Waist, Bouncer).
3. 4th/5th Stump "Corridor of Uncertainty" shading.
4. Off-side and Leg-side Wide ball guidelines.
5. High-visibility delivery arrival badges (#01 to #13) with interactive selection crosshair.
6. Summary analytics breakdown.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import cv2

from stumpvision.trajectory.stumps import (
    OFF_STUMP_X,
    MIDDLE_STUMP_X,
    LEG_STUMP_X,
    POPPING_CREASE_Y,
    WIDE_LINE_OFF_X,
    WIDE_LINE_LEG_X,
)
from stumpvision.trajectory.drs import evaluate_drs, DRSPrediction


def render_beehive_chart(
    trajectories: List[Any],
    selected_delivery_idx: Optional[int] = None,
    width: int = 860,
    height: int = 740,
) -> np.ndarray:
    """
    Render broadcast-quality Front-On Stumps Beehive Diagram.

    Args:
        trajectories: List of DeliveryTrajectory objects or dicts.
        selected_delivery_idx: Optional 1-based delivery index to highlight.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        np.ndarray: BGR image canvas.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (15, 18, 24)  # Deep broadcast navy

    # Dimensions & Scales
    # Physical: Middle stump is at center x = 430
    cx = width // 2
    turf_y = height - 120    # Baseline turf at y = 620
    px_per_cm = 2.80         # Vertical scale: 71.1 cm stump = ~199 px

    stump_h_px = int(round(71.1 * px_per_cm))
    bails_y = turf_y - stump_h_px

    # Stumps spacing: 9 inches = 22.86 cm total width, ~64 px total span
    off_stump_bx = cx - 28
    mid_stump_bx = cx
    leg_stump_bx = cx + 28
    stump_thick = 10

    # 1. Subtle Pitch Turf Grass Strip
    turf_overlay = canvas.copy()
    cv2.rectangle(turf_overlay, (0, turf_y), (width, height), (22, 38, 25), -1)
    cv2.addWeighted(turf_overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.line(canvas, (0, turf_y), (width, turf_y), (220, 225, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "POPPING CREASE / TURF BASE", (24, turf_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 150), 1, cv2.LINE_AA)

    # 2. Horizontal Anatomical / Height Zones
    height_markers = [
        (45.0, "KNEE ROLL (45 cm)", (70, 90, 110)),
        (71.1, "STUMPS TOP / BAILS (71.1 cm)", (100, 140, 180)),
        (105.0, "WAIST / RIB HEIGHT (105 cm)", (70, 90, 110)),
        (140.0, "BOUNCER THRESHOLD (140 cm)", (60, 80, 100)),
    ]

    for cm_val, label, col in height_markers:
        hy = int(round(turf_y - cm_val * px_per_cm))
        if 60 <= hy <= turf_y:
            # Subtle dashed line
            for dx_dash in range(60, width - 60, 16):
                cv2.line(canvas, (dx_dash, hy), (dx_dash + 8, hy), col, 1, cv2.LINE_AA)
            cv2.putText(canvas, label, (width - 240, hy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)

    # 3. 4th/5th Stump Channel (The "Corridor of Uncertainty") Shading
    # Located just outside off stump (bx: cx - 35 to cx - 95)
    corridor_x1 = off_stump_bx - 70
    corridor_x2 = off_stump_bx - 6
    corr_overlay = canvas.copy()
    cv2.rectangle(corr_overlay, (corridor_x1, bails_y - 40), (corridor_x2, turf_y), (40, 75, 95), -1)
    cv2.addWeighted(corr_overlay, 0.35, canvas, 0.65, 0, canvas)
    cv2.putText(canvas, "CORRIDOR OF UNCERTAINTY", (corridor_x1 - 10, bails_y - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 180, 240), 1, cv2.LINE_AA)

    # 4. Wide Guidelines
    # Off-side wide: 35 inches (88.9 cm) from middle stump = 88.9 * 2.80 = ~249 px
    wide_off_bx = cx - int(round(88.9 * px_per_cm))
    # Leg-side wide: stricter margin
    wide_leg_bx = cx + int(round(45.0 * px_per_cm))

    cv2.line(canvas, (wide_off_bx, 80), (wide_off_bx, turf_y), (80, 160, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WIDE LINE (OFF)", (wide_off_bx - 55, turf_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 160, 240), 1, cv2.LINE_AA)

    cv2.line(canvas, (wide_leg_bx, 80), (wide_leg_bx, turf_y), (80, 160, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WIDE LINE (LEG)", (wide_leg_bx + 10, turf_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 160, 240), 1, cv2.LINE_AA)

    # 5. Three Wooden Stumps & Bails (Polished Realistic 3D Styling)
    stump_specs = [
        (off_stump_bx, "OFF", (220, 180, 100)),
        (mid_stump_bx, "MID", (230, 210, 140)),
        (leg_stump_bx, "LEG", (220, 180, 100)),
    ]

    for sx, label, _ in stump_specs:
        # Stump shadow
        cv2.rectangle(canvas, (sx - 4, bails_y), (sx + 6, turf_y), (25, 20, 15), -1)
        # Wood body
        cv2.rectangle(canvas, (sx - 5, bails_y), (sx + 5, turf_y), (145, 185, 230), -1)
        # Wood highlight
        cv2.line(canvas, (sx - 2, bails_y + 2), (sx - 2, turf_y - 2), (200, 225, 255), 2, cv2.LINE_AA)
        # Stump label at base
        cv2.putText(canvas, label, (sx - 8, turf_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 215, 225), 1, cv2.LINE_AA)

    # Bails resting on top
    cv2.rectangle(canvas, (off_stump_bx - 10, bails_y - 6), (mid_stump_bx + 2, bails_y), (160, 200, 240), -1)
    cv2.rectangle(canvas, (mid_stump_bx - 2, bails_y - 6), (leg_stump_bx + 10, bails_y), (160, 200, 240), -1)
    cv2.rectangle(canvas, (off_stump_bx - 10, bails_y - 6), (leg_stump_bx + 10, bails_y), (25, 30, 40), 1, cv2.LINE_AA)

    # 6. Plot Delivery Arrival Points
    color_map = {
        "yorker": (40, 50, 235),        # Red
        "full": (30, 140, 255),         # Orange
        "good_length": (50, 210, 80),   # Green
        "short": (225, 140, 30),        # Cyan / Blue
    }

    selected_pt_info = None

    for traj in trajectories:
        if isinstance(traj, dict):
            pts = traj.get("points", [])
            d_idx = traj.get("delivery_idx", 1)
            length_cat = traj.get("length_classification", "good_length")
            line_lbl = traj.get("stump_line_display", "Middle")
        else:
            pts = getattr(traj, "points", [])
            d_idx = getattr(traj, "delivery_idx", 1)
            length_cat = getattr(traj, "length_classification", "good_length")
            line_lbl = getattr(traj, "stump_line_display", "Middle")

        if not pts:
            continue

        drs = evaluate_drs(traj)

        # Horizontal arrival mapping:
        # Camera 1 crease x is centered at MIDDLE_STUMP_X (901.5 px)
        # 1 px in Camera 1 ≈ 2.08 cm
        x_crease = pts[-1].x if hasattr(pts[-1], "x") else pts[-1]["x"]
        offset_from_mid_cm = (x_crease - MIDDLE_STUMP_X) * 2.08

        bx_pos = int(round(cx + offset_from_mid_cm * px_per_cm))
        bx_pos = max(50, min(width - 50, bx_pos))

        # Vertical arrival height in cm above turf:
        arrival_cm = drs.arrival_height_cm
        by_pos = int(round(turf_y - arrival_cm * px_per_cm))
        by_pos = max(60, min(turf_y, by_pos))

        is_sel = (selected_delivery_idx is not None and d_idx == selected_delivery_idx)
        dot_col = color_map.get(length_cat.lower(), (255, 255, 0))

        if is_sel:
            selected_pt_info = (bx_pos, by_pos, d_idx, drs, length_cat, line_lbl)
        else:
            # Unselected delivery badge
            cv2.circle(canvas, (bx_pos, by_pos), 12, (20, 25, 32), -1, cv2.LINE_AA)
            cv2.circle(canvas, (bx_pos, by_pos), 10, dot_col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (bx_pos, by_pos), 11, (240, 245, 255), 1, cv2.LINE_AA)
            num_str = f"{d_idx}"
            cv2.putText(canvas, num_str, (bx_pos - (4 if len(num_str) == 1 else 7), by_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    # 7. Render Selected Delivery with Focus Crosshair
    if selected_pt_info:
        sbx, sby, sd_idx, sdrs, slen, sline = selected_pt_info
        # Glowing outer pulse rings
        cv2.circle(canvas, (sbx, sby), 24, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (sbx, sby), 18, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (sbx, sby), 12, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (sbx, sby), 9, (40, 50, 235), -1, cv2.LINE_AA)

        num_str = f"{sd_idx}"
        cv2.putText(canvas, num_str, (sbx - (4 if len(num_str) == 1 else 7), sby + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Crosshair lines
        cv2.line(canvas, (sbx - 30, sby), (sbx - 20, sby), (0, 240, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (sbx + 20, sby), (sbx + 30, sby), (0, 240, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (sbx, sby - 30), (sbx, sby - 20), (0, 240, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (sbx, sby + 20), (sbx, sby + 30), (0, 240, 255), 2, cv2.LINE_AA)

        # Callout card
        call_x = min(width - 240, max(20, sbx + 25))
        call_y = max(70, min(height - 180, sby - 50))
        cv2.rectangle(canvas, (call_x, call_y), (call_x + 215, call_y + 85), (20, 28, 38), -1)
        cv2.rectangle(canvas, (call_x, call_y), (call_x + 215, call_y + 85), (0, 240, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"DELIVERY #{sd_idx:02d} SELECTED", (call_x + 10, call_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Height: {sdrs.arrival_height_cm:.1f} cm", (call_x + 10, call_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 230, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Line:   {sline}", (call_x + 10, call_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 230, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Speed:  {sdrs.speed_release_kmh:.1f} km/h", (call_x + 10, call_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 235, 140), 1, cv2.LINE_AA)

    # 8. Header & Telemetry Overlay
    cv2.putText(canvas, "STUMPVISION AI — STRIKER CREASE BEEHIVE", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Front-On Batsman Arrival Grid & Corridor Calibration", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 175, 210), 1, cv2.LINE_AA)

    # Bottom Legend
    leg_y = height - 25
    leg_items = [
        ("Yorker", (40, 50, 235)),
        ("Full", (30, 140, 255)),
        ("Good Length", (50, 210, 80)),
        ("Short", (225, 140, 30)),
    ]
    cur_lx = 40
    for l_name, l_col in leg_items:
        cv2.circle(canvas, (cur_lx, leg_y), 6, l_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, l_name, (cur_lx + 12, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 225, 230), 1, cv2.LINE_AA)
        cur_lx += 140

    return canvas

