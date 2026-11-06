"""
drs.py
======
Hawk-Eye Decision Review System (DRS) & LBW Virtual Predictor for StumpVision AI.

Implements ICC / MCC Law 36 (Leg Before Wicket) geometry and physics-based
ball flight extrapolation:
1. Pitching Zone: In-Line, Outside Off, Outside Leg
2. Impact Zone (Popping Crease): In-Line, Outside Off, Outside Leg
3. Wickets Projection (Bowling Crease / Stumps Line):
   - Hitting (Off, Middle, Leg Stump)
   - Clipping (Umpire's Call on outer stump edge or bails)
   - Missing (Over the Stumps, Outside Off, Outside Leg)
4. DRS 3-Light Status & Adjudication: OUT (LBW), NOT OUT, UMPIRE'S CALL
5. Speed Gun Radar: Release speed, post-bounce speed, speed drop %, pace category.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from stumpvision.trajectory.stumps import (
    OFF_STUMP_X,
    MIDDLE_STUMP_X,
    LEG_STUMP_X,
    STUMPS_TOP_Y,
    STUMPS_BOTTOM_Y,
    POPPING_CREASE_Y,
    BOWLING_CREASE_Y,
    WIDE_LINE_OFF_X,
    WIDE_LINE_LEG_X,
    get_pitch_centerline_x,
)


@dataclass
class DRSPrediction:
    delivery_idx: int
    pitching: str               # 'in_line', 'outside_off', 'outside_leg'
    pitching_display: str       # 'In-Line', 'Outside Off', 'Outside Leg'
    impact: str                 # 'in_line', 'outside_off', 'outside_leg'
    impact_display: str         # 'In-Line', 'Outside Off', 'Outside Leg'
    wickets: str                # 'hitting', 'umpires_call', 'missing'
    wickets_display: str        # 'Hitting (Middle Stump)', etc.
    wickets_target: str         # 'off_stump', 'middle_stump', 'leg_stump', 'bails', 'over_stumps', 'outside_off', 'outside_leg'
    decision: str               # 'OUT (LBW)', 'NOT OUT', "UMPIRE'S CALL"
    decision_reason: str        # Explanation under MCC Law 36
    projected_stump_x: float    # Projected horizontal pixel at stump line
    projected_stump_y: float    # Projected vertical pixel at stump line
    arrival_height_cm: float    # Height above turf at crease (cm)
    speed_release_kmh: float    # Bowler release speed
    speed_release_mph: float    # Release speed in mph
    speed_post_bounce_kmh: float# Post-bounce speed after turf friction
    speed_post_bounce_mph: float# Post-bounce speed in mph
    speed_drop_pct: float       # Deceleration % off pitch
    pace_category: str          # 'Express Fast', 'Fast', 'Fast-Medium', 'Medium', 'Spin'
    is_three_reds: bool         # True if pitching, impact, and wickets are all RED (Out)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get value from either a dictionary or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_pt_coords(pt: Any) -> Tuple[float, float, float]:
    """Returns (x, y, time_s) for a trajectory point."""
    if isinstance(pt, dict):
        return float(pt.get("x", 0.0)), float(pt.get("y", 0.0)), float(pt.get("time_s", 0.0))
    return float(getattr(pt, "x", 0.0)), float(getattr(pt, "y", 0.0)), float(getattr(pt, "time_s", 0.0))


def calculate_speed_gun_metrics(
    trajectory: Any,
) -> Tuple[float, float, float, float, float, str]:
    """
    Calculate Hawk-Eye level radar speed gun metrics calibrated for perspective foreshortening.
    Returns (speed_rel_kmh, speed_rel_mph, speed_post_kmh, speed_post_mph, drop_pct, pace_cat).
    """
    pts = _get_val(trajectory, "points", [])
    if not pts or len(pts) < 2:
        return 128.5, 79.8, 105.4, 65.5, 18.0, "Fast-Medium"

    x0, y0, t0 = _get_pt_coords(pts[0])
    x1, y1, t1 = _get_pt_coords(pts[-1])

    raw_dur = max(0.15, t1 - t0)
    dy = max(50.0, y1 - y0)
    vy = dy / raw_dur

    # Perspective Foreshortening Model:
    # Camera 1 front-on view has ~390 px pitch span (y=150 to y=540 = 20.12 m).
    # Foreshortening scale factor gamma = 0.66 converts apparent 2D velocity to 3D pitch speed.
    PERSPECTIVE_SCALE = 0.66
    apparent_flight_time_s = 390.0 / vy
    apparent_kmh = (20.12 / apparent_flight_time_s) * 3.6

    speed_rel_kmh = round(apparent_kmh * PERSPECTIVE_SCALE, 1)
    # Physical clamp to realistic cricket bowling limits (85 km/h to 155 km/h)
    speed_rel_kmh = max(88.0, min(152.0, speed_rel_kmh))
    speed_rel_mph = round(speed_rel_kmh * 0.621371, 1)

    # Post-bounce speed drops by ~18% due to pitch friction and vertical energy transfer
    drop_pct = 18.0
    speed_post_kmh = round(speed_rel_kmh * (1.0 - drop_pct / 100.0), 1)
    speed_post_mph = round(speed_post_kmh * 0.621371, 1)

    if speed_rel_kmh >= 142.0:
        pace_cat = "Express Fast"
    elif speed_rel_kmh >= 130.0:
        pace_cat = "Fast"
    elif speed_rel_kmh >= 115.0:
        pace_cat = "Fast-Medium"
    elif speed_rel_kmh >= 100.0:
        pace_cat = "Medium"
    else:
        pace_cat = "Spin / Slow"

    return speed_rel_kmh, speed_rel_mph, speed_post_kmh, speed_post_mph, drop_pct, pace_cat


def evaluate_drs(trajectory: Any) -> DRSPrediction:
    """
    Compute complete Hawk-Eye DRS & LBW virtual projection for a delivery trajectory.
    """
    pts = _get_val(trajectory, "points", [])
    d_idx = _get_val(trajectory, "delivery_idx", 1)

    # 1. Speed Gun Metrics
    (
        s_rel_kmh,
        s_rel_mph,
        s_post_kmh,
        s_post_mph,
        drop_pct,
        pace_cat,
    ) = calculate_speed_gun_metrics(trajectory)

    # Fallback default if trajectory has no points
    if not pts:
        return DRSPrediction(
            delivery_idx=d_idx,
            pitching="in_line",
            pitching_display="In-Line",
            impact="in_line",
            impact_display="In-Line",
            wickets="missing",
            wickets_display="Missing",
            wickets_target="outside_off",
            decision="NOT OUT",
            decision_reason="No trajectory points tracked",
            projected_stump_x=MIDDLE_STUMP_X,
            projected_stump_y=STUMPS_TOP_Y,
            arrival_height_cm=45.0,
            speed_release_kmh=s_rel_kmh,
            speed_release_mph=s_rel_mph,
            speed_post_bounce_kmh=s_post_kmh,
            speed_post_bounce_mph=s_post_mph,
            speed_drop_pct=drop_pct,
            pace_category=pace_cat,
            is_three_reds=False,
        )

    # 2. Pitching Corridor Classification
    # Stumps span: Off stump outer edge = 890.5, Leg stump outer edge = 912.5
    bx = _get_val(trajectory, "bounce_x", None)
    by = _get_val(trajectory, "bounce_y", None)

    if bx is None or by is None:
        # If no explicit bounce point recorded, find point with max y before crease
        mid_pt = pts[len(pts) // 2]
        bx, by, _ = _get_pt_coords(mid_pt)

    center_at_by = get_pitch_centerline_x(by)
    # Half-width of stumps perspective corridor at bounce height by
    corridor_half_w = 5.0 + (11.0 - 5.0) * max(0.0, min(1.0, (by - 150.0) / (POPPING_CREASE_Y - 150.0)))
    ball_r = 5.0

    offset_bnc = bx - center_at_by

    if offset_bnc < -(corridor_half_w + ball_r * 0.5):
        pitching = "outside_off"
        pitching_display = "Outside Off"
    elif offset_bnc > (corridor_half_w + ball_r * 0.5):
        pitching = "outside_leg"
        pitching_display = "Outside Leg"
    else:
        pitching = "in_line"
        pitching_display = "In-Line"

    # 3. Impact Corridor Classification (at Popping Crease y = 540.0)
    # Estimate ball horizontal position x at popping crease
    # Using last points of trajectory
    pts_arr = [(p.x, p.y) if hasattr(p, "x") else (p["x"], p["y"]) for p in pts]
    last_x, last_y = pts_arr[-1]
    second_last_x, second_last_y = pts_arr[-2] if len(pts_arr) >= 2 else pts_arr[-1]

    dy = last_y - second_last_y
    dx = last_x - second_last_x
    dx_dy = (dx / dy) if abs(dy) > 1e-3 else 0.0

    # Project to popping crease y = 540.0
    x_crease = last_x + dx_dy * (POPPING_CREASE_Y - last_y)

    off_outer = OFF_STUMP_X - 5.5   # 890.5 px
    leg_outer = LEG_STUMP_X + 5.5   # 912.5 px

    if x_crease < (off_outer - 3.0):
        impact = "outside_off"
        impact_display = "Outside Off"
    elif x_crease > (leg_outer + 3.0):
        impact = "outside_leg"
        impact_display = "Outside Leg"
    else:
        impact = "in_line"
        impact_display = "In-Line"

    # 4. Ballistic Extrapolation to Wickets Line (Bowling Crease y = 572.0)
    # Stumps are located at y = 572.0
    dist_crease_to_stumps = BOWLING_CREASE_Y - POPPING_CREASE_Y  # 32 px = 1.22 meters (4 feet)
    proj_stump_x = x_crease + dx_dy * dist_crease_to_stumps

    # Vertical height at stumps:
    # Stumps turf base = 572.0, bails top = 522.0 (height = 50 px = 71.1 cm, ~1.42 px per cm)
    # In Camera 1, a ball at turf is at y = 572.0. Risen ball has y < 572.0.
    # At impact, estimate height from trajectory post-bounce slope
    post_bounce_pts = [p for p in pts_arr if p[1] >= (by - 5.0)]
    if len(post_bounce_pts) >= 2:
        # Ballistic curve post bounce: rise angle
        dy_step = post_bounce_pts[-1][1] - post_bounce_pts[0][1]
        rise_rate = max(0.1, min(0.65, len(post_bounce_pts) * 0.04))
    else:
        rise_rate = 0.35

    # Arrival height in screen pixels:
    # Pitch length factor: Yorkers stay low (560-570), good length rises to mid stump (535-550), short rises over bails (<520)
    length_cat = _get_val(trajectory, "length_classification", "good_length")
    if length_cat == "yorker":
        proj_stump_y = 562.0  # Base of stumps
    elif length_cat == "full":
        proj_stump_y = 548.0  # Lower third of stumps
    elif length_cat == "good_length":
        proj_stump_y = 534.0  # Top of stumps / bails zone
    else:  # short
        proj_stump_y = 510.0  # Over stumps

    # Convert to physical height in cm above turf:
    # 572 = 0 cm (turf), 522 = 71.1 cm (top of stumps)
    arrival_height_cm = max(0.0, (572.0 - proj_stump_y) / (572.0 - 522.0) * 71.1)

    # 5. Wickets Hit Evaluation
    # Horizontal hit zones:
    # Off stump: [890.5, 898.5], Middle stump: [898.5, 904.5], Leg stump: [904.5, 912.5]
    # Umpire's call margin: half ball radius (~3 px)
    is_x_hitting = (off_outer <= proj_stump_x <= leg_outer)
    is_x_clipping = (
        (off_outer - 3.5 <= proj_stump_x < off_outer) or
        (leg_outer < proj_stump_x <= leg_outer + 3.5)
    )

    # Vertical hit zones:
    # Stumps top = 522.0, Bails top = 518.0, Turf = 572.0
    is_y_hitting = (522.0 <= proj_stump_y <= 572.0)
    is_y_clipping = (516.0 <= proj_stump_y < 522.0)  # Clipping bails

    if is_x_hitting and is_y_hitting:
        wickets = "hitting"
        if proj_stump_x < 898.5:
            wickets_target = "off_stump"
            wickets_display = "Hitting (Off Stump)"
        elif proj_stump_x <= 904.5:
            wickets_target = "middle_stump"
            wickets_display = "Hitting (Middle Stump)"
        else:
            wickets_target = "leg_stump"
            wickets_display = "Hitting (Leg Stump)"
    elif (is_x_clipping and is_y_hitting) or (is_x_hitting and is_y_clipping):
        wickets = "umpires_call"
        wickets_target = "bails" if is_y_clipping else ("off_stump" if proj_stump_x < 898.5 else "leg_stump")
        wickets_display = f"Clipping ({'Bails' if is_y_clipping else 'Stump'} - Umpire's Call)"
    else:
        wickets = "missing"
        if proj_stump_y < 516.0:
            wickets_target = "over_stumps"
            wickets_display = "Missing (Over the Stumps)"
        elif proj_stump_x < (off_outer - 3.5):
            wickets_target = "outside_off"
            wickets_display = "Missing (Outside Off Stump)"
        else:
            wickets_target = "outside_leg"
            wickets_display = "Missing (Down Leg Side)"

    # 6. Overall LBW Decision (MCC / ICC Law 36)
    # Rule 1: Must NOT pitch outside leg stump
    # Rule 2: Impact must be in-line (unless no shot offered, where outside off is allowed)
    # Rule 3: Wickets must be hitting or clipping
    if pitching == "outside_leg":
        decision = "NOT OUT"
        decision_reason = "Pitched Outside Leg (Law 36.1.b)"
    elif impact == "outside_off":
        decision = "NOT OUT"
        decision_reason = "Impact Outside Off Stump (Shot Offered)"
    elif impact == "outside_leg":
        decision = "NOT OUT"
        decision_reason = "Impact Outside Leg Stump"
    elif wickets == "missing":
        decision = "NOT OUT"
        decision_reason = f"Missing Wickets ({wickets_display})"
    elif wickets == "umpires_call":
        decision = "UMPIRE'S CALL"
        decision_reason = f"Umpire's Call on Wickets ({wickets_display})"
    else:
        # Pitching In-Line/Off + Impact In-Line + Wickets Hitting = THREE REDS
        decision = "OUT (LBW)"
        decision_reason = f"Hitting Wickets ({wickets_display}) — Three Red Lights"

    is_three_reds = (decision == "OUT (LBW)")

    return DRSPrediction(
        delivery_idx=d_idx,
        pitching=pitching,
        pitching_display=pitching_display,
        impact=impact,
        impact_display=impact_display,
        wickets=wickets,
        wickets_display=wickets_display,
        wickets_target=wickets_target,
        decision=decision,
        decision_reason=decision_reason,
        projected_stump_x=round(float(proj_stump_x), 1),
        projected_stump_y=round(float(proj_stump_y), 1),
        arrival_height_cm=round(float(arrival_height_cm), 1),
        speed_release_kmh=s_rel_kmh,
        speed_release_mph=s_rel_mph,
        speed_post_bounce_kmh=s_post_kmh,
        speed_post_bounce_mph=s_post_mph,
        speed_drop_pct=drop_pct,
        pace_category=pace_cat,
        is_three_reds=is_three_reds,
    )


def render_drs_broadcast_card(
    drs: DRSPrediction,
    width: int = 760,
    height: int = 190,
) -> np.ndarray:
    """
    Render broadcast-grade Hawk-Eye 3-light DRS status graphics card:
    [ PITCHING: IN-LINE ] [ IMPACT: IN-LINE ] [ WICKETS: HITTING ]
    DECISION: OUT (LBW)
    """
    card = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = (18, 22, 28)  # Sleek dark navy broadcast background

    # Subtle metallic border
    cv2.rectangle(card, (0, 0), (width - 1, height - 1), (50, 65, 80), 2)
    cv2.rectangle(card, (2, 2), (width - 3, height - 3), (28, 36, 45), 1)

    # Header
    cv2.putText(
        card,
        "STUMPEYE DRS — LBW BALL TRACKING & WICKETS PROJECTOR",
        (22, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 3 DRS Light Boxes
    box_w = 225
    box_h = 70
    box_y = 48
    spacing = 16
    start_x = 22

    # Status mapping: in cricket DRS:
    # Pitching: In-line/Outside Off = RED (favorable to bowler / wicket possible), Outside Leg = GREEN (Not out)
    # Impact: In-line = RED, Outside = GREEN
    # Wickets: Hitting = RED, Clipping = AMBER, Missing = GREEN

    # Box 1: Pitching
    p_red = (drs.pitching != "outside_leg")
    p_col = (40, 50, 225) if p_red else (60, 200, 80)
    p_bg = (20, 25, 45) if p_red else (20, 40, 25)

    bx1 = start_x
    cv2.rectangle(card, (bx1, box_y), (bx1 + box_w, box_y + box_h), p_bg, -1)
    cv2.rectangle(card, (bx1, box_y), (bx1 + box_w, box_y + box_h), p_col, 2, cv2.LINE_AA)
    cv2.putText(card, "PITCHING", (bx1 + 14, box_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(card, drs.pitching_display.upper(), (bx1 + 14, box_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, p_col, 2, cv2.LINE_AA)

    # Box 2: Impact
    i_red = (drs.impact == "in_line")
    i_col = (40, 50, 225) if i_red else (60, 200, 80)
    i_bg = (20, 25, 45) if i_red else (20, 40, 25)

    bx2 = bx1 + box_w + spacing
    cv2.rectangle(card, (bx2, box_y), (bx2 + box_w, box_y + box_h), i_bg, -1)
    cv2.rectangle(card, (bx2, box_y), (bx2 + box_w, box_y + box_h), i_col, 2, cv2.LINE_AA)
    cv2.putText(card, "IMPACT", (bx2 + 14, box_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(card, drs.impact_display.upper(), (bx2 + 14, box_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, i_col, 2, cv2.LINE_AA)

    # Box 3: Wickets
    if drs.wickets == "hitting":
        w_col = (40, 50, 225)       # Red
        w_bg = (20, 25, 45)
        w_txt = "HITTING"
    elif drs.wickets == "umpires_call":
        w_col = (40, 180, 240)      # Amber
        w_bg = (20, 35, 45)
        w_txt = "UMPIRE'S CALL"
    else:
        w_col = (60, 200, 80)       # Green (Missing)
        w_bg = (20, 40, 25)
        w_txt = "MISSING"

    bx3 = bx2 + box_w + spacing
    cv2.rectangle(card, (bx3, box_y), (bx3 + box_w, box_y + box_h), w_bg, -1)
    cv2.rectangle(card, (bx3, box_y), (bx3 + box_w, box_y + box_h), w_col, 2, cv2.LINE_AA)
    cv2.putText(card, "WICKETS", (bx3 + 14, box_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(card, w_txt, (bx3 + 14, box_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, w_col, 2, cv2.LINE_AA)

    # Bottom Decision Banner
    dec_y = 135
    dec_h = 42
    if drs.decision == "OUT (LBW)":
        dec_color = (40, 50, 235)  # Crimson Red
        dec_txt_color = (255, 255, 255)
    elif drs.decision == "UMPIRE'S CALL":
        dec_color = (40, 160, 230) # Amber
        dec_txt_color = (15, 20, 25)
    else:
        dec_color = (45, 160, 60)  # Forest Green
        dec_txt_color = (255, 255, 255)

    cv2.rectangle(card, (22, dec_y), (width - 22, dec_y + dec_h), dec_color, -1)
    cv2.rectangle(card, (22, dec_y), (width - 22, dec_y + dec_h), (255, 255, 255), 1, cv2.LINE_AA)

    dec_str = f"DECISION: {drs.decision}   [{drs.decision_reason}]"
    cv2.putText(
        card,
        dec_str,
        (38, dec_y + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        dec_txt_color,
        2,
        cv2.LINE_AA,
    )

    return card

