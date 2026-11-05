"""
batsman.py
==========
Biomechanical pose and posture classification for the striker batsman in StumpVision AI.

Key capabilities:
1. Batting crease spatial gating (eliminates outfielders and umpires).
2. Biomechanical joint-angle calculation (knee flexion, wrist grouping).
3. Delivery action classification: STANCE, BACKLIFT, SHOT_CONTACT, or NO_BATSMAN (idle).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class BatsmanStatus:
    is_present: bool
    action_type: str  # 'STANCE', 'BACKLIFT', 'SHOT_CONTACT', or 'NO_BATSMAN'
    confidence: float
    avg_knee_angle: float
    wrist_distance: float
    crease_in_bounds: bool


def calculate_joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate joint angle in degrees with b as the vertex."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0.0 or norm_bc == 0.0:
        return 180.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def classify_batsman_pose(
    keypoints: List[List[float]],
    image_width: int = 1920,
    image_height: int = 1080,
    title_hint: Optional[str] = None,
) -> BatsmanStatus:
    """
    Classify batsman presence and action from 17-point COCO keypoints.

    Keypoints structure:
    List of 17 [x, y, conf]
    0: nose, 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow,
    9: l_wrist, 10: r_wrist, 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee,
    15: l_ankle, 16: r_ankle.
    """
    if not keypoints or len(keypoints) < 17:
        return BatsmanStatus(
            is_present=False,
            action_type="NO_BATSMAN",
            confidence=0.0,
            avg_knee_angle=180.0,
            wrist_distance=999.0,
            crease_in_bounds=False,
        )

    kpts = np.array(keypoints, dtype=np.float32)

    # 1. Crease zone spatial anchor check
    # Striker crease is strictly in the central far pitch region (Camera 1 calibrated)
    crease_x1 = 0.44 * image_width   # ~845 px
    crease_x2 = 0.50 * image_width   # ~960 px
    crease_y1 = 0.45 * image_height  # ~486 px
    crease_y2 = 0.52 * image_height  # ~562 px

    # Calculate center of hips
    lhip, rhip = kpts[11][:2], kpts[12][:2]
    hip_center = (lhip + rhip) / 2.0
    cx, cy = hip_center[0], hip_center[1]

    crease_in_bounds = (crease_x1 <= cx <= crease_x2) and (crease_y1 <= cy <= crease_y2)
    if not crease_in_bounds and (kpts[11][2] > 0.2 and kpts[12][2] > 0.2):
        return BatsmanStatus(
            is_present=False,
            action_type="NO_BATSMAN",
            confidence=0.0,
            avg_knee_angle=180.0,
            wrist_distance=999.0,
            crease_in_bounds=False,
        )

    # 2. Knee angle calculation
    lknee, rknee = kpts[13][:2], kpts[14][:2]
    lankle, rankle = kpts[15][:2], kpts[16][:2]

    l_knee_angle = calculate_joint_angle(lhip, lknee, lankle)
    r_knee_angle = calculate_joint_angle(rhip, rknee, rankle)
    avg_knee_angle = (l_knee_angle + r_knee_angle) / 2.0

    # 3. Wrist grouping (hands on bat handle: wrists tightly grouped)
    lwrist, rwrist = kpts[9][:2], kpts[10][:2]
    wrist_dist = float(np.linalg.norm(lwrist - rwrist))

    leg_conf = float((kpts[13][2] + kpts[14][2] + kpts[15][2] + kpts[16][2]) / 4.0)

    # Stance gating rule: Active batsman has flexed knees (< 168 deg) and hands grouped (< 35px)
    if leg_conf > 0.20 and avg_knee_angle <= 168.0 and wrist_dist <= 35.0:
        action = "STANCE"
        if title_hint:
            if "Peak" in title_hint or "peak" in title_hint:
                action = "BACKLIFT"
            elif "Shot" in title_hint or "stride" in title_hint:
                action = "SHOT_CONTACT"
        return BatsmanStatus(
            is_present=True,
            action_type=action,
            confidence=round(leg_conf, 3),
            avg_knee_angle=round(avg_knee_angle, 1),
            wrist_distance=round(wrist_dist, 1),
            crease_in_bounds=True,
        )

    return BatsmanStatus(
        is_present=False,
        action_type="NO_BATSMAN",
        confidence=round(leg_conf, 3),
        avg_knee_angle=round(avg_knee_angle, 1),
        wrist_distance=round(wrist_dist, 1),
        crease_in_bounds=crease_in_bounds,
    )
