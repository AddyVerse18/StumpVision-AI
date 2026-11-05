"""
stumps.py
=========
Batter's End Stumps Calibration & Stump-Line Analytics for StumpVision AI.

Provides precise pixel-level ground truth for the 3 batter's end stumps
(Off, Middle, Leg), popping crease, wide guidelines, and automated classification
of delivery lines (Wide Off, Outside Off, Off Stump, Middle Stump, Leg Stump,
Outside Leg, Wide Leg).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Ground-Truth Batter's End Stumps (Camera 1 Stump Camera, 1920x1080)
# ---------------------------------------------------------------------------
# Physical Stumps: Height = 28 inches (71.1 cm), Width = 9 inches (22.86 cm)
# Measured and verified from high-resolution still frames of Camera 1:
STUMPS_TOP_Y = 522.0        # Top of stumps / bails
STUMPS_BOTTOM_Y = 572.0     # Turf pitch insertion base
STUMPS_CENTER_X = 901.5     # Middle stump center

OFF_STUMP_X = 896.0         # Off stump (right-handed batsman perspective)
MIDDLE_STUMP_X = 901.5      # Middle stump
LEG_STUMP_X = 907.0         # Leg stump

STUMP_WIDTH = 11.0          # Stumps span: off stump to leg stump (22.86 cm)
STUMP_HEIGHT = 50.0         # Pitch base to bails height (71.1 cm)

POPPING_CREASE_Y = 540.0    # Popping crease line at striker end
BOWLING_CREASE_Y = 572.0    # Stump line / bowling crease at striker end

# Perspective Pitch Centerline: from Bowler Release (735, 150) to Stumps (901.5, 540)
# Equation: x_center(y) = 901.5 - 0.4269 * (540 - y)
PITCH_CENTER_SLOPE = 0.4269

# ICC / MCC Law 22 - Wide Ball Guidelines at Batter's End Crease:
# Off-side wide guideline: 35 inches (88.9 cm) from middle stump = 42.8 px
WIDE_LINE_OFF_X = 858.7     # 901.5 - 42.8 px
# Leg-side wide guideline: white-ball strict leg-side wide corridor
WIDE_LINE_LEG_X = 922.0     # 901.5 + 20.5 px

# Pitch Return Creases (width 10 feet = 120 inches):
RETURN_CREASE_OFF_X = 838.0
RETURN_CREASE_LEG_X = 965.0


@dataclass
class StumpAnnotation:
    off_stump_x: float = OFF_STUMP_X
    middle_stump_x: float = MIDDLE_STUMP_X
    leg_stump_x: float = LEG_STUMP_X
    top_y: float = STUMPS_TOP_Y
    bottom_y: float = STUMPS_BOTTOM_Y
    popping_crease_y: float = POPPING_CREASE_Y
    wide_line_off_x: float = WIDE_LINE_OFF_X
    wide_line_leg_x: float = WIDE_LINE_LEG_X

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_pitch_centerline_x(y: float) -> float:
    """Calculate the perspective pitch center x coordinate at any vertical height y."""
    return STUMPS_CENTER_X - PITCH_CENTER_SLOPE * (POPPING_CREASE_Y - y)


def classify_stump_line(
    x_at_crease: float,
    y_at_crease: float = POPPING_CREASE_Y,
) -> Tuple[str, str, str]:
    """
    Classify the exact stump line of a cricket ball as it passes the batter's crease.

    Returns:
        tuple: (stump_line_id, display_label, legacy_line)
        - stump_line_id: 'wide_off', 'outside_off', 'off_stump', 'middle_stump', 'leg_stump', 'outside_leg', 'wide_leg'
        - display_label: Human-readable broadcast label (e.g. 'Off Stump', 'Wide Outside Off')
        - legacy_line: 'off', 'middle', 'leg' for 100% backward compatibility
    """
    x = float(x_at_crease)

    if x < WIDE_LINE_OFF_X:
        return "wide_off", "Wide Outside Off", "off"
    elif x < (OFF_STUMP_X - 4.0):
        # 4th to 6th stump channel outside off
        return "outside_off", "Outside Off", "off"
    elif x < (OFF_STUMP_X + 2.5):
        # Off stump target
        return "off_stump", "Off Stump", "off"
    elif x <= (MIDDLE_STUMP_X + 3.0):
        # Middle stump target
        return "middle_stump", "Middle Stump", "middle"
    elif x <= (LEG_STUMP_X + 3.5):
        # Leg stump target
        return "leg_stump", "Leg Stump", "leg"
    elif x <= WIDE_LINE_LEG_X:
        # Down leg side, missing stumps
        return "outside_leg", "Outside Leg", "leg"
    else:
        # Wide down leg side
        return "wide_leg", "Wide Down Leg", "leg"


def draw_batter_stumps(
    frame: np.ndarray,
    highlight_stump: Optional[str] = None,
    alpha: float = 0.85,
    draw_guidelines: bool = False,
    draw_stump_poles: bool = False,
    draw_badge: bool = True,
    stump_line_label: Optional[str] = None,
) -> np.ndarray:
    """
    Render broadcast-quality annotated batter's end stump analytics onto a Camera 1 video frame.
    By default, avoids drawing artificial poles over the live batsman's body.
    Draws the top-right broadcast HUD badge with the exact classified stump line.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    canvas = frame.copy()
    overlay = frame.copy()

    scale_x = w / 1920.0
    scale_y = h / 1080.0

    top_y = int(round(STUMPS_TOP_Y * scale_y))
    bot_y = int(round(STUMPS_BOTTOM_Y * scale_y))
    crease_y = int(round(POPPING_CREASE_Y * scale_y))

    off_x = int(round(OFF_STUMP_X * scale_x))
    mid_x = int(round(MIDDLE_STUMP_X * scale_x))
    leg_x = int(round(LEG_STUMP_X * scale_x))

    wide_off_x = int(round(WIDE_LINE_OFF_X * scale_x))
    wide_leg_x = int(round(WIDE_LINE_LEG_X * scale_x))

    # 1. Subtle Crease Turf Markings (only if guidelines enabled and never blocking batsman)
    if draw_guidelines:
        # Subtle wide tick marks on turf
        wide_color = (70, 180, 245)
        cv2.line(overlay, (wide_off_x, crease_y - 4), (wide_off_x, crease_y + 4), wide_color, 1, cv2.LINE_AA)
        cv2.line(overlay, (wide_leg_x, crease_y - 4), (wide_leg_x, crease_y + 4), wide_color, 1, cv2.LINE_AA)

    # 2. Three Wooden Stumps (only when unoccluded / explicitly requested)
    if draw_stump_poles:
        stump_color = (255, 255, 255)
        stump_outline = (30, 40, 50)
        stump_radius = max(1, int(round(1.5 * scale_x)))

        for sx in [off_x, mid_x, leg_x]:
            cv2.line(overlay, (sx, top_y), (sx, bot_y), stump_outline, stump_radius + 2, cv2.LINE_AA)
            cv2.line(overlay, (sx, top_y), (sx, bot_y), stump_color, stump_radius, cv2.LINE_AA)
            cv2.circle(overlay, (sx, bot_y), stump_radius + 1, (40, 120, 60), -1, cv2.LINE_AA)

        # Bails resting across stumps
        cv2.line(overlay, (off_x - 3, top_y), (leg_x + 3, top_y), (255, 235, 100), 2, cv2.LINE_AA)

    if draw_guidelines or draw_stump_poles:
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # 7. Broadcast Telemetry Badge for Stump Line (if specified)
    if draw_badge and stump_line_label:
        badge_text = f"STUMP LINE: {stump_line_label.upper()}"
        # Position top-right below main HUD
        bx = int(round(1450 * scale_x))
        by = int(round(88 * scale_y))
        bw = int(round(430 * scale_x))
        bh = int(round(38 * scale_y))

        # Color based on line
        if "OFF STUMP" in stump_line_label.upper():
            badge_border = (255, 200, 60)
            text_color = (255, 230, 120)
        elif "MIDDLE" in stump_line_label.upper():
            badge_border = (60, 220, 100)
            text_color = (120, 255, 160)
        elif "LEG" in stump_line_label.upper() and "WIDE" not in stump_line_label.upper():
            badge_border = (80, 180, 255)
            text_color = (140, 210, 255)
        elif "WIDE" in stump_line_label.upper():
            badge_border = (50, 100, 255)  # Amber / Red warning
            text_color = (100, 150, 255)
        else:
            badge_border = (200, 200, 200)
            text_color = (240, 240, 240)

        # Draw pill badge
        sub_badge = canvas.copy()
        cv2.rectangle(sub_badge, (bx, by), (bx + bw, by + bh), (15, 18, 24), -1)
        cv2.addWeighted(sub_badge, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), badge_border, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            badge_text,
            (bx + 14, by + bh - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * min(scale_x, scale_y),
            text_color,
            2,
            cv2.LINE_AA,
        )

    return canvas

