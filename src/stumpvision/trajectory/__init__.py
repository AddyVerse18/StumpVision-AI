"""
trajectory package for StumpVision AI.
Ball pitch trajectory tracking, delivery event windowing, and bounce localization (M5).
"""

from stumpvision.trajectory.tracker import (
    TrajectoryPoint,
    DeliveryTrajectory,
    BallTrajectoryTracker,
    track_all_deliveries,
)
from stumpvision.trajectory.viz import (
    render_trajectory_on_frame,
    draw_2d_pitch_map,
    draw_pitch_heatmap_diagram,
    draw_perspective_pitch_zone_diagram,
)
from stumpvision.trajectory.stumps import (
    classify_stump_line,
    draw_batter_stumps,
    StumpAnnotation,
)
from stumpvision.trajectory.drs import (
    DRSPrediction,
    evaluate_drs,
    render_drs_broadcast_card,
    calculate_speed_gun_metrics,
)
from stumpvision.trajectory.beehive import (
    render_beehive_chart,
)

__all__ = [
    "TrajectoryPoint",
    "DeliveryTrajectory",
    "BallTrajectoryTracker",
    "track_all_deliveries",
    "render_trajectory_on_frame",
    "draw_2d_pitch_map",
    "draw_pitch_heatmap_diagram",
    "draw_perspective_pitch_zone_diagram",
    "classify_stump_line",
    "draw_batter_stumps",
    "StumpAnnotation",
    "DRSPrediction",
    "evaluate_drs",
    "render_drs_broadcast_card",
    "calculate_speed_gun_metrics",
    "render_beehive_chart",
]
