"""
signals package for StumpVision AI Delivery Finder.
Provides streaming temporal signal extraction and multimodal biomechanical fusion.
"""

from stumpvision.signals.motion import extract_motion_signal, save_signal_csv
from stumpvision.signals.batsman import (
    classify_batsman_pose,
    calculate_joint_angle,
    BatsmanStatus,
)
from stumpvision.signals.fused_detector import (
    fuse_motion_with_batsman_pose,
    load_poses_as_dataframe,
    FusedFilterStats,
)

__all__ = [
    "extract_motion_signal",
    "save_signal_csv",
    "classify_batsman_pose",
    "calculate_joint_angle",
    "BatsmanStatus",
    "fuse_motion_with_batsman_pose",
    "load_poses_as_dataframe",
    "FusedFilterStats",
]
