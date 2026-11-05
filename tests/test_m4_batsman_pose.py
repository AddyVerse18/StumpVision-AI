"""
test_m4_batsman_pose.py
=======================
Automated verification tests for batsman 17-point skeletal pose estimation dataset
and biomechanical posture gating.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from stumpvision.signals.batsman import (
    calculate_joint_angle,
    classify_batsman_pose,
    BatsmanStatus,
)

POSE_DATA_PATH = Path("labels/m4_poses/batsman_pose_dev.json")
MANIFEST_PATH = Path("labels/m4_extracted_frames.csv")


def test_batsman_pose_file_exists():
    """Verify batsman pose data file exists and is valid JSON."""
    assert POSE_DATA_PATH.exists(), f"Pose data file not found at {POSE_DATA_PATH}"
    with open(POSE_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 54, f"Expected 54 pose records, found {len(data)}"


def test_batsman_pose_keypoints_structure():
    """Verify each pose record has valid 17-joint structure and valid ranges."""
    with open(POSE_DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        assert "file_name" in rec
        assert "crop_coords" in rec
        assert "keypoints" in rec
        
        kpts = rec["keypoints"]
        assert len(kpts) == 17, f"Expected 17 joints, got {len(kpts)} in {rec['file_name']}"
        
        for joint in kpts:
            assert len(joint) == 3, f"Joint must be [x, y, conf]: {joint}"
            x, y, conf = joint
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"
            if conf > 0.0:
                assert 0.0 <= x <= 1920.0, f"X coordinate out of bounds: {x}"
                assert 0.0 <= y <= 1080.0, f"Y coordinate out of bounds: {y}"


def test_batsman_pose_anatomical_integrity():
    """Verify key anatomical joints (head, shoulders, ankles) have solid detection rates."""
    with open(POSE_DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # COCO keypoint index mapping:
    # 0: nose, 5: left_shoulder, 6: right_shoulder, 15: left_ankle, 16: right_ankle
    nose_detected = sum(1 for r in records if r["keypoints"][0][2] > 0.3)
    shoulders_detected = sum(1 for r in records if (r["keypoints"][5][2] > 0.3 or r["keypoints"][6][2] > 0.3))
    
    assert nose_detected >= 40, f"Nose detection too low: {nose_detected}/54"
    assert shoulders_detected >= 45, f"Shoulder detection too low: {shoulders_detected}/54"


def test_calculate_joint_angle():
    """Verify knee/elbow angle trigonometry."""
    hip = np.array([100.0, 50.0])
    knee = np.array([100.0, 100.0])
    ankle = np.array([100.0, 150.0])
    # Straight line = 180 degrees
    angle_straight = calculate_joint_angle(hip, knee, ankle)
    assert abs(angle_straight - 180.0) < 1e-3

    # Right angle = 90 degrees
    ankle_right = np.array([150.0, 100.0])
    angle_90 = calculate_joint_angle(hip, knee, ankle_right)
    assert abs(angle_90 - 90.0) < 1e-3


def test_classify_batsman_posture_gating():
    """Verify stance gating identifies active batsman vs out-of-bounds/idle fielder."""
    with open(POSE_DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Frame 1 is a delivery frame with batsman in stance
    r0 = records[0]
    status = classify_batsman_pose(r0["keypoints"], title_hint="Stride")
    assert status.is_present is True
    assert status.action_type in ("STANCE", "SHOT_CONTACT")
    assert status.crease_in_bounds is True

    # Synthetic out-of-crease test (e.g. outfield fielder at x=200, y=800)
    outfield_kpts = [[200.0, 800.0, 0.9] for _ in range(17)]
    outfield_status = classify_batsman_pose(outfield_kpts)
    assert outfield_status.is_present is False
    assert outfield_status.action_type == "NO_BATSMAN"
    assert outfield_status.crease_in_bounds is False
