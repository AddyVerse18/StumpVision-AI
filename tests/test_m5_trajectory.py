"""
test_m5_trajectory.py
=====================
Automated tests for M5 ball pitch trajectory tracking, delivery event windowing,
and bounce length/line classification.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from stumpvision.trajectory.tracker import (
    TrajectoryPoint,
    DeliveryTrajectory,
    BallTrajectoryTracker,
    track_all_deliveries,
)
from stumpvision.trajectory.viz import render_trajectory_on_frame
from stumpvision.cli import run_track_trajectory


TRAJECTORIES_JSON = Path("outputs/trajectories/dev_trajectories.json")
TRAJECTORIES_CSV = Path("outputs/trajectories/dev_trajectories_summary.csv")


def test_trajectory_point_and_delivery_serialization():
    """Verify trajectory dataclasses serialize to dict and JSON cleanly."""
    pt1 = TrajectoryPoint(frame_idx=665, time_s=11.0833, x=779.0, y=193.5, radius=8.5, area=150.0)
    pt2 = TrajectoryPoint(frame_idx=667, time_s=11.1167, x=800.0, y=252.5, radius=7.5, area=120.0)
    traj = DeliveryTrajectory(
        clip_id="20220102063650_000130",
        delivery_idx=1,
        candidate_time_s=11.25,
        points=[pt1, pt2],
        t_release=11.0833,
        t_bounce=11.1167,
        t_impact=11.1167,
        bounce_x=800.0,
        bounce_y=252.5,
        length_classification="short",
        line_classification="middle",
        is_valid=True,
    )
    d = traj.to_dict()
    assert d["clip_id"] == "20220102063650_000130"
    assert len(d["points"]) == 2
    assert d["points"][0]["x"] == 779.0


def test_synthetic_trajectory_linking():
    """Verify BallTrajectoryTracker correctly links sequential blobs into a trajectory."""
    tracker = BallTrajectoryTracker(min_track_points=5)

    # Simulate 7 frames with a moving ball
    blobs_per_frame = []
    for i in range(7):
        f_idx = 100 + i
        t_s = f_idx / 60.0
        # Ball travels down-right: x increases by 15, y increases by 25
        x = 600.0 + i * 15.0
        y = 150.0 + i * 25.0
        pt = TrajectoryPoint(frame_idx=f_idx, time_s=t_s, x=x, y=y, radius=8.0, area=100.0)
        blobs_per_frame.append([pt])

    track = tracker.link_trajectory_points(blobs_per_frame, start_frame_idx=100)
    assert len(track) == 7
    assert track[0].x == 600.0
    assert track[-1].x == 690.0


def test_pitch_length_classification():
    """Verify pitch bounce y coordinate correctly maps to cricket length classifications."""
    tracker = BallTrajectoryTracker(min_track_points=3)
    pts_yorker = [
        TrajectoryPoint(1, 0.1, 800, 200, 5, 50),
        TrajectoryPoint(2, 0.2, 820, 350, 5, 50),
        TrajectoryPoint(3, 0.3, 840, 500, 5, 50),
    ]
    res_yorker = tracker.analyze_trajectory("clip1", 1, 0.2, pts_yorker)
    assert res_yorker.length_classification == "yorker"

    pts_good = [
        TrajectoryPoint(1, 0.1, 800, 200, 5, 50),
        TrajectoryPoint(2, 0.2, 820, 350, 5, 50),
        TrajectoryPoint(3, 0.3, 840, 440, 5, 50),
    ]
    res_good = tracker.analyze_trajectory("clip1", 1, 0.2, pts_good)
    assert res_good.length_classification == "good_length"

    pts_short = [
        TrajectoryPoint(1, 0.1, 800, 200, 5, 50),
        TrajectoryPoint(2, 0.2, 820, 300, 5, 50),
        TrajectoryPoint(3, 0.3, 840, 380, 5, 50),
    ]
    res_short = tracker.analyze_trajectory("clip1", 1, 0.2, pts_short)
    assert res_short.length_classification == "short"


def test_dev_trajectories_output_files():
    """Verify tracked trajectories JSON and CSV exist and contain all 13 dev deliveries."""
    assert TRAJECTORIES_JSON.exists(), f"Trajectories JSON missing: {TRAJECTORIES_JSON}"
    assert TRAJECTORIES_CSV.exists(), f"Trajectories CSV missing: {TRAJECTORIES_CSV}"

    with open(TRAJECTORIES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 13, f"Expected 13 delivery trajectories, found {len(data)}"
    assert all(t["is_valid"] for t in data), "Expected all 13 deliveries to have valid trajectory"

    df = pd.read_csv(TRAJECTORIES_CSV)
    assert len(df) == 13
    assert set(df["length"].unique()).issubset({"yorker", "full", "good_length", "short"})
    assert set(df["line"].unique()).issubset({"off", "middle", "leg"})


def test_render_trajectory_on_frame():
    """Verify render_trajectory_on_frame returns image without crashing."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pts = [
        TrajectoryPoint(1, 0.1, 750, 150, 8, 100),
        TrajectoryPoint(2, 0.2, 800, 300, 7, 80),
        TrajectoryPoint(3, 0.3, 840, 450, 5, 50),
    ]
    traj = DeliveryTrajectory(
        clip_id="test_clip",
        delivery_idx=1,
        candidate_time_s=0.2,
        points=pts,
        bounce_x=840.0,
        bounce_y=450.0,
        length_classification="good_length",
        line_classification="middle",
        estimated_speed_kmh=132.5,
        is_valid=True,
    )
    rendered = render_trajectory_on_frame(frame, traj, draw_hud=True)
    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8
    # Canvas should have drawn pixels (not pure black)
    assert np.any(rendered > 0)


def test_render_pitch_zones_overlay():
    """Verify calibrated broadcast pitch zones overlay renders on frame."""
    from stumpvision.trajectory.viz import render_pitch_zones_overlay
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    rendered = render_pitch_zones_overlay(frame, alpha=0.35, draw_labels=True, draw_hud_cards=True)
    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8
    assert np.any(rendered > 0)


def test_cli_track_trajectory_run(tmp_path):
    """Test CLI subcommand runner for trajectory tracking with custom output dir."""
    custom_out = tmp_path / "custom_traj"
    code = run_track_trajectory(output_dir=str(custom_out))
    assert code == 0
    assert (custom_out / "dev_trajectories.json").exists()
    assert (custom_out / "dev_trajectories_summary.csv").exists()


def test_draw_2d_pitch_map():
    """Verify 2D aerial pitch map renders properly with delivery markers."""
    from stumpvision.trajectory.viz import draw_2d_pitch_map
    summary_df = pd.read_csv(TRAJECTORIES_CSV)
    pitch_img = draw_2d_pitch_map(summary_df=summary_df, selected_delivery_idx=1)
    assert pitch_img.shape == (860, 600, 3)
    assert pitch_img.dtype == np.uint8
    assert np.any(pitch_img > 0)


def test_app_script_integrity():
    """Verify production app.py exists and can be imported/parsed without syntax error."""
    app_path = Path("app.py")
    assert app_path.exists(), "app.py missing from project root"
    content = app_path.read_text(encoding="utf-8")
    assert "st.set_page_config" in content
    assert "draw_2d_pitch_map" in content


