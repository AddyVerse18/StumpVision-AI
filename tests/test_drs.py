"""
test_drs.py
===========
Unit tests for Hawk-Eye DRS & LBW Virtual Predictor.
Validates ICC / MCC Law 36 rules, ballistic wicket projection, and radar speed gun.
"""

import pytest
import numpy as np

from stumpvision.trajectory.tracker import DeliveryTrajectory, TrajectoryPoint
from stumpvision.trajectory.drs import (
    evaluate_drs,
    calculate_speed_gun_metrics,
    render_drs_broadcast_card,
    DRSPrediction,
)
from stumpvision.trajectory.stumps import (
    OFF_STUMP_X,
    MIDDLE_STUMP_X,
    LEG_STUMP_X,
    POPPING_CREASE_Y,
    BOWLING_CREASE_Y,
    STUMPS_TOP_Y,
    STUMPS_BOTTOM_Y,
)


def make_test_trajectory(
    bounce_x: float,
    bounce_y: float,
    bat_x: float,
    bat_y: float,
    length: str = "good_length",
    dur: float = 0.35,
) -> DeliveryTrajectory:
    """Helper to synthesize a test trajectory for DRS evaluation."""
    p1 = TrajectoryPoint(frame_idx=100, time_s=10.0, x=735.0, y=150.0, radius=5.0, area=25.0)
    p2 = TrajectoryPoint(frame_idx=112, time_s=10.0 + dur * 0.6, x=bounce_x, y=bounce_y, radius=5.0, area=25.0)
    p3 = TrajectoryPoint(frame_idx=120, time_s=10.0 + dur, x=bat_x, y=bat_y, radius=5.0, area=25.0)

    return DeliveryTrajectory(
        clip_id="test_clip",
        delivery_idx=1,
        candidate_time_s=10.2,
        points=[p1, p2, p3],
        t_release=10.0,
        t_bounce=10.0 + dur * 0.6,
        t_impact=10.0 + dur,
        bounce_x=bounce_x,
        bounce_y=bounce_y,
        length_classification=length,
        line_classification="middle",
        is_valid=True,
    )


class TestSpeedGun:
    def test_speed_gun_metrics_calculation(self):
        traj = make_test_trajectory(bounce_x=860.0, bounce_y=430.0, bat_x=900.0, bat_y=540.0, dur=0.36)
        s_rel, s_mph, s_post, s_post_mph, drop, pace = calculate_speed_gun_metrics(traj)

        assert 100.0 <= s_rel <= 155.0
        assert s_mph == pytest.approx(round(s_rel * 0.621371, 1), abs=0.2)
        assert s_post < s_rel
        assert drop == 18.0
        assert pace in ["Express Fast", "Fast", "Fast-Medium", "Medium"]

    def test_speed_gun_fallback_on_empty(self):
        empty_traj = DeliveryTrajectory(clip_id="c", delivery_idx=1, candidate_time_s=0.0)
        s_rel, s_mph, s_post, s_post_mph, drop, pace = calculate_speed_gun_metrics(empty_traj)
        assert s_rel == 128.5
        assert pace == "Fast-Medium"


class TestDRSLegBeforeWicket:
    def test_law36_pitched_outside_leg_is_not_out(self):
        # Ball bounces far down leg side (x = 880 at y = 430, centerline at y=430 is ~855)
        # Offset > corridor -> pitched outside leg
        traj = make_test_trajectory(bounce_x=885.0, bounce_y=430.0, bat_x=901.5, bat_y=540.0)
        drs = evaluate_drs(traj)

        assert drs.pitching == "outside_leg"
        assert drs.decision == "NOT OUT"
        assert "Outside Leg" in drs.decision_reason

    def test_law36_impact_outside_off_is_not_out(self):
        # Pitching in-line, but bat impact far outside off (x = 850, off stump outer is 890.5)
        traj = make_test_trajectory(bounce_x=855.0, bounce_y=430.0, bat_x=850.0, bat_y=540.0)
        drs = evaluate_drs(traj)

        assert drs.pitching == "in_line"
        assert drs.impact == "outside_off"
        assert drs.decision == "NOT OUT"
        assert "Impact Outside Off" in drs.decision_reason

    def test_three_red_lights_lbw_out(self):
        # Perfectly straight delivery hitting stumps:
        # Pitched in-line at centerline (854.5 at y=430), impact in-line at 895.0, good length -> hitting stumps
        traj = make_test_trajectory(bounce_x=854.5, bounce_y=430.0, bat_x=895.0, bat_y=540.0, length="good_length")
        drs = evaluate_drs(traj)

        assert drs.pitching == "in_line"
        assert drs.impact == "in_line"
        assert drs.wickets == "hitting"
        assert drs.decision == "OUT (LBW)"
        assert drs.is_three_reds is True
        assert "Three Red Lights" in drs.decision_reason

    def test_missing_over_the_stumps_is_not_out(self):
        # Short pitch rising over bails (pitched in-line at y=380, centerline=833.2, length="short")
        traj = make_test_trajectory(bounce_x=833.2, bounce_y=380.0, bat_x=895.0, bat_y=540.0, length="short")
        drs = evaluate_drs(traj)

        assert drs.pitching == "in_line"
        assert drs.impact == "in_line"
        assert drs.wickets == "missing"
        assert drs.wickets_target == "over_stumps"
        assert drs.decision == "NOT OUT"
        assert "Over the Stumps" in drs.decision_reason

    def test_render_drs_broadcast_card(self):
        traj = make_test_trajectory(bounce_x=855.0, bounce_y=430.0, bat_x=901.5, bat_y=540.0)
        drs = evaluate_drs(traj)
        card = render_drs_broadcast_card(drs, width=700, height=190)

        assert isinstance(card, np.ndarray)
        assert card.shape == (190, 700, 3)
        assert np.any(card > 0)
