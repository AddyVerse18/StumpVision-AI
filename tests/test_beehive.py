"""
test_beehive.py
===============
Unit tests for the Front-On Stumps Beehive Visualizer.
"""

import pytest
import numpy as np

from stumpvision.trajectory.tracker import DeliveryTrajectory, TrajectoryPoint
from stumpvision.trajectory.beehive import render_beehive_chart


def make_sample_trajectories():
    trajs = []
    for i in range(1, 6):
        pts = [
            TrajectoryPoint(frame_idx=10, time_s=0.1, x=735.0, y=150.0, radius=5.0, area=25.0),
            TrajectoryPoint(frame_idx=25, time_s=0.25, x=850.0 + i * 10, y=430.0, radius=5.0, area=25.0),
            TrajectoryPoint(frame_idx=35, time_s=0.35, x=890.0 + i * 5, y=540.0, radius=5.0, area=25.0),
        ]
        t = DeliveryTrajectory(
            clip_id="test_clip",
            delivery_idx=i,
            candidate_time_s=1.0 * i,
            points=pts,
            t_release=0.1,
            t_bounce=0.25,
            t_impact=0.35,
            bounce_x=850.0 + i * 10,
            bounce_y=430.0,
            length_classification="good_length",
            line_classification="off",
            stump_line_display="Off Stump",
            is_valid=True,
        )
        trajs.append(t)
    return trajs


class TestBeehiveChart:
    def test_render_beehive_chart_dimensions(self):
        trajs = make_sample_trajectories()
        chart = render_beehive_chart(trajs, selected_delivery_idx=1, width=800, height=700)

        assert isinstance(chart, np.ndarray)
        assert chart.shape == (700, 800, 3)
        assert np.any(chart > 0)

    def test_render_beehive_with_dict_trajectories(self):
        trajs = [t.to_dict() for t in make_sample_trajectories()]
        chart = render_beehive_chart(trajs, selected_delivery_idx=3, width=800, height=700)

        assert isinstance(chart, np.ndarray)
        assert chart.shape == (700, 800, 3)

    def test_render_beehive_empty_list(self):
        chart = render_beehive_chart([], selected_delivery_idx=None, width=800, height=700)
        assert chart.shape == (700, 800, 3)
        assert np.any(chart > 0)

