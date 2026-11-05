"""
conftest.py
===========
Pytest fixtures and synthetic video generator for StumpVision testing.
Generates small synthetic test videos in-memory/temp folders without touching real footage.
"""

from pathlib import Path
import numpy as np
import pytest
import cv2


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """
    Generate a 1-second, 30 fps, 640x360 synthetic test video with drawn patterns.
    """
    video_path = tmp_path / "synthetic_cam1_000001.mp4"
    fps = 30.0
    width = 640
    height = 360
    total_frames = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert writer.isOpened(), "Could not open VideoWriter for synthetic video fixture"

    for i in range(total_frames):
        # Create synthetic frame with frame index text and moving rectangle
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Moving box across frame
        bx = int((i / total_frames) * (width - 60))
        cv2.rectangle(frame, (bx, 150), (bx + 60, 210), (0, 255, 0), -1)
        cv2.putText(
            frame, f"Frame {i:03d}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        writer.write(frame)

    writer.release()
    return video_path
