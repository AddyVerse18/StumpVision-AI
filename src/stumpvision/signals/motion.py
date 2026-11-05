"""
motion.py
=========
Streaming motion signal extraction for StumpVision (M3).
Computes mean absolute frame differences on downscaled grayscale frames.
Strictly stream-based: never loads full video into memory.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
import pandas as pd


def extract_motion_signal(
    video_path: Path,
    downscale_width: int = 160,
    downscale_height: int = 90,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Stream video and compute normalized frame-difference motion signal.

    Args:
        video_path: Path to video file.
        downscale_width: Width to resize frames for difference computation.
        downscale_height: Height to resize frames for difference computation.
        max_frames: Optional cap on frames to decode (useful for fast tests).

    Returns:
        dict containing:
            - frame_indices: np.ndarray of shape (N,)
            - time_s: np.ndarray of shape (N,)
            - motion: np.ndarray of shape (N,)
            - fps: float
            - total_frames: int
            - duration_s: float
            - width: int (original width)
            - height: int (original height)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or np.isnan(fps):
        fps = 60.0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        return {
            "frame_indices": np.array([], dtype=int),
            "time_s": np.array([], dtype=float),
            "motion": np.array([], dtype=float),
            "motion_top": np.array([], dtype=float),
            "motion_mid": np.array([], dtype=float),
            "motion_bot": np.array([], dtype=float),
            "fps": fps,
            "total_frames": 0,
            "duration_s": 0.0,
            "width": orig_w,
            "height": orig_h,
        }

    prev_gray = cv2.resize(
        cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY),
        (downscale_width, downscale_height),
    )

    h_third1 = downscale_height // 3
    h_third2 = (2 * downscale_height) // 3

    frame_indices = [0]
    time_stamps = [0.0]
    motion_values = [0.0]  # Frame 0 has 0 motion by definition
    motion_top_values = [0.0]
    motion_mid_values = [0.0]
    motion_bot_values = [0.0]

    frame_count = 1
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (downscale_width, downscale_height),
        )

        diff_map = cv2.absdiff(gray, prev_gray)
        diff = float(np.mean(diff_map))
        diff_top = float(np.mean(diff_map[0:h_third1, :]))
        diff_mid = float(np.mean(diff_map[h_third1:h_third2, :]))
        diff_bot = float(np.mean(diff_map[h_third2:, :]))

        frame_indices.append(frame_count)
        time_stamps.append(frame_count / fps)
        motion_values.append(diff)
        motion_top_values.append(diff_top)
        motion_mid_values.append(diff_mid)
        motion_bot_values.append(diff_bot)

        prev_gray = gray
        frame_count += 1

    cap.release()

    n = len(motion_values)
    dur_s = (n / fps) if fps > 0 else 0.0

    return {
        "frame_indices": np.array(frame_indices, dtype=int),
        "time_s": np.array(time_stamps, dtype=float),
        "motion": np.array(motion_values, dtype=float),
        "motion_top": np.array(motion_top_values, dtype=float),
        "motion_mid": np.array(motion_mid_values, dtype=float),
        "motion_bot": np.array(motion_bot_values, dtype=float),
        "fps": fps,
        "total_frames": n,
        "duration_s": dur_s,
        "width": orig_w,
        "height": orig_h,
    }


def save_signal_csv(signal_data: Dict[str, Any], output_path: Path) -> Path:
    """Save extracted motion signal to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "frame_index": signal_data["frame_indices"],
        "time_s": np.round(signal_data["time_s"], 4),
        "motion": np.round(signal_data["motion"], 4),
    })
    df.to_csv(output_path, index=False)
    return output_path

