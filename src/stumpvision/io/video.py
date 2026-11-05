"""
video.py
========
Streamed video inspection and frame decoding for StumpVision.
Guaranteed strictly read-only: never modifies, moves, or writes to video source files.
"""

import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2

logger = logging.getLogger("stumpvision.io.video")


def parse_sequence_number(filename: str) -> Optional[int]:
    """
    Extract the numeric sequence number suffix from filename.
    Example: '20220102063650_000130.MP4' -> 130
    Does NOT use the leading date timestamp (camera clock unreliable).
    """
    stem = Path(filename).stem
    # Match trailing numeric block after underscore, or trailing digits
    match = re.search(r"_(\d+)$", stem)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", stem)
    return int(digits[-1]) if digits else None


def verify_cfr_ffprobe(video_path: Path) -> Tuple[str, Optional[str]]:
    """
    Check if ffprobe is available to verify Constant Frame Rate (CFR).
    Returns (status, detail).
    status is 'verified_cfr', 'variable_frame_rate', or 'not verified'.
    """
    ffprobe_cmd = shutil.which("ffprobe")
    if not ffprobe_cmd:
        return "not verified", "ffprobe not found on system PATH"

    cmd = [
        ffprobe_cmd,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "json",
        str(video_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        streams = data.get("streams", [])
        if not streams:
            return "not verified", "no video streams found by ffprobe"
        st = streams[0]
        r_fps = st.get("r_frame_rate", "")
        avg_fps = st.get("avg_frame_rate", "")
        if r_fps and avg_fps and r_fps == avg_fps:
            return "verified_cfr", f"r_fps={r_fps} matches avg_fps={avg_fps}"
        else:
            return "variable_frame_rate", f"r_fps={r_fps} != avg_fps={avg_fps}"
    except Exception as e:
        return "not verified", f"ffprobe error: {e}"


def count_decoded_frames(video_path: Path) -> Tuple[int, float]:
    """
    Stream and decode all frames to count actual decodable frames.
    Uses cap.grab() to avoid full decompress overhead while verifying stream integrity.
    Never loads full video into memory.
    Returns (decoded_count, elapsed_seconds).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file for decoding: {video_path}")

    t0 = time.time()
    count = 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break
        count += 1
    cap.release()
    elapsed = time.time() - t0
    return count, elapsed


def probe_video(
    video_path: Path,
    camera: str,
    data_dir: Path,
    count_frames: bool = False,
    schema_version: str = "1.0",
) -> Dict[str, Any]:
    """
    Probe video metadata using OpenCV (and ffprobe if present).
    Guarantees read-only access and verifies file mtime/size remain unchanged.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Record stat before probe
    stat_before = video_path.stat()
    size_before = stat_before.st_size
    mtime_before = stat_before.st_mtime

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Verify stat after probe
    stat_after = video_path.stat()
    if stat_after.st_size != size_before or stat_after.st_mtime != mtime_before:
        raise RuntimeError(f"CRITICAL SAFETY VIOLATION: File modified during probe: {video_path}")

    duration_s = round(n_frames_reported / fps, 2) if fps > 0 else 0.0

    # Constant frame rate check
    cfr_status, _ = verify_cfr_ffprobe(video_path)

    # Decoded frame count if requested
    n_frames_decoded = None
    if count_frames:
        decoded_count, _ = count_decoded_frames(video_path)
        n_frames_decoded = decoded_count

    try:
        rel_path = str(video_path.relative_to(data_dir)).replace("\\", "/")
    except ValueError:
        rel_path = str(video_path.name)

    seq_no = parse_sequence_number(video_path.name)

    return {
        "clip_id": video_path.stem,
        "relative_path": rel_path,
        "camera": camera,
        "sequence_no": seq_no if seq_no is not None else -1,
        "file_size_bytes": size_before,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "n_frames_reported": n_frames_reported,
        "n_frames_decoded": n_frames_decoded if n_frames_decoded is not None else "",
        "duration_s": duration_s,
        "cfr_status": cfr_status,
        "schema_version": schema_version,
    }
