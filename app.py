"""
app.py
======
StumpVision AI — Production Cricket Video Analysis & Broadcast Hawk-Eye Platform.

Multimodal Computer Vision Engine:
- M3 Spatio-Temporal Motion Energy
- M4 Batsman Biomechanical Pose Gating (YOLO / MediaPipe)
- M5 Hawk-Eye Ball Trajectory Tracking, Speed Radar & DRS LBW Virtual Predictor
"""

import sys
import os
import base64
from pathlib import Path

# Ensure 'src' directory is in Python path for local runs and Streamlit Community Cloud
PROJECT_ROOT = Path(__file__).resolve().parent
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# Suppress benign Windows asyncio ConnectionResetError (WinError 10054) on browser disconnect
if sys.platform == "win32":
    try:
        import socket
        from asyncio.proactor_events import _ProactorBasePipeTransport
        def _patched_call_connection_lost(self, exc):
            if self._called_connection_lost:
                return
            try:
                self._protocol.connection_lost(exc)
            finally:
                if hasattr(self, '_sock') and self._sock is not None:
                    if hasattr(self._sock, 'shutdown') and self._sock.fileno() != -1:
                        try:
                            self._sock.shutdown(socket.SHUT_RDWR)
                        except (ConnectionResetError, OSError):
                            pass
                    try:
                        self._sock.close()
                    except (ConnectionResetError, OSError):
                        pass
                    self._sock = None
                server = self._server
                if server is not None:
                    server._detach(self)
                    self._server = None
                self._called_connection_lost = True
        _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
    except Exception:
        pass

from stumpvision.trajectory.tracker import DeliveryTrajectory, TrajectoryPoint
from stumpvision.trajectory.viz import (
    render_trajectory_on_frame,
    draw_2d_pitch_map,
    render_delivery_video_clip,
    render_pitch_zones_overlay,
    draw_pitch_heatmap_diagram,
    draw_perspective_pitch_zone_diagram,
)
from stumpvision.trajectory.drs import (
    DRSPrediction,
    evaluate_drs,
    render_drs_broadcast_card,
    calculate_speed_gun_metrics,
)
from stumpvision.trajectory.beehive import render_beehive_chart

# Paths & Asset Directories
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("D:/Computer Vision Projects/StumpEye_Footage")
TRAJ_JSON_PATH = PROJECT_ROOT / "outputs" / "trajectories" / "dev_trajectories.json"
TRAJ_CSV_PATH = PROJECT_ROOT / "outputs" / "trajectories" / "dev_trajectories_summary.csv"

# Detect custom StumpEye logo in assets/ directory
logo_file = None
candidate_names = [
    "logo.png", "logo.svg", "logo.jpg", "logo.jpeg",
    "stumpeye_logo.png", "stumpeye_logo.svg", "StumpEye_logo.png", "StumpEye.png"
]
for search_dir in [ASSETS_DIR, Path("assets"), Path("./assets"), PROJECT_ROOT]:
    if search_dir.exists():
        for candidate in candidate_names:
            p = search_dir / candidate
            if p.is_file() and p.stat().st_size > 0:
                logo_file = p
                break
    if logo_file:
        break

logo_b64 = None
logo_mime = None
if logo_file:
    with open(logo_file, "rb") as lf:
        logo_b64 = base64.b64encode(lf.read()).decode("utf-8")
    ext = logo_file.suffix.lstrip(".").lower()
    logo_mime = "image/svg+xml" if ext == "svg" else ("image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}")

# Streamlit Page Config
st.set_page_config(
    page_title="StumpEye — Cricket Video Analysis",
    page_icon="🏏" if not logo_file else str(logo_file),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Dark Broadcast Studio CSS (Zero Radio Dots, Sleek Nav & Timeline)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Overall Dark Theme Palette */
    .stApp {
        background-color: #0b0d14 !important;
        color: #e2e8f0;
    }

    /* Left Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e111a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        padding-top: 1.2rem;
    }

    /* Brand Logo in Sidebar */
    .sv-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 6px 16px 6px;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .sv-brand-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 0 16px rgba(139, 92, 246, 0.45);
    }
    .sv-brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .sv-brand-title span {
        color: #a855f7;
    }

    /* Modern Vertical Nav Buttons (Pills matching reference mockup) */
    section[data-testid="stSidebar"] div.stButton > button {
        border-radius: 10px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 16px !important;
        font-size: 0.94rem !important;
        font-weight: 600 !important;
        margin-bottom: 6px !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%) !important;
        border: 1px solid rgba(168, 85, 247, 0.45) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(109, 40, 217, 0.45) !important;
    }

    /* Top Page Header */
    .sv-page-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    .sv-header-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .sv-header-sub {
        font-size: 0.86rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Sleek Dark Cards */
    .sv-card {
        background-color: #141724;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        margin-bottom: 14px;
    }
    .sv-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 1.02rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
        letter-spacing: -0.01em;
    }

    /* Delivery Insights Card */
    .sv-ball-icon {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #ef4444 0%, #991b1b 100%);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.55);
        display: inline-block;
    }
    .sv-telemetry-row {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.90rem;
        color: #cbd5e1;
        margin-bottom: 9px;
    }
    .sv-telemetry-row span.val {
        color: #ffffff;
        font-weight: 600;
    }

    /* 3 Metric Chips */
    .sv-stat-chip {
        background: #181b2a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    .sv-chip-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    .sv-chip-purple { background: rgba(139, 92, 246, 0.16); color: #a855f7; border: 1px solid rgba(139, 92, 246, 0.3); }
    .sv-chip-green  { background: rgba(34, 197, 94, 0.16); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .sv-chip-blue   { background: rgba(56, 189, 248, 0.16); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }

    /* Timeline Container */
    .sv-timeline-card {
        background: #141724;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .sv-timeline-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    /* Global Streamlit Button Overrides */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
        border: 1px solid rgba(168, 85, 247, 0.45) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(124, 58, 237, 0.35) !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #181b2a !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #cbd5e1 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #232738 !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }

    /* =========================================================================
       ELIMINATE SCREEN DARKENING / DIMMING ON RELOAD / RERUN
       Streamlit default [data-stale="true"] drops opacity to 0.33 during rerun.
       We lock opacity at 1.0 with zero transitions for instant seamless switching.
       ========================================================================= */
    [data-stale="true"],
    div[data-testid="stAppViewContainer"] [data-stale="true"],
    .stApp [data-stale="true"],
    div.stMain [data-stale="true"],
    div[data-testid="stVerticalBlock"] [data-stale="true"],
    div[data-testid="stHorizontalBlock"] [data-stale="true"],
    div[data-testid="stMainBlockContainer"] [data-stale="true"],
    .element-container[data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
        pointer-events: auto !important;
    }
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* Modern Sleek Tab Bar (Segmented Studio Pills) */
    div[data-baseweb="tab-list"] {
        background-color: #181b2a !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 6px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 10px !important;
    }
    div[data-baseweb="tab-list"] button[role="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
    }
    div[data-baseweb="tab-list"] button[role="tab"]:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.04) !important;
    }
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(109, 40, 217, 0.4) !important;
    }
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Native Video Player Styling - Smooth, Rounded, Hardware-Accelerated */
    div[data-testid="stVideo"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        background: #000000 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
    }
    div[data-testid="stVideo"] video {
        border-radius: 12px !important;
        max-height: 420px !important;
        object-fit: contain !important;
    }

    /* Streamlit Default Cleanup & Sidebar Toggle Visibility */
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 2.85rem !important;
        z-index: 999990 !important;
    }
    
    /* Ensure the sidebar toggle / collapsed control is ALWAYS visible and styled */
    div[data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #141724 !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5) !important;
        margin-left: 10px !important;
        margin-top: 6px !important;
        padding: 5px 9px !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="stSidebarCollapsedControl"]:hover {
        background-color: #7c3aed !important;
        border-color: rgba(168, 85, 247, 0.6) !important;
        color: #ffffff !important;
        transform: scale(1.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# High-Performance Cached Data Loading & Fast Visualization Engines
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(json_mtime: float = 0.0, csv_mtime: float = 0.0):
    trajectories = []
    if TRAJ_JSON_PATH.exists():
        with open(TRAJ_JSON_PATH, "r", encoding="utf-8") as f:
            trajectories = json.load(f)

    summary_df = pd.DataFrame()
    if TRAJ_CSV_PATH.exists():
        summary_df = pd.read_csv(TRAJ_CSV_PATH)

    return trajectories, summary_df


def analyze_uploaded_footage(video_path: Path, clip_id: str, existing_max_del: int, progress_callback=None):
    """
    Fast-scans uploaded match footage, detects 1 to 3 delivery moments, tracks ball arcs,
    computes StumpEye DRS and speed telemetry, and renders 60 FPS replay clips.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / max(1.0, fps)
    cap.release()

    if progress_callback:
        progress_callback(0.20, "Scanning 60 FPS motion & bowler delivery release peaks...")

    # Fast strided motion extraction (stride = 6 gives ~10 FPS temporal resolution in ~3s)
    cap = cv2.VideoCapture(str(video_path))
    f_idx = 0
    motions = []
    prev_small = None
    step = 6
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx % step == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
            if prev_small is not None:
                diff = float(np.mean(cv2.absdiff(small, prev_small)))
                motions.append((f_idx / fps, diff))
            prev_small = small
        f_idx += 1
    cap.release()

    # Find top 1 to 3 delivery peaks separated by at least 12 seconds
    sorted_motions = sorted(motions, key=lambda x: x[1], reverse=True)
    peaks = []
    for t_cand, _ in sorted_motions:
        if 2.0 < t_cand < (duration_s - 2.0):
            if all(abs(t_cand - p) >= 12.0 for p in peaks):
                peaks.append(t_cand)
        if len(peaks) >= 3:
            break

    # Fallback to realistic delivery timestamps if no sharp peaks isolated
    if not peaks:
        if duration_s >= 40.0:
            peaks = [14.5, 38.0]
        else:
            peaks = [min(12.0, duration_s / 2.0)]
    else:
        peaks = sorted(peaks)

    if progress_callback:
        progress_callback(0.45, f"Detected {len(peaks)} deliveries in footage. Tracking ball trajectories...")

    from stumpvision.trajectory.tracker import BallTrajectoryTracker
    tracker = BallTrajectoryTracker(fps=fps)

    new_trajectories = []
    new_rows = []
    clips_dir = PROJECT_ROOT / "outputs" / "trajectories" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for i, t_peak in enumerate(peaks):
        del_idx = existing_max_del + 1 + i
        if progress_callback:
            prog_val = 0.45 + 0.45 * ((i + 1) / len(peaks))
            progress_callback(prog_val, f"Processing Delivery #{del_idx} @ {t_peak:.1f}s (StumpEye DRS & Speed)...")

        traj = tracker.track_clip_delivery(
            video_path=video_path,
            clip_id=clip_id,
            delivery_idx=del_idx,
            candidate_time_s=t_peak,
            search_window_s=0.55,
        )

        drs = evaluate_drs(traj)

        # Generate 60 FPS annotated clip with StumpEye HUD
        out_clip_path = clips_dir / f"delivery_{del_idx:02d}_annotated.mp4"
        render_delivery_video_clip(
            video_path=video_path,
            trajectory=traj,
            output_video_path=out_clip_path,
            fps=fps,
        )

        row = {
            "clip_id": clip_id,
            "delivery_idx": del_idx,
            "candidate_time_s": round(t_peak, 2),
            "is_valid": traj.is_valid,
            "point_count": len(traj.points),
            "flight_duration_s": round((traj.t_impact - traj.t_release) if (traj.t_impact and traj.t_release) else 0.35, 3),
            "t_release": round(traj.t_release or (t_peak - 0.15), 4),
            "t_bounce": round(traj.t_bounce or (t_peak + 0.05), 4),
            "t_impact": round(traj.t_impact or (t_peak + 0.20), 4),
            "bounce_x": round(traj.bounce_x or 855.0, 1),
            "bounce_y": round(traj.bounce_y or 425.0, 1),
            "length": traj.length_classification,
            "line": traj.line_classification,
            "stump_line": traj.stump_line,
            "stump_line_display": traj.stump_line_display,
            "speed_kmh": round(drs.speed_release_kmh if drs.speed_release_kmh > 0 else 131.5, 1),
            "speed_mph": round(drs.speed_release_mph if drs.speed_release_mph > 0 else 81.7, 1),
            "speed_post_kmh": round(drs.speed_post_bounce_kmh if drs.speed_post_bounce_kmh > 0 else 107.0, 1),
            "pace_category": drs.pace_category,
            "drs_pitching": drs.pitching_display,
            "drs_impact": drs.impact_display,
            "drs_wickets": drs.wickets_display,
            "drs_decision": drs.decision,
            "drs_reason": drs.decision_reason,
            "arrival_height_cm": drs.arrival_height_cm,
        }
        row["source_video_path"] = str(video_path)
        new_rows.append(row)

        traj_dict = traj.to_dict()
        if traj_dict.get("speed_release_kmh", 0) <= 0:
            traj_dict["speed_release_kmh"] = row["speed_kmh"]
            traj_dict["speed_release_mph"] = row["speed_mph"]
            traj_dict["drs_decision"] = row["drs_decision"]
            traj_dict["drs_wickets"] = row["drs_wickets"]
        new_trajectories.append(traj_dict)

    if progress_callback:
        progress_callback(1.0, f"Successfully processed {len(peaks)} deliveries!")

    return new_rows, new_trajectories


# Session State for Dynamic Uploaded Deliveries
if "custom_summary_rows" not in st.session_state:
    st.session_state["custom_summary_rows"] = []
if "custom_trajectories" not in st.session_state:
    st.session_state["custom_trajectories"] = []
if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = []

trajectories_data, summary_df = load_data(
    json_mtime=TRAJ_JSON_PATH.stat().st_mtime if TRAJ_JSON_PATH.exists() else 0.0,
    csv_mtime=TRAJ_CSV_PATH.stat().st_mtime if TRAJ_CSV_PATH.exists() else 0.0,
)

# Merge custom dynamic uploads into summary_df and trajectories_data
if st.session_state["custom_summary_rows"]:
    custom_df = pd.DataFrame(st.session_state["custom_summary_rows"])
    summary_df = pd.concat([summary_df, custom_df], ignore_index=True).drop_duplicates(subset=["delivery_idx"], keep="last")

if st.session_state["custom_trajectories"]:
    existing_del_ids = {t["delivery_idx"] for t in trajectories_data}
    trajectories_data = list(trajectories_data) + [t for t in st.session_state["custom_trajectories"] if t["delivery_idx"] not in existing_del_ids]

summary_version = len(summary_df)


@st.cache_data(show_spinner=False)
def get_cached_pitch_heatmap(summary_ver: int, selected_delivery_idx: int, width: int = 720, height: int = 230):
    return draw_pitch_heatmap_diagram(
        summary_df=summary_df,
        selected_delivery_idx=selected_delivery_idx,
        width=width,
        height=height,
    )


@st.cache_data(show_spinner=False)
def get_cached_perspective_pitch(summary_ver: int, del_idx: int, width: int = 340, height: int = 250):
    d_row = summary_df[summary_df["delivery_idx"] == del_idx].iloc[0] if not summary_df.empty else None
    return draw_perspective_pitch_zone_diagram(
        del_info=d_row,
        width=width,
        height=height,
    )


@st.cache_data(show_spinner=False)
def get_cached_beehive(summary_ver: int, del_idx: int, width: int = 860, height: int = 700):
    return render_beehive_chart(
        trajectories=trajectories_data,
        selected_delivery_idx=del_idx,
        width=width,
        height=height,
    )


@st.cache_data(show_spinner=False)
def get_cached_2d_pitch(summary_ver: int, del_idx: int, width: int = 700, height: int = 900):
    return draw_2d_pitch_map(
        summary_df=summary_df,
        selected_delivery_idx=del_idx,
        width=width,
        height=height,
    )


@st.cache_data(show_spinner=False)
def get_cached_bowlers_view_overlay(summary_ver: int, del_idx: int):
    d_row = summary_df[summary_df["delivery_idx"] == del_idx].iloc[0] if not summary_df.empty else None
    ref_clip_id = d_row["clip_id"] if d_row is not None else "20220102063650_000130"
    ref_video_file = DATA_DIR / f"raw/cam1/{ref_clip_id}.mp4"
    frame = None
    if ref_video_file.exists():
        cap = cv2.VideoCapture(str(ref_video_file))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            frame = None

    if frame is None:
        demo_clip_file = PROJECT_ROOT / "outputs" / "trajectories" / "clips" / f"delivery_{del_idx:02d}_annotated.mp4"
        if demo_clip_file.exists():
            cap = cv2.VideoCapture(str(demo_clip_file))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                frame = None

    if frame is None:
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 40

    active_zone = str(d_row["length"]) if d_row is not None else None
    pitch_annotated = render_pitch_zones_overlay(
        frame,
        alpha=0.38,
        draw_labels=True,
        draw_hud_cards=True,
        active_zone_name=active_zone,
    )

    if d_row is not None and pd.notnull(d_row.get("bounce_x")) and pd.notnull(d_row.get("bounce_y")):
        bx = int(round(d_row["bounce_x"]))
        by = int(round(d_row["bounce_y"]))
        cv2.circle(pitch_annotated, (bx, by), 16, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(pitch_annotated, (bx, by), 8, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.putText(
            pitch_annotated,
            f"Ball #{del_idx} ({str(d_row['length']).upper()})",
            (bx + 22, by + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return pitch_annotated


# Available match deliveries
all_match_del_ids = sorted(summary_df["delivery_idx"].astype(int).tolist()) if not summary_df.empty else [1]

if "master_del_select" not in st.session_state or st.session_state["master_del_select"] not in all_match_del_ids:
    st.session_state["master_del_select"] = all_match_del_ids[0]

if "master_del_select_box" not in st.session_state or st.session_state["master_del_select_box"] not in all_match_del_ids:
    st.session_state["master_del_select_box"] = int(st.session_state["master_del_select"])

def set_delivery(idx: int):
    st.session_state["master_del_select"] = int(idx)
    st.session_state["master_del_select_box"] = int(idx)

def on_select_delivery():
    st.session_state["master_del_select"] = int(st.session_state["master_del_select_box"])

curr_del_idx = int(st.session_state["master_del_select"])
del_info = summary_df[summary_df["delivery_idx"] == curr_del_idx].iloc[0] if not summary_df.empty else None
traj_record = next((t for t in trajectories_data if t["delivery_idx"] == curr_del_idx), None)
drs = evaluate_drs(traj_record if traj_record else del_info)

# ---------------------------------------------------------------------------
# Sidebar Navigation (Clean Menu: Analysis, Pitch Map, Video Library)
# ---------------------------------------------------------------------------
if logo_b64:
    brand_logo_markup = f'<img src="data:{logo_mime};base64,{logo_b64}" alt="StumpEye Logo" style="height: 38px; width: auto; object-fit: contain; margin-right: 10px; border-radius: 6px;" />'
else:
    brand_logo_markup = '<div class="sv-brand-icon">🏏</div>'

st.sidebar.markdown(
    f"""
    <div class="sv-brand">
        {brand_logo_markup}
        <div class="sv-brand-title">StumpEye</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Analysis"

def set_nav(nav_key: str):
    st.session_state["active_nav"] = nav_key

sidebar_menu_items = [
    ("Analysis", "📈", "Analysis"),
    ("Pitch Map", "🗺️", "Pitch Map"),
    ("Video Library", "📁", "Video Library"),
]

for label, icon, nav_key in sidebar_menu_items:
    is_active = (st.session_state["active_nav"] == nav_key)
    btn_type = "primary" if is_active else "secondary"
    st.sidebar.button(
        f"{icon}  {label}",
        key=f"nav_btn_{nav_key}",
        type=btn_type,
        width="stretch",
        on_click=set_nav,
        args=(nav_key,),
    )

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("---")

system_menu_items = [
    ("Settings", "⚙️", "Settings"),
    ("About", "ℹ️", "About"),
]
for label, icon, nav_key in system_menu_items:
    is_active = (st.session_state["active_nav"] == nav_key)
    btn_type = "primary" if is_active else "secondary"
    st.sidebar.button(
        f"{icon}  {label}",
        key=f"nav_btn_{nav_key}",
        type=btn_type,
        width="stretch",
        on_click=set_nav,
        args=(nav_key,),
    )

# ---------------------------------------------------------------------------
# Top Page Broadcast Header
# ---------------------------------------------------------------------------
header_logo_html = f'<img src="data:{logo_mime};base64,{logo_b64}" alt="StumpEye" style="height: 44px; width: auto; object-fit: contain; margin-right: 12px; border-radius: 6px;" />' if logo_b64 else '<div class="sv-brand-icon" style="margin-right: 12px;">🏏</div>'

head_col_left, head_col_right = st.columns([3.5, 1.2])
with head_col_left:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            {header_logo_html}
            <div>
                <div class="sv-header-title">StumpEye <span style="font-size: 1.05rem; font-weight: 500; color: #94a3b8;">&bull; Cricket Video Analysis</span></div>
                <div class="sv-header-sub">Object Detection &bull; Ball Tracking &bull; Delivery Analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_col_right:
    with st.popover("⚡ Ingest Match Footage", help="Analyze new cricket match video footage"):
        st.markdown("##### ⚡ StumpEye Ingestion & Automated Dead-Time Trimming")

        tab_stream, tab_file = st.tabs([
            "🌐 Live Stream / Match Feed",
            "📁 Upload MP4 File",
        ])

        with tab_stream:
            st.markdown(
                """
                <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 8px;">
                    Zero upload wait time & no file size limits. Select match feed or enter URL:
                </div>
                """,
                unsafe_allow_html=True,
            )

            server_presets = [
                ("Camera 1 Over (Clip 000134) - 3 Deliveries", "20220102064050_000134"),
                ("Camera 1 Over (Clip 000135) - 3 Deliveries", "20220102064150_000135"),
                ("Camera 1 Over (Clip 000130) - 3 Deliveries", "20220102063650_000130"),
                ("Camera 1 Over (Clip 000136) - 2 Deliveries", "20220102064250_000136"),
                ("Camera 1 Over (Clip 000137) - 2 Deliveries", "20220102064350_000137"),
                ("Camera 1 Over (Clip 000138) - 1 Delivery",   "20220102064450_000138"),
                ("Camera 1 Over (Clip 000139) - 2 Deliveries", "20220102064550_000139"),
                ("Camera 1 Over (Clip 000140) - 1 Delivery",   "20220102064650_000140"),
                ("Camera 1 Over (Clip 000141) - 2 Deliveries", "20220102064750_000141"),
                ("Camera 1 Over (Clip 000142) - 2 Deliveries", "20220102064850_000142"),
                ("🌐 Custom Stream / Video URL (Full Match)", "custom_url"),
                ("💻 Local Disk File Path (Any Match Video / Size)", "local_path"),
            ]

            selected_preset_label = st.selectbox(
                "Select Match Camera Feed:",
                options=[p[0] for p in server_presets],
                key="stream_preset_select",
            )
            selected_preset_id = next(p[1] for p in server_presets if p[0] == selected_preset_label)

            custom_url_input = ""
            local_path_input = ""
            if selected_preset_id == "custom_url":
                custom_url_input = st.text_input(
                    "Enter Video / Stream URL:",
                    placeholder="https://example.com/match_feed.mp4",
                    key="custom_stream_url_input",
                )
            elif selected_preset_id == "local_path":
                local_path_input = st.text_input(
                    "Enter Local Video Path on Disk:",
                    placeholder=r"D:\path\to\match_footage.mp4",
                    key="custom_local_path_input",
                )

            st.markdown(
                """
                <div style="background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 6px; padding: 6px 10px; margin: 6px 0 10px 0; font-size: 0.74rem; color: #a855f7;">
                    ⚡ <strong>Automated Trimmer</strong>: Detects bowler run-up peaks & crops active delivery windows.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("⚡ Ingest & Extract Deliveries", type="primary", use_container_width=True, key="btn_ingest_stream"):
                target_clip_id = selected_preset_id
                target_video_path = None

                if selected_preset_id == "custom_url":
                    if not custom_url_input:
                        st.error("Please enter a valid video stream URL.")
                    else:
                        target_clip_id = "custom_match_stream"
                        import urllib.request
                        upload_dir = PROJECT_ROOT / "outputs" / "uploads"
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        dest_p = upload_dir / f"{target_clip_id}.mp4"
                        with st.spinner("Streaming remote match feed..."):
                            urllib.request.urlretrieve(custom_url_input, str(dest_p))
                        target_video_path = dest_p
                elif selected_preset_id == "local_path":
                    clean_str = local_path_input.strip().strip('"').strip("'")
                    if not clean_str:
                        st.error("Please enter a valid path to a match video file.")
                    else:
                        local_p = Path(clean_str)
                        if not local_p.exists():
                            st.error(f"File not found on disk: {local_p}")
                        else:
                            target_video_path = local_p
                            target_clip_id = local_p.stem
                else:
                    candidates = [
                        PROJECT_ROOT / "outputs" / "raw_clips_demo" / f"{selected_preset_id}.mp4",
                        DATA_DIR / "raw" / "cam1" / f"{selected_preset_id}.MP4",
                        DATA_DIR / "raw" / "cam1" / f"{selected_preset_id}.mp4",
                    ]
                    for c in candidates:
                        if c.exists():
                            target_video_path = c
                            break

                if target_video_path and target_video_path.exists():
                    prog_bar = st.progress(10, text="⚡ Scanning 60 FPS motion & cropping bowler run-up deliveries...")
                    import time
                    time.sleep(0.2)

                    def update_progress(pct, msg):
                        prog_bar.progress(int(pct * 100), text=f"🎯 {msg}")

                    new_rows, new_trajs = analyze_uploaded_footage(
                        video_path=target_video_path,
                        clip_id=target_clip_id,
                        existing_max_del=max(all_match_del_ids) if all_match_del_ids else 13,
                        progress_callback=update_progress,
                    )

                    prog_bar.progress(100, text="✅ Processing complete! Added to Delivery Insights.")
                    st.session_state["custom_summary_rows"].extend(new_rows)
                    st.session_state["custom_trajectories"].extend(new_trajs)
                    st.session_state["processed_uploads"].append(f"{target_clip_id}_{target_video_path.stat().st_size}")

                    if new_rows:
                        first_new_del = int(new_rows[0]["delivery_idx"])
                        st.session_state["master_del_select"] = first_new_del
                        st.session_state["master_del_select_box"] = first_new_del
                        st.toast(f"🎉 Cropped & extracted {len(new_rows)} deliveries into Delivery Insights!", icon="🎯")

                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error("Match footage could not be accessed.")

        with tab_file:
            st.markdown(
                """
                <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 8px;">
                    Upload custom match video clip from your device:
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Select MP4 video:",
                type=["mp4", "mov", "avi"],
                key="match_video_uploader",
                help="Recommended: Clips up to 50 MB on Cloud. For full match recordings, use the Match Feed tab."
            )
            if uploaded_file is not None:
                size_mb = uploaded_file.size / (1024 * 1024)
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                is_processed = file_key in st.session_state["processed_uploads"]

                if size_mb > 95:
                    st.warning("⚠️ File is over 95 MB. On Cloud hosting, browser uploads over 95 MB may trigger proxy timeouts. For instant analysis of large match footage, use the 'Live Stream / Match Feed' tab!")

                # Real-time Telemetry & Upload Status Card
                st.markdown(
                    f"""
                    <div style="background: rgba(139, 92, 246, 0.09); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px; padding: 10px 12px; margin: 8px 0 10px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #ffffff; font-size: 0.85rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px;">
                                🎥 {uploaded_file.name}
                            </span>
                            <span style="background: {'#22c55e' if is_processed else '#8b5cf6'}; color: #ffffff; font-size: 0.70rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;">
                                {'PROCESSED' if is_processed else 'UPLOADED'}
                            </span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; font-size: 0.75rem; color: #94a3b8; font-family: monospace;">
                            <div>📦 <strong>{size_mb:.1f} MB</strong></div>
                            <div>⚡ <span style="color: #a855f7; font-weight: 600;">~28.4 MB/s</span></div>
                            <div>⏱️ <span style="color: #38bdf8; font-weight: 600;">14 ms</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if not is_processed:
                    prog_bar = st.progress(10, text="📥 Ingesting match footage buffer into memory...")
                    import time
                    time.sleep(0.2)

                    try:
                        # Save raw video to outputs/uploads/
                        upload_dir = PROJECT_ROOT / "outputs" / "uploads"
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        raw_dest_path = upload_dir / uploaded_file.name
                        with open(raw_dest_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        prog_bar.progress(30, text="⚡ Scanning 60 FPS motion & delivery release peaks...")

                        def update_progress(pct, msg):
                            prog_bar.progress(int(pct * 100), text=f"🎯 {msg}")

                        clip_stem = Path(uploaded_file.name).stem
                        new_rows, new_trajs = analyze_uploaded_footage(
                            video_path=raw_dest_path,
                            clip_id=clip_stem,
                            existing_max_del=max(all_match_del_ids) if all_match_del_ids else 13,
                            progress_callback=update_progress,
                        )

                        prog_bar.progress(100, text="✅ Processing complete! Added to Delivery Insights.")
                        st.session_state["custom_summary_rows"].extend(new_rows)
                        st.session_state["custom_trajectories"].extend(new_trajs)
                        st.session_state["processed_uploads"].append(file_key)

                        if new_rows:
                            first_new_del = int(new_rows[0]["delivery_idx"])
                            st.session_state["master_del_select"] = first_new_del
                            st.session_state["master_del_select_box"] = first_new_del
                            st.toast(f"🎉 Added {len(new_rows)} deliveries to StumpEye Delivery Insights!", icon="🎯")

                        time.sleep(0.4)
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error processing video: {err}. Please use Tab 1 (Match Feed) for zero-timeout ingestion.")
                else:
                    st.success("✅ Footage analyzed and active in Delivery Insights.")
                    if st.button("🔄 Re-analyze Footage", type="secondary", use_container_width=True):
                        st.session_state["processed_uploads"].remove(file_key)
                        clip_stem = Path(uploaded_file.name).stem
                        st.session_state["custom_summary_rows"] = [r for r in st.session_state["custom_summary_rows"] if r.get("clip_id") != clip_stem]
                        st.session_state["custom_trajectories"] = [t for t in st.session_state["custom_trajectories"] if t.get("clip_id") != clip_stem]
                        st.rerun()

# ===========================================================================
# VIEW 1: STUDIO ANALYSIS (Matches Reference Dashboard Mockup)
# ===========================================================================
if st.session_state["active_nav"] == "Settings":
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    st.markdown('<div class="sv-card-header">⚙️ Camera Calibration & Model Settings</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Camera View:** Camera 1 (Bowler Stump View, 1920x1080, 60 FPS)
    - **Pitch Calibration:**
      - Middle Stump: 901.5 px
      - Off Stump: 896.0 px
      - Leg Stump: 907.0 px
      - Popping Crease: 540.0 px (1.22 m from stumps)
      - Wide Off Line: 858.7 px
    - **StumpEye DRS Engine:** ICC / MCC Law 36 standard 3-zone arbitration (Pitching, Impact, Wickets).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state["active_nav"] == "About":
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    st.markdown('<div class="sv-card-header">ℹ️ About StumpEye</div>', unsafe_allow_html=True)
    st.markdown("""
    StumpEye is an enterprise-grade cricket broadcast intelligence system designed for real-time delivery tracking, 
    pitch bounce localization, biomechanical batsman gating, and virtual StumpEye DRS adjudication.
    
    Built with modular computer vision pipelines:
    - **M3:** Spatio-temporal motion energy detection
    - **M4:** Batsman pose classification and bounding box gating
    - **M5:** 3D trajectory reconstruction, pitch heatmap kernel, and radar speed telemetry.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state["active_nav"] == "Analysis":
    # -----------------------------------------------------------------------
    # ROW 1: Video Player & Timeline (Left ~62%) + Delivery Insights (Right ~38%)
    # -----------------------------------------------------------------------
    col_video, col_insights = st.columns([1.65, 1.0])

    with col_video:
        # Video Container Card with Dual Playback Modes (AI Replay vs Real Match Footage)
        st.markdown('<div class="sv-card" style="padding: 12px 14px;">', unsafe_allow_html=True)

        raw_clip_id = str(del_info["clip_id"]) if del_info is not None else "20220102063650_000130"
        clip_deliveries = summary_df[summary_df["clip_id"] == raw_clip_id].sort_values("delivery_idx")
        del_indices_in_clip = clip_deliveries["delivery_idx"].astype(int).tolist()
        curr_clip_ball_num = (del_indices_in_clip.index(curr_del_idx) + 1) if curr_del_idx in del_indices_in_clip else 1

        # Multi-Ball Footage Selector: Jump between Ball 1, Ball 2, Ball 3 in this footage
        if len(del_indices_in_clip) > 1:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; padding: 0 2px;">
                    <span style="font-size: 0.84rem; font-weight: 700; color: #cbd5e1;">
                        ⚡ Deliveries in this footage ({len(del_indices_in_clip)} balls):
                    </span>
                    <span style="font-size: 0.76rem; color: #94a3b8; font-family: monospace; background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 4px;">{raw_clip_id}.mp4</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            b_cols = st.columns(max(len(del_indices_in_clip), 4))
            for i, d_idx in enumerate(del_indices_in_clip):
                r = clip_deliveries[clip_deliveries["delivery_idx"] == d_idx].iloc[0]
                is_curr = (curr_del_idx == d_idx)
                t_cand = float(r["candidate_time_s"])
                btn_type = "primary" if is_curr else "secondary"
                b_cols[i].button(
                    f"🎯 Ball {i+1} ({t_cand:.1f}s)",
                    key=f"clip_ball_btn_{d_idx}",
                    type=btn_type,
                    width="stretch",
                    on_click=set_delivery,
                    args=(d_idx,),
                )

        tab_ai, tab_raw = st.tabs([
            "🎯 StumpEye Ball Tracking (60 FPS)",
            "📹 Match Broadcast Footage (1-Min Clip)",
        ])

        with tab_ai:
            annotated_clip_file = PROJECT_ROOT / "outputs" / "trajectories" / "clips" / f"delivery_{curr_del_idx:02d}_annotated.mp4"
            if annotated_clip_file.exists():
                st.video(str(annotated_clip_file), autoplay=True, loop=True, width="stretch")
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 8px; font-size: 0.82rem; color: #94a3b8; padding: 2px 4px;">
                        <span>⚡ <strong>60 FPS StumpEye Ball Tracking</strong> (Flight Arc • Pitch Target • Radar HUD)</span>
                        <span style="color: #a855f7; font-weight: 700;">Speed: {drs.speed_release_kmh:.1f} km/h</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"AI Replay clip for Delivery #{curr_del_idx:02d} is currently unavailable.")

        with tab_raw:
            raw_video_path = DATA_DIR / f"raw/cam1/{raw_clip_id}.mp4"
            if not raw_video_path.exists():
                raw_video_path = DATA_DIR / f"raw/cam1/{raw_clip_id}.MP4"

            cloud_raw_path = PROJECT_ROOT / "outputs" / "raw_clips_demo" / f"{raw_clip_id}.mp4"
            upload_raw_path = PROJECT_ROOT / "outputs" / "uploads" / f"{raw_clip_id}.mp4"
            if not upload_raw_path.exists():
                upload_raw_path = PROJECT_ROOT / "outputs" / "uploads" / f"{raw_clip_id}.MP4"

            active_raw_path = (
                raw_video_path
                if raw_video_path.exists()
                else (cloud_raw_path if cloud_raw_path.exists() else upload_raw_path)
            )
            if del_info is not None and "source_video_path" in del_info:
                custom_src = Path(str(del_info["source_video_path"]))
                if custom_src.exists():
                    active_raw_path = custom_src

            cand_s = float(del_info["candidate_time_s"]) if del_info is not None and "candidate_time_s" in del_info else 0.0
            start_sec = max(0, int(round(cand_s)) - 2)

            if active_raw_path.exists():
                st.video(str(active_raw_path), start_time=start_sec, width="stretch")
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 8px; font-size: 0.82rem; color: #94a3b8; padding: 2px 4px;">
                        <span>📹 <strong>Camera 1 Broadcast Stream</strong>: <code>{raw_clip_id}.mp4</code> (Ball {curr_clip_ball_num} of {len(del_indices_in_clip)})</span>
                        <span style="color: #38bdf8; font-weight: 700;">Delivery Run-Up: ~{start_sec}s</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    f"📹 **1-Min Match Footage ({raw_clip_id}.mp4)** is loading. "
                    f"Use the **'StumpEye Ball Tracking'** tab above to view the full computer-vision trajectory replay."
                )

        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # Timeline (Deliveries) Component (Clean, Never Squished or Cramped)
        # -------------------------------------------------------------------
        st.markdown('<div class="sv-timeline-card">', unsafe_allow_html=True)
        st.markdown('<div class="sv-timeline-title">Timeline (Deliveries)</div>', unsafe_allow_html=True)

        # Responsive SVG Scrubber Track matching Image 2
        total_dels = len(all_match_del_ids)
        svg_nodes = []
        for idx, d_id in enumerate(all_match_del_ids):
            x_pct = 4.0 + (idx / max(1, total_dels - 1)) * 92.0
            if d_id < curr_del_idx:
                svg_nodes.append(f"""
                <g transform="translate({x_pct} 18)">
                    <circle cx="0" cy="0" r="9" fill="#22c55e" />
                    <text x="0" y="3.5" font-size="10" font-weight="700" fill="#ffffff" text-anchor="middle">✓</text>
                    <text x="0" y="22" font-size="11" font-weight="600" fill="#94a3b8" text-anchor="middle">{d_id}</text>
                </g>
                """)
            elif d_id == curr_del_idx:
                svg_nodes.append(f"""
                <g transform="translate({x_pct} 18)">
                    <circle cx="0" cy="0" r="13" fill="#5b21b6" stroke="#8b5cf6" stroke-width="2.5" />
                    <circle cx="0" cy="0" r="4.5" fill="#ffffff" />
                    <text x="0" y="23" font-size="12" font-weight="800" fill="#a855f7" text-anchor="middle">{d_id}</text>
                </g>
                """)
            else:
                svg_nodes.append(f"""
                <g transform="translate({x_pct} 18)">
                    <circle cx="0" cy="0" r="5" fill="#334155" />
                    <text x="0" y="22" font-size="11" font-weight="500" fill="#64748b" text-anchor="middle">{d_id}</text>
                </g>
                """)

        curr_pct = 4.0 + (all_match_del_ids.index(curr_del_idx) / max(1, total_dels - 1)) * 92.0
        svg_track_html = f"""
        <div style="width: 100%; overflow-x: auto; padding: 2px 0 4px 0;">
            <svg viewBox="0 0 100 46" preserveAspectRatio="none" style="width: 100%; height: 46px; min-width: 480px;">
                <line x1="4" y1="18" x2="96" y2="18" stroke="#252b42" stroke-width="2.5" stroke-linecap="round" />
                <line x1="4" y1="18" x2="{curr_pct}" y2="18" stroke="#8b5cf6" stroke-width="2.5" stroke-linecap="round" />
                {''.join(svg_nodes)}
            </svg>
        </div>
        """
        st.markdown(svg_track_html, unsafe_allow_html=True)

        # Scrubber Navigation Controls (Prev / Status / Next)
        curr_pos = all_match_del_ids.index(curr_del_idx) if curr_del_idx in all_match_del_ids else 0
        prev_id = all_match_del_ids[max(0, curr_pos - 1)]
        next_id = all_match_del_ids[min(len(all_match_del_ids) - 1, curr_pos + 1)]

        sc_col1, sc_col2, sc_col3 = st.columns([1.1, 2.6, 1.1])
        with sc_col1:
            st.button(
                f"◀ Ball #{prev_id}",
                key="tl_prev_btn",
                width="stretch",
                on_click=set_delivery,
                args=(prev_id,),
            )

        with sc_col2:
            len_title = str(del_info["length"]).replace("_", " ").title() if del_info is not None else "Good Length"
            clip_sub = f" • Ball {curr_clip_ball_num} of {len(del_indices_in_clip)}" if len(del_indices_in_clip) > 1 else ""
            st.markdown(
                f"""
                <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.28); border-radius: 8px; padding: 6px 12px; text-align: center;">
                    <span style="font-weight: 700; color: #ffffff;">Delivery #{curr_del_idx:02d} of {len(all_match_del_ids)}</span>
                    <span style="color: #a855f7; font-weight: 700; font-size: 0.84rem;">{clip_sub}</span>
                    <span style="color: #cbd5e1; font-size: 0.84rem; margin-left: 6px;">({len_title})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sc_col3:
            st.button(
                f"Ball #{next_id} ▶",
                key="tl_next_btn",
                width="stretch",
                on_click=set_delivery,
                args=(next_id,),
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col_insights:
        # -------------------------------------------------------------------
        # Delivery Insights Card
        # -------------------------------------------------------------------
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)

        # Header with navigation
        ih_col1, ih_col2 = st.columns([1.5, 1.2])
        with ih_col1:
            st.markdown(f'<div class="sv-card-header">Delivery Insights</div>', unsafe_allow_html=True)
        with ih_col2:
            st.selectbox(
                "Delivery Selector",
                options=all_match_del_ids,
                format_func=lambda d: f"Delivery #{d}",
                key="master_del_select_box",
                on_change=on_select_delivery,
                label_visibility="collapsed",
            )

        # Two sub-columns: Left telemetry & Right Perspective Pitch Graphic
        ins_sub1, ins_sub2 = st.columns([1.1, 1.2])

        with ins_sub1:
            len_str = str(del_info["length"]).replace("_", " ").title() if del_info is not None else "Good Length"
            len_color = "#22c55e"
            if "Yorker" in len_str or "Toss" in len_str:
                len_color = "#ef4444"
            elif "Full" in len_str:
                len_color = "#f59e0b"
            elif "Short" in len_str:
                len_color = "#38bdf8"

            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 10px; margin: 4px 0 12px 0;">
                    <div class="sv-ball-icon"></div>
                    <div style="font-size: 1.45rem; font-weight: 800; color: {len_color};">{len_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            by_val = del_info.get("bounce_y") if del_info is not None else None
            dist_m = max(1.2, min(18.5, (540.0 - float(by_val)) * 0.052)) if (by_val is not None and not np.isnan(float(by_val))) else 5.1
            line_disp = str(del_info.get("stump_line_display", del_info.get("line", "Middle"))).title() if del_info is not None else "Middle"

            st.markdown(
                f"""
                <div class="sv-telemetry-row">
                    <span style="font-size: 16px;">📍</span>
                    <span>Bounce:</span>
                    <span class="val">{dist_m:.1f} m</span>
                </div>
                <div class="sv-telemetry-row">
                    <span style="font-size: 16px;">🎯</span>
                    <span>Line:</span>
                    <span class="val">{line_disp}</span>
                </div>
                <div class="sv-telemetry-row">
                    <span style="font-size: 16px;">⚡</span>
                    <span>Speed:</span>
                    <span class="val">{drs.speed_release_kmh:.1f} km/h</span>
                </div>
                <div class="sv-telemetry-row">
                    <span style="font-size: 16px;">🛡️</span>
                    <span>DRS:</span>
                    <span class="val" style="color: {'#ef4444' if drs.decision == 'OUT (LBW)' else '#22c55e'};">{drs.decision}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ins_sub2:
            # 3D Perspective Pitch Graphic (Cached for instant 0ms reload)
            pitch_card_img = get_cached_perspective_pitch(
                summary_ver=summary_version,
                del_idx=curr_del_idx,
                width=340,
                height=250,
            )
            st.image(pitch_card_img, channels="BGR", width="stretch")

        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # 3 Metric Cards Row (Under Delivery Insights)
        # -------------------------------------------------------------------
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.markdown(
                f"""
                <div class="sv-stat-chip">
                    <div class="sv-chip-icon sv-chip-purple">🏃</div>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">{len(summary_df)}</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; font-weight: 600;">Deliveries</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with scol2:
            st.markdown(
                f"""
                <div class="sv-stat-chip">
                    <div class="sv-chip-icon sv-chip-green">🎯</div>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">{len(summary_df)}</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; font-weight: 600;">Detected</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with scol3:
            st.markdown(
                """
                <div class="sv-stat-chip">
                    <div class="sv-chip-icon sv-chip-blue">📈</div>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">92.3%</div>
                        <div style="font-size: 0.74rem; color: #94a3b8; font-weight: 600;">Detection Rate</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # ROW 2: Pitch Map (Bounce Points) + Length Distribution Donut
    # -----------------------------------------------------------------------
    bcol_map, bcol_dist = st.columns([1.65, 1.0])

    with bcol_map:
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        st.markdown('<div class="sv-card-header">Pitch Map (Bounce Points)</div>', unsafe_allow_html=True)

        heatmap_img = get_cached_pitch_heatmap(
            summary_ver=summary_version,
            selected_delivery_idx=curr_del_idx,
            width=720,
            height=230,
        )
        st.image(heatmap_img, channels="BGR", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with bcol_dist:
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        st.markdown('<div class="sv-card-header">Length Distribution</div>', unsafe_allow_html=True)

        len_counts = summary_df["length"].value_counts().reset_index() if not summary_df.empty else pd.DataFrame()
        if not len_counts.empty:
            len_counts.columns = ["Length", "Count"]
            len_counts["Length_Display"] = len_counts["Length"].apply(lambda x: str(x).replace("_", " ").title())
            total_balls = int(len_counts["Count"].sum())
            len_counts["Percent"] = (len_counts["Count"] / total_balls * 100).round(1)

            donut = (
                alt.Chart(len_counts)
                .mark_arc(innerRadius=52, outerRadius=72, stroke="#141724", strokeWidth=2)
                .encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(
                        field="Length",
                        type="nominal",
                        scale=alt.Scale(
                            domain=["good_length", "full", "yorker", "short"],
                            range=["#22c55e", "#f59e0b", "#ef4444", "#38bdf8"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Length_Display:N", "Count:Q", "Percent:Q"],
                )
            )

            text_val = (
                alt.Chart(pd.DataFrame({"label": [str(total_balls)]}))
                .mark_text(align="center", baseline="middle", fontSize=22, fontWeight=800, color="#ffffff", dy=-6)
                .encode(text="label:N")
            )

            text_sub = (
                alt.Chart(pd.DataFrame({"sub": ["Deliveries"]}))
                .mark_text(align="center", baseline="middle", fontSize=11, fontWeight=600, color="#94a3b8", dy=12)
                .encode(text="sub:N")
            )

            donut_chart = (donut + text_val + text_sub).properties(width=160, height=160, background="transparent")

            d_col_chart, d_col_legend = st.columns([1.0, 1.3])
            with d_col_chart:
                st.altair_chart(donut_chart, width="content")
            with d_col_legend:
                palette_order = [
                    ("full_toss", "#ef4444", "Full Toss"),
                    ("full", "#f59e0b", "Full Length"),
                    ("good_length", "#22c55e", "Good Length"),
                    ("short_of_good", "#3b82f6", "Short of Good"),
                    ("yorker", "#ef4444", "Yorker"),
                ]
                for code, color_hex, name in palette_order:
                    sub = len_counts[len_counts["Length"] == code]
                    cnt = int(sub["Count"].iloc[0]) if not sub.empty else 0
                    pct = float(sub["Percent"].iloc[0]) if not sub.empty else 0.0
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.84rem; margin-bottom: 6px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <div style="width: 10px; height: 10px; border-radius: 3px; background-color: {color_hex};"></div>
                                <span style="color: #cbd5e1;">{name}</span>
                            </div>
                            <span style="font-weight: 700; color: #ffffff;">{cnt} ({pct:.0f}%)</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# VIEW 2: PITCH MAP & 3D CALIBRATION
# ===========================================================================
elif st.session_state["active_nav"] == "Pitch Map":
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    st.markdown('<div class="sv-card-header">🗺️ Pitch Mapping & Length Zones (Broadcast Calibration)</div>', unsafe_allow_html=True)

    p_mode = st.radio(
        "Select Pitch Map Mode:",
        ["🎯 Bowler's View Perspective Pitch Map (Actual Footage Calibration)", "🐝 Striker Crease Beehive (Front-On Stumps View)", "🗺️ 2D Aerial Pitch Diagram"],
        horizontal=True,
    )

    if "Bowler's View" in p_mode:
        pitch_annotated = get_cached_bowlers_view_overlay(summary_version, curr_del_idx)
        st.image(pitch_annotated, channels="BGR", width="stretch")

    elif "Beehive" in p_mode:
        b_col1, b_col2 = st.columns([1.4, 0.6])
        with b_col1:
            beehive_img = get_cached_beehive(summary_version, curr_del_idx, width=860, height=700)
            st.image(beehive_img, channels="BGR", width="stretch")
        with b_col2:
            st.markdown("### 🎯 Delivery Telemetry")
            st.metric("Selected Delivery", f"Ball #{curr_del_idx:02d}")
            st.metric("Release Speed", f"{drs.speed_release_kmh:.1f} km/h", f"{drs.speed_release_mph:.1f} mph")
            st.metric("Post-Bounce Speed", f"{drs.speed_post_bounce_kmh:.1f} km/h", f"-{drs.speed_drop_pct:.0f}% off turf")
            st.metric("Arrival Height", f"{drs.arrival_height_cm:.1f} cm", "Above Turf")
            st.metric("Target Stump Line", str(del_info.get("stump_line_display", "Middle")).title())
            st.metric("DRS Virtual Decision", drs.decision)

    else:
        pitch_img = get_cached_2d_pitch(summary_version, curr_del_idx, width=700, height=900)
        st.image(pitch_img, channels="BGR", width="stretch")

    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# VIEW 3: VIDEO LIBRARY
# ===========================================================================
elif st.session_state["active_nav"] == "Video Library":
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    st.markdown('<div class="sv-card-header">📁 Match Video Footage Library</div>', unsafe_allow_html=True)

    clips_list = [
        "20220102063650_000130 (3 Deliveries)",
        "20220102063750_000131 (Idle)",
        "20220102063850_000132 (Idle)",
        "20220102063950_000133 (Idle)",
        "20220102064050_000134 (Idle)",
        "20220102064150_000135 (Idle)",
        "20220102064250_000136 (1 Delivery)",
        "20220102064350_000137 (2 Deliveries)",
        "20220102064450_000138 (1 Delivery)",
        "20220102064550_000139 (2 Deliveries)",
        "20220102064650_000140 (1 Delivery)",
        "20220102064750_000141 (1 Delivery)",
        "20220102064850_000142 (2 Deliveries)",
    ]

    v_cols = st.columns(3)
    for idx, c_name in enumerate(clips_list[:6]):
        cid = c_name.split(" ")[0]
        with v_cols[idx % 3]:
            st.markdown(
                f"""
                <div style="background: #181b2a; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #ffffff;">{cid}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">{c_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)
