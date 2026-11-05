# StumpVision AI 🏏

> Production-grade Cricket Broadcast Intelligence, Ball Trajectory Tracking & Virtual Hawk-Eye DRS Platform.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Tests-104%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

StumpVision AI is an end-to-end computer vision platform designed for real-time cricket broadcast analysis, automated delivery extraction, sub-pixel ball trajectory tracking, and virtual Decision Review System (DRS) arbitration following ICC/MCC Law 36 standards.

---

## ⚡ Core Capabilities

- **M3 Spatio-Temporal Motion Gating:** High-speed motion energy segmentation detecting bowling release and batsman action across 60 FPS broadcast camera streams.
- **M4 Biomechanical Batsman Gating:** Keypoint pose estimation (YOLO / MediaPipe) validating batsman stance inside the popping crease prior to delivery evaluation.
- **M5 Hawk-Eye Ball Trajectory Engine:**
  - Multi-phase physical flight reconstruction: *Release $\to$ Pitch Bounce $\to$ Bat Contact*.
  - Strict physical forward-momentum constraints preventing mid-air curvature or false reversals.
  - Sub-pixel turf bounce localization calibrated against actual pitch crease geometry.
- **Virtual DRS Adjudication (LBW):**
  - **Pitching Zone:** In-line, Outside Off, or Outside Leg.
  - **Impact Zone:** In-line or Outside Off (with umpire's call boundary).
  - **Wickets Projection:** Trajectory extrapolation to stumps determining *Hitting*, *Umpire's Call*, or *Missing*.
- **Speed Radar Telemetry:** Release speed, post-bounce turf deceleration, and arrival height metrics.

---

## 🖥️ Broadcast Studio Interface

The interactive dashboard matches professional television broadcast consoles (Hawk-Eye / Star Sports):

- **Dual-Feed Video Console:** Instant client-side switching between the **AI 60 FPS Trajectory Replay** (with flight arc and speed HUD) and the **Real 1-Minute Broadcast Footage** (auto-jumped to bowler run-up).
- **Multi-Ball Footage Selector:** Quick access to all deliveries occurring within the same match footage clip (`Ball 1`, `Ball 2`, `Ball 3`).
- **Interactive Delivery Timeline:** Responsive visual scrubber spanning match deliveries with instant 0 ms switching.
- **3D Perspective Pitch Zone Card:** Real-time visual display mapping each delivery to length zones (*Yorker, Full, Good Length, Short*).
- **Kernel Density Pitch Heatmap:** Multi-delivery pitch bounce scatter and density distribution.

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- FFmpeg (optional, recommended for hardware-accelerated video rendering)

### 2. Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/your-username/Stumpeye-project.git
cd Stumpeye-project

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

### 3. Launch Broadcast Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Testing

The platform includes a comprehensive test suite covering pitch calibration, trajectory interpolation, DRS arbitration, and video rendering:

```bash
pytest -q
```
```text
........................................................................ [ 69%]
................................                                         [100%]
104 passed in 19.8s
```

---

## 📁 Project Structure

```text
Stumpeye-project/
├── app.py                      # Production Broadcast Studio Dashboard (Streamlit)
├── src/
│   └── stumpvision/
│       ├── trajectory/
│       │   ├── tracker.py      # Ball Trajectory Tracking & Physical Momentum Engine
│       │   ├── drs.py          # ICC Law 36 Virtual DRS & Speed Radar Engine
│       │   ├── viz.py          # Broadcast Overlay, Heatmap & Replay Video Renderers
│       │   ├── stumps.py       # Camera 1 Crease & Stump Line Calibration
│       │   └── beehive.py      # Striker Crease Beehive Arrival Plotter
│       └── cli.py              # StumpVision Command-Line Interface
├── outputs/
│   └── trajectories/
│       ├── dev_trajectories.json       # Trajectory coordinates & physical models
│       ├── dev_trajectories_summary.csv # Match delivery summary & metrics
│       └── clips/                      # 60 FPS annotated broadcast replay clips
├── tests/                      # 104 unit & integration test cases
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Package configuration
```

---

## 🌐 Deployment Options

| Platform | Best For | Compatibility |
| :--- | :--- | :--- |
| **Streamlit Community Cloud** | Free 1-click deployment directly from your GitHub repo | Recommended (`share.streamlit.io`) |
| **Hugging Face Spaces** | Free hosting with Streamlit runtime & CPU/GPU support | Supported |
| **Render / Railway** | Containerized web service deployment | Supported |

*Note: GitHub Pages (`.github.io`) and Netlify host static HTML/JS files only and cannot execute a Python backend (OpenCV / Streamlit).*

---

## 📜 License

MIT License. Designed and engineered for high-performance sports video analytics.

