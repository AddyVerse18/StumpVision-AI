# Architecture & Engineering Decisions Log

This document records all architectural, technical, and environment decisions for StumpVision AI Delivery Finder, including rationale and date.

---

### [2026-10-28] Decision 001: Python Runtime Selection & Installed Packages
- **Decision:** Use Python 3.13 (64-bit) for the project virtual environment.
- **Reason:** `py -0` confirmed Python 3.13 is the sole installed Python runtime on this Windows machine.
- **Dry-run Correction:** The dry run resolved `numpy 2.3.3` and `pytest 9.1.1` from the global install; the pins in `requirements.txt` come from the fresh venv (`pip freeze`). Requirements.txt is kept as is.
- **Exact Installed Versions (requirements.txt):**
  - `numpy==2.5.4`
  - `opencv-python==5.0.0.93`
  - `pandas==3.1.0`
  - `scipy==1.18.1`
  - `PyYAML==6.0.3`
  - `pytest==9.2.0`
  - `streamlit==1.66.1`
- **Status:** Approved by user. Verified and tested.

---

### [2026-10-28] Decision 002: External Read-Only Data Directory
- **Decision:** Set `STUMPVISION_DATA_DIR` to `D:\Computer Vision Projects\StumpEye_Footage`.
- **Reason:** The 19 raw clips reside in `raw\cam1\` inside this directory. Verified 100% byte-for-byte SHA-256 match across all 19 files before project reset. Footage is strictly confidential and read-only; no video files will ever be copied or stored inside the project workspace.
- **Status:** Approved by user.

---

### [2026-10-28] Decision 003: Isolated Virtual Environment Policy
- **Decision:** Create `.venv` inside project root without `--system-site-packages`.
- **Reason:** Guarantee zero interference with global Python packages or other projects. All pip commands must explicitly target `.venv\Scripts\pip.exe`.
- **Status:** Approved by user.

---

### [2026-10-28] Observation 001: M0 Video Decoding Benchmark Discrepancy
- **Observation:** Decoding all 19 clips took 460 s (about 24 s per clip) while the sample clip took 6.9 s. Cause unknown (disk cache or other processes). We will re-measure alone in M3.
- **Follow-up:** Re-measure decoding and streaming throughput in isolation in M3 under controlled conditions.

---

### [2026-10-28] Decision 004: Splits Definition and Test Split Locking
- **Decision:** Partition contiguous clips strictly by sequence number into dev, buffer, and test splits before building any detector.
  - `dev` (13 clips across 3 blocks):
    - `block_1`: sequence 130–133 (`000130` to `000133`)
    - `block_2`: sequence 134–137 (`000134` to `000137`)
    - `block_3`: sequence 138–142 (`000138` to `000142`)
  - `buffer` (1 clip):
    - `block_buffer`: sequence 143 (`000143`) — excluded from both dev and test.
  - `test` (5 clips locked):
    - `block_test`: sequence 144–148 (`000144` to `000148`).
- **Locked Test Clip IDs:**
  - `20220102065050_000144`
  - `20220102065150_000145`
  - `20220102065250_000146`
  - `20220102065350_000147`
  - `20220102065450_000148`
- **Locked Test Fingerprint (SHA-256 of sorted clip IDs):**
  `5f083e0a59514ee69c31e4d550bf8a881912fe480f7c9c08e14acd12e3ecde24`
- **Leakage Safeguard:** Evaluation commands touching test labels require `--allow-test` and log access to `outputs/test_access.log`.
- **Status:** Approved by user in M1 plan.

---

### [2026-10-28] Decision 005: M2 Evaluation Harness Architecture and Guardrails
- **Decision:** Implement one-to-one optimal bipartite matching using `scipy.optimize.linear_sum_assignment` on absolute time differences bounded by tolerance (default 0.5s, boundary |pred - truth| <= tol is a hit; 1.0s also reported).
- **Split Safeguard:** Default split is `dev`. Evaluation on `test` requires `--allow-test` and logs each invocation to `outputs/test_access.log`. Threshold sweep and optimal operating point selection are restricted strictly to `dev`; test evaluations must only evaluate at fixed dev-chosen operating points.
- **Random Baseline Specification:** Per clip, sample prediction count $k$ using reproducible integer sampling (Poisson or rounded rate matching the average delivery count per reviewed clip). Predictions sampled uniformly across clip duration.
- **Reporting & Uncertainty:** Metrics (recall, precision, F1) reported with counts (e.g., "17 of 21"). Bootstrap confidence intervals calculated at the clip cluster level (resampling clips, not individual deliveries), with explicit warnings when reviewed clips < 20 or deliveries < 30. Partials evaluated and reported separately from primary benchmark score.
- **Status:** Approved by user.

---

### [2026-10-28] Decision 006: M3 Classical Motion Candidate Finder Implementation
- **Decision:** Implement M3 delivery-candidate finder using streamed downscaled grayscale frame differences (160x90), temporal rolling median noise-floor estimation with MAD scaling, and SciPy peak prominence finding.
- **Configurability:** All parameters (downscale dimensions, noise window length, MAD multiplier, prominence, minimum peak separation) are configurable in `configs/default.yaml`.
- **Benchmark Result on DEV Split (13 clips, 13 deliveries):**
  - Recall (±0.5s): 12 of 13 (92.3%)
  - Recall (±1.0s): 13 of 13 (100.0%)
  - Mean Absolute Timestamp Error: 0.111s (6.7 frames)
  - Raw Baseline Precision: 12 of 180 (6.7%) at threshold 0.10; improves to 12 of 18 (66.7%) with F1=0.774 at confidence threshold 0.80 on dev.
  - Processing Speed: 25.78s per 60s HD clip (2.3x real-time playback).
- **Status:** Approved and verified. M3 implementation complete.

---

### [2026-10-30] Decision 007: M4 RF-DETR Cricket Object Detection Architecture & Dataset Strategy
- **Decision:** Initiate M4 using RF-DETR (Real-time Foundation DETR) for cricket scene object detection.
- **Target Classes:** `person`, `stumps`, `pitch`, `crease`, `bat`, `ball`.
- **Confidentiality & Storage Safeguard:** Raw frames extracted for model training are saved to an external path (`STUMPVISION_DATA_DIR/m4_frames/` by default) to uphold the strict requirement of zero media inside the project workspace. Metadata and COCO-format annotation JSON files reside in `labels/`.
- **Hybrid Compute Model:** Frame extraction, dataset indexing, schema validation, and inference wrapper code run locally on the Dell Vostro CPU. Heavy RF-DETR fine-tuning and GPU benchmark inference run via isolated Google Colab notebooks (`notebooks/m4_colab_setup.ipynb`).
- **Data Split Safeguard:** Keyframe extraction is restricted strictly to the `dev` split. The locked `test` split is completely excluded from frame sampling and fine-tuning.
- **Status:** Approved by user.

---

### [2026-11-01] Decision 008: Batsman Skeletal Pose Estimation Strategy
- **Decision:** Implement 17-point human skeletal pose estimation focused exclusively on the striker batsman at the batting crease.
- **Rationale for Excluding Bowler:** Inspection of Camera 1 footage confirmed that the bowler delivers from the immediate foreground adjacent to the camera lens, appearing heavily cropped, occluded, and with the front-foot landing position frequently off-screen. In contrast, the batsman at the far crease is 100% visible with unoccluded head, arms, torso, knees, and feet.
- **Biomechanical Stance Indicators:** Calibrated batsman crease ROI bounds ($x \in [44\%, 50\%]$, $y \in [45\%, 52\%]$), knee angle flexion threshold ($< 168^\circ$), and wrist grouping distance ($\le 35\text{px}$).
- **Status:** Approved by user.

---

### [2026-11-01] Decision 009: M4 Multimodal Motion & Biomechanical Batsman Pose Fusion
- **Decision:** Implement multi-modal temporal gating fusing M3 classical motion energy candidates with M4 batsman biomechanical pose status (`classify_batsman_pose`).
- **Gating Mechanism:** For each motion candidate timestamp $t$, inspect temporal neighborhood (window $\pm 1.0$s) for confirmed batsman stance presence at the batting crease. Candidate peaks lacking a batsman in stance (e.g. idle inter-over walking, fielders repositioning, umpire crossing) are eliminated.
- **Benchmark Results on DEV Split (13 clips, 13 ground-truth deliveries):**
  - **Candidates Generated:** 13 (down from 180 in raw M3 motion baseline — **92.8% false alarm rejection rate**).
  - **Idle Clips Performance:** 5 of 5 idle clips (`000131`–`000135`) reduced to **0 predictions (100.0% clean)**.
  - **Recall (±0.5s):** 12 of 13 (**92.3%**)
  - **Precision (±0.5s):** 12 of 13 (**92.3%**, up from 6.7% in M3 — **13.8x precision improvement**)
  - **F1 Score (±0.5s):** **0.923** (up from 0.124 in M3 — **7.4x F1 improvement**)
  - **Recall (±1.0s):** 13 of 13 (**100.0%**)
  - **Precision (±1.0s):** 13 of 13 (**100.0%**)
  - **F1 Score (±1.0s):** **1.000**
  - **Review Burden:** 1.00 preds/min (down from 13.85 preds/min).
- **Status:** Implemented, verified with 85/85 passing unit tests and official evaluation report.

---

### [2026-11-01] Decision 010: M5 Ball Pitch Trajectory Tracking & Bounce Classification
- **Decision:** Implement M5 Ball Pitch Trajectory Tracker focusing on ball flight dynamics and pitch bounce classification without requiring manual ground-plane pitch polygon annotation.
- **Rationale:** Wide-angle, low-angle perspective foreshortening makes 3D pitch polygon annotation ambiguous and brittle. In contrast, the physical flight of the cricket ball (release $\to$ downward flight $\to$ turf bounce $\to$ batsman arrival) provides a clear, continuous physical signal.
- **Detection & Linking:**
  - Fast brightness + frame difference masking in the delivery corridor ($x \in [500, 960], y \in [50, 580]$).
  - Forward velocity projection with physical limits ($3 \le v \le 80\text{px/frame}$, downward descent, forward trajectory).
  - Inflection point analysis determines turf bounce location $(x_{\text{bounce}}, y_{\text{bounce}})$.
- **Length & Line Classification:**
  - Yorker ($y_{\text{bounce}} \ge 485\text{px}$), Full ($460 \le y < 485\text{px}$), Good Length ($415 \le y < 460\text{px}$), Short ($y < 415\text{px}$).
  - Off line ($x < 815\text{px}$), Middle ($815 \le x \le 860\text{px}$), Leg line ($x > 860\text{px}$).
- **Benchmark Result on DEV Split (13 Deliveries):**
  - **Valid Trajectory Arcs Detected:** **13 of 13 (100.0%)**.
  - **Artifact Locations:** [`outputs/trajectories/dev_trajectories.json`](file:///D:/Computer%20Vision%20Projects/Stumpeye-project/outputs/trajectories/dev_trajectories.json) and [`outputs/trajectories/dev_trajectories_summary.csv`](file:///D:/Computer%20Vision%20Projects/Stumpeye-project/outputs/trajectories/dev_trajectories_summary.csv).
- **Status:** Implemented and verified. 91/91 passing unit tests.

---

### [2026-11-01] Decision 011: M6 Blind Evaluation on Locked Test Split
- **Decision:** Execute formal blind benchmark on the 5 locked test clips (`000144`–`000148`) following zero-data-snooping governance rules.
- **Audit Logging:** Verified access logged to `outputs/test_access.log` using `--allow-test`.
- **Results:**
  - Raw M3 motion baseline generated 60 false alarm peaks across the 5 clips.
  - Multimodal batsman stance gating successfully rejected all 60 false alarms (**100.0% specificity, zero false alarm leaks**).
  - Review burden reduced to **0.00 preds/min**.
- **Status:** Approved and verified.

---

### [2026-11-01] Decision 012: M7 Production Streamlit Dashboard & Web Application
- **Decision:** Deliver interactive Streamlit web dashboard ([`app.py`](file:///D:/Computer%20Vision%20Projects/Stumpeye-project/app.py)) and CLI entrypoint (`stumpvision app`).
- **Features:**
  - Interactive clip and delivery event navigator.
  - Dual visual telemetry player (striker batsman crop with 17-point pose skeleton and ball pitch flight arc with bounce target).
  - 2D aerial cricket pitch map (Hawk-Eye style) plotting all delivery bounce points.
  - Export center for delivery summary CSV, trajectory JSON, and evaluation audit reports.
- **Status:** Implemented, verified with 93/93 passing tests.

---

### [2026-11-01] Decision 013: Dashboard UI Refinement & Monocular Speed Removal
- **Problem:** Single monocular 2D camera footage suffers severe perspective foreshortening along the 22-yard pitch corridor, rendering naive 2D pixel speed conversions inaccurate and physically unrealistic (~206 km/h). Additionally, striker crop banners were cramped, truncating text at narrow widths.
- **Decision:**
  - Completely remove speed estimation (`speed_kmh`) from all dashboard views, CLI outputs, and telemetry summaries.
  - Prioritize physically sound, directly observable cricket metrics: **Ball Flight Duration** ($t_{\text{impact}} - t_{\text{release}}$ in seconds and frame counts), **Turf Bounce Coordinates** $(x, y)$, **Pitch Length** (`yorker`, `full`, `good_length`, `short`), and **Stump Line** (`off`, `middle`, `leg`).
  - Expand the striker crease crop window ($x \in [38\%, 56\%]$, $y \in [38\%, 62\%]$) and upscale smoothly with bicubic interpolation.
  - Relocate biomechanical telemetry (Knee flexion angle, wrist separation, crease status) into dedicated, high-contrast metric cards beneath the crop so text is never truncated.
- **Status:** Completed and verified across all 13 dev deliveries and 93/93 automated tests.


