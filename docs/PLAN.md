# RESET AND RESTART: StumpVision AI Delivery Finder (PLAN v4, no git)

## 0. Rules for the agent
- Work ONE step at a time. Before changing anything, post a short plan and wait for my approval. Keep changes small. Show REAL test output. Never invent metrics, dataset sizes or results.
- Ask before adding any dependency. Log decisions in docs/DECISIONS.md (decision, reason, date).
- NO GIT: do not run git commands or set up GitHub or any remote. If a .git folder already exists, leave it alone and report it. I will tell you when to use git.
- CONFIDENTIALITY: the footage belongs to StumpEye and is private. I download it myself into a folder OUTSIDE the project. You must NOT download, scrape or fetch footage, and must not open any footage links or browser sessions. Never upload, view or send footage or frames to any external service or model API. Derived numbers and plots are fine. No hosted labelling tools (CVAT cloud, Roboflow), Colab or Docker unless I approve in writing.
- Independent project, not an official StumpEye product. Don't imply endorsement.

## 1. RESET PROTOCOL (in order)
Step 1: INSPECT ONLY. Report: project root path, total size, top-level listing, any virtual environment inside the project, whether a .git folder exists, and any video/media files or files over 20 MB inside the project. Delete nothing.
Step 2: PROPOSE the reset: list exactly what you will delete (everything inside the project root except what I name). Confirm you will NOT touch parent folders, other projects, global Python installs or my data folder. I have made my own backup copy of the old project.
Step 3: Wait for me to type exactly: CONFIRM RESET. Only then delete, only inside the project root. Then create a fresh scaffold and save this whole message as docs/PLAN.md.
If anything unexpected appears (paths outside the project, large media inside it), STOP and ask me.

## 2. Goal
A small, useful tool for StumpEye: given stump-camera clips, find WHEN deliveries happen so a reviewer can jump to them and skip empty stretches.
Per clip: view type (bowler_end / batter_end / unknown), candidate delivery timestamps with scores, inactive segments. Plus a CSV/JSON export, a simple review page, and a measured evaluation.
Design for HIGH RECALL: a missed delivery is worse than a false alarm. Report recall first, then precision.
Out of scope (v1): ball detection or tracking, bat or batter detection, shots or stance analysis, outcomes (4/6/out), score, speed, LBW or umpiring, real-time use. No box annotation. Outputs are "candidates" for a human to check, never decisions.

## 3. What we know (from ONE clip; verify on others)
- 1920x1080, 60 fps constant, 60.0 s, 3600 frames, H.264 plus AAC, about 124 MB. Clips are fixed-length chunks of continuous recording, not one delivery each.
- Camera at ground level at one end, wide-angle, static, looking down the pitch. At the bowler's end the bowler runs past the lens; the far batter is small (about 60 to 70 px tall).
- Frame-difference signal: flat noise floor when nothing moves; big spikes at two deliveries (about 0:02 and 0:43). A batter walking past the lens (about 0:16) also spiked (false-positive risk). About 0:18 to 0:42 was empty.
- Audio loudness did NOT line up with deliveries, so audio is not a cue.
- The bowling end alternates each over, so bowler's-end views probably appear in both camera folders. At batter's-end views the batter's back or side reportedly fills the frame. Not verified.
Everything is a hypothesis until measured.

## 4. Environment and data
- Windows 11, Dell Vostro 3500, 11th-gen i5, 24 GB RAM, CPU only. No training on GPU, no cloud.
- NEW virtual environment inside the project only. Do not touch other environments or global packages. Check compatibility and tell me the Python version and why (prefer 3.11 or 3.12).
- Core dependencies: numpy, opencv-python, pandas, scipy, PyYAML, pytest, streamlit. Optional, only with my approval: scikit-learn (small models in M4), ffmpeg/ffprobe, a pretrained person detector (e.g. Ultralytics YOLO nano on CPU; note its AGPL-3.0 licence).
- Data dir from env var STUMPVISION_DATA_DIR (outside the project): raw/cam1/, raw/cam2/ hold the clips I download. Outputs go to outputs/ inside the project.
- DATA SIZE: I will download only 12 to 16 full-quality original clips (about 2 GB): 3 or 4 contiguous blocks of 3 to 5 clips from early, middle and late in the match, from both cameras. Keep the originals; no proxies by default. Optional CLI `make_proxies` (640 wide, 30 fps, no audio, under STUMPVISION_DATA_DIR/proxy/) only if processing speed or disk makes it necessary; ask me first. Never delete my files.
- Process video frame by frame in a stream. Never load a whole video into memory. Warn me if free disk space looks low.

## 5. Project layout
configs/ docs/ (PLAN.md, DECISIONS.md, LABELING_GUIDE.md, EVALUATION.md) src/stumpvision/ (io, signals, detect, eval, viz, export, schemas.py, cli.py) app/ scripts/ tests/ labels/ outputs/
No footage inside the project.

## 6. Data contracts (frame indexes 0-based; time_s = frame_index / fps; add schema_version to every output)
- manifest.csv: clip_id, relative_path, camera, fps, width, height, n_frames, duration_s, block_id, view (manual), split (dev | test)
- labels.csv: clip_id, delivery_time_s (one row per delivery; empty time = no delivery), notes. I create it in a spreadsheet by hand.
- signal files per clip: frame_index, time_s, motion
- predictions: clip_id, time_s, score, status="candidate", method_version
- segments: clip_id, start_s, end_s, kind (active | inactive)

## 7. Milestones (do not start the next until I approve)
M0 Setup and audit: scaffold, venv, config, `audit` CLI that scans the clips in the data dir and writes the manifest (warn on variable frame rate), logging, pytest with a synthetic video made in code. Acceptance: tests pass; my sample clip reports 3600 frames, 60 fps, 60.0 s; no footage in the project; other environments untouched.
M1 Labels and splits: docs/LABELING_GUIDE.md defining "delivery time" (default: the moment the bowler is closest to the lens; I decide). Labels template and splits script. Split by contiguous block, never by random clip or frame. Assign 1 or 2 blocks as a LOCKED test set; the rest is dev. With so few clips, evaluate dev by leave-one-block-out. Leakage check must pass. Report the number of clips and deliveries honestly in every result.
M2 Evaluation harness BEFORE any detector: one-to-one matching within a tolerance (default 0.5 s, also report 1.0 s); recall, precision, F1, mean absolute timestamp error, per-clip misses and false positives. Unit tests (perfect, miss, false positive, duplicates, tolerance edge). "Predict nothing" and "random times" baselines run through it.
M3 Motion-signal baseline: streamed downscaled grayscale frame differences, noise-floor estimate, peak finding with minimum separation, inactive segments, debug plots, thresholds in config. Measure real processing time per clip. Evaluate on DEV only; list failure cases (e.g. batter walking past). Do not touch the test set.
M4 Reduce false positives (only if M3 shows a need; one idea at a time, keep only what improves dev metrics, log each in docs/EVALUATION.md):
  (a) spike shape and duration, and where in the frame the motion is;
  (b) a small model (logistic regression or a shallow decision tree) on features of each candidate spike, trained on my timestamp labels with leave-one-block-out validation. Report how small the training set is;
  (c) a sanity check of a pretrained person detector on about 30 frames (size growth and direction cues), only with my approval;
  (d) a simple view classifier (bowler_end / batter_end / unknown).
M5 Review app and exports: Streamlit app reading saved results only. Pick a clip, play from a candidate time, list candidates with scores, show the signal plot, optional "mark wrong" saved to a feedback CSV. CSV/JSON exports. Stretch: cut short clips around candidates.
M6 Final evaluation: run the locked test set (log every run; at most twice), reproduce from a fresh venv with documented commands, README with problem, method, measured results, failure analysis, limits, and an "implemented vs experimental" list. Every number must trace to a saved report. State limits: one match, one ground, a small number of clips, mostly bowler's-end views.
