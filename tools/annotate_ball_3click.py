"""
annotate_ball_3click.py
========================
Interactive 3-Click Ball Annotation Tool for StumpVision AI.

For each delivery, opens an OpenCV window where you:
  1. Scrub to the BOWLER RELEASE frame  -> click on ball in bowler's hand
  2. Scrub to the TURF BOUNCE frame     -> click on ball at pitch bounce
  3. Scrub to the BAT CONTACT frame     -> click on ball at bat face

Keyboard Controls:
  D / Right Arrow  = Forward 1 frame
  A / Left Arrow   = Back 1 frame
  W / Up Arrow     = Forward 5 frames
  S / Down Arrow   = Back 5 frames
  Q               = Forward 20 frames
  E               = Back 20 frames
  SPACE            = Confirm click for current stage
  R               = Reset current delivery
  ESC             = Skip this delivery
  N               = Move to Next delivery
  P               = Move to Previous delivery

Outputs:
  - outputs/trajectories/manual_annotations.json
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("D:/Computer Vision Projects/StumpEye_Footage")

TRAJ_JSON = PROJECT_ROOT / "outputs" / "trajectories" / "dev_trajectories.json"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "trajectories" / "manual_annotations.json"

STAGES = ["BOWLER_RELEASE", "TURF_BOUNCE", "BAT_CONTACT"]
STAGE_COLORS = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]  # Red, Yellow, Green
STAGE_LABELS = [
    "Click on BALL in bowler's hand (Release)",
    "Click on BALL at pitch turf bounce spot",
    "Click on BALL at bat contact point",
]

# Global state for mouse callback
click_x, click_y = -1, -1
click_received = False


def mouse_callback(event, x, y, flags, param):
    global click_x, click_y, click_received
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y
        click_received = True


def draw_hud(frame, delivery_idx, clip_id, stage_idx, frame_idx, total_frames,
             annotations_so_far, candidate_time_s):
    """Draw informative HUD overlay on frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Semi-transparent dark bar at top
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.82, display, 0.18, 0, display)

    # Title
    title = "DELIVERY #%02d  [Clip %s]  |  Ball at %.2fs" % (
        delivery_idx, clip_id[-6:], candidate_time_s)
    cv2.putText(display, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Stage indicator
    stage_text = "Stage %d/3: %s" % (stage_idx + 1, STAGE_LABELS[stage_idx])
    cv2.putText(display, stage_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, STAGE_COLORS[stage_idx], 2, cv2.LINE_AA)

    # Frame info
    frame_text = "Frame: %d / %d  |  Time: %.3fs  |  [A/D]=+-1f  [W/S]=+-5f  [Q/E]=+-20f  SPACE=Confirm  R=Reset  ESC=Skip" % (
        frame_idx, total_frames, frame_idx / 60.0)
    cv2.putText(display, frame_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (180, 195, 210), 1, cv2.LINE_AA)

    # Draw already-placed annotations
    for i, ann in enumerate(annotations_so_far):
        if ann is not None:
            ax, ay, af = ann
            # Draw circle + label
            cv2.circle(display, (int(ax), int(ay)), 12, STAGE_COLORS[i], 2, cv2.LINE_AA)
            cv2.circle(display, (int(ax), int(ay)), 4, (255, 255, 255), -1, cv2.LINE_AA)
            label = "%s (f%d)" % (STAGES[i], af)
            cv2.putText(display, label, (int(ax) + 15, int(ay) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, STAGE_COLORS[i], 1, cv2.LINE_AA)

    # Crosshair in center of frame
    cx, cy = w // 2, h // 2
    cv2.line(display, (cx - 20, cy), (cx + 20, cy), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(display, (cx, cy - 20), (cx, cy + 20), (100, 100, 100), 1, cv2.LINE_AA)

    return display


def annotate_delivery(cap, delivery_idx, clip_id, candidate_time_s,
                      existing_annotation=None):
    """Interactive 3-click annotation for a single delivery."""
    global click_x, click_y, click_received

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    # Start near the delivery event (1.5s before candidate time)
    start_frame = max(0, int(round((candidate_time_s - 1.5) * fps)))
    current_frame = start_frame

    # Annotations: list of 3 tuples (x, y, frame_idx) or None
    annotations = [None, None, None]
    if existing_annotation:
        for i, key in enumerate(["release", "bounce", "bat_contact"]):
            if key in existing_annotation and existing_annotation[key] is not None:
                a = existing_annotation[key]
                annotations[i] = (a["x"], a["y"], a["frame_idx"])

    current_stage = 0
    # Find first unfilled stage
    for i in range(3):
        if annotations[i] is None:
            current_stage = i
            break
    else:
        current_stage = 0  # All filled, start from scratch

    win_name = "StumpVision - Delivery #%02d Annotation" % delivery_idx
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)
    cv2.setMouseCallback(win_name, mouse_callback)

    click_received = False

    while True:
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            current_frame = max(0, current_frame - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

        # Draw HUD
        display = draw_hud(frame, delivery_idx, clip_id, current_stage,
                          current_frame, total_frames, annotations, candidate_time_s)

        # Show pending click marker
        if click_received and current_stage < 3:
            cv2.circle(display, (click_x, click_y), 14, STAGE_COLORS[current_stage], 2, cv2.LINE_AA)
            cv2.circle(display, (click_x, click_y), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(display, "SPACE to confirm | Click again to adjust",
                       (click_x + 18, click_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(win_name, display)

        key = cv2.waitKey(30) & 0xFF

        # Navigation
        if key == ord('d') or key == 83:  # Right arrow
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key == ord('a') or key == 81:  # Left arrow
            current_frame = max(0, current_frame - 1)
        elif key == ord('w') or key == 82:  # Up arrow
            current_frame = min(total_frames - 1, current_frame + 5)
        elif key == ord('s') or key == 84:  # Down arrow
            current_frame = max(0, current_frame - 5)
        elif key == ord('q'):
            current_frame = min(total_frames - 1, current_frame + 20)
        elif key == ord('e'):
            current_frame = max(0, current_frame - 20)

        # Confirm click for current stage
        elif key == 32:  # SPACE
            if click_received and current_stage < 3:
                annotations[current_stage] = (click_x, click_y, current_frame)
                print("  [OK] %s: x=%d y=%d frame=%d (t=%.3fs)" % (
                    STAGES[current_stage], click_x, click_y, current_frame,
                    current_frame / fps))
                click_received = False
                current_stage += 1

                if current_stage >= 3:
                    # All 3 done — show final result briefly
                    final_display = draw_hud(frame, delivery_idx, clip_id, 2,
                                            current_frame, total_frames,
                                            annotations, candidate_time_s)
                    cv2.putText(final_display,
                               "ALL 3 POINTS SET! Press SPACE to save, R to redo",
                               (15, frame.shape[0] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                               cv2.LINE_AA)
                    cv2.imshow(win_name, final_display)

                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 == 32:  # SPACE = save
                            cv2.destroyWindow(win_name)
                            return {
                                "delivery_idx": delivery_idx,
                                "clip_id": clip_id,
                                "candidate_time_s": candidate_time_s,
                                "release": {
                                    "x": annotations[0][0],
                                    "y": annotations[0][1],
                                    "frame_idx": annotations[0][2],
                                },
                                "bounce": {
                                    "x": annotations[1][0],
                                    "y": annotations[1][1],
                                    "frame_idx": annotations[1][2],
                                },
                                "bat_contact": {
                                    "x": annotations[2][0],
                                    "y": annotations[2][1],
                                    "frame_idx": annotations[2][2],
                                },
                            }
                        elif k2 == ord('r'):
                            annotations = [None, None, None]
                            current_stage = 0
                            click_received = False
                            break
                        elif k2 == 27:  # ESC
                            cv2.destroyWindow(win_name)
                            return None

        # Reset
        elif key == ord('r'):
            annotations = [None, None, None]
            current_stage = 0
            click_received = False
            print("  [RESET] Cleared all points for Delivery #%02d" % delivery_idx)

        # Skip / ESC
        elif key == 27:
            cv2.destroyWindow(win_name)
            return None

    cv2.destroyWindow(win_name)
    return None


def main():
    """Run 3-click annotation for all deliveries."""
    print("=" * 70)
    print("  StumpVision AI — 3-Click Ball Annotation Tool")
    print("=" * 70)
    print()

    # Load existing trajectories to get delivery list
    if not TRAJ_JSON.exists():
        print("ERROR: Trajectory JSON not found at", TRAJ_JSON)
        sys.exit(1)

    with open(TRAJ_JSON, "r") as f:
        trajectories = json.load(f)

    print("Found %d deliveries to annotate." % len(trajectories))
    print()

    # Load existing manual annotations if any
    existing_annotations = {}
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r") as f:
            existing_data = json.load(f)
        for ann in existing_data:
            existing_annotations[ann["delivery_idx"]] = ann
        print("Loaded %d existing manual annotations." % len(existing_annotations))
    print()

    all_annotations = dict(existing_annotations)  # Start with existing

    for t in trajectories:
        d_idx = t["delivery_idx"]
        clip_id = t["clip_id"]
        cand_t = t["candidate_time_s"]

        # Find video file (case-insensitive)
        video_path = DATA_DIR / "raw" / "cam1" / ("%s.mp4" % clip_id)
        if not video_path.exists():
            video_path = DATA_DIR / "raw" / "cam1" / ("%s.MP4" % clip_id)
        if not video_path.exists():
            print("SKIP Delivery #%02d: Video not found for clip %s" % (d_idx, clip_id))
            continue

        existing = all_annotations.get(d_idx)
        status = " [ALREADY ANNOTATED]" if existing else ""
        print("--- Delivery #%02d  Clip=%s  t=%.2fs%s ---" % (
            d_idx, clip_id[-6:], cand_t, status))

        if existing:
            print("  Existing: Release=(%d,%d)f%d  Bounce=(%d,%d)f%d  Bat=(%d,%d)f%d" % (
                existing["release"]["x"], existing["release"]["y"], existing["release"]["frame_idx"],
                existing["bounce"]["x"], existing["bounce"]["y"], existing["bounce"]["frame_idx"],
                existing["bat_contact"]["x"], existing["bat_contact"]["y"], existing["bat_contact"]["frame_idx"],
            ))
            print("  Press ENTER to re-annotate, or type 's' to skip: ", end="")
            user_input = input().strip().lower()
            if user_input == 's':
                continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("  ERROR: Cannot open video file")
            continue

        result = annotate_delivery(cap, d_idx, clip_id, cand_t, existing)
        cap.release()

        if result is not None:
            all_annotations[d_idx] = result
            # Save after each delivery (incremental)
            OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                sorted_anns = sorted(all_annotations.values(), key=lambda a: a["delivery_idx"])
                json.dump(sorted_anns, f, indent=2)
            print("  SAVED. (%d/%d deliveries annotated)" % (
                len(all_annotations), len(trajectories)))
        else:
            print("  Skipped Delivery #%02d" % d_idx)

        print()

    cv2.destroyAllWindows()

    # Final summary
    print()
    print("=" * 70)
    print("  ANNOTATION COMPLETE")
    print("  Total annotated: %d / %d deliveries" % (
        len(all_annotations), len(trajectories)))
    print("  Output saved to: %s" % OUTPUT_JSON)
    print("=" * 70)


if __name__ == "__main__":
    main()

