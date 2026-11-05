"""
ui.py
=====
Local OpenCV interactive labeling interface for StumpVision AI.
Zero network access. Footage is accessed strictly read-only.
Keys are processed using cv2.waitKeyEx.
"""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from stumpvision.schemas import VALID_VIEWS
from stumpvision.labeling.store import LabelStore


# Keycode constants (supporting both standard ASCII and Windows extended arrow codes)
KEY_LEFT_ARROW = {0x250000, 2424832}
KEY_RIGHT_ARROW = {0x270000, 2555904}
KEY_ESC = 27


def run_labeling_gui(
    data_dir: Path,
    manifest_path: Path,
    labels_path: Path,
    status_path: Path,
    blind: bool = False,
    display_width: int = 1280,
    display_height: int = 720,
) -> None:
    """
    Launch the local OpenCV labeling HUD.
    """
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)
    labels_path = Path(labels_path)
    status_path = Path(status_path)

    store = LabelStore(
        labels_path=labels_path,
        clip_status_path=status_path,
        manifest_path=manifest_path,
        blind=blind,
    )

    if store.manifest_df.empty:
        print("[ERROR] Manifest is empty or missing. Run `stumpvision audit` first.")
        return

    clips = store.manifest_df.to_dict(orient="records")
    total_clips = len(clips)
    clip_idx = 0

    window_name = "StumpVision AI Delivery Finder — Labeling HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    active_note = ""

    while 0 <= clip_idx < total_clips:
        clip_info = clips[clip_idx]
        clip_id = str(clip_info["clip_id"]).strip()
        rel_path = str(clip_info["relative_path"]).strip()
        split = str(clip_info.get("split", "")).strip()
        video_full_path = data_dir / rel_path

        if not video_full_path.exists():
            print(f"[ERROR] Video file not found: {video_full_path}")
            clip_idx += 1
            continue

        cap = cv2.VideoCapture(str(video_full_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_full_path}")
            clip_idx += 1
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = store.get_fps_for_clip(clip_id)
        current_frame = 0
        is_playing = False
        speed_multiplier = 1.0  # 0.25, 0.5, 1.0

        current_view = store.get_view_for_clip(clip_id)
        if not current_view:
            current_view = "bowler_end"
            store.set_clip_view(clip_id, current_view)

        jump_frames = max(1, int(round(fps)))

        change_clip = False

        while cap.isOpened() and not change_clip:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                # Loop or pause at end
                current_frame = max(0, total_frames - 1)
                is_playing = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret:
                    break

            # Resize for smooth display and drawing
            display_frame = cv2.resize(frame, (display_width, display_height))

            # ----------------------------------------------------
            # RENDER HUD OVERLAY
            # ----------------------------------------------------
            overlay = display_frame.copy()
            # Top banner
            cv2.rectangle(overlay, (0, 0), (display_width, 88), (15, 15, 15), -1)
            # Bottom banner
            cv2.rectangle(overlay, (0, display_height - 60), (display_width, display_height), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.82, display_frame, 0.18, 0, display_frame)

            reviewed = store.is_clip_reviewed(clip_id)
            rev_text = "REVIEWED" if reviewed else "UNREVIEWED"
            rev_color = (60, 220, 60) if reviewed else (60, 60, 240)

            # Line 1: Header
            line1 = f"Clip [{clip_idx + 1}/{total_clips}]: {clip_id} | Split: {split or 'unassigned'} | View: {current_view.upper()} | Status: {rev_text}"
            cv2.putText(display_frame, line1, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(display_frame, rev_text, (display_width - 150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rev_color, 2, cv2.LINE_AA)

            # Line 2: Playback & Timeline
            time_s = current_frame / fps
            dur_s = total_frames / fps
            play_text = "PLAYING" if is_playing else "PAUSED"
            play_color = (80, 220, 80) if is_playing else (200, 200, 80)
            line2 = f"Frame: {current_frame:04d}/{total_frames} ({time_s:05.2f}s/{dur_s:05.2f}s) | FPS: {fps:.1f} | Speed: {speed_multiplier}x | [{play_text}]"
            cv2.putText(display_frame, line2, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, play_color, 1, cv2.LINE_AA)

            # Line 3: Marks on clip
            if store.blind:
                marks_str = "[BLIND RETEST MODE - LABELS HIDDEN]"
                marks_color = (200, 180, 50)
            else:
                marks = store.get_marks_for_clip(clip_id)
                if not marks:
                    marks_str = "Marks: None"
                else:
                    items = []
                    for m in marks:
                        k = m.get("kind", "deliv")
                        f_idx = m.get("frame_index", "")
                        items.append(f"{k}@{f_idx}")
                    marks_str = "Marks: " + ", ".join(items)
                marks_color = (80, 240, 240) if "delivery" in marks_str else (180, 180, 180)

            if active_note:
                marks_str += f" | Tag: [{active_note}]"

            cv2.putText(display_frame, marks_str, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, marks_color, 1, cv2.LINE_AA)

            # Timeline bar
            bar_y = display_height - 66
            cv2.line(display_frame, (16, bar_y), (display_width - 16, bar_y), (80, 80, 80), 3)
            progress_x = int(16 + (display_width - 32) * (current_frame / max(1, total_frames)))
            cv2.circle(display_frame, (progress_x, bar_y), 6, (0, 140, 255), -1)

            # Draw marks on timeline bar
            if not store.blind:
                for m in store.get_marks_for_clip(clip_id):
                    try:
                        mf = int(float(m.get("frame_index", -1)))
                        if mf >= 0:
                            mx = int(16 + (display_width - 32) * (mf / max(1, total_frames)))
                            m_col = (0, 255, 0) if m.get("kind") == "delivery" else (0, 165, 255)
                            cv2.line(display_frame, (mx, bar_y - 6), (mx, bar_y + 6), m_col, 2)
                    except (ValueError, TypeError):
                        pass

            # Bottom Controls Help
            controls_line1 = "[A/D] Step -/+1 frame | [J/L] Jump -/+1s | [Space] Play/Pause | [1/2/3] Speed 0.25x/0.5x/1x"
            controls_line2 = "[M] Mark Delivery | [P] Part End | [O] Part Start | [U] Undo | [V] View | [R] Reviewed | [4-7] Tags | [[]/]] Clip | [Q] Exit"
            cv2.putText(display_frame, controls_line1, (16, display_height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display_frame, controls_line2, (16, display_height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

            # Delay calculation based on speed
            base_delay = int(1000.0 / (fps * speed_multiplier)) if is_playing else 30
            key = cv2.waitKeyEx(max(1, base_delay))

            if key != -1:
                # Key processing
                if key in [ord("q"), ord("Q"), KEY_ESC]:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n[INFO] Exited labeling tool. All actions saved.")
                    return

                elif key == ord(" "):
                    is_playing = not is_playing

                elif key in [ord("a"), ord("A")] or key in KEY_LEFT_ARROW:
                    is_playing = False
                    current_frame = max(0, current_frame - 1)

                elif key in [ord("d"), ord("D")] or key in KEY_RIGHT_ARROW:
                    is_playing = False
                    current_frame = min(total_frames - 1, current_frame + 1)

                elif key in [ord("j"), ord("J")]:
                    is_playing = False
                    current_frame = max(0, current_frame - jump_frames)

                elif key in [ord("l"), ord("L")]:
                    is_playing = False
                    current_frame = min(total_frames - 1, current_frame + jump_frames)

                elif key == ord("1"):
                    speed_multiplier = 0.25
                elif key == ord("2"):
                    speed_multiplier = 0.5
                elif key == ord("3"):
                    speed_multiplier = 1.0

                elif key == ord("4"):
                    active_note = "batter_walkpast"
                    print(f"[TAG] Quick note tag set: {active_note}")
                elif key == ord("5"):
                    active_note = "fielder"
                    print(f"[TAG] Quick note tag set: {active_note}")
                elif key == ord("6"):
                    active_note = "camera_moved"
                    print(f"[TAG] Quick note tag set: {active_note}")
                elif key == ord("7"):
                    active_note = "other"
                    print(f"[TAG] Quick note tag set: {active_note}")

                elif key in [ord("m"), ord("M")]:
                    new_mark = store.add_mark(
                        clip_id=clip_id,
                        frame_index=current_frame,
                        kind="delivery",
                        notes=active_note,
                    )
                    active_note = ""
                    print(f"[MARK] Added DELIVERY at frame {current_frame} (time {new_mark['delivery_time_s']}s) on {clip_id}")

                elif key in [ord("p"), ord("P")]:
                    new_mark = store.add_mark(
                        clip_id=clip_id,
                        frame_index=current_frame,
                        kind="partial_end",
                        notes=active_note,
                    )
                    active_note = ""
                    print(f"[MARK] Added PARTIAL_END at frame {current_frame} (time {new_mark['delivery_time_s']}s) on {clip_id}")

                elif key in [ord("o"), ord("O")]:
                    new_mark = store.add_mark(
                        clip_id=clip_id,
                        frame_index=current_frame,
                        kind="partial_start",
                        notes=active_note,
                    )
                    active_note = ""
                    print(f"[MARK] Added PARTIAL_START at frame {current_frame} (time {new_mark['delivery_time_s']}s) on {clip_id}")

                elif key in [ord("u"), ord("U")]:
                    undone = store.undo_last_mark(clip_id)
                    if undone:
                        print(f"[UNDO] Removed mark {undone.get('kind')} at frame {undone.get('frame_index')}")
                    else:
                        print("[UNDO] No marks to undo for this clip.")

                elif key in [ord("v"), ord("V")]:
                    idx = VALID_VIEWS.index(current_view) if current_view in VALID_VIEWS else 0
                    current_view = VALID_VIEWS[(idx + 1) % len(VALID_VIEWS)]
                    store.set_clip_view(clip_id, current_view)
                    print(f"[VIEW] Clip {clip_id} view set to: {current_view}")

                elif key in [ord("r"), ord("R")]:
                    new_status = not store.is_clip_reviewed(clip_id)
                    store.set_clip_reviewed(clip_id, reviewed=new_status)
                    status_str = "REVIEWED" if new_status else "UNREVIEWED"
                    print(f"[STATUS] Clip {clip_id} marked as: {status_str}")

                elif key in [ord("n"), ord("N")]:
                    is_playing = False
                    user_note = input(f"\n[NOTE] Enter note for frame {current_frame} on clip {clip_id} (e.g. linked_to=...): ").strip()
                    if user_note:
                        active_note = user_note
                        print(f"[NOTE] Note set: '{active_note}'. Press M/P/O to apply.")

                elif key == ord("["):
                    # Previous clip
                    if clip_idx > 0:
                        clip_idx -= 1
                        change_clip = True

                elif key == ord("]"):
                    # Next clip
                    if clip_idx < total_clips - 1:
                        clip_idx += 1
                        change_clip = True

            elif is_playing:
                current_frame = min(total_frames - 1, current_frame + 1)
                if current_frame >= total_frames - 1:
                    is_playing = False

        cap.release()

    cv2.destroyAllWindows()
    print("\n[INFO] Finished reviewing all clips.")
