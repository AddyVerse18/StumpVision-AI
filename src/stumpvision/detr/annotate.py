"""
annotate.py
===========
Local OpenCV interactive bounding-box annotation tool for StumpVision M4 (RF-DETR).
Supports:
- 6 cricket categories: person, stumps, pitch, crease, bat, ball.
- Image navigation, zoom, pan, box drawing, editing, deleting, undo.
- Safe atomic persistence with automatic timestamped backups.
- Resumes at first unannotated image or previously viewed index.
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from stumpvision.detr.coco_schema import (
    DEFAULT_CATEGORIES,
    load_coco_json,
    save_coco_with_backup,
    validate_coco_dataset,
)

# Colors in BGR format for visualization
CATEGORY_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (0, 220, 0),     # person: vibrant green
    2: (0, 230, 255),   # stumps: bright yellow
    3: (255, 120, 0),   # pitch: blue
    4: (255, 0, 255),   # crease: magenta
    5: (255, 255, 0),   # bat: cyan
    6: (0, 0, 255),     # ball: red
}

# Windows keycode constants
KEY_LEFT_ARROW = {0x250000, 2424832}
KEY_RIGHT_ARROW = {0x270000, 2555904}
KEY_DELETE = {0x2E0000, 3014656, 127}
KEY_ESC = 27


class AnnotationSession:
    """Headless state manager for COCO bounding-box annotation."""

    def __init__(
        self,
        coco_path: Path,
        frames_dir: Path,
        backup_dir: Optional[Path] = None,
        auto_resume: bool = True,
    ):
        self.coco_path = Path(coco_path)
        self.frames_dir = Path(frames_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else self.coco_path.parent / "backups"

        self.coco_data: Dict[str, Any] = load_coco_json(self.coco_path)
        self.images: List[Dict[str, Any]] = self.coco_data.get("images", [])
        if not self.images:
            raise ValueError(f"No images found in COCO dataset: {self.coco_path}")

        self.categories: List[Dict[str, Any]] = self.coco_data.get("categories", list(DEFAULT_CATEGORIES))
        self.cat_id_to_name: Dict[int, str] = {c["id"]: c["name"] for c in self.categories}
        self.cat_name_to_id: Dict[str, int] = {c["name"]: c["id"] for c in self.categories}

        # Index annotations by image_id
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {
            img["id"]: [] for img in self.images
        }
        max_ann_id = 0
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in self.annotations_by_image:
                self.annotations_by_image[img_id].append(deepcopy(ann))
            max_ann_id = max(max_ann_id, ann["id"])
        self.max_annotation_id = max_ann_id

        self.active_category_id: int = 1  # person by default
        self.selected_bbox_idx: Optional[int] = None
        self.undo_stack: List[Dict[str, Any]] = []
        self.has_unsaved_changes: bool = False

        self.image_idx: int = 0
        if auto_resume:
            self.image_idx = self.find_first_unannotated_index()

    def current_image(self) -> Dict[str, Any]:
        return self.images[self.image_idx]

    def current_annotations(self) -> List[Dict[str, Any]]:
        img_id = self.current_image()["id"]
        return self.annotations_by_image[img_id]

    def set_image_index(self, idx: int) -> int:
        self.image_idx = max(0, min(len(self.images) - 1, idx))
        self.selected_bbox_idx = None
        return self.image_idx

    def next_image(self) -> bool:
        if self.image_idx < len(self.images) - 1:
            self.image_idx += 1
            self.selected_bbox_idx = None
            return True
        return False

    def prev_image(self) -> bool:
        if self.image_idx > 0:
            self.image_idx -= 1
            self.selected_bbox_idx = None
            return True
        return False

    def find_first_unannotated_index(self) -> int:
        for idx, img in enumerate(self.images):
            if len(self.annotations_by_image.get(img["id"], [])) == 0:
                return idx
        return 0

    def set_active_category(self, cat_id: int) -> bool:
        if cat_id in self.cat_id_to_name:
            self.active_category_id = cat_id
            return True
        return False

    def add_bounding_box(
        self,
        bbox_xywh: Tuple[float, float, float, float],
        category_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Add a bounding box to the current image with bounds clamping and validation."""
        curr_img = self.current_image()
        img_w = float(curr_img.get("width", 1920))
        img_h = float(curr_img.get("height", 1080))
        cat_id = category_id if category_id is not None else self.active_category_id

        if cat_id not in self.cat_id_to_name:
            raise ValueError(f"Unknown category ID: {cat_id}")

        x, y, w, h = bbox_xywh
        # Handle inverted coordinates
        if w < 0:
            x += w
            w = abs(w)
        if h < 0:
            y += h
            h = abs(h)

        # Clamp to image boundaries
        x_clamped = max(0.0, min(img_w - 1.0, float(x)))
        y_clamped = max(0.0, min(img_h - 1.0, float(y)))
        w_clamped = max(0.0, min(img_w - x_clamped, float(w)))
        h_clamped = max(0.0, min(img_h - y_clamped, float(h)))

        # Reject zero or tiny boxes (e.g. accidental clicks)
        if w_clamped < 2.0 or h_clamped < 2.0:
            return None

        # Check duplicate in same image
        for ann in self.current_annotations():
            if ann["category_id"] == cat_id:
                bx, by, bw, bh = ann["bbox"]
                if (
                    abs(bx - x_clamped) < 1.0
                    and abs(by - y_clamped) < 1.0
                    and abs(bw - w_clamped) < 1.0
                    and abs(bh - h_clamped) < 1.0
                ):
                    return None

        self.max_annotation_id += 1
        new_ann: Dict[str, Any] = {
            "id": self.max_annotation_id,
            "image_id": curr_img["id"],
            "category_id": cat_id,
            "bbox": [
                round(x_clamped, 2),
                round(y_clamped, 2),
                round(w_clamped, 2),
                round(h_clamped, 2),
            ],
            "area": round(w_clamped * h_clamped, 2),
            "iscrowd": 0,
        }

        self.annotations_by_image[curr_img["id"]].append(new_ann)
        self.selected_bbox_idx = len(self.annotations_by_image[curr_img["id"]]) - 1
        self.has_unsaved_changes = True

        self.undo_stack.append({
            "action": "add",
            "image_id": curr_img["id"],
            "annotation": deepcopy(new_ann),
        })
        return new_ann

    def delete_selected_bounding_box(self) -> Optional[Dict[str, Any]]:
        """Delete currently selected bounding box."""
        curr_anns = self.current_annotations()
        if self.selected_bbox_idx is None or self.selected_bbox_idx >= len(curr_anns):
            return None

        deleted = curr_anns.pop(self.selected_bbox_idx)
        self.selected_bbox_idx = None
        self.has_unsaved_changes = True

        self.undo_stack.append({
            "action": "delete",
            "image_id": self.current_image()["id"],
            "annotation": deepcopy(deleted),
        })
        return deleted

    def undo(self) -> bool:
        """Undo last box action."""
        if not self.undo_stack:
            return False

        last_op = self.undo_stack.pop()
        action = last_op["action"]
        img_id = last_op["image_id"]
        ann = last_op["annotation"]

        if action == "add":
            # Remove the added annotation
            self.annotations_by_image[img_id] = [
                a for a in self.annotations_by_image[img_id] if a["id"] != ann["id"]
            ]
            self.selected_bbox_idx = None
            self.has_unsaved_changes = True
            return True
        elif action == "delete":
            # Restore the deleted annotation
            self.annotations_by_image[img_id].append(deepcopy(ann))
            self.selected_bbox_idx = len(self.annotations_by_image[img_id]) - 1
            self.has_unsaved_changes = True
            return True
        return False

    def save(self) -> Tuple[Path, Optional[Path]]:
        """Reconstruct COCO annotations and atomically write with backup."""
        all_anns = []
        for img in self.images:
            all_anns.extend(self.annotations_by_image[img["id"]])

        # Re-sort annotations deterministically by image_id and bbox
        all_anns.sort(key=lambda a: (a["image_id"], a["bbox"][0], a["bbox"][1]))

        self.coco_data["annotations"] = all_anns
        saved_p, backup_p = save_coco_with_backup(
            dataset=self.coco_data,
            output_path=self.coco_path,
            backup_dir=self.backup_dir,
        )
        self.has_unsaved_changes = False
        return saved_p, backup_p

    def get_statistics(self) -> Dict[str, Any]:
        """Compute live session annotation counts."""
        total_images = len(self.images)
        annotated = sum(
            1 for img in self.images if len(self.annotations_by_image[img["id"]]) > 0
        )
        cat_counts = {c["name"]: 0 for c in self.categories}
        total_boxes = 0

        for img in self.images:
            for a in self.annotations_by_image[img["id"]]:
                cname = self.cat_id_to_name.get(a["category_id"])
                if cname:
                    cat_counts[cname] += 1
                total_boxes += 1

        return {
            "total_images": total_images,
            "annotated_images": annotated,
            "unannotated_images": total_images - annotated,
            "total_boxes": total_boxes,
            "category_counts": cat_counts,
        }


def run_annotation_gui(
    coco_path: Path,
    frames_dir: Path,
    display_width: int = 1280,
    display_height: int = 720,
) -> None:
    """Launch the OpenCV GUI for bounding-box annotation."""
    coco_path = Path(coco_path)
    frames_dir = Path(frames_dir)

    session = AnnotationSession(coco_path=coco_path, frames_dir=frames_dir)
    window_name = "StumpVision M4 — Cricket BBox Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Mouse interaction state
    mouse_state = {
        "drawing": False,
        "start_pt": (0, 0),
        "curr_pt": (0, 0),
        "pan_start": (0, 0),
        "panning": False,
    }

    # View zoom and pan offsets
    view_state = {
        "zoom": 1.0,
        "pan_x": 0.0,
        "pan_y": 0.0,
    }

    def window_to_image_coords(wx: int, wy: int, img_w: int, img_h: int) -> Tuple[float, float]:
        """Convert displayed window coordinates back to original frame pixel coordinates."""
        # Active canvas size is display_width x display_height
        norm_x = (wx - view_state["pan_x"]) / (display_width * view_state["zoom"])
        norm_y = (wy - view_state["pan_y"]) / (display_height * view_state["zoom"])
        ix = norm_x * img_w
        iy = norm_y * img_h
        return ix, iy

    def on_mouse(event, x, y, flags, param):
        img_info = session.current_image()
        w_orig = img_info.get("width", 1920)
        h_orig = img_info.get("height", 1080)

        # Right-click or middle-click pan
        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
            mouse_state["panning"] = True
            mouse_state["pan_start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse_state["panning"]:
            dx = x - mouse_state["pan_start"][0]
            dy = y - mouse_state["pan_start"][1]
            view_state["pan_x"] += dx
            view_state["pan_y"] += dy
            mouse_state["pan_start"] = (x, y)
        elif event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_MBUTTONUP:
            mouse_state["panning"] = False

        # Left-click drawing and selection
        elif event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = window_to_image_coords(x, y, w_orig, h_orig)
            mouse_state["drawing"] = True
            mouse_state["start_pt"] = (ix, iy)
            mouse_state["curr_pt"] = (ix, iy)

            # Check if clicked inside an existing box to select it
            hit_idx = None
            for idx, ann in enumerate(session.current_annotations()):
                bx, by, bw, bh = ann["bbox"]
                if bx <= ix <= bx + bw and by <= iy <= by + bh:
                    hit_idx = idx
            session.selected_bbox_idx = hit_idx

        elif event == cv2.EVENT_MOUSEMOVE and mouse_state["drawing"]:
            ix, iy = window_to_image_coords(x, y, w_orig, h_orig)
            mouse_state["curr_pt"] = (ix, iy)

        elif event == cv2.EVENT_LBUTTONUP and mouse_state["drawing"]:
            mouse_state["drawing"] = False
            ix, iy = window_to_image_coords(x, y, w_orig, h_orig)
            sx, sy = mouse_state["start_pt"]
            bw = abs(ix - sx)
            bh = abs(iy - sy)
            bx = min(sx, ix)
            by = min(sy, iy)

            # If user dragged a meaningful box (>= 4 pixels), create it
            if bw >= 4.0 and bh >= 4.0:
                session.add_bounding_box((bx, by, bw, bh))

    cv2.setMouseCallback(window_name, on_mouse)

    status_msg = "Ready. [1-6] Class | Left-drag to draw | [D/A] Next/Prev"
    status_time = datetime.now()

    while True:
        img_info = session.current_image()
        fname = img_info.get("file_name", "")
        fpath = frames_dir / fname

        if not fpath.exists():
            frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"Missing frame file: {fname}",
                (50, display_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
        else:
            orig_frame = cv2.imread(str(fpath))
            if orig_frame is None:
                orig_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            h_orig, w_orig = orig_frame.shape[:2]

            # Render canvas with zoom/pan
            scaled_w = int(display_width * view_state["zoom"])
            scaled_h = int(display_height * view_state["zoom"])
            resized = cv2.resize(orig_frame, (scaled_w, scaled_h))

            # Apply pan offsets to display buffer
            frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            px = int(round(view_state["pan_x"]))
            py = int(round(view_state["pan_y"]))

            # Coordinates on canvas
            src_x1 = max(0, -px)
            src_y1 = max(0, -py)
            src_x2 = min(scaled_w, display_width - px)
            src_y2 = min(scaled_h, display_height - py)

            dst_x1 = max(0, px)
            dst_y1 = max(0, py)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            if src_x2 > src_x1 and src_y2 > src_y1:
                frame[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

            def img_to_window_coords(ix: float, iy: float) -> Tuple[int, int]:
                norm_x = ix / w_orig
                norm_y = iy / h_orig
                wx = int(round(norm_x * display_width * view_state["zoom"] + view_state["pan_x"]))
                wy = int(round(norm_y * display_height * view_state["zoom"] + view_state["pan_y"]))
                return wx, wy

            # Draw existing bounding boxes
            curr_anns = session.current_annotations()
            for a_idx, ann in enumerate(curr_anns):
                bx, by, bw, bh = ann["bbox"]
                cid = ann["category_id"]
                cname = session.cat_id_to_name.get(cid, "obj")
                color = CATEGORY_COLORS.get(cid, (255, 255, 255))

                pt1 = img_to_window_coords(bx, by)
                pt2 = img_to_window_coords(bx + bw, by + bh)

                is_selected = (a_idx == session.selected_bbox_idx)
                thickness = 3 if is_selected else 2

                cv2.rectangle(frame, pt1, pt2, color, thickness)
                tag_label = f"{cid}:{cname}"
                cv2.putText(
                    frame,
                    tag_label,
                    (pt1[0], max(15, pt1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2 if is_selected else 1,
                    cv2.LINE_AA,
                )

            # Draw currently drawing box preview
            if mouse_state["drawing"]:
                sx, sy = mouse_state["start_pt"]
                cx, cy = mouse_state["curr_pt"]
                pt1 = img_to_window_coords(min(sx, cx), min(sy, cy))
                pt2 = img_to_window_coords(max(sx, cx), max(sy, cy))
                act_color = CATEGORY_COLORS.get(session.active_category_id, (0, 255, 0))
                cv2.rectangle(frame, pt1, pt2, act_color, 1, cv2.LINE_4)

        # ----------------- HUD Header / Overlay -----------------
        # Dark top bar
        cv2.rectangle(frame, (0, 0), (display_width, 60), (25, 25, 25), -1)
        stype = img_info.get("sample_type", "frame")
        ts_val = img_info.get("time_s", 0.0)
        idx_str = f"Image [{session.image_idx + 1}/{len(session.images)}]"
        info_str = f"{idx_str}  {fname} (t={ts_val:.2f}s, {stype})  Boxes: {len(session.current_annotations())}"
        cv2.putText(frame, info_str, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Categories bar
        cat_bar_x = 15
        for cat in session.categories:
            cid = cat["id"]
            cname = cat["name"]
            is_active = (cid == session.active_category_id)
            color = CATEGORY_COLORS.get(cid, (255, 255, 255))
            label = f"[{cid}] {cname}"
            if is_active:
                cv2.putText(frame, f"*{label}*", (cat_bar_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, label, (cat_bar_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            cat_bar_x += len(label) * 11 + 25

        # ----------------- HUD Footer Bar -----------------
        cv2.rectangle(frame, (0, display_height - 35), (display_width, display_height), (25, 25, 25), -1)
        help_bar = "[1-6] Class | [D/->] Next | [A/<-] Prev | [X/Del] Delete | [U] Undo | [S] Save | [+/-/0] Zoom | [Q/Esc] Exit"
        cv2.putText(frame, help_bar, (15, display_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Status flash
        if (datetime.now() - status_time).total_seconds() < 4.0:
            cv2.putText(frame, status_msg, (display_width - 450, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKeyEx(20)

        if key in (KEY_ESC, ord("q"), ord("Q")):
            if session.has_unsaved_changes:
                session.save()
                print("[INFO] Auto-saved changes before exiting.")
            break

        # Category hotkeys: 1-6
        elif ord("1") <= key <= ord("6"):
            chosen_id = key - ord("0")
            session.set_active_category(chosen_id)
            status_msg = f"Selected class: {session.cat_id_to_name.get(chosen_id)}"
            status_time = datetime.now()

        # Next image: D or Right Arrow
        elif key in (ord("d"), ord("D")) or key in KEY_RIGHT_ARROW:
            if session.has_unsaved_changes:
                session.save()
            if session.next_image():
                status_msg = f"Moved to image {session.image_idx + 1}"
            else:
                status_msg = "Already at final image"
            status_time = datetime.now()

        # Previous image: A or Left Arrow
        elif key in (ord("a"), ord("A")) or key in KEY_LEFT_ARROW:
            if session.has_unsaved_changes:
                session.save()
            if session.prev_image():
                status_msg = f"Moved to image {session.image_idx + 1}"
            else:
                status_msg = "Already at first image"
            status_time = datetime.now()

        # Delete selected box: X or Delete
        elif key in (ord("x"), ord("X")) or key in KEY_DELETE:
            deleted = session.delete_selected_bounding_box()
            if deleted:
                status_msg = f"Deleted bbox {deleted['id']}"
            else:
                status_msg = "No bbox selected to delete"
            status_time = datetime.now()

        # Undo: U
        elif key in (ord("u"), ord("U")):
            if session.undo():
                status_msg = "Reversed last action"
            else:
                status_msg = "Undo stack empty"
            status_time = datetime.now()

        # Save: S
        elif key in (ord("s"), ord("S")):
            saved_p, backup_p = session.save()
            status_msg = f"Saved! Backup: {backup_p.name if backup_p else 'none'}"
            status_time = datetime.now()

        # Zoom controls
        elif key in (ord("+"), ord("=")):
            view_state["zoom"] = min(4.0, view_state["zoom"] * 1.25)
            status_msg = f"Zoom: {view_state['zoom']:.2f}x"
            status_time = datetime.now()
        elif key == ord("-"):
            view_state["zoom"] = max(1.0, view_state["zoom"] / 1.25)
            if view_state["zoom"] <= 1.05:
                view_state["zoom"] = 1.0
                view_state["pan_x"] = 0.0
                view_state["pan_y"] = 0.0
            status_msg = f"Zoom: {view_state['zoom']:.2f}x"
            status_time = datetime.now()
        elif key == ord("0"):
            view_state["zoom"] = 1.0
            view_state["pan_x"] = 0.0
            view_state["pan_y"] = 0.0
            status_msg = "Reset zoom and pan"
            status_time = datetime.now()

    cv2.destroyAllWindows()

