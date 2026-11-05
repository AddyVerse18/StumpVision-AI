# StumpVision AI — M4 RF-DETR Cricket Annotation Guide

This guide establishes the official bounding-box labeling protocol, geometric conventions, and desktop tool usage for the StumpVision AI M4 RF-DETR object detection dataset.

---

## 1. Dataset & File Locations

- **Extracted Frames (Read-Only):** `D:\Computer Vision Projects\StumpEye_Footage\m4_frames\` (54 keyframes extracted from the 13 dev clips).
- **Frame Index Manifest:** `labels/m4_extracted_frames.csv`
- **COCO Dataset:** `labels/m4_annotations/cricket_dev_coco.json`
- **Automatic Backups:** `labels/m4_annotations/backups/cricket_dev_coco_backup_<timestamp>.json`
- **Configuration:** `configs/m4_rf_detr.yaml`

---

## 2. Category Definitions & Bounding Conventions

All annotations use the standard COCO format: axis-aligned bounding boxes `[x, y, width, height]` (in 0-indexed pixel coordinates relative to the 1920×1080 source image).

### Category 1: `person`
- **What to enclose:** Any visible human player or match official (bowler, batter, wicket-keeper, fielders, umpires).
- **Boundary rule:** Draw a tight box enclosing the entire person from helmet/head to boots, including protective gear (pads, gloves) and clothing.
- **Exclusions:** Do *not* include the cricket bat in the `person` bounding box; the bat is annotated separately under category 5 (`bat`).

### Category 2: `stumps`
- **What to enclose:** The set of three vertical wickets, bails, and the immediate stump base.
- **Boundary rule:** Draw a tight box around the three stumps and bails.
- **Perspective note:** In bowler-end views, the bowling-end stumps are directly in front of the lens; the batting-end stumps are visible at the far end of the pitch. Annotate both sets when visible.

### Category 3: `pitch`
- **What to enclose:** The 22-yard prepared turf strip extending between the wickets.
- **Boundary rule:** An axis-aligned box encompassing the visible surface of the cricket pitch.
- **Representation Limitation & Note:** In ground-level perspective from behind the stumps, the pitch appears as a perspective trapezoid (narrowing toward the horizon). In standard 2D object detection (RF-DETR), the bounding box represents the AABB surrounding this visible strip.

### Category 4: `crease`
- **What to enclose:** The white painted crease markings (popping crease, bowling crease, or return creases).
- **Boundary rule:** A transverse box enclosing the visible white crease line markings on the pitch turf.

### Category 5: `bat`
- **What to enclose:** The cricket bat.
- **Boundary rule:** Tight box enclosing the blade and handle of the bat when held, swung, or carried by the batter.

### Category 6: `ball`
- **What to enclose:** The cricket ball.
- **Boundary rule:** Small box enclosing the ball.
- **Motion Blur Rule:** At 1080p 60fps, a delivery traveling 120–140 km/h often appears as a linear streak. If the streak is clearly identifiable as the ball in flight, enclose the streak. Do *not* annotate or hallucinate a ball when it is completely occluded or indistinguishable from grass texture.

---

## 3. Launching the Annotation Interface

To launch the local desktop annotation tool:

```powershell
.\.venv\Scripts\stumpvision.exe m4-annotate
```

### Keyboard Shortcuts

| Key | Action | Description |
| :--- | :--- | :--- |
| `1` – `6` | **Select Category** | Switch active drawing class (1: person, 2: stumps, 3: pitch, 4: crease, 5: bat, 6: ball) |
| `Left Click + Drag` | **Draw BBox** | Click and drag on canvas to draw a bounding box for the active category |
| `Left Click` | **Select BBox** | Click inside an existing box to select it (highlighted with thicker stroke) |
| `D` or `Right Arrow` | **Next Frame** | Advance to next image (automatically saves current annotations) |
| `A` or `Left Arrow` | **Previous Frame** | Go back to previous image (automatically saves current annotations) |
| `X` or `Delete` | **Delete BBox** | Delete the currently selected bounding box |
| `U` | **Undo** | Undo the most recent add or delete operation |
| `S` | **Save** | Manually save annotations to COCO JSON with timestamped backup |
| `+` / `-` | **Zoom In / Out** | Zoom canvas (1.0x up to 4.0x) |
| `0` | **Reset View** | Reset zoom and pan offsets to default |
| `Right Click + Drag` | **Pan Canvas** | Pan around the image when zoomed in |
| `Q` or `Esc` | **Save & Exit** | Save unsaved changes and close the interface |

---

## 4. Safe Persistence & Backup Strategy

1. **Atomic Write:** The dataset is written to a temporary file before replacing the target file, preventing corruption if interrupted.
2. **Automatic Backup:** Every time `save()` is triggered (manually or via navigation/exit), an exact copy of the previous state is preserved in `labels/m4_annotations/backups/cricket_dev_coco_backup_<timestamp>.json`.
3. **Session Resumption:** When launched, the tool automatically scans the dataset and opens directly at the first unannotated image, allowing annotation across multiple sessions.

---

## 5. Validating Dataset Integrity

To verify that all annotations on disk conform to the COCO standard, have positive areas, and remain within image boundaries:

```powershell
.\.venv\Scripts\stumpvision.exe m4-validate
```

Output displays:
- Total registered images and unannotated image count
- Total bounding boxes and per-category breakdown
- Any invalid, out-of-bounds, zero-area, or duplicate boxes

