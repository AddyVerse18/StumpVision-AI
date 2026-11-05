# StumpVision AI — Delivery Labeling Guide (M1)

This document establishes the official labeling protocol for cricket delivery ground truth in the StumpVision AI Delivery Finder.

---

## 1. Ground Truth Definition: Delivery Point

- **Stored Truth:** `frame_index` (0-indexed integer) is the definitive ground truth.
- **Timestamp:** `delivery_time_s = frame_index / fps` (computed directly using the clip's constant frame rate from `labels/manifest.csv`).
- **Bowler's Closest Approach:** For a standard delivery from the bowler's end, mark the exact frame in the delivery stride where the bowler's body is largest in the camera image (the point of closest approach to the stumps/camera).
- **Multiple Deliveries Per Clip:** Every complete delivery within a 60-second clip receives its own distinct row in `labels/labels.csv`.

---

## 2. Views and Camera Orientation

Each clip is assigned an overall `view` tag written to `labels/manifest.csv`:
- `bowler_end`: Camera is behind the stumps at the bowler's end, facing the batter.
- `batter_end`: Camera is behind the stumps at the batter's end, facing the bowler running in.
- `mixed`: Significant camera displacement, mid-over rotation, or hybrid perspective.
- `unclear`: Occluded, poorly lit, or ambiguous viewpoint.

**Labeling across views:**
- For `batter_end`, `mixed`, and `unclear` views: mark the bowler's delivery stride (around ball release).
- Evaluation will report detection metrics broken down by view category.

---

## 3. Partial Deliveries

When a clip boundary truncates an active delivery:
- `partial_start`: The delivery is already under way as the clip begins (e.g. bowler follow-through or release).
  - **Stored `frame_index`:** Mark the *last visible frame* of the delivery action.
  - **Kind:** `partial_start`.
- `partial_end`: A delivery begins near the end of a clip but is cut off before completion.
  - **Stored `frame_index`:** Mark the *first visible frame* where the run-up / delivery stride starts.
  - **Kind:** `partial_end`.
- **Linking Consecutive Halves:**
  - In the `notes` column, include the reference to the adjacent clip (e.g. `linked_to=20220102063750_000131`).
  - Reporting tools count a delivery cut between two consecutive clips as *one delivery* when both halves are marked and linked.
  - **Evaluation Policy:** All partial deliveries are reported separately and excluded from the primary benchmark score.

---

## 4. Clips with No Deliveries (Idle Clips)

- An unreviewed clip is **not** a no-delivery clip.
- When reviewing a clip where no deliveries occur, press `R` (mark reviewed).
- The labeling tool automatically writes exactly one row:
  - `clip_id`: current clip ID
  - `delivery_time_s`: empty (`""` / `NaN`)
  - `frame_index`: empty (`""` / `NaN`)
  - `kind`: `none`
  - `notes`: distractor notes if applicable
- Unreviewed clips remain absent or flagged as unreviewed until explicitly confirmed. Evaluation excludes unreviewed clips.

---

## 5. Distractor Notes & Failure Analysis

Use the `notes` column to flag visual artifacts or potential false-positive triggers for post-hoc error analysis:
- `batter_walkpast`: Batter walks directly in front of the camera or stumps.
- `fielder`: Fielder runs through the frame or crosses near the pitch.
- `camera_moved`: Camera was bumped, repositioned, or vibrated.
- `other`: Umpires, practice swings, ball retrievals, etc.

---

## 6. Daily Relabeling & Consistency Verification Protocol

To verify labeling reliability and human temporal drift:
1. One day after completing initial labels, select 4 designated dev clips for relabeling.
2. Launch the blind labeling interface:
   ```powershell
   stumpvision label --labels-file labels/retest.csv --blind
   ```
   In `--blind` mode, existing annotations are hidden, and the primary `labels/labels.csv` is write-protected.
3. Compare the two labeling sets:
   ```powershell
   stumpvision compare-labels --primary labels/labels.csv --retest labels/retest.csv
   ```
4. Verification criteria:
   - Agreement within **±0.5 seconds** (±30 frames at 60 fps).
   - Zero unexpected missing or extra marks.
   - Mean and maximum difference (in frames and seconds) logged in report.
