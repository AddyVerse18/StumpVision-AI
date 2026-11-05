import cv2
import numpy as np
from pathlib import Path

# Load consecutive frames in d04 and find the bowler hand / ball
for f in range(2890, 2925, 2):
    f1 = cv2.imread(f"outputs/calibration/d04_frames/f{f}.jpg")
    f2 = cv2.imread(f"outputs/calibration/d04_frames/f{f+2}.jpg")
    if f1 is None or f2 is None:
        continue
    # Diff
    diff = cv2.absdiff(f1, f2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Check max motion location in upper half (bowler release area: y in 0 to 250 in crop, which is 50 to 300 in full frame)
    upper = gray[0:250, :]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(upper)
    # Also find bright moving spots
    mask = (upper > 30)
    if mask.sum() > 0:
        ys, xs = np.where(mask)
        # Centroid of moving object in upper region
        print(f"Frame {f}->{f+2}: max motion={max_val} at ({max_loc[0]+600}, {max_loc[1]+50}), upper motion centroid=({xs.mean()+600:.1f}, {ys.mean()+50:.1f}), active px={mask.sum()}")

