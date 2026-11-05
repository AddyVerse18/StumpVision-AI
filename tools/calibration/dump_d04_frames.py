import cv2
import numpy as np
from pathlib import Path

cap = cv2.VideoCapture(r"D:\Computer Vision Projects\StumpEye_Footage\raw\cam1\20220102064250_000136.MP4")

# Save individual frames around bowler release for delivery 4 (around 2870 to 2920)
scratch_dir = Path(r"outputs/calibration\d04_frames")
scratch_dir.mkdir(parents=True, exist_ok=True)

# Let's save every frame from 2880 to 2930
for f in range(2880, 2930):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if ret:
        # Save a crop around bowler & pitch
        crop = frame[50:600, 600:1000]
        cv2.putText(crop, f"f{f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imwrite(str(scratch_dir / f"f{f}.jpg"), crop)
cap.release()
print("Saved frames 2880-2930 to scratch/d04_frames")

