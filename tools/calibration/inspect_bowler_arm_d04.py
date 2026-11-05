import cv2
import numpy as np

cap = cv2.VideoCapture(r"D:\Computer Vision Projects\StumpEye_Footage\raw\cam1\20220102064250_000136.MP4")

# Look at bowler's arm and head in frames 2896 to 2906
# Crop x in [350, 600], y in [150, 500]
for f in range(2896, 2907):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if ret:
        crop = frame[150:500, 350:600].copy()
        cv2.putText(crop, f"f{f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(f"outputs/calibration/bowler_arm_f{f}.jpg", crop)
cap.release()
print("Saved bowler arm crops.")

