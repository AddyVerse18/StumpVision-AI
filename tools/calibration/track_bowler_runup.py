import cv2
import numpy as np

cap = cv2.VideoCapture(r"D:\Computer Vision Projects\StumpEye_Footage\raw\cam1\20220102064250_000136.MP4")

# Check bowler position in every 5th frame from 2870 to 2915
for f in range(2870, 2915, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if ret:
        # Find where the green person is
        # Crop x in [200, 600], y in [200, 700]
        sub = frame[200:700, 100:600]
        # Look for green jersey
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        # bright green jersey: h in [40, 80], sat > 80, val > 80
        mask = (hsv[:,:,0] >= 40) & (hsv[:,:,0] <= 80) & (hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 80)
        if mask.sum() > 50:
            ys, xs = np.where(mask)
            print(f"Frame {f}: Bowler green jersey at x={xs.mean()+100:.1f}, y={ys.mean()+200:.1f}, px_count={mask.sum()}")
        else:
            print(f"Frame {f}: No green jersey found in x:[100,600]")
cap.release()

