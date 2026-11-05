import cv2
import numpy as np

# In crops: x offset is 350, y offset is 150
for f in range(2898, 2906):
    img = cv2.imread(f"outputs/calibration/bowler_arm_f{f}.jpg")
    # Bowler's skin/hand or white ball at top of arm
    # Let's find top-most non-sky pixels in column range 50 to 200 (x in [400, 550])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Sky is blue: hsv hue ~ 100-130
    not_sky = (hsv[:,:,0] < 95) | (hsv[:,:,1] < 30) | (hsv[:,:,2] < 120)
    # in region y in [0, 250], x in [40, 220]
    sub = not_sky[0:250, 40:220]
    ys, xs = np.where(sub)
    if len(ys) > 0:
        top_y = ys.min() + 150
        top_x = xs[ys == ys.min()].mean() + 350 + 40
        print(f"Frame {f}: Highest point of bowler/arm at x={top_x:.1f}, y={top_y:.1f}")

