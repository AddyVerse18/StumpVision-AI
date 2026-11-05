import cv2
import numpy as np

# Find ball in frames 2906 to 2925
for f in range(2905, 2925):
    f1 = cv2.imread(f"outputs/calibration/d04_frames/f{f}.jpg")
    f2 = cv2.imread(f"outputs/calibration/d04_frames/f{f+2}.jpg")
    if f1 is None or f2 is None:
        continue
    diff = cv2.absdiff(f1, f2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Brightness mask on f2
    gray_f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    cand = (gray > 15) & (gray_f2 > 160)
    # find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cand.astype(np.uint8)*255)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        # in crop: x offset 600, y offset 50
        if 4 <= area <= 150:
            print(f"f{f}->f{f+2}: blob at ({cx+600:.1f}, {cy+50:.1f}), area={area}")

