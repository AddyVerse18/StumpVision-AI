import cv2
import numpy as np

# Load f2905 and f2895 and compute diff to find the bowler who is running
f2905 = cv2.imread(r"outputs/calibration\d04_full_f2905.jpg")
f2895 = cv2.imread(r"outputs/calibration\d04_full_f2895.jpg")

diff = cv2.absdiff(f2905, f2895)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
# Find moving blobs
ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
# Find contours
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    area = cv2.contourArea(c)
    if area > 1000:
        x, y, w, h = cv2.boundingRect(c)
        print(f"Moving object: x={x}, y={y}, w={w}, h={h}, area={area}")

