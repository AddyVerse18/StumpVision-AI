import cv2
import numpy as np

img = cv2.imread(r"outputs/calibration\d04_full_f2903.jpg")
# Check pixels around x: 430-455, y: 140-165
sub = img[135:170, 425:460]
print("Sub patch mean:", sub.mean(axis=(0,1)))
print("Min brightness:", sub.min(), "Max brightness:", sub.max())

