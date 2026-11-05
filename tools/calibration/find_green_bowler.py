import cv2
import numpy as np

# In d04_full_f2900, let's find the green jersey of the bowler!
# The bowler wears a distinctive green jersey!
f2900 = cv2.imread(r"outputs/calibration\d04_full_f2900.jpg")
f2905 = cv2.imread(r"outputs/calibration\d04_full_f2905.jpg")

# Green jersey in HSV
hsv = cv2.cvtColor(f2900, cv2.COLOR_BGR2HSV)
# Green hue: 35 to 85, sat > 40, val > 50
green_mask = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] >= 40) & (hsv[:,:,2] >= 50)

# Where is the bowler in f2900?
ys, xs = np.where(green_mask)
print(f"Bowler green jersey in f2900: x in [{xs.min()}, {xs.max()}], y in [{ys.min()}, {ys.max()}], center=({xs.mean():.1f}, {ys.mean():.1f})")

# In f2905:
hsv2905 = cv2.cvtColor(f2905, cv2.COLOR_BGR2HSV)
green_mask2 = (hsv2905[:,:,0] >= 35) & (hsv2905[:,:,0] <= 85) & (hsv2905[:,:,1] >= 40) & (hsv2905[:,:,2] >= 50)
ys2, xs2 = np.where(green_mask2)
print(f"Bowler green jersey in f2905: x in [{xs2.min()}, {xs2.max()}], y in [{ys2.min()}, {ys2.max()}], center=({xs2.mean():.1f}, {ys2.mean():.1f})")

