import cv2

# Look at bowler in frame 2903
f2903 = cv2.imread(r"outputs/calibration\d04_full_f2903.jpg")
if f2903 is not None:
    # Draw circle at (441, 150)
    disp = f2903.copy()
    cv2.circle(disp, (441, 150), 10, (0, 255, 255), 2)
    # Save crop around bowler
    crop = disp[100:600, 300:600]
    cv2.imwrite(r"outputs/calibration\bowler_release_d04_verified.jpg", crop)
    print("Saved bowler_release_d04_verified.jpg")

