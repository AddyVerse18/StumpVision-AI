import cv2

f2905 = cv2.imread(r"outputs/calibration\d04_full_f2905.jpg")
# Crop around the bowler shown in user's screenshot
cv2.imwrite(r"outputs/calibration\bowler_crop_f2905.jpg", f2905[250:700, 250:500])
print("Saved bowler crop.")

