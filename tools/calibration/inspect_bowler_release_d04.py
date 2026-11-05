import cv2

# Look at bowler in frames 2900 to 2908
cap = cv2.VideoCapture(r"D:\Computer Vision Projects\StumpEye_Footage\raw\cam1\20220102064250_000136.MP4")
for f in range(2900, 2912):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if ret:
        # Crop bowler from x: 600 to 800, y: 50 to 450
        crop = frame[50:450, 600:800]
        cv2.imwrite(f"outputs/calibration/bowler_f{f}.jpg", crop)
cap.release()
print("Saved bowler crops.")

