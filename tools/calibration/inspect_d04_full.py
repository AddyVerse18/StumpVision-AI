import cv2

cap = cv2.VideoCapture(r"D:\Computer Vision Projects\StumpEye_Footage\raw\cam1\20220102064250_000136.MP4")

# Let's save full frames 2895, 2900, 2903, 2905, 2908, 2912, 2915, 2920
scratch_dir = "outputs/calibration"
for f in [2895, 2900, 2903, 2905, 2908, 2912, 2915, 2920]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"{scratch_dir}/d04_full_f{f}.jpg", frame)
cap.release()
print("Saved full frames for d04.")

