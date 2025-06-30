import cv2
import os

# Define output path
output_dir = "reconn_image/test1"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "new_image.jpg")

# Open the first camera (index 0)
cap = cv2.VideoCapture(0)

# Check if camera is accessible
if not cap.isOpened():
    print("❌ Could not open camera.")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"✅ Image saved to {output_path}")
    else:
        print("❌ Failed to capture image.")
    cap.release()

