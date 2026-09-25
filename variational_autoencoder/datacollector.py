import os
import time

import cv2

OUTPUT_FOLDER = "my_dataset"
TOTAL_IMAGES = 200
DELAY_SECONDS = 2.0

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Starting capture... Saving {TOTAL_IMAGES} images to '{OUTPUT_FOLDER}'.")
print("Press 'q' in the video window to quit early.")

count = 0
last_capture_time = time.time()

while count < TOTAL_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Live Webcam Feed - Press q to quit", frame)

    current_time = time.time()
    if current_time - last_capture_time >= DELAY_SECONDS:
        filename = os.path.join(OUTPUT_FOLDER, f"img_{count:03d}.jpg")

        cv2.imwrite(filename, frame)
        print(f"Captured {filename} ({count + 1}/{TOTAL_IMAGES})")

        count += 1
        last_capture_time = current_time

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Capture interrupted by user.")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")
