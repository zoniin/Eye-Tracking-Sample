"""
Quick test to verify camera is working and face detection parameters
"""
import cv2
import dlib
import numpy as np

# Test camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[ERROR] Cannot open camera 1")
    exit(1)

print("[INFO] Camera 1 opened successfully")

# Get camera properties
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Camera resolution: {width}x{height} @ {fps} FPS")

# Load face detector
detector = dlib.get_frontal_face_detector()
print("[INFO] Face detector loaded")

print("[INFO] Testing face detection - press 'q' to quit")
print("[INFO] Trying different upsampling values for better detection...")

frame_count = 0
faces_detected_total = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame")
        break

    frame_count += 1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try with different upsampling levels
    # 0 = no upsampling (faster, less sensitive)
    # 1 = upsample once (slower, more sensitive)
    faces_0 = detector(frame_gray, 0)
    faces_1 = detector(frame_gray, 1)

    # Use the higher upsampling result
    faces = faces_1 if len(faces_1) > 0 else faces_0
    upsample_used = 1 if len(faces_1) > 0 else 0

    if len(faces) > 0:
        faces_detected_total += 1

    # Draw faces
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Face detected (upsample={upsample_used})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Status text
    status = f"Faces: {len(faces)} | Upsample: {upsample_used} | Frame: {frame_count}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    detection_rate = (faces_detected_total / frame_count) * 100 if frame_count > 0 else 0
    cv2.putText(frame, f"Detection rate: {detection_rate:.1f}%",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("Camera Test - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n[INFO] Test completed")
print(f"[INFO] Total frames: {frame_count}")
print(f"[INFO] Frames with faces: {faces_detected_total}")
print(f"[INFO] Detection rate: {detection_rate:.1f}%")
