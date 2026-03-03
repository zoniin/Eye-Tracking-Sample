"""
Eye Tracking Sample using dlib
Tracks gaze direction (left, right, center, up, down) using the
68-point facial landmark predictor from dlib.

Usage:
    python eye_tracker.py
    python eye_tracker.py --predictor path/to/shape_predictor_68_face_landmarks.dat
    python eye_tracker.py --source path/to/video.mp4
"""

import argparse
import sys
import cv2
import dlib
import numpy as np


# ---------------------------------------------------------------------------
# Facial landmark indices (dlib 68-point model)
# ---------------------------------------------------------------------------
# Left eye:  landmarks 36-41
# Right eye: landmarks 42-47
LEFT_EYE_POINTS  = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

# Iris region: inner 4 points of each eye
LEFT_IRIS_POINTS  = [37, 38, 40, 41]
RIGHT_IRIS_POINTS = [43, 44, 46, 47]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """Compute the Eye Aspect Ratio (EAR) to detect blinks."""
    # Vertical distances
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    # Horizontal distance
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C)


def landmarks_to_np(shape, dtype=np.float32) -> np.ndarray:
    """Convert a dlib full_object_detection to a (68, 2) NumPy array."""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_eye_region(frame_gray: np.ndarray, landmarks: np.ndarray,
                   eye_points: list, padding: int = 5):
    """
    Extract the eye ROI from the grayscale frame.

    Returns:
        eye_roi   -- cropped grayscale eye image
        eye_rect  -- (x, y, w, h) bounding rectangle in frame coordinates
    """
    pts = landmarks[eye_points].astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame_gray.shape[1] - x, w + 2 * padding)
    h = min(frame_gray.shape[0] - y, h + 2 * padding)
    return frame_gray[y:y + h, x:x + w], (x, y, w, h)


def detect_iris_position(eye_roi: np.ndarray):
    """
    Estimate iris centre within the eye ROI using thresholding + contours.

    Returns:
        (cx, cy) relative position in [0, 1] x [0, 1], or None on failure.
    """
    if eye_roi.size == 0:
        return None

    # Blur and threshold to isolate the dark iris
    blurred = cv2.GaussianBlur(eye_roi, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest contour (likely the iris)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"] / eye_roi.shape[1]
    cy = M["m01"] / M["m00"] / eye_roi.shape[0]
    return cx, cy


def classify_gaze(left_pos, right_pos) -> str:
    """
    Classify gaze direction from normalised iris positions.

    left_pos / right_pos: (cx, cy) in [0,1] or None
    """
    positions = [p for p in (left_pos, right_pos) if p is not None]
    if not positions:
        return "undetected"

    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])

    # Thresholds (tune to your setup)
    if cx < 0.38:
        h_dir = "right"   # iris near the left edge → looking right
    elif cx > 0.62:
        h_dir = "left"    # iris near the right edge → looking left
    else:
        h_dir = "center"

    if cy < 0.35:
        v_dir = "up"
    elif cy > 0.65:
        v_dir = "down"
    else:
        v_dir = ""

    if h_dir == "center" and not v_dir:
        return "center"
    if v_dir and h_dir == "center":
        return v_dir
    if v_dir:
        return f"{v_dir}-{h_dir}"
    return h_dir


def draw_gaze_arrow(frame: np.ndarray, gaze: str,
                    origin, length: int = 60):
    """Draw an arrow on the frame indicating gaze direction."""
    direction_map = {
        "center":       (0,  0),
        "left":         (-1, 0),
        "right":        (1,  0),
        "up":           (0, -1),
        "down":         (0,  1),
        "up-left":      (-1, -1),
        "up-right":     (1, -1),
        "down-left":    (-1,  1),
        "down-right":   (1,  1),
    }
    if gaze not in direction_map:
        return

    dx, dy = direction_map[gaze]
    ox, oy = origin
    tip = (int(ox + dx * length), int(oy + dy * length))
    color = (0, 255, 0) if gaze == "center" else (0, 165, 255)
    cv2.arrowedLine(frame, (ox, oy), tip, color, 3, tipLength=0.3)


# ---------------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------------

def run(predictor_path: str, source):
    # Load dlib models
    detector  = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(predictor_path)
    except RuntimeError as exc:
        print(f"[ERROR] Could not load shape predictor: {exc}")
        print("  Download it with:  bash setup.sh")
        sys.exit(1)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source!r}")
        sys.exit(1)

    # Blink detection state
    EAR_THRESHOLD = 0.21
    blink_count   = 0
    blink_frame   = 0
    BLINK_CONSEC  = 3          # consecutive frames below threshold = blink

    print("[INFO] Eye tracker running — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = detector(frame_gray, 0)

        gaze = "no face"

        for face in faces:
            shape     = predictor(frame_gray, face)
            landmarks = landmarks_to_np(shape)

            # --- Draw face bounding box ---
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            # --- Draw all 68 landmarks ---
            for (lx, ly) in landmarks.astype(np.int32):
                cv2.circle(frame, (lx, ly), 1, (0, 255, 255), -1)

            # --- Extract eye ROIs ---
            left_roi,  left_rect  = get_eye_region(frame_gray, landmarks,
                                                    LEFT_EYE_POINTS)
            right_roi, right_rect = get_eye_region(frame_gray, landmarks,
                                                    RIGHT_EYE_POINTS)

            # Draw eye rectangles
            for rect in (left_rect, right_rect):
                rx, ry, rw, rh = rect
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                              (0, 200, 255), 1)

            # --- Iris detection ---
            left_pos  = detect_iris_position(left_roi)
            right_pos = detect_iris_position(right_roi)

            # Draw iris centres in frame coordinates
            for pos, rect in ((left_pos, left_rect), (right_pos, right_rect)):
                if pos is None:
                    continue
                rx, ry, rw, rh = rect
                ix = int(rx + pos[0] * rw)
                iy = int(ry + pos[1] * rh)
                cv2.circle(frame, (ix, iy), 4, (0, 0, 255), -1)

            # --- EAR / blink detection ---
            left_eye_pts  = landmarks[LEFT_EYE_POINTS]
            right_eye_pts = landmarks[RIGHT_EYE_POINTS]
            ear = (eye_aspect_ratio(left_eye_pts) +
                   eye_aspect_ratio(right_eye_pts)) / 2.0

            if ear < EAR_THRESHOLD:
                blink_frame += 1
            else:
                if blink_frame >= BLINK_CONSEC:
                    blink_count += 1
                blink_frame = 0

            # --- Gaze classification ---
            gaze = classify_gaze(left_pos, right_pos)

            # --- Gaze arrow (centred above the face) ---
            arrow_origin = (int((x1 + x2) / 2), max(0, y1 - 30))
            draw_gaze_arrow(frame, gaze, arrow_origin)

            # --- Per-face info overlay ---
            cv2.putText(frame, f"EAR: {ear:.2f}",
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1)

            # Only process the first face for clarity
            break

        # --- Global HUD ---
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Gaze: {gaze}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 150, 150), 1)

        cv2.imshow("Eye Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Session ended. Total blinks detected: {blink_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="dlib eye tracker")
    p.add_argument(
        "--predictor",
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib 68-point shape predictor .dat file "
             "(default: shape_predictor_68_face_landmarks.dat)",
    )
    p.add_argument(
        "--source",
        default=0,
        help="Video source: 0 for webcam (default), or path to a video file",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Allow passing an integer camera index via --source
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass
    run(args.predictor, source)
