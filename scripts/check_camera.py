"""
Check that a camera opens and that MediaPipe finds a face in it.

    python scripts/check_camera.py            # camera 0
    python scripts/check_camera.py --source 1

Prints resolution / FPS and the share of frames with a detected face; press Q
to quit the preview.  (Replaces the old dlib-only test_camera.py, which was
hard-wired to camera 1.)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eyetrack.landmarks import FaceLandmarkTracker  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index or video file")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until Q)")
    args = p.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video source {args.source}")
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[INFO] Opened {args.source}: {w:.0f}x{h:.0f} @ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS")
    tracker = FaceLandmarkTracker()
    n = faces = 0
    t0 = time.monotonic()
    try:
        while not args.frames or n < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            obs = tracker.process(frame, time.monotonic() - t0)
            n += 1
            if obs is not None:
                faces += 1
                for i in (468, 473):
                    x, y = obs.landmarks[i, :2]
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            label = f"face in {faces}/{n} frames"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Camera check (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    fps = n / max(time.monotonic() - t0, 1e-6)
    print(f"[INFO] {n} frames, face detected in {faces} ({100 * faces / max(n, 1):.0f}%), {fps:.0f} FPS")


if __name__ == "__main__":
    main()
