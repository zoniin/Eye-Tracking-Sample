# Research tracking improvements

Run the improved application with `python eye_tracker_v2.py`.
Use `python eye_tracker_v2.py --export-data output/session.csv` to collect measurements.
The existing eye_tracker.py entry point is unchanged.

Changes:
- Supports both legacy MediaPipe FaceMesh and current FaceLandmarker Tasks. The bundled face_landmarker.task is resolved relative to the script.
- Applies the configured smoothing window independently to each iris. Missing faces and closed eyes clear smoothing; closed eyes produce undetected gaze.
- Counts blinks by elapsed duration (default 0.06 to 0.8 seconds), rather than frame count. Longer closures remain separate measurements.
- Exports EAR, current closure duration, completed closure duration, closed-time ratio, observed duration, and raw left/right iris coordinates to CSV and JSON. Existing columns are retained.
- Eye measurements use a monotonic clock for webcams and frame index / source FPS for recorded video. Gaps over 0.25 seconds are excluded. FocusAnalyzer retains its legacy timing and scoring.
- YAML is needed only for YAML configuration operations; the default launch does not import it.

The closed-time ratio is time below the configured EAR threshold divided by valid observed time over the last 30 seconds. It is NOT a validated PERCLOS measurement or fatigue diagnosis. No face means missing evidence. These measurements support future independently labeled attention/fatigue training. Existing focus labels remain heuristic.

## MPIIGaze and EVE integration status

The supplied repositories do not contain usable training datasets or model checkpoints. MPIIGaze/Data contains placeholder .DS_Store files; its .gitignore excludes .mat. EVE excludes .h5, .mp4, .npy and .pth. Result plots and illustrative eye images are not a labeled training dataset.

`canonical_gaze_angles` provides a tested angle-order boundary for future models: MPIIGaze's scripts emit pitch/yaw radians, EVE's declare yaw/pitch radians. It does not convert reference frames or make a webcam crop equivalent to a normalized MPIIGaze image. No learned gaze model is active, no neural-network dependencies are added, and no improved gaze accuracy is claimed.

To enable learned gaze estimation, obtain the datasets from their original providers or Taha's trained checkpoints together with preprocessing metadata. Reproduce normalization, head-pose representation, eye layout and timestamp alignment before adding inference. Evaluate on held-out people before using model predictions in behavior scoring.

Run the dependency-free measurement tests from this directory:
`python -m unittest discover -s tests -p test_temporal_eyes.py -v`
