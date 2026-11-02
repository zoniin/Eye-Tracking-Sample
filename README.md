# Eye Tracking Sample

Webcam eye tracking for studying student behaviour in computing education
(SIGCSE 2026 poster: *Capturing Real-Time Student Behavior using a Webcam-Based Eye-Tracking Pipeline*).

```
webcam → face & eye detection → fixations / blinks → features → behaviour state → educator alerts + research logs
```

## Which version?

| | Use it for | Entry point | Docs |
|---|---|---|---|
| **V3** (recommended) | Research studies: screen calibration, areas of interest, fixations, blinks/PERCLOS, behaviour states (engaged / distracted / fatigued / confused / away), self-report probes, model training, gaze CNNs from the MPIIGaze & EVE work | `python -m eyetrack` | [docs/v3-pipeline.md](docs/v3-pipeline.md) |
| V2 | Single-file tracker with a focus-score dashboard, CSV/JSON export, time-based blink measures | `python eye_tracker_v2.py` | [docs/v2-guide.md](docs/v2-guide.md) |
| V1 | Minimal demo; MediaPipe or legacy dlib backend | `python eye_tracker.py` | [docs/v1-guide.md](docs/v1-guide.md) |

## Quick start

```bash
pip install -r requirements.txt

# V3: calibrate, track areas of interest, ask self-report probes, log the session to output/
python -m eyetrack live --participant P01 --calibrate --aoi configs/aoi_programming.yaml --probes

# analyse a recorded video instead of the webcam
python -m eyetrack process recording.mp4 --participant P01
```

Check that your camera works first with `python scripts/check_camera.py` (use `--source 1` for a second camera).
Training gaze models or the behaviour classifier needs `pip install -r requirements-train.txt`; see
[docs/v3-pipeline.md](docs/v3-pipeline.md#study-workflow-turning-the-rules-into-a-validated-model).

## Repository layout

```
eyetrack/            V3 pipeline package (perception, events, features, behaviour, CLI)
training/            train gaze CNNs (MPIIGaze, EVE) and the behaviour classifier
configs/             default.yaml (pipeline settings), aoi_programming.yaml (screen regions)
eye_tracker.py       V1
eye_tracker_v2.py    V2  (+ temporal_eyes.py, its time-based blink/closure measures)
eye_tracker_v3.py    wrapper for `python -m eyetrack`
examples/            analysing exported V2 session data
scripts/             quick start menus, camera check, optional dlib setup
tests/               pytest suite (no webcam needed): python -m pytest tests
docs/                guides for each version, design notes and reports (index: docs/README.md)
face_landmarker.task MediaPipe face model used by all versions
```

Session data is written to `output/` (ignored by git). It contains derived eye/face measurements
only, no video. See the privacy notes in the V3 docs before running studies.

## Tests

```bash
python -m pytest tests
```

See [CHANGELOG.md](CHANGELOG.md) for the version history.
