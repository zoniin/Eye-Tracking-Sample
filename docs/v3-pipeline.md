# Eye Tracker V3 — behaviour-classification pipeline

V3 implements the full pipeline from the SIGCSE 2026 poster
(*Capturing Real-Time Student Behavior using Webcam-Based Eye-Tracking Pipeline*, Figure 1),
including the stages that were still dotted lines in the prototype:

```
Webcam data capture ─► Face & eye detection ─► Eye event detection ─► Feature calculation ─► Behavior classification
   (OpenCV)            (MediaPipe Tasks,       (I-DT fixations,       (Table 1 metrics +     (rules baseline or model
                        head pose, gaze)        saccades, blinks)      fatigue/AOI/expression) trained on self-reports)
                                                                                                   │
                                                              Educators: real-time alerts ◄────────┤
                                                              Researchers: time-stamped logs ◄──────┘
```

It lives in the `eyetrack/` package. V1 (`eye_tracker.py`) and V2 (`eye_tracker_v2.py`) remain
available as separate entry points. V3 also adopts the time-based measurement approach from the `v2`
branch (`temporal_eyes.py`): PERCLOS and the per-minute rates are weighted by valid observed time, with
tracking gaps treated as missing evidence, so results don't depend on the camera's frame rate.

---

## What is new compared to V2

| Area | V2 | V3 |
|---|---|---|
| Face/eye detection | `mp.solutions.face_mesh` (removed from current MediaPipe) | MediaPipe **Tasks** FaceLandmarker + 52 blendshapes (brow lowering, jaw opening, eye blink) |
| Head pose | 6-point generic model (pitch came out near ±180°) | solvePnP against MediaPipe's **canonical 3-D face mesh** (25 rigid landmarks), verified to recover synthetic poses exactly |
| Gaze | 9 discrete directions from an iris bounding box | continuous gaze angles: geometric (iris offset + head pose) **or** CNNs from Model_MPIIGaze / Model_EVE on normalised eye patches |
| Calibration | re-centres the "center" thresholds | 5/9/13-point **screen calibration** → regression to screen coordinates, held-out (leave-one-point-out) error reported |
| Eye events | blink frame counter | **I-DT fixations** (Salvucci & Goldberg 2000, poster ref. [3]), saccades, blinks timed in ms, long eye closures |
| Features | % centre gaze, blink rate, face presence | fixation count/duration, blink count/rate (Table 1) + time-weighted PERCLOS, blink duration, yawns, AOI dwell, AOI switching & transition entropy, gaze dispersion, head motion, brow lowering |
| AOIs | – | configurable screen regions (code, instructions, compiler output, feedback) |
| Behaviour | weighted "focus score" | **engaged / distracted / fatigued / confused / away** with reasons, smoothing, alerts; or a **trained classifier** |
| Ground truth | – | **thought probes** (self-reports) during sessions → labelled dataset → leave-one-participant-out training |
| Data | whole session buffered, written at exit | streamed per-session folder (`frames`, `events`, `windows`, `probes`, `alerts`) |

---

## How the MPIIGaze and EVE work is incorporated

Both research repos contain training scripts but no trained weights, and their models
could only run on the datasets' pre-normalised eye images. V3 brings them in as follows.

**Model_MPIIGaze** (`model.py`, `model_lenet.py`, `model_resnet.py`, `model_shallow.py`)
* `LeNetGaze` and `ResNet8Gaze` are ported to `eyetrack/models/gaze_nets.py`.
* `training/train_mpiigaze.py` replaces the four scripts with one trainer that:
  * uses every participant and day, not just one;
  * evaluates **leave-one-person-out**, the standard MPIIGaze protocol (a 70/30 split of one person's frames leaks identity and overstates accuracy);
  * mirrors right-eye images so one network handles both eyes;
  * adds an angular loss and augmentation for webcam noise;
  * exports **ONNX** plus a JSON model card.
* `eyetrack/normalization.py` implements the MPIIGaze **data normalisation** (Zhang et al. 2018) from live MediaPipe landmarks. This is the missing link that lets a model trained on `Data/Normalized` run on a webcam: the eye centre is located in 3-D, a virtual camera is rotated to face it, a 60×36 patch is warped out, and predictions are rotated back to camera space.

**Model_EVE** (`model_gru_temp.py`, `model_eyenet_*`, `model_gazerefine_yawpitch_gru.py`, …)
* EyeNet-GRU (MobileNetV3-Small + GRU) had the lowest error in your comparison (~5.4° vs 6.3° for single-frame MobileNet and 7.1° for GazeRefine-GRU, from `results/`). It is ported with an **explicit hidden-state input/output**, so it runs frame by frame in real time with the same temporal smoothing it learned on sequences. A unit test checks that streaming output equals sequence output.
* `training/train_eve.py` has three commands:
  * `extract` crops each eye. The original script squashed the two-eye strip into one 64×64 image, which a live pipeline can't reproduce.
  * `preview` draws gaze arrows on the crops so you can check eye order and label order.
  * `train` adds a temporal-smoothness loss and exports streaming ONNX.
* **Label-order note:** EVE stores `*_g_tobii` as `[pitch, yaw]`. Model_EVE read it as `[yaw, pitch]`, which doesn't change what the network learns but does miscompute the reported angular error. Run `preview` before training to confirm.
* GazeRefine-GRU needs the screen-content video, which isn't available live, so it isn't included. The streaming GRU covers the temporal-refinement part.

The geometric estimator remains the default because it needs no weights. After calibration the screen
mapping absorbs most person/webcam-specific bias for either estimator.

---

## Quick start

```bash
pip install -r requirements.txt

# live webcam: 9-point calibration, programming AOIs, thought probes every 1.5-3 min
python -m eyetrack live --participant P01 --calibrate --aoi configs/aoi_programming.yaml --probes

# same, with a trained gaze CNN and/or a trained behaviour model
python -m eyetrack live --participant P01 --calibrate --gaze-model models/gaze_resnet8_mpiigaze.onnx \
    --behavior-model models/behavior.joblib --alerts-jsonl output/alerts.jsonl

# analyse a recorded video offline
python -m eyetrack process recording.mp4 --participant P01 --calibration output/calibrations/P01.json

# recompute windows/states for a session with a different window length
python -m eyetrack features output/P01_20261102_101500 --window 60 --hop 5
```

`python eye_tracker_v3.py ...` is equivalent to `python -m eyetrack ...`.
Keys in the live window: **Q** quit, **C** (re)calibrate, **P** show a probe now, **M** toggle mesh.

---

## Study workflow (turning the rules into a validated model)

1. **Collect.** Run `live --calibrate --probes` with each participant. At random intervals (default 90–180 s) a probe asks *"What were you doing just before this appeared?"*: 1 focused, 2 distracted/mind-wandering, 3 confused, 4 tired, 5 frustrated. Answers go to `probes.csv`. You can also add a `labels.csv` (`t,label`) per session for observer coding or task-performance events (for example, compile errors or test failures).
2. **Build the dataset.** Each label becomes the window of features ending just before it:
   ```bash
   python training/train_behavior.py build --sessions "output/P*" --window 30 --out data/behavior.csv
   ```
3. **Train and validate** with leave-one-participant-out cross-validation, against majority-class and rule-based baselines:
   ```bash
   python training/train_behavior.py train --dataset data/behavior.csv --participant-normalize --out models/behavior.joblib
   ```
   This writes `models/behavior.report.json` with balanced accuracy, macro-F1, the confusion matrix, per-participant results and feature importance.
   `--participant-normalize` z-scores features against each session's own baseline. The baseline is computed from all of the session's windows, labeled or not, so no label information leaks. Live, the model collects about 60 s of baseline before it starts predicting. The model also remembers its window length, and the live pipeline uses it automatically.
4. **Deploy** with `--behavior-model models/behavior.joblib`.

To try the workflow without a study, `python -m eyetrack simulate --participants 8` writes **synthetic**
sessions. They exist to exercise the code; numbers trained on them say nothing about real students.

Gaze CNNs (optional, needs the datasets and `pip install -r requirements-train.txt`):
```bash
python training/train_mpiigaze.py --data MPIIGaze/Data/Normalized --arch resnet8 --lopo --final
python training/train_eve.py extract --data /path/to/eve_dataset_2
python training/train_eve.py preview --data /path/to/eve_dataset_2
python training/train_eve.py train   --data /path/to/eve_dataset_2 --epochs 10
```

---

## Session outputs (`output/<participant>_<timestamp>/`)

| File | Content |
|---|---|
| `frames.csv` | per frame: face, EAR, eye closure, gaze pitch/yaw, gaze in degrees, screen x/y, AOI, head pose and distance, blendshapes, raw per-eye iris offsets (for re-calibrating offline) |
| `events.csv` | fixations (start, end, duration, position, AOI), saccades (amplitude, from/to AOI), blinks (kind) |
| `windows.csv` | every second: all window features, state, raw state, class probabilities, reasons |
| `probes.csv` | probe shown/answered time, latency, label |
| `alerts.csv` | educator alerts (also `--alerts-jsonl` for dashboards) |
| `meta.json` | settings, AOI names, calibration accuracy, models used |

## Behaviour states (rule-based baseline)

| State | Main evidence |
|---|---|
| engaged | gaze on screen, long fixations, high fixation time, stays within an AOI, normal blinks |
| distracted | off-screen gaze, head turning, scattered gaze, face repeatedly lost |
| fatigued | PERCLOS, slow blinks, long eye closures, yawning |
| confused | rapid AOI switching, short scanning fixations, unsystematic transitions, brow lowering |
| away | face visible less than 50% of the window |

The thresholds in `eyetrack/behavior.py` come from the literature and are **starting points, not
validated cut-offs**. Adjust them in `configs/default.yaml` (`behavior_rules`) or replace the rules
with a trained model. Every estimate lists the evidence that produced it, so an alert is never a black box.

## Limitations and next steps

* No gaze CNN weights are bundled: the datasets have to be downloaded and the trainers run. Expect several degrees of person-independent error; calibration is still recommended.
* Webcam fixation detection is coarse (2.5° dispersion). Fixation *durations and counts* are more reliable than exact positions, so AOIs should be large screen regions.
* Probes are shown in an OpenCV window that has to be focused to answer. For classroom use, a web or IDE-plugin probe would be less intrusive.
* Behaviour labels must come from real participants before reporting accuracy. Use leave-one-participant-out results only.
* Privacy (poster section 4): sessions store derived signals only, no video (unless `process` is pointed at your own recording). Keep `output/` out of version control (already in `.gitignore`).

## Code map

```
eyetrack/
  landmarks.py      MediaPipe Tasks wrapper, EAR, roll-invariant iris offsets, blendshapes
  geometry.py       gaze conventions, canonical face model, head pose (solvePnP)
  normalization.py  MPIIGaze data normalisation (eye patches for CNNs)
  gaze.py           geometric + ONNX appearance gaze estimators (streaming EyeNet-GRU)
  filters.py        1-Euro filter
  calibration.py    screen calibration (ridge regression, leave-one-point-out)
  events.py         I-DT fixations, saccades, blinks
  aoi.py            AOI layouts
  features.py       windowed features
  behavior.py       rules, learned classifier, smoothing, alerts
  probes.py         thought-probe scheduler
  session.py        streaming CSV/JSON logs
  dataset.py        sessions -> labelled feature table
  pipeline.py       end-to-end orchestration
  simulate.py       synthetic data for tests/demos
  app.py            CLI (live / process / features / simulate)
  models/gaze_nets.py  LeNet, ResNet-8 (MPIIGaze), EyeNet-GRU (EVE)
training/
  train_mpiigaze.py  train_eve.py  train_behavior.py  common.py
tests/                   pytest suite, no webcam needed (python -m pytest tests)
```
