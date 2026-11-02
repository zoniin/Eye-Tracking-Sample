# Eye Tracker — How It Works

A plain-English walkthrough of every feature in this project, written for someone who hasn't worked with computer vision or Python before.

---

## What Does This Program Do?

It uses your webcam to watch your face in real time. Every frame of video is analyzed to figure out:
- Where your eyes are pointing (gaze direction)
- Whether you just blinked
- Whether you seem focused or distracted

All of this happens live on-screen with a visual overlay drawn on top of the video feed.

---

## Recent Changes (What's New)

### 1. Switched the main face-detection engine to MediaPipe
**Before:** The program used a library called **dlib**, which required downloading a large external model file and was relatively slow (~30–50 frames per second).

**After:** The default is now **MediaPipe Face Mesh** from Google, which runs at ~120–180 frames per second, works right out of the box, and is significantly more accurate — especially in poor lighting or when your head is tilted.

You can still use the old dlib engine if you need it by passing `--backend dlib` when you run the program.

### 2. Native iris tracking (no more guessing)
**Before:** To figure out where your iris (the colored part of your eye) was pointing, the program had to do image processing tricks — darkening the image, finding blobs, and guessing which blob was your iris.

**After:** MediaPipe directly tells the program the exact pixel location of your iris center. No guessing required. This is much more reliable, especially in varied lighting conditions.

### 3. Focus detection dashboard
A new panel was added to the right side of the video window that tracks whether you're paying attention over time. It analyzes your gaze, blink rate, and whether your face is even visible to produce a live "focus score."

### 4. Default camera fixed
The program was accidentally defaulting to camera index `1` (the second camera on your system) instead of `0` (your main webcam). This has been corrected.

---

## Feature Breakdown

### Face Detection

The program reads each video frame and looks for a human face. When it finds one, it draws a rectangle around it on screen.

MediaPipe does this by running a small AI model that was trained on millions of faces. It outputs 478 specific "landmark" points — tiny dots mapped to precise locations like the tip of your nose, the corner of your mouth, the center of each iris, etc.

Think of it like a dot-to-dot drawing of your face, redrawn 120+ times per second.

---

### Eye Landmark Mapping

Once the face is found, the program picks out specific landmark points that belong to each eye. It uses:
- **6 points** per eye for blink detection (the opening and closing points)
- **16 points** per eye for drawing the full eye outline on screen
- **1 point** per eye for the iris center (points 468 and 473 in MediaPipe's numbering)

These points are drawn as tiny yellow dots on the video feed so you can see exactly what the program is tracking.

---

### Gaze Direction Detection

To figure out where you're looking, the program measures where your iris sits *within* your eye socket.

- If your iris is shifted toward the left edge of your eye → you're looking **right**
- If it's shifted right → you're looking **left**
- If it's centered → you're looking straight ahead (**center**)
- Same logic applies vertically for **up** and **down**

It averages the position of both irises together to get the final gaze direction. The result is shown as a text label ("Gaze: center") and an arrow drawn above your face pointing in that direction.

The thresholds that define "too far left/right/up/down" are tunable values in the code (currently set to 38% and 62% of the eye width horizontally, 35% and 65% vertically).

---

### Blink Detection (Eye Aspect Ratio)

Blink detection uses a measurement called the **Eye Aspect Ratio (EAR)**.

Here's the idea: your eye opening has a shape. When your eye is wide open, the height of the opening is relatively large compared to the width. When you blink, the eye closes and the height shrinks to nearly zero.

The program measures the vertical and horizontal distances between specific eye landmark points and computes a ratio. If that ratio drops below **0.21** for at least **3 consecutive frames**, it counts as a blink.

This threshold prevents false positives — a single glitchy frame won't be counted as a blink.

The running blink count is displayed in the top-left corner of the screen.

---

### Focus Detection

This is the newest and most complex feature. It watches your behavior over a rolling **30-second window** and assigns a **focus score from 0 to 100**.

The score is made up of three sub-scores, weighted differently:

| Signal | Weight | What it measures |
|---|---|---|
| Gaze | 50% | How much of the time your eyes are pointed at the screen (center gaze) |
| Blink rate | 20% | Whether you're blinking at a normal rate (15–30 times/min) |
| Face presence | 30% | Whether your face is actually visible — are you looking away from the screen entirely? |

**How each signal is scored:**

- **Gaze score**: Simple ratio. If you looked at center for 25 of the last 30 seconds, your gaze score is ~83.
- **Blink score**: Normal blink rate (15–30/min) = 100. Too low (staring) or too high (stressed/distracted) = lower score.
- **Presence score**: Face visible >80% of the time = full score. Less than 50% = zero.

**Focus states:**

| Score | State |
|---|---|
| 75–100 | Focused |
| 50–74 | Semi-focused |
| Below 50 | Distracted |
| Face mostly gone | Away |

To avoid the label flipping back and forth rapidly, the program requires that a new state persist for at least **3 seconds** before switching. This is called hysteresis.

---

### Focus Dashboard (On-Screen Panel)

The panel on the right side of the video window shows:

- **Current state** — "FOCUSED", "DISTRACTED", etc., color-coded (green = good, orange = distracted, red = away)
- **Focus score** — the composite number out of 100
- **Session timer** — how long you've been running the program
- **Focused vs. distracted time** — a running total and percentage breakdown
- **Window stats** — center gaze %, blink rate, face visibility for the current 30-second window
- **Three bar graphs** — visual bars for Gaze, Blink, and Presence scores

---

### Dual Backend Support

You can run the program in two modes:

**MediaPipe (default)**
```
python eye_tracker.py
```
Uses Google's MediaPipe library. Faster, more accurate, no extra files needed.

**dlib (legacy)**
```
python eye_tracker.py --backend dlib
```
Uses the older dlib library. Requires a separate model file (`shape_predictor_68_face_landmarks.dat`). Slower, but kept for compatibility.

---

### Command-Line Options

| Flag | What it does |
|---|---|
| `--backend mediapipe` | Use MediaPipe (default) |
| `--backend dlib` | Use dlib instead |
| `--source 0` | Use webcam 0 (default — your main webcam) |
| `--source 1` | Use a different camera (e.g. an external webcam) |
| `--source video.mp4` | Run on a video file instead of live camera |
| `--no-focus-tracking` | Turn off the focus dashboard (lighter, faster) |
| `--predictor path/to/file.dat` | Point to your dlib model file (dlib mode only) |

---

### Session Summary (When You Quit)

When you press `q` to close the program, it prints a summary in the terminal showing:
- Total session length
- How much time you were focused vs. distracted (with percentages)
- Total blink count for the session
