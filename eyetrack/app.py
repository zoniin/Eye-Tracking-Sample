"""
Command-line entry points.

    python -m eyetrack live      [--calibrate] [--probes] [--aoi configs/aoi_programming.yaml] ...
    python -m eyetrack process   video.mp4 [--calibration calib.json]
    python -m eyetrack features  output/P01_20260101_120000 [--window 30 --hop 5]
    python -m eyetrack simulate  --participants 8 --minutes 15      (synthetic demo data)

Run ``python -m eyetrack <command> -h`` for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .aoi import OFF_SCREEN, AOILayout
from .behavior import load_classifier
from .calibration import CalibrationSession, ScreenCalibration
from .events import EventConfig, detect_events
from .features import feature_names, windows_from_session
from .pipeline import EyeTrackingPipeline, PipelineConfig
from .probes import PROMPT, ProbeScheduler
from .sample import read_frames_csv
from .session import SessionWriter

STATE_COLORS = {  # BGR
    "engaged": (80, 200, 80), "distracted": (0, 165, 255), "fatigued": (200, 120, 60),
    "confused": (200, 80, 200), "away": (60, 60, 230), "unknown": (160, 160, 160),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _apply(dc, values: Dict) -> None:
    for f in fields(dc):
        if f.name in values:
            cur = getattr(dc, f.name)
            if is_dataclass(cur) and isinstance(values[f.name], dict):
                _apply(cur, values[f.name])
            else:
                setattr(dc, f.name, type(cur)(values[f.name]) if cur is not None else values[f.name])


def load_config(path: Optional[str]) -> Tuple[PipelineConfig, Dict]:
    cfg = PipelineConfig()
    raw: Dict = {}
    if path:
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        _apply(cfg, raw.get("pipeline", {}))
    return cfg, raw


def screen_resolution(default=(1920, 1080)) -> Tuple[int, int]:
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _text(img, txt, org, scale=0.5, color=(255, 255, 255), thick=1):
    import cv2
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame, res, pipe: EyeTrackingPipeline, fps: float, show_mesh: bool) -> None:
    import cv2
    h, w = frame.shape[:2]
    obs = res.observation
    if obs is not None:
        lm = obs.landmarks
        if show_mesh:
            for x, y, _ in lm[:468:3]:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 200, 200), -1)
        for c in (468, 473):
            cv2.circle(frame, (int(lm[c, 0]), int(lm[c, 1])), 3, (0, 0, 255), -1)
        s = res.sample
        if np.isfinite(s.gaze_yaw):
            nose = (int(lm[1, 0]), int(lm[1, 1]))
            dx = -math.sin(math.radians(s.gaze_yaw)) * 80
            dy = -math.sin(math.radians(s.gaze_pitch)) * 80
            cv2.arrowedLine(frame, nose, (int(nose[0] + dx), int(nose[1] + dy)), (0, 255, 0), 2, tipLength=0.25)

    # side panel
    pw = 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - pw, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    x, y = w - pw + 12, 28
    est, f = pipe.last_estimate, pipe.last_features or {}
    state = est.state if est else "unknown"
    _text(frame, state.upper(), (x, y + 6), 0.9, STATE_COLORS.get(state, (255, 255, 255)), 2)
    y += 30
    if est:
        _text(frame, f"confidence {est.confidence * 100:.0f}%  ({est.source})", (x, y), 0.45, (200, 200, 200))
        y += 20
        for r in est.reasons[:3]:
            _text(frame, f"- {r}", (x, y), 0.42, (220, 220, 220))
            y += 18

    def fmt(k, unit="", scale=1.0, nd=0):
        v = f.get(k, float("nan"))
        return "-" if v is None or not np.isfinite(v) else f"{v * scale:.{nd}f}{unit}"

    y += 6
    rows = [
        ("Fixations", f"{fmt('fixation_count')} ({fmt('fixation_rate', '/min')})"),
        ("Fixation dur", fmt("fixation_dur_mean", " ms")),
        ("Blinks", f"{fmt('blink_count')} ({fmt('blink_rate', '/min')})"),
        ("Blink dur", fmt("blink_dur_mean", " ms")),
        ("PERCLOS", fmt("perclos", "%", 100)),
        ("On screen", fmt("on_screen_ratio", "%", 100)),
        ("AOI switches", fmt("aoi_switch_ratio", "%", 100)),
        ("Face visible", fmt("face_present_ratio", "%", 100)),
        ("FPS", f"{fps:.0f}"),
    ]
    _text(frame, f"Last {pipe.cfg.window_s:.0f}s window", (x, y), 0.45, (255, 255, 255))
    y += 20
    for k, v in rows:
        _text(frame, f"{k}:", (x, y), 0.42, (180, 180, 180))
        _text(frame, v, (x + 120, y), 0.42, (255, 255, 255))
        y += 18

    # screen minimap with AOIs + gaze
    mw, mh = pw - 24, int((pw - 24) * 9 / 16)
    mx, my = x, min(y + 10, h - mh - 30)
    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (90, 90, 90), 1)
    for a in pipe.aoi_layout.aois:
        p0 = (int(mx + a.x * mw), int(my + a.y * mh))
        p1 = (int(mx + (a.x + a.w) * mw), int(my + (a.y + a.h) * mh))
        active = res.sample.aoi == a.name
        cv2.rectangle(frame, p0, p1, (0, 200, 255) if active else (80, 80, 80), 1)
        _text(frame, a.name[:10], (p0[0] + 3, p0[1] + 12), 0.35, (200, 200, 200))
    s = res.sample
    if pipe.calibration is None:
        _text(frame, "not calibrated - press C", (mx + 8, my + mh // 2), 0.42, (0, 200, 255))
    elif np.isfinite(s.screen_x):
        gx = int(mx + np.clip(s.screen_x, -0.05, 1.05) * mw)
        gy = int(my + np.clip(s.screen_y, -0.05, 1.05) * mh)
        cv2.circle(frame, (gx, gy), 5, (0, 0, 255) if s.aoi == OFF_SCREEN else (0, 255, 0), -1)
    _text(frame, "Q quit  C calibrate  P probe  M mesh", (x, h - 10), 0.4, (170, 170, 170))


def draw_probe(canvas_size, options) -> np.ndarray:
    w, h = canvas_size
    img = np.full((h, w, 3), 30, np.uint8)
    _text(img, PROMPT, (30, 50), 0.7, (255, 255, 255), 2)
    for i, (k, label) in enumerate(options):
        _text(img, f"[{k}]  {label}", (60, 100 + 36 * i), 0.7, (180, 230, 255), 2)
    _text(img, "Press a number key (this window must be focused)", (30, h - 20), 0.5, (170, 170, 170))
    return img


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _build_pipeline(args, cfg: PipelineConfig, raw_cfg: Dict, writer=None, tracker_mode="video"):
    from .landmarks import FaceLandmarkTracker
    layout = AOILayout.from_yaml(args.aoi) if args.aoi else AOILayout()
    calib = ScreenCalibration.load(args.calibration) if getattr(args, "calibration", None) else None
    classifier = load_classifier(args.behavior_model, raw_cfg.get("behavior_rules"))
    tracker = FaceLandmarkTracker(mode=tracker_mode)
    pipe = EyeTrackingPipeline(cfg, tracker=tracker, calibration=calib, aoi_layout=layout,
                               classifier=classifier, writer=writer)
    return pipe


def _maybe_gaze_model(pipe: EyeTrackingPipeline, args, w: int, h: int) -> None:
    if args.gaze_model:
        from .gaze import AppearanceGazeEstimator
        from .geometry import default_camera_matrix
        pipe.camera_matrix = default_camera_matrix(w, h, pipe.cfg.camera_focal_scale)
        pipe.gaze_estimator = AppearanceGazeEstimator(args.gaze_model, pipe.camera_matrix)


def _session_meta(args, pipe: EyeTrackingPipeline) -> Dict:
    return {"source": str(args.source), "aoi_names": pipe.aoi_layout.names,
            "aoi_layout": args.aoi, "gaze_model": args.gaze_model,
            "behavior_model": args.behavior_model, "classifier": pipe.classifier.source,
            "window_s": pipe.cfg.window_s, "screen_cm": [pipe.cfg.screen_width_cm,
                                                         pipe.cfg.screen_height_cm]}


def cmd_live(args) -> None:
    import cv2
    cfg, raw = load_config(args.config)
    if args.window:
        cfg.window_s = args.window
    writer = None if args.no_log else SessionWriter(args.out, args.participant)
    pipe = _build_pipeline(args, cfg, raw, writer)
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source {args.source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from camera")
    _maybe_gaze_model(pipe, args, frame.shape[1], frame.shape[0])
    if writer:
        writer.write_meta(**_session_meta(args, pipe))
    probe_cfg = raw.get("probes", {})
    probes = (ProbeScheduler(**{k: v for k, v in probe_cfg.items() if k != "options"})
              if args.probes else None)
    screen = tuple(args.screen) if args.screen else screen_resolution()
    alert_log = open(args.alerts_jsonl, "a") if args.alerts_jsonl else None
    calib: Optional[CalibrationSession] = CalibrationSession(args.calib_points) if args.calibrate else None
    show_mesh = False
    t_start = time.monotonic()
    last = t_start
    fps = 0.0
    print("Running. Q = quit, C = calibrate, P = probe now.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            t = now - t_start
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now
            res = pipe.process_frame(frame, t)
            if res.alert:
                print(f"[ALERT {t:7.1f}s] {res.alert.message}")
                if alert_log:
                    alert_log.write(json.dumps({"t": res.alert.t, "state": res.alert.state,
                                                "duration_s": res.alert.duration_s,
                                                "participant": args.participant,
                                                "message": res.alert.message}) + "\n")
                    alert_log.flush()

            # --- calibration -----------------------------------------------------
            if calib is not None:
                valid = res.sample.face and not (res.sample.closure >= cfg.events.blink_close)
                calib.update(t, res.gaze.features if res.gaze else None, valid)
                if calib.done:
                    cv2.destroyWindow("Calibration")
                    try:
                        result = calib.fit(pipe.gaze_estimator.source)
                        pipe.calibration = result
                        pipe._screen_filter.reset()
                        deg = result.error_degrees(cfg.screen_width_cm, cfg.default_distance_cm)
                        print(f"Calibration done: held-out error {result.loo_error * 100:.1f}% "
                              f"of screen width (~{deg:.1f} deg), degree={result.degree}")
                        path = Path(args.out) / "calibrations" / f"{args.participant}.json"
                        result.save(str(path))
                        if writer:
                            writer.write_meta(calibration={"loo_error": result.loo_error,
                                                           "error_deg": deg, "file": str(path)})
                    except ValueError as exc:
                        print(f"Calibration failed: {exc}")
                    calib = None
                else:
                    canvas = np.zeros((screen[1], screen[0], 3), np.uint8)
                    tx, ty = calib.current_target
                    c = (int(tx * screen[0]), int(ty * screen[1]))
                    r = int(30 * (1.0 - calib.progress(t))) + 6
                    cv2.circle(canvas, c, r, (255, 255, 255), 2)
                    cv2.circle(canvas, c, 6, (0, 0, 255), -1)
                    _text(canvas, "Follow the dot with your eyes, keep your head still",
                          (40, 40), 0.8, (200, 200, 200), 2)
                    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("Calibration", canvas)

            # --- probes ------------------------------------------------------------
            if probes is not None and calib is None:
                r = probes.update(t)
                if r and writer:
                    writer.probe(r)
                if probes.active:
                    cv2.imshow("Probe", draw_probe((720, 320), probes.options))
                    try:
                        cv2.setWindowProperty("Probe", cv2.WND_PROP_TOPMOST, 1)
                    except Exception:
                        pass
                else:
                    try:
                        if cv2.getWindowProperty("Probe", cv2.WND_PROP_VISIBLE) >= 0:
                            cv2.destroyWindow("Probe")
                    except cv2.error:
                        pass

            if not args.headless:
                draw_overlay(frame, res, pipe, fps, show_mesh)
                cv2.imshow("Eye Tracker V3", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calib = CalibrationSession(args.calib_points)
            elif key == ord("m"):
                show_mesh = not show_mesh
            elif key == ord("p") and probes is not None:
                probes.trigger(t)
            elif probes is not None and probes.active and key != 255:
                r = probes.key(t, chr(key))
                if r and writer:
                    writer.probe(r)
    finally:
        pipe.flush()
        cap.release()
        cv2.destroyAllWindows()
        pipe.tracker.close()
        if alert_log:
            alert_log.close()
        if writer:
            writer.close(duration_s=time.monotonic() - t_start)
            print(f"Session saved to {writer.dir}")


def cmd_process(args) -> None:
    import cv2
    cfg, raw = load_config(args.config)
    if args.window:
        cfg.window_s = args.window
    writer = SessionWriter(args.out, args.participant)
    pipe = _build_pipeline(args, cfg, raw, writer)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    i = 0
    first = True
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if first:
                _maybe_gaze_model(pipe, args, frame.shape[1], frame.shape[0])
                writer.write_meta(**_session_meta(args, pipe), video_fps=fps)
                first = False
            res = pipe.process_frame(frame, i / fps)
            i += 1
            if args.show:
                draw_overlay(frame, res, pipe, fps, False)
                cv2.imshow("Eye Tracker V3 (offline)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if i % 300 == 0:
                st = pipe.last_estimate.state if pipe.last_estimate else "-"
                print(f"  {i}/{n or '?'} frames  t={i / fps:6.1f}s  state={st}", flush=True)
    finally:
        pipe.flush()
        cap.release()
        pipe.tracker.close()
        if args.show:
            cv2.destroyAllWindows()
        writer.close(frames=i)
    print(f"Processed {i} frames -> {writer.dir}")


def cmd_features(args) -> None:
    d = Path(args.session)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
    aoi_names = meta.get("aoi_names", [])
    samples = read_frames_csv(d / "frames.csv")
    stream = detect_events(samples, EventConfig())
    rows = windows_from_session(samples, stream, args.window, args.hop, aoi_names)
    clf = load_classifier(args.behavior_model)
    for r in rows:
        est = clf.predict(r)
        r["state"] = est.state
        r.update({f"p_{k}": round(v, 4) for k, v in est.probs.items()})
    out = d / args.output
    names = ["t", "state"] + [k for k in rows[0] if k.startswith("p_")] + feature_names(aoi_names) if rows else []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print(f"{len(stream.fixations)} fixations, {len(stream.blinks)} blinks, "
          f"{len(rows)} windows -> {out}; states: {counts}")


def cmd_simulate(args) -> None:
    """Synthetic sessions with probe labels (for trying the training workflow)."""
    from .probes import ProbeResponse
    from .simulate import SIM_STATES, default_layout, simulate_session
    rng = np.random.default_rng(args.seed)
    layout = default_layout()
    for p in range(args.participants):
        segs = []
        total = 0.0
        while total < args.minutes * 60:
            st = str(rng.choice(SIM_STATES))
            dur = float(rng.uniform(60, 150))
            segs.append((st, dur))
            total += dur
        samples, truth = simulate_session(segs, seed=int(rng.integers(1 << 30)), layout=layout)
        writer = SessionWriter(args.out, f"SIM{p:02d}",
                               meta={"synthetic": True, "aoi_names": layout.names,
                                     "note": "generated by eyetrack.simulate - not real data"})
        pipe = EyeTrackingPipeline(PipelineConfig(), aoi_layout=layout, writer=writer)
        for s in samples:
            pipe.process_sample(s)
        pipe.flush()
        for t_end, state in truth:
            label = state if rng.random() > args.label_noise else str(rng.choice(SIM_STATES))
            writer.probe(ProbeResponse(t_end, t_end + 2.0, label, ""))
        writer.close(segments=segs)
        print(f"{writer.dir}: {len(samples)} frames, {len(truth)} probes")


# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m eyetrack", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, source_default=None):
        sp.add_argument("--config", help="YAML config (see configs/default.yaml)")
        sp.add_argument("--aoi", help="AOI layout YAML (e.g. configs/aoi_programming.yaml)")
        sp.add_argument("--gaze-model", help="ONNX gaze CNN from training/ (default: geometric)")
        sp.add_argument("--behavior-model", help="joblib from training/train_behavior.py (default: rules)")
        sp.add_argument("--calibration", help="saved calibration JSON")
        sp.add_argument("--participant", default="anon")
        sp.add_argument("--window", type=float, help="feature window seconds (default 30)")
        sp.add_argument("--out", default="output", help="where session folders are written")

    lv = sub.add_parser("live", help="real-time webcam tracking")
    common(lv)
    lv.add_argument("--source", default="0", help="camera index or stream URL")
    lv.add_argument("--width", type=int, default=1280)
    lv.add_argument("--height", type=int, default=720)
    lv.add_argument("--calibrate", action="store_true", help="start with screen calibration")
    lv.add_argument("--calib-points", type=int, default=9, choices=[5, 9, 13])
    lv.add_argument("--screen", type=int, nargs=2, metavar=("W", "H"), help="screen resolution in px")
    lv.add_argument("--probes", action="store_true", help="ask self-report thought probes")
    lv.add_argument("--alerts-jsonl", help="append educator alerts as JSON lines to this file")
    lv.add_argument("--headless", action="store_true", help="no camera preview window")
    lv.add_argument("--no-log", action="store_true", help="do not write a session folder")
    lv.set_defaults(func=cmd_live)

    pr = sub.add_parser("process", help="analyse a recorded video file")
    common(pr)
    pr.add_argument("source", help="video file")
    pr.add_argument("--show", action="store_true")
    pr.set_defaults(func=cmd_process)

    fe = sub.add_parser("features", help="recompute windows/states from a session's frames.csv")
    fe.add_argument("session")
    fe.add_argument("--window", type=float, default=30.0)
    fe.add_argument("--hop", type=float, default=5.0)
    fe.add_argument("--behavior-model")
    fe.add_argument("--output", default="windows_offline.csv")
    fe.set_defaults(func=cmd_features)

    si = sub.add_parser("simulate", help="write SYNTHETIC labelled sessions for testing training")
    si.add_argument("--participants", type=int, default=6)
    si.add_argument("--minutes", type=float, default=15)
    si.add_argument("--label-noise", type=float, default=0.1)
    si.add_argument("--out", default="output/simulated")
    si.add_argument("--seed", type=int, default=0)
    si.set_defaults(func=cmd_simulate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
