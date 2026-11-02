"""Tests for the V3 ``eyetrack`` pipeline (no webcam required)."""

import csv
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from eyetrack import geometry as geo
from eyetrack.aoi import AOI, OFF_SCREEN, OTHER, AngularScreenModel, AOILayout
from eyetrack.behavior import (AlertManager, BehaviorEstimate, Evidence,
                               RuleBasedClassifier, StateSmoother)
from eyetrack.calibration import (CalibrationSession, ScreenCalibration,
                                  fit_calibration, grid_points)
from eyetrack.dataset import build_examples
from eyetrack.events import EventConfig, detect_events
from eyetrack.features import compute_window_features, transition_entropy
from eyetrack.filters import OneEuroFilter
from eyetrack.landmarks import eye_aspect_ratio, iris_offset
from eyetrack.normalization import EyeNormalizer
from eyetrack.pipeline import EyeTrackingPipeline, PipelineConfig
from eyetrack.probes import ProbeScheduler
from eyetrack.sample import FrameSample, read_frames_csv, write_frames_csv
from eyetrack.session import SessionWriter
from eyetrack.simulate import default_layout, simulate_session

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def rot(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
    rx, _ = cv2.Rodrigues(np.array([math.radians(pitch_deg), 0, 0.0]))
    ry, _ = cv2.Rodrigues(np.array([0, math.radians(yaw_deg), 0.0]))
    rz, _ = cv2.Rodrigues(np.array([0, 0, math.radians(roll_deg)]))
    return ry @ rx @ rz


def project_face(R, t, K):
    pts = (R @ geo.canonical_face_model().T).T + t
    p2, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, None)
    lm = np.zeros((478, 3))
    lm[:468, :2] = p2.reshape(-1, 2)
    return lm


def gaze_stream(points, fps=30.0, noise=0.0, seed=0, t0=0.0):
    """points: list of (x_deg, y_deg, seconds) fixation targets."""
    rng = np.random.default_rng(seed)
    out, t = [], t0
    for x, y, dur in points:
        for _ in range(int(round(dur * fps))):
            out.append(FrameSample(t=t, face=True, closure=0.0,
                                   gaze_x_deg=x + rng.normal(0, noise),
                                   gaze_y_deg=y + rng.normal(0, noise)))
            t += 1.0 / fps
    return out


# ---------------------------------------------------------------------------
# geometry / normalisation
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_pitchyaw_roundtrip(self):
        py = np.array([[0.1, -0.3], [-0.4, 0.2], [0.0, 0.0]])
        back = geo.vector_to_pitchyaw(geo.pitchyaw_to_vector(py))
        assert np.allclose(py, back)

    def test_looking_at_camera_is_minus_z(self):
        assert np.allclose(geo.pitchyaw_to_vector([[0, 0]])[0], [0, 0, -1])

    def test_angular_error(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.0, math.radians(10)]])
        assert geo.angular_error_deg(a, b)[0] == pytest.approx(10.0, abs=1e-6)

    @pytest.mark.parametrize("pitch,yaw,roll", [(0, 0, 0), (0, 20, 0), (-15, 0, 0), (10, -25, 5)])
    def test_head_pose_recovery(self, pitch, yaw, roll):
        K = geo.default_camera_matrix(1280, 720)
        R = rot(pitch, yaw, roll)
        lm = project_face(R, np.array([2.0, -1.0, 55.0]), K)
        hp = geo.HeadPoseEstimator(K).estimate(lm)
        assert hp is not None
        assert np.allclose(hp.rotation, R, atol=1e-3)
        assert hp.distance_cm == pytest.approx(np.linalg.norm([2, -1, 55]), rel=1e-3)
        assert hp.reprojection_error < 0.01

    def test_head_pose_sign_conventions(self):
        K = geo.default_camera_matrix(1280, 720)
        # Rotating about +y by a positive angle turns the face towards the
        # camera's left (the subject's right) -> positive yaw.
        hp = geo.HeadPoseEstimator(K).estimate(project_face(rot(yaw_deg=20), np.array([0, 0, 60.0]), K))
        assert hp.yaw == pytest.approx(20, abs=0.1)
        # Head tilted down (chin towards chest) -> negative pitch
        hp = geo.HeadPoseEstimator(K).estimate(project_face(rot(pitch_deg=15), np.array([0, 0, 60.0]), K))
        assert hp.pitch == pytest.approx(-15, abs=0.1)

    def test_eye_centres_between_corners(self):
        K = geo.default_camera_matrix(640, 480)
        lm = project_face(np.eye(3), np.array([0, 0, 60.0]), K)
        hp = geo.HeadPoseEstimator(K).estimate(lm)
        right, left = geo.eye_centers_3d(hp)
        p, _ = cv2.projectPoints(np.array([right, left]), np.zeros(3), np.zeros(3), K, None)
        p = p.reshape(-1, 2)
        assert np.allclose(p[0], lm[[33, 133], :2].mean(axis=0), atol=0.5)
        assert np.allclose(p[1], lm[[263, 362], :2].mean(axis=0), atol=0.5)
        assert right[0] < left[0]      # subject's right eye is on the image left


class TestNormalization:
    def setup_method(self):
        self.K = geo.default_camera_matrix(640, 480)
        R = rot(5, 15, 3)
        self.pose = geo.HeadPose(rotation=R, translation=np.array([3.0, -2.0, 55.0]))
        self.center = geo.eye_centers_3d(self.pose)[0]
        self.norm = EyeNormalizer(self.K)

    def test_patch_shape_and_axis(self):
        gray = np.random.default_rng(0).integers(0, 255, (480, 640), dtype=np.uint8)
        ne = self.norm.normalize(gray, self.pose, self.center)
        assert ne.patch.shape == (36, 60)
        # the normalised camera looks straight at the eye centre
        assert np.allclose(ne.rotation[2], self.center / np.linalg.norm(self.center))
        assert np.allclose(ne.rotation @ ne.rotation.T, np.eye(3), atol=1e-9)

    @pytest.mark.parametrize("flip", [False, True])
    def test_gaze_denormalize_roundtrip(self, flip):
        gray = np.zeros((480, 640), np.uint8)
        ne = self.norm.normalize(gray, self.pose, self.center, flip=flip)
        g_cam = geo.pitchyaw_to_vector([[0.2, -0.35]])[0]
        py_norm = geo.vector_to_pitchyaw((ne.rotation @ g_cam)[None])[0]
        if flip:
            py_norm[1] = -py_norm[1]       # what a model sees for a mirrored eye
        assert np.allclose(ne.denormalize_gaze(py_norm), g_cam, atol=1e-9)

    def test_eye_centre_maps_to_patch_centre(self):
        gray = np.zeros((480, 640), np.uint8)
        p, _ = cv2.projectPoints(self.center[None], np.zeros(3), np.zeros(3), self.K, None)
        u, v = p.reshape(2)
        gray[int(round(v)) - 1:int(round(v)) + 2, int(round(u)) - 1:int(round(u)) + 2] = 255
        ne = EyeNormalizer(self.K, equalize=False).normalize(gray, self.pose, self.center)
        ys, xs = np.nonzero(ne.patch > 100)
        assert abs(xs.mean() - 30) < 2 and abs(ys.mean() - 18) < 2


# ---------------------------------------------------------------------------
# landmarks / filters
# ---------------------------------------------------------------------------

class TestEyeMeasures:
    def test_ear_open_vs_closed(self):
        open_eye = np.array([[0, 0], [3, -2], [7, -2], [10, 0], [7, 2], [3, 2]], float)
        closed = open_eye * np.array([1.0, 0.1])
        assert eye_aspect_ratio(open_eye) > 0.35 > eye_aspect_ratio(closed)

    def test_iris_offset_invariant_to_roll_and_scale(self):
        outer, inner = np.array([0.0, 0.0]), np.array([30.0, 0.0])
        up, lo = np.array([15.0, -5.0]), np.array([15.0, 5.0])
        iris = np.array([21.0, 2.0])
        h0, v0 = iris_offset(iris, outer, inner, up, lo)
        th = math.radians(25)
        Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        tf = lambda p: 2.5 * (Rm @ p) + np.array([100, 50])  # noqa: E731
        h1, v1 = iris_offset(tf(iris), tf(outer), tf(inner), tf(up), tf(lo))
        assert h0 == pytest.approx(0.7) and h1 == pytest.approx(h0)
        assert v1 == pytest.approx(v0)


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = 5 + rng.normal(0, 1.0, 300)
    out = np.array([f(x, i / 30) for i, x in enumerate(raw)])
    assert out[50:].std() < 0.5 * raw[50:].std()


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_fixations_and_saccades(self):
        samples = gaze_stream([(0, 0, 0.4), (10, 0, 0.6), (10, 8, 0.3)], noise=0.3)
        es = detect_events(samples, EventConfig())
        assert len(es.fixations) == 3
        durs = [f.duration for f in es.fixations]
        assert durs[0] == pytest.approx(0.4, abs=0.12)
        assert durs[1] == pytest.approx(0.6, abs=0.12)
        assert len(es.saccades) == 2
        assert es.saccades[0].amplitude_deg == pytest.approx(10, abs=1.0)

    def test_short_glances_are_not_fixations(self):
        samples = gaze_stream([(0, 0, 0.05), (10, 0, 0.05), (20, 0, 0.05)])
        es = detect_events(samples, EventConfig(median_window=1))
        assert es.fixations == []

    def test_tracking_gap_splits_fixation(self):
        a = gaze_stream([(0, 0, 0.5)])
        gap_t = a[-1].t
        b = gaze_stream([(0, 0, 0.5)], t0=gap_t + 0.5)
        es = detect_events(a + b, EventConfig())
        assert len(es.fixations) == 2

    def test_blinks_and_long_closures(self):
        samples, t = [], 0.0
        pattern = [(0, 1.0), (1, 0.15), (0, 1.0), (1, 0.8), (0, 1.0), (1, 0.01), (0, 0.5)]
        for closed, dur in pattern:
            for _ in range(max(1, int(round(dur * 30)))):
                samples.append(FrameSample(t=t, face=True, closure=float(closed),
                                           gaze_x_deg=0.0, gaze_y_deg=0.0))
                t += 1 / 30
        es = detect_events(samples, EventConfig())
        kinds = [b.kind for b in es.blinks]
        assert kinds == ["blink", "long_closure"]           # 10 ms flicker rejected
        assert es.blinks[0].duration == pytest.approx(0.15, abs=0.04)
        assert es.blinks[1].duration == pytest.approx(0.8, abs=0.04)


# ---------------------------------------------------------------------------
# features / AOIs
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_table1_metrics(self):
        samples = gaze_stream([(0, 0, 0.5), (8, 0, 0.5)] * 6)       # 12 fixations in 6 s
        # add 3 blinks
        for k in (30, 90, 150):
            for j in range(4):
                samples[k + j].closure = 1.0
        es = detect_events(samples)
        f = compute_window_features(samples, es.fixations, es.saccades, es.blinks,
                                    0.0, samples[-1].t)
        assert f["fixation_count"] == pytest.approx(12, abs=1)
        assert 300 <= f["fixation_dur_mean"] <= 520
        assert f["blink_count"] == 3
        assert f["blink_rate"] == pytest.approx(3 / (samples[-1].t / 60), rel=0.05)
        assert f["face_present_ratio"] == 1.0

    def test_transition_entropy_extremes(self):
        assert transition_entropy(["a", "b"] * 20) == pytest.approx(0.0)
        rng = np.random.default_rng(0)
        seq = list(rng.choice(["a", "b", "c", "d"], 400))
        assert transition_entropy(seq) > 0.9

    def test_aoi_layout(self):
        lay = AOILayout([AOI("code", 0.3, 0.0, 0.7, 0.7), AOI("instr", 0.0, 0.0, 0.3, 1.0)])
        assert lay.classify(0.5, 0.3) == "code"
        assert lay.classify(0.1, 0.9) == "instr"
        assert lay.classify(0.8, 0.9) == OTHER
        assert lay.classify(1.3, 0.5) == OFF_SCREEN
        assert lay.classify(float("nan"), 0.5) == ""
        lay2 = AOILayout.from_yaml(str(ROOT / "configs" / "aoi_programming.yaml"))
        assert "code" in lay2.names

    def test_angular_screen_model(self):
        m = AngularScreenModel()
        assert m.classify(-15, 5) == OTHER
        assert m.classify(-15, 45) == OFF_SCREEN


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def _data(self, noise=0.002, seed=0):
        rng = np.random.default_rng(seed)
        feats, targets, groups = [], [], []
        for gi, (x, y) in enumerate(grid_points(9)):
            for _ in range(25):
                # features are an unknown (invertible) function of the target
                z = np.array([x, y])
                f = np.concatenate([z, z ** 2, [x * y], rng.normal(0, 0.01, 3)])
                feats.append(f + rng.normal(0, noise, 8))
                targets.append((x, y))
                groups.append(gi)
        return np.array(feats), np.array(targets), np.array(groups)

    def test_fit_recovers_mapping(self, tmp_path):
        X, Y, g = self._data()
        cal = fit_calibration(X, Y, g)
        assert cal.loo_error < 0.05
        pred = cal.predict(X[g == 4])
        assert np.allclose(pred.mean(axis=0), [0.5, 0.5], atol=0.02)
        p = tmp_path / "cal.json"
        cal.save(str(p))
        cal2 = ScreenCalibration.load(str(p))
        assert np.allclose(cal2.predict(X[:5]), cal.predict(X[:5]))

    def test_needs_four_points(self):
        X, Y, g = self._data()
        with pytest.raises(ValueError):
            fit_calibration(X[g < 3], Y[g < 3], g[g < 3])

    def test_session_flow(self):
        cs = CalibrationSession(9, settle_s=0.2, collect_s=0.3, seed=1)
        assert cs.current_target == (0.5, 0.5)
        t = 0.0
        while not cs.done:
            x, y = cs.current_target
            cs.update(t, np.array([x, y, x * y, x * x]), valid=True)
            t += 1 / 30
        assert len(set(cs.groups)) == 9
        assert cs.fit().loo_error < 0.05


# ---------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------

def _run(states, seed=3):
    lay = default_layout()
    samples, _ = simulate_session(states, seed=seed, layout=lay)
    pipe = EyeTrackingPipeline(PipelineConfig(), aoi_layout=lay)
    seen = []
    for s in samples:
        r = pipe.process_sample(s)
        if r.estimate is not None:
            seen.append((s.t, r.estimate.state))
    return pipe, seen


class TestBehavior:
    @pytest.mark.parametrize("state", ["engaged", "distracted", "confused", "away"])
    def test_rules_on_simulated_states(self, state):
        _, seen = _run([(state, 90)])
        late = [s for t, s in seen if t > 40]
        assert max(set(late), key=late.count) == state

    def test_reasons_are_reported(self):
        pipe, _ = _run([("confused", 60)])
        assert pipe.last_estimate.reasons
        assert pipe.last_estimate.source == "rules"

    def test_missing_features_do_not_crash(self):
        est = RuleBasedClassifier().predict({"face_present_ratio": 1.0})
        assert est.state in ("engaged", "distracted", "fatigued", "confused")

    def test_support_only_evidence(self):
        ev = Evidence("yawn_count", 0.5, 1.5, 1.0, "yawning", support_only=True)
        assert ev.score(0.0) == 0.0 and ev.score(2.0) == 1.0

    def test_rule_overrides(self):
        clf = RuleBasedClassifier(overrides={"fatigued": {"perclos": {"lo": 0.3, "hi": 0.5}}})
        ev = [e for e in clf.rules["fatigued"] if e.feature == "perclos"][0]
        assert (ev.lo, ev.hi) == (0.3, 0.5)

    def test_smoother_hysteresis(self):
        sm = StateSmoother(tau_s=1.0, min_dwell_s=3.0)
        a = BehaviorEstimate("engaged", {"engaged": 0.9, "distracted": 0.1})
        b = BehaviorEstimate("distracted", {"engaged": 0.1, "distracted": 0.9})
        assert sm.update(0, a).state == "engaged"
        assert sm.update(1, b).state == "engaged"       # not yet: needs dwell time
        states = [sm.update(t, b).state for t in range(2, 8)]
        assert states[-1] == "distracted"
        assert sm.update(8, BehaviorEstimate("away", {"away": 1.0})).state == "away"

    def test_alert_manager(self):
        am = AlertManager(after_s=10, cooldown_s=100)
        fired = [am.update(t, "distracted") for t in range(0, 30)]
        alerts = [a for a in fired if a]
        assert len(alerts) == 1 and alerts[0].t == 10
        assert am.update(31, "engaged") is None


def test_probe_scheduler():
    ps = ProbeScheduler(min_interval_s=10, max_interval_s=10, timeout_s=5, seed=0)
    ps.update(0.0)
    assert not ps.active
    ps.update(10.5)
    assert ps.active
    r = ps.key(12.0, "3")
    assert r.label == "confused" and r.latency_s == pytest.approx(1.5)
    ps.update(23.0)
    r2 = ps.update(29.0)
    assert r2.label == "no_response"


# ---------------------------------------------------------------------------
# logging / dataset / training
# ---------------------------------------------------------------------------

def test_frames_csv_roundtrip(tmp_path):
    samples, _ = simulate_session([("engaged", 3)], seed=0)
    p = tmp_path / "frames.csv"
    write_frames_csv(p, samples)
    back = read_frames_csv(p)
    assert len(back) == len(samples)
    assert back[10].aoi == samples[10].aoi
    assert back[10].gaze_x_deg == pytest.approx(samples[10].gaze_x_deg, nan_ok=True)


def _write_sim_sessions(root, n=3, seed=0):
    from eyetrack.probes import ProbeResponse
    lay = default_layout()
    for p in range(n):
        segs = [("engaged", 70), ("confused", 70), ("distracted", 70), ("engaged", 70)]
        samples, truth = simulate_session(segs, seed=seed + p, layout=lay)
        w = SessionWriter(str(root), f"P{p}", meta={"aoi_names": lay.names})
        pipe = EyeTrackingPipeline(PipelineConfig(), aoi_layout=lay, writer=w)
        for s in samples:
            pipe.process_sample(s)
        pipe.flush()
        for t_end, st in truth:
            w.probe(ProbeResponse(t_end, t_end + 1, st))
        w.close()


def test_session_and_dataset(tmp_path):
    _write_sim_sessions(tmp_path, n=1)
    d = next(tmp_path.iterdir())
    for name in ("frames.csv", "events.csv", "windows.csv", "probes.csv", "meta.json"):
        assert (d / name).is_file(), name
    with open(d / "windows.csv") as f:
        row = next(csv.DictReader(f))
    assert "state" in row and "fixation_count" in row and "dwell_code" in row
    rows = build_examples(str(d), window_s=30)
    assert [r["label"] for r in rows] == ["engaged", "confused", "distracted", "engaged"]


def test_train_behavior_cli(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    _write_sim_sessions(tmp_path / "s", n=3)
    ds = tmp_path / "ds.csv"
    script = ROOT / "training" / "train_behavior.py"
    subprocess.run([sys.executable, str(script), "build", "--sessions", str(tmp_path / "s" / "*"),
                    "--out", str(ds)], check=True, capture_output=True)
    out = tmp_path / "behavior.joblib"
    r = subprocess.run([sys.executable, str(script), "train", "--dataset", str(ds), "--out", str(out),
                        "--min-count", "2"], check=True, capture_output=True, text=True)
    assert "leave-one-participant-out" in r.stdout
    from eyetrack.behavior import LearnedBehaviorClassifier
    clf = LearnedBehaviorClassifier(str(out))
    est = clf.predict({"face_present_ratio": 1.0, "fixation_dur_mean": 300.0})
    assert est.state in clf.labels and est.source == "learned"


# ---------------------------------------------------------------------------
# gaze networks (PyTorch) + ONNX inference
# ---------------------------------------------------------------------------

class TestGazeNets:
    def test_frame_models(self):
        torch = pytest.importorskip("torch")
        from eyetrack.models.gaze_nets import build_model
        for arch in ("lenet", "resnet8"):
            m = build_model(arch).eval()
            out = m(torch.zeros(4, 1, 36, 60), torch.zeros(4, 2))
            assert out.shape == (4, 2)

    def test_eyenet_gru_streaming_matches_sequence(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        from eyetrack.models.gaze_nets import EyeNetGRU
        torch.manual_seed(0)
        m = EyeNetGRU(hidden_size=32).eval()
        x = torch.rand(2, 5, 1, 64, 64)
        with torch.no_grad():
            seq, _, _ = m(x)
            h = None
            steps = []
            for t in range(5):
                g, _, h = m(x[:, t:t + 1], h)
                steps.append(g)
        assert torch.allclose(seq, torch.cat(steps, dim=1), atol=1e-5)

    def test_onnx_export_and_appearance_estimator(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        sys.path.insert(0, str(ROOT / "training"))
        from common import export_frame_model_onnx
        from eyetrack.gaze import AppearanceGazeEstimator
        from eyetrack.landmarks import EyeMeasurements
        from eyetrack.models.gaze_nets import build_model
        path = tmp_path / "g.onnx"
        export_frame_model_onnx(build_model("resnet8"), path, (36, 60),
                                {"arch": "resnet8", "input_size": [60, 36]})
        K = geo.default_camera_matrix(640, 480)
        est = AppearanceGazeEstimator(str(path), K)
        pose = geo.HeadPoseEstimator(K).estimate(project_face(rot(0, 10), np.array([0, 0, 55.0]), K))
        eyes = EyeMeasurements(0.3, 0.3, (0.5, 0.0), (0.5, 0.0), 0.0)
        res = est.estimate(np.zeros((480, 640), np.uint8), pose, eyes)
        assert res.source == "appearance" and np.isfinite(res.pitchyaw).all()
        assert res.features.shape == (8,)


# ---------------------------------------------------------------------------
# MediaPipe on a real face image (skipped if unavailable)
# ---------------------------------------------------------------------------

def test_mediapipe_end_to_end():
    skdata = pytest.importorskip("skimage.data")
    from eyetrack.landmarks import FaceLandmarkTracker, measure_eyes
    try:
        tracker = FaceLandmarkTracker(mode="image")
    except (OSError, RuntimeError, FileNotFoundError) as exc:  # e.g. missing libEGL
        pytest.skip(f"MediaPipe unavailable: {exc}")
    img = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
    obs = tracker.process(img)
    tracker.close()
    assert obs is not None and obs.landmarks.shape == (478, 3)
    assert "eyeBlinkLeft" in obs.blendshapes
    eyes = measure_eyes(obs)
    assert 0.2 < eyes.ear < 0.45 and eyes.blink_score < 0.3
    K = geo.default_camera_matrix(*obs.image_size)
    hp = geo.HeadPoseEstimator(K).estimate(obs.landmarks)
    assert hp is not None and hp.reprojection_error < 5.0
