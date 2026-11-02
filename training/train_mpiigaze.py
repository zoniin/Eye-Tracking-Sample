"""
Train an appearance-based gaze CNN on MPIIGaze and export it for the live
pipeline (consolidates Model_MPIIGaze/model.py, model_lenet.py,
model_resnet.py and model_shallow.py).

Improvements over the original scripts
--------------------------------------
* Uses **all** participants/days (the originals loaded one participant).
* Person-independent evaluation: leave-one-person-out (the standard MPIIGaze
  protocol) instead of a 70/30 split of one person's frames, which leaks
  identity and over-states accuracy.
* Right-eye images are mirrored (and their yaw negated) so one network handles
  both eyes - doubling the effective data.
* 2-D normalised head pose input (GazeNet convention), computable at run time.
* Angular (or L1) loss, cosine LR schedule, photometric augmentation for
  webcam robustness.
* Exports ONNX + a JSON model card that ``eyetrack`` loads with
  ``--gaze-model``.

Expected data layout (the official MPIIGaze release)::

    MPIIGaze/Data/Normalized/p00/day01.mat ... p14/dayNN.mat

Examples::

    # one held-out person (fast sanity check)
    python training/train_mpiigaze.py --data MPIIGaze/Data/Normalized --arch resnet8 --test-person p00

    # full leave-one-person-out evaluation (15 folds), then a final model on everyone
    python training/train_mpiigaze.py --data MPIIGaze/Data/Normalized --arch resnet8 --lopo --final
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import (augment_eye_batch, export_frame_model_onnx, get_device,
                    set_seed)
from eyetrack.geometry import rotation_to_pitchyaw, vector_to_pitchyaw
from eyetrack.models.gaze_nets import (angular_loss, build_model,
                                       pitchyaw_to_vector_torch)

INPUT_HW = (36, 60)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _field(obj, name):
    return getattr(obj, name) if hasattr(obj, name) else obj[name]


def load_day(mat_path: Path) -> Dict[str, np.ndarray]:
    """Load one ``dayXX.mat`` -> images (N,36,60) uint8, gaze/head (N,2) rad.

    Right eyes are mirrored so every sample looks like a left eye.
    """
    from scipy.io import loadmat
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    imgs, gazes, heads = [], [], []
    for side in ("left", "right"):
        rec = _field(data, side)
        img = np.asarray(_field(rec, "image"))
        gaze = np.asarray(_field(rec, "gaze"), dtype=np.float64)
        pose = np.asarray(_field(rec, "pose"), dtype=np.float64)
        if img.size == 0:
            continue
        if img.ndim == 2:      # a single sample was squeezed
            img, gaze, pose = img[None], gaze[None], pose[None]
        if img.shape[1:] != INPUT_HW:
            img = np.stack([cv2.resize(i, INPUT_HW[::-1]) for i in img])
        g = vector_to_pitchyaw(gaze)
        hp = np.array([rotation_to_pitchyaw(cv2.Rodrigues(p.reshape(3, 1))[0]) for p in pose])
        if side == "right":
            img = img[:, :, ::-1]
            g[:, 1] *= -1
            hp[:, 1] *= -1
        imgs.append(img.astype(np.uint8))
        gazes.append(g)
        heads.append(hp)
    return {"image": np.concatenate(imgs), "gaze": np.concatenate(gazes),
            "head": np.concatenate(heads)}


def load_person(person_dir: Path, max_samples: int = 0, seed: int = 0) -> Dict[str, np.ndarray]:
    days = [load_day(p) for p in sorted(person_dir.glob("*.mat"))]
    out = {k: np.concatenate([d[k] for d in days]) for k in days[0]}
    if max_samples and len(out["image"]) > max_samples:
        idx = np.random.default_rng(seed).choice(len(out["image"]), max_samples, replace=False)
        out = {k: v[idx] for k, v in out.items()}
    return out


def to_dataset(parts: List[Dict[str, np.ndarray]]) -> TensorDataset:
    img = np.concatenate([p["image"] for p in parts]).astype(np.float32) / 255.0
    head = np.concatenate([p["head"] for p in parts]).astype(np.float32)
    gaze = np.concatenate([p["gaze"] for p in parts]).astype(np.float32)
    return TensorDataset(torch.from_numpy(img[:, None].copy()), torch.from_numpy(head),
                         torch.from_numpy(gaze))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    errs = []
    for img, head, gaze in loader:
        pred = model(img.to(device), head.to(device))
        a = pred.cpu().double()
        b = gaze.double()
        cos = torch.nn.functional.cosine_similarity(
            pitchyaw_to_vector_torch(a), pitchyaw_to_vector_torch(b), dim=1).clamp(-1, 1)
        errs.append(torch.rad2deg(torch.acos(cos)))
    return float(torch.cat(errs).mean())


def train(args, train_parts, val_parts, device, tag: str) -> Tuple[torch.nn.Module, float]:
    model = build_model(args.arch, use_head_pose=not args.no_head_pose).to(device)
    train_ds = to_dataset(train_parts)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = (DataLoader(to_dataset(val_parts), batch_size=512)
                  if val_parts else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    best = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for img, head, gaze in train_loader:
            img, head, gaze = img.to(device), head.to(device), gaze.to(device)
            if not args.no_augment:
                img = augment_eye_batch(img)
            pred = model(img, head)
            loss = (angular_loss(pred, gaze) if args.loss == "angular"
                    else torch.nn.functional.l1_loss(pred, gaze))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        msg = f"[{tag}] epoch {epoch}/{args.epochs} loss={total / len(train_loader):.4f}"
        if val_loader is not None:
            best = evaluate(model, val_loader, device)
            msg += f" val_err={best:.2f} deg"
        print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
    return model, best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="MPIIGaze/Data/Normalized directory")
    p.add_argument("--arch", default="resnet8", choices=["resnet8", "lenet"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", default="angular", choices=["angular", "l1"])
    p.add_argument("--no-head-pose", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--max-per-person", type=int, default=0,
                   help="subsample each participant (0 = all, 3000 = MPIIGaze evaluation subset size)")
    p.add_argument("--test-person", default="p00", help="held-out participant for a single fold")
    p.add_argument("--lopo", action="store_true", help="run all leave-one-person-out folds")
    p.add_argument("--final", action="store_true", help="also train on all people and export")
    p.add_argument("--out", default="models", help="output directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    root = Path(args.data)
    people = sorted(d.name for d in root.iterdir() if d.is_dir() and any(d.glob("*.mat")))
    if not people:
        raise SystemExit(f"No participant folders with .mat files under {root}")
    print(f"Loading {len(people)} participants on {device} ...", flush=True)
    data = {pid: load_person(root / pid, args.max_per_person, args.seed) for pid in people}
    print({k: len(v["image"]) for k, v in data.items()})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = people if args.lopo else [args.test_person]
    results = {}
    for test in folds:
        if test not in data:
            raise SystemExit(f"--test-person {test} not found; have {people}")
        train_parts = [data[q] for q in people if q != test]
        model, err = train(args, train_parts, [data[test]], device, tag=f"test={test}")
        results[test] = err
    mean_err = float(np.mean(list(results.values())))
    print(f"Person-independent error: {mean_err:.2f} deg  per person: "
          + ", ".join(f"{k}={v:.2f}" for k, v in results.items()))

    if args.final:
        model, _ = train(args, list(data.values()), [], device, tag="final")
    name = f"gaze_{args.arch}_mpiigaze"
    torch.save({"model_state_dict": model.state_dict(), "arch": args.arch},
               out_dir / f"{name}.pt")
    meta = {
        "arch": args.arch,
        "dataset": "MPIIGaze (Normalized)",
        "input_size": [INPUT_HW[1], INPUT_HW[0]],
        "uses_head_pose": not args.no_head_pose,
        "flip_right_eye": True,
        "normalization": {"focal": 960.0, "distance_cm": 60.0, "equalize": True},
        "output": "normalized-camera [pitch, yaw] radians (MPIIGaze convention)",
        "person_independent_error_deg": results,
        "mean_error_deg": mean_err,
        "trained_on_all_people": bool(args.final),
        "args": vars(args),
    }
    export_frame_model_onnx(model, out_dir / f"{name}.onnx", INPUT_HW, meta,
                            use_head_pose=not args.no_head_pose)
    (out_dir / f"{name}_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {out_dir / (name + '.onnx')} (+ .json model card, .pt checkpoint)")


if __name__ == "__main__":
    main()
