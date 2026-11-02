"""
Train the EyeNet-GRU temporal gaze model on EVE and export a *streaming* ONNX
model for the live pipeline (consolidates Model_EVE/models/model_gru_temp.py,
extract_basler_eyes_to_npy.py and the k-fold utilities).

EyeNet-GRU was the most accurate model in the Model_EVE comparison
(~5.4 deg vs 6.3 deg for single-frame MobileNet and 7.1 deg for GazeRefine-GRU),
and the GRU smooths gaze over time, which reduces jitter in fixation detection.

Steps
-----
1. Extract per-eye 64x64 patches once (fast random access, ~60 MB per step)::

       python training/train_eve.py extract --data /path/to/eve_dataset_2

2. Check that eye halves / label order look right (writes a PNG grid with
   gaze arrows)::

       python training/train_eve.py preview --data /path/to/eve_dataset_2

3. Train + export::

       python training/train_eve.py train --data /path/to/eve_dataset_2 --epochs 10

Notes
-----
* EVE stores ``{eye}_g_tobii`` as ``[pitch, yaw]`` (EVE's
  ``pitchyaw_to_vector``).  Model_EVE's helper read it as ``[yaw, pitch]``;
  that does not change what the network learns but mis-computes the reported
  angular error.  Use ``--label-order`` if your preview arrows look swapped.
* ``basler_eyes.mp4`` holds both normalised eye patches side by side.  The
  original script resized the *whole* strip to 64x64 (both eyes squashed into
  one image).  Here each eye is cropped separately (``--layout split``), the
  right eye is mirrored, and one network handles both - matching what the live
  pipeline feeds it.  ``--layout full`` reproduces the original behaviour
  (not usable live).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import augment_eye_batch, export_eyenet_gru_onnx, get_device, set_seed
from eyetrack.models.gaze_nets import EyeNetGRU, angular_loss, angular_error_deg_torch

PATCH = 64
EYES_FILE = "eyes64.npy"


def step_dirs(root: str, split: str) -> List[str]:
    return sorted(d for d in glob.glob(os.path.join(root, f"{split}*", "step*")) if os.path.isdir(d))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def crop_eyes(frame_gray: np.ndarray, layout: str, left_half: int) -> np.ndarray:
    """Return (2, 64, 64) [left_eye, right_eye(mirrored)] or (1, 64, 64)."""
    if layout == "full":
        return cv2.resize(frame_gray, (PATCH, PATCH), interpolation=cv2.INTER_AREA)[None]
    h, w = frame_gray.shape
    halves = [frame_gray[:, : w // 2], frame_gray[:, w // 2:]]
    left = cv2.resize(halves[left_half], (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    right = cv2.resize(halves[1 - left_half], (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    return np.stack([left, right[:, ::-1]])


def extract(args) -> None:
    for split in ("train", "val", "test"):
        for sd in step_dirs(args.data, split):
            mp4 = os.path.join(sd, f"{args.camera}_eyes.mp4")
            out = os.path.join(sd, f"{args.camera}_{args.layout}_{EYES_FILE}")
            if not os.path.isfile(mp4) or (os.path.isfile(out) and not args.overwrite):
                continue
            cap = cv2.VideoCapture(mp4)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(crop_eyes(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        args.layout, args.left_half))
            cap.release()
            if frames:
                np.save(out, np.stack(frames).astype(np.uint8))
                print(f"[{split}] {sd}: {len(frames)} frames -> {out}", flush=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_labels(step_dir: str, camera: str, eye: str, order: str) -> Tuple[np.ndarray, np.ndarray]:
    import h5py
    with h5py.File(os.path.join(step_dir, f"{camera}.h5"), "r") as f:
        g = np.array(f[f"{eye}_g_tobii"]["data"], dtype=np.float32)
        valid = np.array(f[f"{eye}_g_tobii"]["validity"]).astype(bool)
        if f"{eye}_p" in f:
            valid &= np.array(f[f"{eye}_p"]["validity"]).astype(bool)
    if order == "yawpitch":
        g = g[:, ::-1].copy()
    return g, valid


class EVESequences(Dataset):
    """Sequences of T frames of one eye (right eyes mirrored, yaw negated)."""

    def __init__(self, root, split, camera="basler", layout="split", seq_len=30,
                 stride=30, order="pitchyaw", max_steps=None):
        self.items = []   # (eyes_path, eye_idx, start, gaze_seq)
        self.seq_len = seq_len
        eyes_list = ["left", "right"] if layout == "split" else ["left"]
        for sd in step_dirs(root, split)[:max_steps]:
            path = os.path.join(sd, f"{camera}_{layout}_{EYES_FILE}")
            if not os.path.isfile(path):
                continue
            n_frames = np.load(path, mmap_mode="r").shape[0]
            for k, eye in enumerate(eyes_list):
                try:
                    g, valid = load_labels(sd, camera, eye, order)
                except (KeyError, OSError):
                    continue
                n = min(n_frames, len(g))
                if eye == "right":
                    g = g.copy()
                    g[:, 1] *= -1
                for s in range(0, n - seq_len + 1, stride):
                    if valid[s:s + seq_len].all():
                        self.items.append((path, k, s, g[s:s + seq_len]))
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, k, s, g = self.items[i]
        arr = self._cache.get(path)
        if arr is None:
            arr = self._cache[path] = np.load(path, mmap_mode="r")
        imgs = np.asarray(arr[s:s + self.seq_len, k], dtype=np.float32) / 255.0
        return torch.from_numpy(imgs[:, None].copy()), torch.from_numpy(np.asarray(g, np.float32))


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview(args) -> None:
    ds = EVESequences(args.data, "train", args.camera, args.layout, 1, 97,
                      args.label_order, max_steps=3)
    if len(ds) == 0:
        raise SystemExit("No extracted samples found - run `extract` first.")
    tiles = []
    for i in np.linspace(0, len(ds) - 1, min(24, len(ds))).astype(int):
        img, g = ds[i]
        tile = cv2.cvtColor((img[0, 0].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(tile, (128, 128))
        pitch, yaw = g[0].tolist()
        dx, dy = -np.sin(yaw) * np.cos(pitch) * 50, -np.sin(pitch) * 50
        cv2.arrowedLine(tile, (64, 64), (int(64 + dx), int(64 + dy)), (0, 0, 255), 2)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)])
    cv2.imwrite(args.out_png, grid)
    print(f"Wrote {args.out_png}: arrows should point where the eye looks "
          "(iris offset direction). If they are mirrored/swapped, change "
          "--left-half or --label-order and re-extract.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    errs, n = 0.0, 0
    for img, g in loader:
        pred, _, _ = model(img.to(device))
        errs += angular_error_deg_torch(pred.cpu(), g).item() * len(img)
        n += len(img)
    return errs / max(n, 1)


def train(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)
    kw = dict(camera=args.camera, layout=args.layout, seq_len=args.seq_len,
              order=args.label_order, max_steps=args.max_steps)
    train_ds = EVESequences(args.data, "train", stride=args.seq_len // 2, **kw)
    val_ds = EVESequences(args.data, "val", stride=args.seq_len, **kw)
    print(f"train seqs={len(train_ds)} val seqs={len(val_ds)} device={device}", flush=True)
    if len(train_ds) == 0:
        raise SystemExit("No training sequences - run `extract` first.")
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)
    model = EyeNetGRU(hidden_size=args.hidden, pretrained_backbone=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, out = float("inf"), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "gaze_eyenet_gru_eve"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for img, g in tl:
            img, g = img.to(device), g.to(device)
            img = augment_eye_batch(img)
            pred, _, _ = model(img)
            loss = angular_loss(pred, g)
            # temporal smoothness prior: discourage frame-to-frame jitter
            loss = loss + args.smooth * (pred[:, 1:] - pred[:, :-1]).abs().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        sched.step()
        val = evaluate(model, vl, device) if len(val_ds) else float("nan")
        print(f"[EyeNet-GRU] epoch {epoch}/{args.epochs} loss={tot / len(tl):.4f} "
              f"val_err={val:.2f} deg ({time.time() - t0:.0f}s)", flush=True)
        if not np.isfinite(val) or val < best:
            best = val
            torch.save({"model_state_dict": model.state_dict(), "val_err_deg": val,
                        "hidden": args.hidden}, out / f"{name}.pt")
    ckpt = torch.load(out / f"{name}.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    meta = {
        "arch": "eyenet_gru", "dataset": "EVE", "input_size": [PATCH, PATCH],
        "hidden_size": args.hidden, "uses_head_pose": False, "flip_right_eye": True,
        # Approximate: EVE eye patches are normalised at ~60 cm; the focal length
        # here maps them to 64 px.  Verify against your EVE preprocessing and
        # adjust if live predictions look biased (calibration absorbs the rest).
        "normalization": {"focal": 650.0, "distance_cm": 60.0, "equalize": False},
        "output": "normalized-camera [pitch, yaw] radians",
        "val_error_deg": ckpt["val_err_deg"], "args": vars(args),
    }
    export_eyenet_gru_onnx(model, out / f"{name}.onnx", (PATCH, PATCH), meta)
    print(f"Saved {out / (name + '.onnx')}  best val error {best:.2f} deg")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["extract", "preview", "train"])
    p.add_argument("--data", required=True, help="EVE root (contains train01.., val01..)")
    p.add_argument("--camera", default="basler")
    p.add_argument("--layout", default="split", choices=["split", "full"])
    p.add_argument("--left-half", type=int, default=0, choices=[0, 1],
                   help="which half of the eyes video is the subject's left eye")
    p.add_argument("--label-order", default="pitchyaw", choices=["pitchyaw", "yawpitch"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out-png", default="eve_preview.png")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--smooth", type=float, default=0.05)
    p.add_argument("--pretrained", action="store_true", help="ImageNet MobileNetV3 weights")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="models")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    {"extract": extract, "preview": preview, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
