"""Shared helpers for the training scripts (PyTorch required)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def augment_eye_batch(img: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
    """Photometric + small geometric jitter for (B, ..., H, W) images in [0,1].

    Webcams are much noisier than the MPIIGaze laptop cameras and the EVE
    Basler camera, so brightness/contrast/noise/blur augmentation helps the
    models transfer.
    """
    shape = img.shape
    x = img.reshape(-1, 1, shape[-2], shape[-1])
    b = x.shape[0]
    dev = x.device
    contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    bright = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.15, 0.15)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * contrast + mean + bright
    if max_shift > 0:
        dx, dy = np.random.randint(-max_shift, max_shift + 1, size=2)
        x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
    blur_mask = torch.rand(b, device=dev) < 0.3
    if blur_mask.any():
        k = torch.tensor([1.0, 2.0, 1.0], device=dev)
        k = (k[:, None] * k[None, :]) / 16.0
        blurred = torch.nn.functional.conv2d(x[blur_mask], k[None, None], padding=1)
        x = x.clone()
        x[blur_mask] = blurred
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1).reshape(shape)


def write_model_card(path: Path, meta: Dict) -> None:
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def export_frame_model_onnx(model: torch.nn.Module, path: Path, input_hw, meta: Dict,
                            use_head_pose: bool = True) -> None:
    """Export a single-frame gaze model (LeNet / ResNet-8) to ONNX."""
    model = model.eval().cpu()
    h, w = input_hw
    img = torch.zeros(2, 1, h, w)
    head = torch.zeros(2, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_head_pose:
        args, names = (img, head), ["image", "head_pose"]
        axes = {"image": {0: "batch"}, "head_pose": {0: "batch"}, "gaze": {0: "batch"}}
    else:
        args, names = (img,), ["image"]
        axes = {"image": {0: "batch"}, "gaze": {0: "batch"}}
    torch.onnx.export(model, args, str(path), input_names=names, output_names=["gaze"],
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    write_model_card(path, meta)


def export_eyenet_gru_onnx(model: torch.nn.Module, path: Path, input_hw, meta: Dict) -> None:
    """Export EyeNet-GRU with explicit hidden state for streaming inference."""
    model = model.eval().cpu()
    h, w = input_hw
    img = torch.zeros(2, 1, 1, h, w)
    h0 = torch.zeros(1, 2, model.hidden_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (img, h0), str(path), input_names=["image", "h0"],
        output_names=["gaze", "pupil", "h1"],
        dynamic_axes={"image": {0: "batch", 1: "time"}, "h0": {1: "batch"},
                      "gaze": {0: "batch", 1: "time"}, "pupil": {0: "batch", 1: "time"},
                      "h1": {1: "batch"}},
        opset_version=17, dynamo=False)
    write_model_card(path, meta)
