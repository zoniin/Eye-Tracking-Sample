"""
PyTorch gaze networks ported from the team's research repos.

* ``LeNetGaze``   - Model_MPIIGaze/model_lenet.py (3 conv + 2 FC, head pose concat)
* ``ResNet8Gaze`` - Model_MPIIGaze/model_resnet.py (pre-activation ResNet-8)
* ``EyeNetGRU``   - Model_EVE/models/model_gru_temp.py (MobileNetV3-Small + GRU),
  the best model in the EVE comparison (~5.4 deg val error).

Changes from the originals:
* Head pose is the 2-angle (pitch, yaw) of the *normalised* head rotation
  (GazeNet convention) instead of the raw 3-D rotation vector, so the same
  input can be computed at run time from a webcam frame.
* ``EyeNetGRU.forward`` accepts and returns the GRU hidden state, so the
  model can be exported to ONNX and run frame-by-frame in real time.
* The EyeNet-GRU first conv takes 1 channel directly instead of repeating
  grayscale to 3 channels (same function, 3x cheaper first layer).

All models take images scaled to [0, 1] and output (pitch, yaw) radians in the
MPIIGaze convention (see ``eyetrack.geometry``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetGaze(nn.Module):
    def __init__(self, input_size=(36, 60), use_head_pose: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        self.use_head_pose = use_head_pose
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        h, w = input_size
        flat = 128 * (h // 8) * (w // 8)
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128 + (2 if use_head_pose else 0), 2)

    def forward(self, image, head_pose=None):
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        if self.use_head_pose:
            x = torch.cat([x, head_pose], dim=1)
        return self.fc_out(x)


class PreActBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
                         if stride != 1 or in_ch != out_ch else None)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + sc


class ResNet8Gaze(nn.Module):
    def __init__(self, use_head_pose: bool = True, dropout: float = 0.5,
                 width: int = 32):
        super().__init__()
        self.use_head_pose = use_head_pose
        self.conv_in = nn.Conv2d(1, width, 3, 1, 1, bias=False)
        self.layer1 = PreActBlock(width, width, 1)
        self.layer2 = PreActBlock(width, width * 2, 2)
        self.layer3 = PreActBlock(width * 2, width * 4, 2)
        self.bn_last = nn.BatchNorm2d(width * 4)
        self.dropout = nn.Dropout(dropout)
        self.fc_img = nn.Linear(width * 4, 128)
        self.fc_out = nn.Linear(128 + (2 if use_head_pose else 0), 2)

    def forward(self, image, head_pose=None):
        x = self.conv_in(image)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = F.relu(self.bn_last(x))
        x = x.mean(dim=(2, 3))
        x = self.dropout(F.relu(self.fc_img(x)))
        if self.use_head_pose:
            x = torch.cat([x, head_pose], dim=1)
        return self.fc_out(x)


class EyeNetGRU(nn.Module):
    """MobileNetV3-Small per-frame encoder + GRU over time (Model_EVE)."""

    def __init__(self, hidden_size: int = 256, pretrained_backbone: bool = False):
        super().__init__()
        import torchvision.models as tvm
        weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
        backbone = tvm.mobilenet_v3_small(weights=weights)
        first = backbone.features[0][0]
        conv1 = nn.Conv2d(1, first.out_channels, first.kernel_size, first.stride,
                          first.padding, bias=False)
        with torch.no_grad():   # grayscale = sum of RGB filters
            conv1.weight.copy_(first.weight.sum(dim=1, keepdim=True))
        backbone.features[0][0] = conv1
        self.features = backbone.features
        feat_dim = backbone.classifier[0].in_features
        self.hidden_size = hidden_size
        self.gru = nn.GRU(feat_dim, hidden_size, batch_first=True)
        self.gaze_head = nn.Linear(hidden_size, 2)
        self.pupil_head = nn.Linear(hidden_size, 1)

    def forward(self, image, h0=None):
        """image: (B, T, 1, H, W); h0: (1, B, hidden) or None.

        Returns gaze (B, T, 2), pupil (B, T, 1), h1 (1, B, hidden).
        """
        b, t = image.shape[:2]
        x = image.reshape(b * t, *image.shape[2:])
        f = self.features(x).mean(dim=(2, 3)).reshape(b, t, -1)
        if h0 is None:
            h0 = torch.zeros(1, b, self.hidden_size, device=image.device, dtype=image.dtype)
        out, h1 = self.gru(f, h0)
        return self.gaze_head(out), self.pupil_head(out), h1


def build_model(arch: str, **kwargs) -> nn.Module:
    arch = arch.lower()
    if arch == "lenet":
        return LeNetGaze(**kwargs)
    if arch == "resnet8":
        return ResNet8Gaze(**kwargs)
    if arch == "eyenet_gru":
        return EyeNetGRU(**kwargs)
    raise ValueError(f"unknown architecture {arch!r}")


def pitchyaw_to_vector_torch(py: torch.Tensor) -> torch.Tensor:
    pitch, yaw = py[..., 0], py[..., 1]
    return torch.stack([-torch.cos(pitch) * torch.sin(yaw),
                        -torch.sin(pitch),
                        -torch.cos(pitch) * torch.cos(yaw)], dim=-1)


def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean angle (radians) between predicted and target pitch/yaw."""
    a = pitchyaw_to_vector_torch(pred.reshape(-1, 2))
    b = pitchyaw_to_vector_torch(target.reshape(-1, 2))
    cos = F.cosine_similarity(a, b, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


def angular_error_deg_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(angular_loss(pred, target))
