"""Parameter-free sinusoidal temporal positional encodings."""

from __future__ import annotations

import math

import torch


def sinusoidal_time_encoding(frame_idx: torch.Tensor, dim: int) -> torch.Tensor:
    """Interleaved sin/cos temporal code evaluated at clip-local frame ordinals (0 .. T-1)."""
    if dim < 1:
        raise ValueError("Time encoding dimension must be positive")
    with torch.autocast(device_type=frame_idx.device.type, enabled=False):
        freq = torch.exp(
            torch.arange(0, dim, 2, device=frame_idx.device, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        phase = frame_idx.float()[..., None] * freq
        code = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
    return code[..., :dim]


def mvtap_time_encoding(frame_idx: torch.Tensor, dim: int) -> torch.Tensor:
    """Concatenated 1D sin/cos temporal positional encoding evaluated at clip-local ordinals (0 .. T-1)."""
    if dim < 2 or dim % 2:
        raise ValueError("Sinusoidal temporal encoding requires an even dimension")
    omega = torch.arange(dim // 2, dtype=torch.float64, device=frame_idx.device) / (dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    out = frame_idx.double()[..., None] * omega
    return torch.cat([out.sin(), out.cos()], dim=-1).float()
