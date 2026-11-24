"""Projective Positional Encoding (PRoPE) for calibrated multi-view cross-attention.

Adapted from liruilong940607/prope/prope/torch.py (Cameras as Relative Positional Encoding).
Projection algebra is evaluated in float32 for numerical stability under mixed precision
and cast back for fused scaled dot-product attention.
"""
# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_se3(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = R
    out[..., :3, 3] = -(R @ T[..., :3, 3, None]).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def crop_projection_pair(
    K: torch.Tensor,
    T: torch.Tensor,
    image_size: int,
    origin: torch.Tensor | None = None,
    focal_units: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized lifted camera projection pairs (P, P^-1) for calibrated crop cameras.

    Args:
        K: (BT, V, 3, 3) virtual crop camera intrinsics.
        T: (BT, V, 4, 4) world-to-crop-camera SE(3) extrinsics (meters).
        image_size: square crop resolution S.
        origin: optional (BT, 3) triangulated 3D hand center in world coordinates; shifts the
            origin to the hand center and normalizes translations by the mean camera-to-hand distance.
        focal_units: normalize intrinsics by each crop camera's focal length rather than image size.
    """
    with torch.autocast(device_type=K.device.type, enabled=False):
        K, T = K.float(), T.float()
        if origin is not None:
            d = (T[..., :3, :3] @ origin.float()[:, None, :, None]).squeeze(-1) + T[..., :3, 3]
            s = d.norm(dim=-1).mean(-1).clamp_min(1e-6)
            T = T.clone()
            T[..., :3, 3] = d / s[:, None, None]
        T = T @ inverse_se3(T[:, :1])
        Kn = torch.zeros_like(T)
        if focal_units:
            Kn[..., 0, 0] = Kn[..., 1, 1] = 1
            Kn[..., 0, 2] = (K[..., 0, 2] - image_size / 2) / K[..., 0, 0]
            Kn[..., 1, 2] = (K[..., 1, 2] - image_size / 2) / K[..., 1, 1]
        else:
            Kn[..., 0, 0] = K[..., 0, 0] / image_size
            Kn[..., 1, 1] = K[..., 1, 1] / image_size
            Kn[..., 0, 2] = K[..., 0, 2] / image_size - 0.5
            Kn[..., 1, 2] = K[..., 1, 2] / image_size - 0.5
        Kn[..., 2, 2] = Kn[..., 3, 3] = 1
        Ki = torch.zeros_like(Kn)
        Ki[..., 0, 0] = 1 / Kn[..., 0, 0]
        Ki[..., 1, 1] = 1 / Kn[..., 1, 1]
        Ki[..., 0, 2] = -Kn[..., 0, 2] / Kn[..., 0, 0]
        Ki[..., 1, 2] = -Kn[..., 1, 2] / Kn[..., 1, 1]
        Ki[..., 2, 2] = Ki[..., 3, 3] = 1
        return Kn @ T, inverse_se3(T) @ Ki


class RelativeCameraAttention(nn.Module):
    """Parameter-free relative projective attention across calibrated views of the same frame."""

    def __init__(self, head_dim: int, grid: int, geometry: bool = True, rope2d: bool = False):
        super().__init__()
        if head_dim % 8:
            raise ValueError("PRoPE requires head_dim divisible by 8")
        if not rope2d and not geometry:
            raise ValueError("RelativeCameraAttention requires geometry=True or rope2d=True")
        self.geometry, self.grid, self.rope2d = geometry, grid, rope2d
        freqs = 100.0 ** (-torch.arange(head_dim // 8).float() / (head_dim // 8))
        xy = (torch.arange(grid).repeat(grid), torch.arange(grid).repeat_interleave(grid))
        for name, pos in zip(("x", "y"), xy):
            angles = pos[:, None] * freqs
            self.register_buffer(f"cos_{name}", angles.cos()[None, None], persistent=False)
            self.register_buffer(f"sin_{name}", angles.sin()[None, None], persistent=False)

    def _rope(self, x: torch.Tensor, axis: str, inverse: bool) -> torch.Tensor:
        repeats = x.shape[-2] // (self.grid ** 2)
        cos = getattr(self, f"cos_{axis}").repeat(1, 1, repeats, 1)
        sin = getattr(self, f"sin_{axis}").repeat(1, 1, repeats, 1)
        a, b = x.chunk(2, -1)
        if inverse:
            return torch.cat((cos * a - sin * b, sin * a + cos * b), -1)
        return torch.cat((cos * a + sin * b, -sin * a + cos * b), -1)

    def _transform(self, x: torch.Tensor, matrix: torch.Tensor | None = None, inverse_rope: bool = False) -> torch.Tensor:
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            d = x.shape[-1]
            if self.rope2d:
                proj, rx, ry = x.float().split((d // 2, d // 4, d // 4), -1)
            else:
                proj = x.float()
            if matrix is not None:
                B, H, L, D = proj.shape
                V = matrix.shape[1]
                proj = torch.einsum(
                    "bcij,bncpkj->bncpki",
                    matrix,
                    proj.reshape(B, H, V, L // V, D // 4, 4),
                ).reshape(B, H, L, D)
            out = (
                torch.cat((proj, self._rope(rx, "x", inverse_rope), self._rope(ry, "y", inverse_rope)), -1)
                if self.rope2d
                else proj
            )
        return out.to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, geometry=None) -> torch.Tensor:
        if q.shape[-2] % (self.grid ** 2):
            raise ValueError("PRoPE expects complete, row-major patch grids")
        P = Pi = None
        if self.geometry:
            if geometry is None:
                raise ValueError("Missing PRoPE crop projection matrices")
            P, Pi = geometry
        q = self._transform(q, P.transpose(-1, -2) if P is not None else None)
        k, v = self._transform(k, Pi), self._transform(v, Pi)
        out = F.scaled_dot_product_attention(q, k, v)
        return self._transform(out, P, inverse_rope=True)
