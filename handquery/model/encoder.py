"""Multi-view spatiotemporal ViT encoder with divided temporal attention and PRoPE cross-view blocks."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from handquery.model.layers import Attention, DropPath, Mlp, PatchEmbed
from handquery.model.prope import RelativeCameraAttention, crop_projection_pair
from handquery.model.time_encoding import mvtap_time_encoding, sinusoidal_time_encoding


class STBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        temporal: bool = True,
        position: nn.Module | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.position = position
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)
        self.temporal = temporal
        if temporal:
            self.norm_t = nn.LayerNorm(dim, eps=1e-6)
            self.attn_t = Attention(dim, heads)
            nn.init.zeros_(self.attn_t.proj.weight)
            nn.init.zeros_(self.attn_t.proj.bias)

    def forward(self, x: torch.Tensor, shape: tuple[int, int, int, int], mode: str, geometry=None) -> torch.Tensor:
        B, V, T, N = shape
        D = x.shape[-1]
        if self.temporal and T > 1:
            xt = x.view(B, V, T, N, D).transpose(2, 3).reshape(B * V * N, T, D)
            xt = xt + self.drop_path(self.attn_t(self.norm_t(xt)))
            x = xt.view(B, V, N, T, D).transpose(2, 3).reshape(B * V * T, N, D)
        if mode == "view" and (V > 1 or self.position is not None):
            xv = x.view(B, V, T, N, D).transpose(1, 2).reshape(B * T, V * N, D)
            xv = xv + self.drop_path(self.attn(self.norm1(xv), self.position, geometry))
            x = xv.view(B, T, V, N, D).transpose(1, 2).reshape(B * V * T, N, D)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VideoEncoder(nn.Module):
    """Shared ViT-B/16 video encoder with divided temporal attention and calibrated cross-view blocks."""

    def __init__(
        self,
        img_size: int = 256,
        patch: int = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        view_blocks: Sequence[int] = (4, 6, 8, 10),
        temporal: bool = True,
        max_frames: int = 64,
        camera_encoding: str = "prope",
        time_encoding: str = "mvtap",
        prope_scale: str = "hand_unit",
        prope_intrinsics: str = "focal",
        prope_rope2d: bool = False,
    ):
        super().__init__()
        if camera_encoding not in ("prope", "rope", "none"):
            raise ValueError(f"Unsupported camera_encoding: {camera_encoding}")
        if time_encoding not in ("mvtap", "sincos", "learned"):
            raise ValueError(f"Unsupported encoder time_encoding: {time_encoding}")
        self.img_size = img_size
        self.patch = patch
        self.max_frames = max_frames
        self.camera_encoding = camera_encoding
        self.time_encoding = time_encoding
        self.prope_scale = prope_scale
        self.prope_intrinsics = prope_intrinsics

        self.patch_embed = PatchEmbed(patch, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch) ** 2 + 1, dim))
        if time_encoding == "learned":
            self.time_embed = nn.Parameter(torch.zeros(max_frames, dim))

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.modes = ["view" if i in set(view_blocks) else "frame" for i in range(depth)]
        grid = img_size // patch
        self.blocks = nn.ModuleList(
            [
                STBlock(
                    dim=dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    temporal=temporal,
                    position=(
                        RelativeCameraAttention(
                            dim // heads,
                            grid,
                            geometry=(camera_encoding == "prope"),
                            rope2d=prope_rope2d,
                        )
                        if self.modes[i] == "view" and camera_encoding in ("prope", "rope")
                        else None
                    ),
                )
                for i in range(depth)
            ]
        )
        self.last_norm = nn.LayerNorm(dim, eps=1e-6)

    def time_code(self, idx: torch.Tensor, dim: int) -> torch.Tensor:
        if self.time_encoding == "learned":
            return self.time_embed[idx]
        if self.time_encoding == "mvtap":
            return mvtap_time_encoding(idx, dim)
        return sinusoidal_time_encoding(idx, dim)

    def forward(
        self,
        images: torch.Tensor,
        K_crop: torch.Tensor | None = None,
        T_crop: torch.Tensor | None = None,
        origin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode normalized RGB crops `(B, V, T, 3, S, S)` -> patch tokens `(B * V * T, P, D)`."""
        B, V, T = images.shape[:3]
        if T > self.max_frames:
            raise ValueError(f"Window length {T} exceeds max_frames={self.max_frames}")

        geometry = None
        if self.camera_encoding == "prope":
            if K_crop is None or T_crop is None:
                raise ValueError("PRoPE requires K_crop and T_crop")
            Kt = K_crop.transpose(1, 2).reshape(B * T, V, 3, 3)
            Tt = T_crop.transpose(1, 2).reshape(B * T, V, 4, 4)
            o = None
            if self.prope_scale == "hand_unit":
                if origin is None:
                    raise ValueError("prope_scale='hand_unit' requires triangulated clip origin")
                o = origin.repeat_interleave(T, 0)
            geometry = crop_projection_pair(
                Kt, Tt, self.img_size, o, focal_units=(self.prope_intrinsics == "focal")
            )

        x = self.patch_embed(images.flatten(0, 2))
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        time = self.time_code(torch.arange(T, device=x.device), x.shape[-1])
        x = x + time.repeat(B * V, 1)[:, None]

        shape = (B, V, T, x.shape[1])
        for blk, mode in zip(self.blocks, self.modes):
            x = blk(x, shape, mode, geometry)
        return self.last_norm(x)
