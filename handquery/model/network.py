"""HandQuery network: multi-view spatiotemporal ViT encoder + independent dense vertex query decoder."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from handquery.model.encoder import VideoEncoder
from handquery.model.layers import CrossBlock
from handquery.model.time_encoding import mvtap_time_encoding

MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255.0
STD = torch.tensor([0.229, 0.224, 0.225]) * 255.0
LEVELS = ("frame", "clip", "image")


def resolve_side_offset(side, per_hand: int, device: torch.device) -> torch.Tensor:
    """Map hand side ('left' / 1 -> 0, 'right' / 0 -> `per_hand`) to `(B,)` slot offsets."""
    if isinstance(side, str):
        val = 0 if side.lower() == "left" else per_hand
        return torch.tensor([val], dtype=torch.long, device=device)
    if isinstance(side, (list, tuple)) and side and isinstance(side[0], str):
        vals = [0 if s.lower() == "left" else per_hand for s in side]
        return torch.tensor(vals, dtype=torch.long, device=device)
    t = torch.as_tensor(side, device=device)
    if t.ndim == 0:
        t = t[None]
    return torch.where(t == 1, 0, per_hand).long()


class HandQueryNet(nn.Module):
    """Multi-view video encoder + independent dense MANO vertex query decoder.

    Each query corresponds to `(hand_vertex_slot, view, frame)`. Queries never attend to one
    another; across decoder layers they cycle through cross-view (`frame`), cross-time (`clip`),
    and per-crop (`image`) memory scopes, predicting normalized 2D coordinates `uv` in `[-1, 1]`,
    aleatoric pixel uncertainty `sigma` (in crop pixels), and visibility probability `vis`.
    """

    def __init__(
        self,
        landmark_ids: Sequence[int] | np.ndarray | None = None,
        encoder: dict | None = None,
        decoder: dict | None = None,
        image_size: int = 256,
        max_frames: int = 64,
    ):
        super().__init__()
        enc_cfg = dict(encoder or {})
        enc_cfg.pop("type", None)
        enc_cfg.pop("grad_ckpt", None)
        if "patch_size" in enc_cfg and "patch" not in enc_cfg:
            enc_cfg["patch"] = enc_cfg.pop("patch_size")
        enc_cfg.setdefault("camera_encoding", "prope")
        self.encoder = VideoEncoder(img_size=image_size, **enc_cfg)
        self.grid = image_size // enc_cfg.get("patch", 16)

        dec_cfg = dict(decoder or {})
        D = int(dec_cfg.get("dim", 768))
        enc_dim = int(enc_cfg.get("dim", 768))
        depth = int(dec_cfg.get("depth", 6))
        heads = int(dec_cfg.get("heads", 12))
        mlp_ratio = float(dec_cfg.get("mlp_ratio", 4.0))

        self.time_code = dec_cfg.get("time_encoding", dec_cfg.get("time_code", "mvtap"))
        if self.time_code not in ("mvtap", "learned"):
            raise ValueError(f"Unsupported decoder time_encoding: {self.time_code}")

        levels = dec_cfg.get("levels", ["frame", "clip", "image", "frame", "clip", "image"])
        self.levels = list(levels)
        if len(self.levels) != depth or any(lv not in LEVELS for lv in self.levels):
            raise ValueError(f"Decoder levels must have length {depth} with entries in {LEVELS}, got {self.levels}")

        self.memory_proj = nn.Linear(enc_dim, D) if enc_dim != D else nn.Identity()

        ids = np.arange(778, dtype=np.int64) if landmark_ids is None else np.asarray(landmark_ids, dtype=np.int64)
        if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or ids.min() < 0 or ids.max() >= 778:
            raise ValueError("landmark_ids must be distinct MANO vertex indices in [0, 777]")
        self.per_hand = len(ids)
        self.n_slots = 2 * self.per_hand

        self.slot_embed = nn.Embedding(self.n_slots, D)
        if self.time_code == "learned":
            self.time_embed = nn.Embedding(max_frames, D)
        self.blocks = nn.ModuleList([CrossBlock(D, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 4)  # u, v, log_sigma (crop px), visibility logit

        pos = np.full(778, -1, dtype=np.int64)
        pos[ids] = np.arange(self.per_hand)
        self.register_buffer("landmark_ids", torch.from_numpy(ids), persistent=False)
        self.register_buffer("vertex_to_pos", torch.from_numpy(pos), persistent=False)
        self.image_size = image_size
        self.max_frames = max_frames
        self.register_buffer("mean", MEAN.view(1, 1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("std", STD.view(1, 1, 1, 3, 1, 1), persistent=False)

    def encode(
        self,
        images: torch.Tensor,
        K_crop: torch.Tensor | None = None,
        T_crop: torch.Tensor | None = None,
        origin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Encode uint8 RGB crops `(B, V, T, S, S, 3)` -> `(B, V, T, P, D)`."""
        x = (images.permute(0, 1, 2, 5, 3, 4).float() - self.mean) / self.std
        B, V, T = x.shape[:3]
        if T > self.max_frames:
            raise ValueError(f"Window length {T} exceeds max_frames={self.max_frames}")
        tok = self.encoder(x, K_crop=K_crop, T_crop=T_crop, origin=origin)
        tok = tok.view(B, V, T, -1, tok.shape[-1])
        tok = self.memory_proj(tok)
        return tok, (B, V, T)

    def query_time(self, frames: torch.Tensor, dim: int) -> torch.Tensor:
        if self.time_code == "learned":
            return self.time_embed(frames)
        return mvtap_time_encoding(frames, dim)

    def decode_levels(self, mem: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode `(B, V, T, K)` query slots against encoded multi-view clip memory `(B, V, T, P, D)`."""
        B, V, T, P, D = mem.shape
        tcode = self.query_time(torch.arange(T, device=slots.device), D)[None, None, :, None]
        q = self.slot_embed(slots) + tcode
        K = q.shape[3]
        layout: dict[str, torch.Tensor] = {}
        for blk, level in zip(self.blocks, self.levels):
            if level not in layout:
                layout[level] = (
                    mem.transpose(1, 2).reshape(B * T, V * P, D)
                    if level == "frame"
                    else mem.reshape(B * V, T * P, D)
                    if level == "clip"
                    else mem.reshape(B * V * T, P, D)
                )
            if level == "frame":
                q = blk(q.transpose(1, 2).reshape(B * T, V * K, D), layout[level]).view(B, T, V, K, D).transpose(1, 2)
            elif level == "clip":
                q = blk(q.reshape(B * V, T * K, D), layout[level]).view(B, V, T, K, D)
            else:
                q = blk(q.reshape(B * V * T, K, D), layout[level]).view(B, V, T, K, D)
        out = self.head(self.norm(q)).float()
        return out[..., :2], out[..., 2], out[..., 3]

    @torch.no_grad()
    def predict(
        self,
        batch: dict,
        vert_ids: Sequence[int] | np.ndarray | None = None,
        chunk: int = 8192,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict dense 2D vertices, pixel uncertainties, and visibilities in every view and frame.

        Args:
            batch: dict containing:
                - `images`: `(B, V, T, S, S, 3)` uint8 RGB crops
                - `K_crop`: `(B, V, T, 3, 3)` virtual crop intrinsics
                - `T_crop`: `(B, V, T, 4, 4)` world-to-crop extrinsics (meters)
                - `origin`: `(B, 3)` triangulated hand center in world coordinates (meters)
                - `side`: `'right'` / `'left'` or integer tensor (`0` = right, `1` = left)
            vert_ids: optional subset of MANO vertex indices in `[0, 777]` (default: all 778).
            chunk: maximum `(T * K_chunk)` queries evaluated per decoder pass.

        Returns:
            `(uv, sigma, vis)` where:
              - `uv`: `(B, V, T, N, 2)` normalized crop coordinates in `[-1, 1]`
              - `sigma`: `(B, V, T, N)` standard deviation in crop pixels
              - `vis`: `(B, V, T, N)` visibility probability in `[0, 1]`
        """
        images = batch["images"]
        K_crop = batch.get("K_crop")
        T_crop = batch.get("T_crop")
        origin = batch.get("origin")
        mem, (B, V, T) = self.encode(images, K_crop=K_crop, T_crop=T_crop, origin=origin)

        dev = images.device
        ids = (
            self.landmark_ids
            if vert_ids is None
            else torch.as_tensor(np.asarray(vert_ids), device=dev, dtype=torch.long)
        )
        pos = self.vertex_to_pos[ids]
        if (pos < 0).any():
            raise ValueError(f"Requested vertices are not covered by this model's {self.per_hand} hand slots")

        offset = resolve_side_offset(batch["side"], self.per_hand, dev)
        if offset.shape[0] == 1 and B > 1:
            offset = offset.expand(B)
        slot = offset[:, None] + pos[None]
        N = slot.shape[1]

        slots = slot[:, None, None, :].expand(B, V, T, N)
        n = max(1, chunk // T)
        outs = [self.decode_levels(mem, slots[..., i : i + n]) for i in range(0, N, n)]
        uv, ls, vl = (torch.cat([o[k] for o in outs], dim=3) for k in range(3))
        return uv, torch.exp(ls), torch.sigmoid(vl)
