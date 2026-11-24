"""Transformer building blocks for the multi-view video encoder and query decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool = True):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, position: nn.Module | None = None, geometry=None) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v) if position is None else position(q, k, v, geometry)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        M, h = mem.shape[1], self.heads
        q = self.q(x).view(B, N, h, C // h).transpose(1, 2)
        k, v = self.kv(mem).view(B, M, 2, h, C // h).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(1.0 - self.p)
        return x * keep / (1.0 - self.p)


class CrossBlock(nn.Module):
    """Independent query cross-attention block (queries attend to encoded memory, never to each other)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = q + self.attn(self.norm_q(q), self.norm_kv(mem))
        return q + self.mlp(self.norm2(q))


class PatchEmbed(nn.Module):
    """ViTPose-compatible 16x16 stride-16 patch embedding with padding=2 (256x256 -> 16x16 tokens)."""

    def __init__(self, patch: int = 16, dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)
