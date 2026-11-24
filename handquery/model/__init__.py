"""Model construction and checkpoint loading utilities."""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
import torch

from handquery.model.network import HandQueryNet

DEFAULT_CONFIG: dict[str, Any] = {
    "image_size": 256,
    "window_size": 48,
    "crop_margin": 1.4,
    "num_vertices": 778,
    "encoder": {
        "patch_size": 16,
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "mlp_ratio": 4.0,
        "drop_path": 0.1,
        "view_blocks": [4, 6, 8, 10],
        "temporal": True,
        "max_frames": 64,
        "camera_encoding": "prope",
        "prope_scale": "hand_unit",
        "prope_intrinsics": "focal",
        "prope_rope2d": False,
        "time_encoding": "mvtap",
    },
    "decoder": {
        "dim": 768,
        "depth": 6,
        "heads": 12,
        "mlp_ratio": 4.0,
        "levels": ["frame", "clip", "image", "frame", "clip", "image"],
        "time_encoding": "mvtap",
    },
}


def _normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize checkpoint metadata into a uniform inference configuration dictionary."""
    if not raw:
        return copy.deepcopy(DEFAULT_CONFIG)
    if "image_size" in raw and "encoder" in raw and "decoder" in raw:
        return copy.deepcopy(raw)
    data_cfg = raw.get("data", {})
    model_cfg = raw.get("model", raw)
    enc = dict(model_cfg.get("encoder", DEFAULT_CONFIG["encoder"]))
    enc.pop("type", None)
    enc.pop("grad_ckpt", None)
    dec_raw = dict(model_cfg.get("decoder", DEFAULT_CONFIG["decoder"]))
    dec = {
        "dim": dec_raw.get("dim", 768),
        "depth": dec_raw.get("depth", 6),
        "heads": dec_raw.get("heads", 12),
        "mlp_ratio": dec_raw.get("mlp_ratio", 4.0),
        "levels": list(dec_raw.get("levels", ["frame", "clip", "image", "frame", "clip", "image"])),
        "time_encoding": dec_raw.get("time_encoding", dec_raw.get("time_code", "mvtap")),
    }
    return {
        "image_size": int(data_cfg.get("S", 256)),
        "window_size": int(data_cfg.get("T", 48)),
        "crop_margin": float(data_cfg.get("margin", 1.4)),
        "num_vertices": 778,
        "encoder": enc,
        "decoder": dec,
    }


def build_model(config: dict[str, Any] | None = None) -> HandQueryNet:
    """Instantiate `HandQueryNet` from a configuration dictionary."""
    cfg = _normalize_config(config)
    num_verts = int(cfg.get("num_vertices", 778))
    max_frames = int(cfg["encoder"].get("max_frames", 64))
    return HandQueryNet(
        landmark_ids=np.arange(num_verts, dtype=np.int64),
        encoder=cfg["encoder"],
        decoder=cfg["decoder"],
        image_size=int(cfg.get("image_size", 256)),
        max_frames=max_frames,
    )


def load_model(
    ckpt_path: str | os.PathLike,
    device: str | torch.device = "cuda",
) -> tuple[HandQueryNet, dict[str, Any]]:
    """Load a trained `HandQueryNet` checkpoint in eval mode along with its normalized config."""
    path = os.path.expanduser(os.fspath(ckpt_path))
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    raw_cfg = ckpt.get("config") or ckpt.get("cfg")
    config = _normalize_config(raw_cfg)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = build_model(config)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model, config
