"""3D hand reconstruction evaluation metrics (in millimeters) and frame-subset helpers.

Implements the standard multi-view benchmark metrics reported in the paper:
  - `MPVPE_mm`: Mean per-vertex position error over all 778 MANO vertices in calibrated rig coordinates
  - `RRV_mm`:   Root-relative (wrist-aligned) per-vertex position error
  - `PAV_mm`:   Procrustes-aligned per-vertex position error
  - `AUC_V`:    Area under the 3D vertex PCK curve over 20 uniformly spaced thresholds in `[0, auc_max_mm]`
  - `MPJPE_poem_mm` (`MPJPE*`), `RRJ_poem_mm` (`RR-J*`), `PAJ_poem_mm` (`PA-J*`):
                Joint errors under POEM's 21-joint definition (MANO `J_regressor` + fingertip vertices
                `[744, 320, 443, 555, 672]`), applied identically to predicted and ground-truth meshes
  - `MPJPE_mm`, `RRJ_mm`, `PA_MPJPE_mm`: Joint errors against each dataset's native 21-joint labels
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

import numpy as np

from handquery.mano import joints21, load_mano_pkl, resolve_mano_paths

POEM_TIPS = [744, 320, 443, 555, 672]
TABLE1_KEYS = [
    "MPVPE_mm",
    "RRV_mm",
    "PAV_mm",
    "AUC_V",
    "MPJPE_poem_mm",
    "RRJ_poem_mm",
    "PAJ_poem_mm",
]
ALL_METRIC_KEYS = TABLE1_KEYS + [
    "AUC_J",
    "MPJPE_mm",
    "RRJ_mm",
    "PA_MPJPE_mm",
    "wrist_mm",
    "tips_mm",
]

_JREG_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _procrustes_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Align `(N, 3)` point cloud `A` onto `B` via similarity Procrustes `(s, R, t)`."""
    ma = A.mean(axis=0)
    mb = B.mean(axis=0)
    A0 = A - ma
    B0 = B - mb
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    denom = float((A0 ** 2).sum())
    s = float(S.sum()) / denom if denom > 1e-12 else 1.0
    return (s * (R @ A.T)).T + (mb - s * (R @ ma))


def mano_regressor(mano_root: str | os.PathLike, side: str) -> np.ndarray:
    root_abs = os.path.abspath(os.path.expanduser(os.fspath(mano_root)))
    key = (root_abs, side.lower())
    if key not in _JREG_CACHE:
        _, pkl_path = resolve_mano_paths(root_abs, side)
        d = load_mano_pkl(pkl_path)
        J = d["J_regressor"]
        _JREG_CACHE[key] = np.asarray(J.toarray() if hasattr(J, "toarray") else J, dtype=np.float64)
    return _JREG_CACHE[key]


def poem_joints(verts: np.ndarray, side: str, mano_root: str | os.PathLike) -> np.ndarray:
    """Compute `(T, 21, 3)` joints under POEM's 21-joint definition from `(T, 778, 3)` vertices."""
    J_reg = mano_regressor(mano_root, side)
    j16 = np.einsum("jv,tvc->tjc", J_reg, verts)
    return joints21(verts, j16, POEM_TIPS)


def auc_pck(err_mm: np.ndarray, max_mm: float, steps: int = 20) -> float:
    """Compute 3D PCK AUC over `steps` thresholds in `[0, max_mm]` for `(T, K)` errors in mm."""
    thresholds = np.linspace(0.0, float(max_mm), steps)
    pck = (err_mm[None] <= thresholds[:, None, None]).mean(axis=1)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(np.mean(trapz_fn(pck, thresholds, axis=0) / trapz_fn(np.ones_like(thresholds), thresholds)))


def eval_frames(valid: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Frames eligible for multi-view scoring: GT valid AND hand boxed in >= 2 cameras of the full rig."""
    return np.asarray(valid, dtype=bool) & ((~np.isnan(boxes[..., 0])).sum(axis=0) >= 2)


def load_subset(subset_json_path: str | os.PathLike) -> dict[str, np.ndarray]:
    with open(os.path.expanduser(os.fspath(subset_json_path)), "r", encoding="utf-8") as f:
        d = json.load(f)
    return {k: np.asarray(v, dtype=np.int64) for k, v in d["sequences"].items()}


def subset_mask(subset: dict[str, np.ndarray], dataset: str, seq_name: str, num_frames: int) -> np.ndarray:
    m = np.zeros(num_frames, dtype=bool)
    idx = subset.get(f"{dataset}/{seq_name}")
    if idx is not None:
        m[idx] = True
    return m


def sequence_metrics(
    verts: np.ndarray,
    j21: np.ndarray,
    gt_verts: np.ndarray,
    gt_j21: np.ndarray,
    mask: np.ndarray,
    side: str,
    mano_root: str | os.PathLike,
    auc_max_mm: float = 20.0,
) -> dict[str, float | int]:
    """Compute all 3D vertex and joint metrics over boolean frame mask `mask`."""
    mask = np.asarray(mask, dtype=bool)
    v = np.nan_to_num(verts[mask]).astype(np.float64)
    j = np.nan_to_num(j21[mask]).astype(np.float64)
    gv = np.nan_to_num(gt_verts[mask]).astype(np.float64)
    gj = np.nan_to_num(gt_j21[mask]).astype(np.float64)

    err_j = np.linalg.norm(j - gj, axis=-1) * 1000.0
    err_v = np.linalg.norm(v - gv, axis=-1) * 1000.0
    rr_j = np.linalg.norm((j - j[:, :1]) - (gj - gj[:, :1]), axis=-1) * 1000.0
    rr_v = np.linalg.norm((v - j[:, :1]) - (gv - gj[:, :1]), axis=-1) * 1000.0

    pa_j = [np.linalg.norm(_procrustes_align(a, b) - b, axis=-1).mean() * 1000.0 for a, b in zip(j, gj)]
    pa_v = [np.linalg.norm(_procrustes_align(va, vb) - vb, axis=-1).mean() * 1000.0 for va, vb in zip(v, gv)]

    jp = poem_joints(v, side, mano_root)
    gp = poem_joints(gv, side, mano_root)
    err_jp = np.linalg.norm(jp - gp, axis=-1) * 1000.0
    rr_jp = np.linalg.norm((jp - jp[:, :1]) - (gp - gp[:, :1]), axis=-1) * 1000.0
    pa_jp = [np.linalg.norm(_procrustes_align(a, b) - b, axis=-1).mean() * 1000.0 for a, b in zip(jp, gp)]

    return {
        "frames_with_gt": int(mask.sum()),
        "MPVPE_mm": float(err_v.mean()),
        "RRV_mm": float(rr_v.mean()),
        "PAV_mm": float(np.mean(pa_v)),
        "AUC_V": auc_pck(err_v, auc_max_mm),
        "MPJPE_poem_mm": float(err_jp.mean()),
        "RRJ_poem_mm": float(rr_jp.mean()),
        "PAJ_poem_mm": float(np.mean(pa_jp)),
        "AUC_J": auc_pck(err_jp, auc_max_mm),
        "MPJPE_mm": float(err_j.mean()),
        "RRJ_mm": float(rr_j.mean()),
        "PA_MPJPE_mm": float(np.mean(pa_j)),
        "wrist_mm": float(err_j[:, 0].mean()),
        "tips_mm": float(err_j[:, [4, 8, 12, 16, 20]].mean()),
    }


def weighted_mean(rows: Sequence[dict[str, Any]], keys: Sequence[str] = ALL_METRIC_KEYS) -> dict[str, Any]:
    """Frame-weighted average of per-sequence metric dictionaries."""
    w = np.array([r["frames_with_gt"] for r in rows], dtype=np.float64)
    out: dict[str, Any] = {
        "sequences": len(rows),
        "frames_with_gt": int(w.sum()),
    }
    for k in keys:
        if rows and all(k in r for r in rows):
            out[k] = float(np.sum(np.array([r[k] for r in rows], dtype=np.float64) * w) / w.sum())
    return out
