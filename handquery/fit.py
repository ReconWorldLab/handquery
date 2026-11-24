"""Uncertainty-aware multi-view MANO sequence fitting.

Pipeline per sequence:
  1. Uncertainty- and visibility-weighted linear DLT triangulation across views (threaded batched SVD);
     points > 15 cm from each frame's median are rejected as outliers.
  2. Per-frame similarity Procrustes alignment of the rest-pose MANO template onto the triangulated
     point cloud to initialize global wrist orientation and 3D translation.
  3. Three-stage Adam optimization over shared sequence shape `betas (1, 10)` and per-frame
     `global_orient (T, 3)`, `pca (T, 45)`, and `transl (T, 3)` using Geman-McClure (`delta = 8.0`)
     robust reprojection residuals normalized by predicted `sigma`, accelerated via CUDA graphs.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import cv2
import numpy as np
import torch

from handquery.mano import FINGERTIP_VERTS, ManoPCALayer, joints21


def triangulate_np(lm: np.ndarray, w: np.ndarray, P: np.ndarray, threads: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Weighted linear DLT triangulation of `(C, T, K, 2)` 2D landmarks across `C` calibrated views."""
    u, v = lm[..., 0], lm[..., 1]
    P0, P1, P2 = P[:, None, None, 0], P[:, None, None, 1], P[:, None, None, 2]
    r1 = (u[..., None] * P2 - P0) * np.sqrt(w)[..., None]
    r2 = (v[..., None] * P2 - P1) * np.sqrt(w)[..., None]
    A = np.concatenate([r1, r2], axis=0).transpose(1, 2, 0, 3)  # (T, K, 2C, 4)
    X = np.empty(A.shape[:2] + (4,), dtype=A.dtype)
    cut = np.linspace(0, len(A), min(len(A), 4 * max(1, threads)) + 1).astype(int)

    def _svd_chunk(k: int) -> None:
        X[cut[k] : cut[k + 1]] = np.linalg.svd(A[cut[k] : cut[k + 1]], full_matrices=False)[2][..., -1, :]

    with ThreadPoolExecutor(max(1, threads)) as ex:
        list(ex.map(_svd_chunk, range(len(cut) - 1)))
    X = X[..., :3] / np.where(np.abs(X[..., 3:]) < 1e-9, 1e-9, X[..., 3:])
    return X.astype(np.float32), (w > 0).sum(0) >= 2


def procrustes(A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity Procrustes alignment `(s, R, t)` minimizing `||s R A + t - B||` for `(N, 3)` point sets."""
    w = np.ones(len(A)) if w is None else w
    ma = (A * w[:, None]).sum(0) / w.sum()
    mb = (B * w[:, None]).sum(0) / w.sum()
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd((A0 * w[:, None]).T @ B0)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    s = S.sum() / ((A0 ** 2) * w[:, None]).sum()
    return float(s), R, mb - s * R @ ma


def geman_mcclure(x: torch.Tensor, delta: float = 8.0) -> torch.Tensor:
    x2 = x ** 2
    return x2 / (delta ** 2 + x2)


_SIDE_STREAM: dict[torch.device, torch.cuda.Stream] = {}


def _adam_steps(opt: torch.optim.Optimizer, backward_fn, n: int, graph: bool, warm: int = 2) -> None:
    """Run `n` Adam steps, optionally capturing forward+backward in a CUDA graph after `warm` warmup steps."""
    params = [p for g in opt.param_groups for p in g["params"]]
    if not graph or n <= warm:
        for _ in range(n):
            opt.zero_grad()
            backward_fn()
            opt.step()
        return

    dev = params[0].device
    with torch.cuda.device(dev):
        s = _SIDE_STREAM.setdefault(dev, torch.cuda.Stream(dev))
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(warm):
                opt.zero_grad()
                backward_fn()
                opt.step()
        torch.cuda.current_stream(dev).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        opt.zero_grad()
        with torch.cuda.graph(g, stream=s):
            backward_fn()
        for _ in range(n - warm):
            g.replay()
            opt.step()
        for p in params:
            p.grad = None
        del g


def fit_sequence(
    lm: np.ndarray,
    sig: np.ndarray,
    vis: np.ndarray,
    valid: np.ndarray,
    Ks: Sequence[np.ndarray] | np.ndarray,
    Es: Sequence[np.ndarray] | np.ndarray,
    side: str,
    mano_root: str | os.PathLike,
    lm_idx: Sequence[int] | np.ndarray | None = None,
    tips: Sequence[int] | str = "smplx",
    iters: tuple[int, int, int] = (150, 300, 200),
    lr: float = 0.01,
    temporal_w: float = 0.0,
    beta_w: float = 0.0,
    device: str | torch.device = "cuda",
    graph: bool = True,
    threads: int = 8,
) -> dict[str, np.ndarray | float]:
    """Fit a 4D MANO mesh sequence to fused multi-view 2D vertex predictions.

    Args:
        lm: `(C, T, K, 2)` 2D vertex coordinates in original image pixels.
        sig: `(C, T, K)` per-vertex pixel standard deviations.
        vis: `(C, T, K)` visibility probabilities in `[0, 1]`.
        valid: `(C, T)` boolean/float mask indicating views & frames with valid observations.
        Ks: `(C, 3, 3)` pinhole camera intrinsics in pixels.
        Es: `(C, 4, 4)` world-to-camera SE(3) extrinsics in meters.
        side: `'right'` or `'left'`.
        mano_root: directory containing `mano/MANO_{RIGHT,LEFT}.pkl` (or `MANO_{RIGHT,LEFT}.pkl`).
        lm_idx: MANO vertex indices of the `K` landmarks (defaults to `0..K-1`, i.e. all 778 vertices).
        tips: fingertip convention name in `FINGERTIP_VERTS` (e.g. `'smplx'`, `'dexycb'`) or 5 vertex IDs.
        iters: iteration counts `(stage1_rigid, stage2_joint, stage3_refine)`.
        lr: Adam learning rate (`0.5 * lr` in stage 3).
        temporal_w: optional temporal acceleration smoothness weight (default `0.0`).
        beta_w: optional L2 shape regularization weight (default `0.0`).
        device: torch device (`'cuda'` or `'cpu'`).
        graph: capture stage forward+backward in CUDA graphs when running on CUDA.
        threads: worker threads for batched DLT triangulation SVDs.

    Returns:
        Dictionary with world-frame `verts (T, 778, 3)`, `joints21 (T, 21, 3)`, `faces (1538, 3)`,
        `betas (1, 10)`, `global_orient (T, 3)`, `pca (T, 45)`, `hand_pose (T, 45)`, `transl (T, 3)`,
        `reproj_px`, and `tri_ok_frac`.
    """
    side = side.lower()
    if isinstance(tips, str):
        if tips not in FINGERTIP_VERTS:
            raise ValueError(f"Unknown tip convention {tips!r}; choose from {sorted(FINGERTIP_VERTS)}")
        tip_ids = FINGERTIP_VERTS[tips][side]
    else:
        tip_ids = list(tips)

    C, T, K = sig.shape
    if lm_idx is None:
        lm_idx = np.arange(K, dtype=np.int64)
    Ks_arr = [np.asarray(k, dtype=np.float32) for k in Ks]
    Es_arr = [np.asarray(e, dtype=np.float32) for e in Es]

    sig_c = np.clip(np.asarray(sig, dtype=np.float32), 1.0, 50.0)
    w = (vis > 0.5) * np.asarray(valid, dtype=np.float32)[..., None] / (sig_c ** 2)
    P = np.stack([Ks_arr[c] @ Es_arr[c][:3] for c in range(C)], axis=0)
    X3, ok = triangulate_np(lm, w, P, threads=threads)
    med = np.array([np.median(X3[t][ok[t]], axis=0) if ok[t].any() else np.zeros(3) for t in range(T)])
    ok &= np.linalg.norm(X3 - med[:, None], axis=-1) < 0.15

    mano_layer = ManoPCALayer(mano_root, side, device)
    idx = torch.as_tensor(lm_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        z3 = torch.zeros(1, 3, device=device)
        v0, j0 = mano_layer(z3, torch.zeros(1, 45, device=device), torch.zeros(1, 10, device=device), z3)
    tmpl = v0[:, idx][0].cpu().numpy()
    wrist = j0[0, 0].cpu().numpy()

    glob0 = np.zeros((T, 3), dtype=np.float32)
    tr0 = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        m = ok[t]
        if m.sum() >= 10:
            _, R_p, tt = procrustes(tmpl[m], X3[t][m])
            glob0[t] = cv2.Rodrigues(R_p)[0][:, 0]
            tr0[t] = R_p @ wrist + tt - wrist
        elif t > 0:
            glob0[t], tr0[t] = glob0[t - 1], tr0[t - 1]
    good = [t for t in range(T) if ok[t].sum() >= 10]
    for t in range(good[0] if good else 0):
        glob0[t], tr0[t] = glob0[good[0]], tr0[good[0]]

    glob = torch.tensor(glob0, device=device, requires_grad=True)
    transl = torch.tensor(tr0, device=device, requires_grad=True)
    coeffs = torch.zeros(T, 45, device=device, requires_grad=True)
    betas = torch.zeros(1, 10, device=device, requires_grad=True)

    lm_t = torch.as_tensor(lm, dtype=torch.float32, device=device)
    sig_t = torch.as_tensor(sig_c, dtype=torch.float32, device=device)
    vis_t = torch.as_tensor(vis, dtype=torch.float32, device=device)
    val_t = torch.as_tensor(valid, dtype=torch.float32, device=device)
    X3_t = torch.as_tensor(X3, dtype=torch.float32, device=device)
    ok_t = torch.as_tensor(ok, dtype=torch.float32, device=device)
    Kt = torch.as_tensor(np.stack(Ks_arr), dtype=torch.float32, device=device)
    Et = torch.as_tensor(np.stack(Es_arr), dtype=torch.float32, device=device)

    def _project(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Xc = torch.einsum("cij,tnj->ctni", Et[:, :3, :3], X) + Et[:, None, None, :3, 3]
        uv = torch.einsum("cij,ctnj->ctni", Kt, Xc)
        return uv[..., :2] / uv[..., 2:3].clamp(min=1e-4), Xc[..., 2]

    def _stage_loss(stage: int) -> torch.Tensor:
        verts, joints = (
            mano_layer(glob, coeffs, betas, transl)
            if stage > 1
            else mano_layer(glob, coeffs.detach(), betas.detach(), transl)
        )
        lmk = verts[:, idx]
        uv, z = _project(lmk)
        res = (uv - lm_t).norm(dim=-1) / sig_t
        wgt = vis_t.clamp(min=0.2 if stage >= 2 else 0.8) * val_t[..., None] * (z > 0.05)
        l_rep = (geman_mcclure(res) * wgt).sum() / wgt.sum().clamp(min=1)
        l_3d = ((lmk - X3_t).norm(dim=-1) * ok_t).sum() / ok_t.sum().clamp(min=1)
        if stage == 1:
            return 5.0 * l_rep + 100.0 * l_3d
        l_pose = (coeffs ** 2).mean()
        l_beta = (betas ** 2).mean()
        if stage == 2:
            return 20.0 * l_rep + 20.0 * l_3d + 0.02 * l_pose + beta_w * l_beta
        total = 20.0 * l_rep + 5.0 * l_3d + 0.02 * l_pose + beta_w * l_beta
        if temporal_w > 0 and T > 2:
            j21 = joints21(verts, joints, tip_ids)
            total = (
                total
                + temporal_w * ((j21[2:] - 2 * j21[1:-1] + j21[:-2]) ** 2).sum(-1).mean()
                + 0.5 * ((coeffs[1:] - coeffs[:-1]) ** 2).mean()
            )
        return total

    use_graph = bool(graph and torch.device(device).type == "cuda")
    for stage, n_steps in zip((1, 2, 3), iters):
        params = [glob, transl] if stage == 1 else [glob, transl, coeffs, betas]
        opt = torch.optim.Adam(params, lr=lr if stage < 3 else lr * 0.5)
        _adam_steps(opt, lambda s=stage: _stage_loss(s).backward(), n_steps, use_graph)

    with torch.no_grad():
        verts, joints = mano_layer(glob, coeffs, betas, transl)
        uv, _ = _project(verts[:, idx])
        m = (vis_t > 0.5) * val_t[..., None]
        rep = float(((uv - lm_t).norm(dim=-1) * m).sum() / m.sum().clamp(min=1))
        j21 = joints21(verts, joints, tip_ids)
        hand_pose = mano_layer.pca_to_axis_angle(coeffs)

    return {
        "verts": verts.cpu().numpy(),
        "joints21": j21.cpu().numpy(),
        "faces": mano_layer.faces.copy(),
        "betas": betas.detach().cpu().numpy(),
        "global_orient": glob.detach().cpu().numpy(),
        "pca": coeffs.detach().cpu().numpy(),
        "hand_pose": hand_pose.detach().cpu().numpy(),
        "transl": transl.detach().cpu().numpy(),
        "reproj_px": rep,
        "tri_ok_frac": float(ok.mean()),
    }
