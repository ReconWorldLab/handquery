"""Pinhole camera geometry, perspective virtual-camera crops, and linear triangulation.

Conventions:
  - OpenCV pinhole camera model with pixel centers at integer coordinates.
  - World-to-camera extrinsics: X_cam = R @ X_world + t (in meters).
  - Perspective hand crops rotate a virtual pinhole camera about the source camera's
    optical center so its optical axis passes through the 2D hand box center, avoiding
    off-axis perspective distortion while keeping exact intrinsics and extrinsics.
"""

from __future__ import annotations

import numpy as np
import torch


def look_at_rotation(direction: np.ndarray, up: tuple[float, float, float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Rotation matrix whose rows are virtual camera axes in the source camera frame (X_virt = R_v @ X_cam)."""
    z = np.asarray(direction, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross((1.0, 0.0, 0.0), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)


def rot_z(theta: float) -> np.ndarray:
    """In-plane 3x3 rotation about the optical (z) axis by angle `theta` (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def virtual_camera(
    K: np.ndarray,
    center_px: np.ndarray,
    crop_px: float,
    image_size: int = 256,
    roll: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a virtual pinhole crop camera of side `crop_px` around `center_px`, resampled to `image_size x image_size`.

    Args:
        K: (3, 3) source camera intrinsics in pixels.
        center_px: (2,) crop center (cx, cy) in source image pixels.
        crop_px: crop width/height in source image pixels at the virtual principal point.
        image_size: output square crop resolution S.
        roll: optional in-plane camera roll in radians (0.0 at inference).

    Returns:
        R_v: (3, 3) source-camera-to-virtual-camera rotation matrix.
        K_v: (3, 3) virtual crop camera intrinsics in crop pixels.
        H: (3, 3) homography mapping source image pixels to crop pixels for `cv2.warpPerspective`.
    """
    K = np.asarray(K, dtype=np.float64)
    d = np.linalg.solve(K, np.array([center_px[0], center_px[1], 1.0], dtype=np.float64))
    R_v = look_at_rotation(d)
    if roll:
        R_v = rot_z(roll) @ R_v
    f_v = 0.5 * (K[0, 0] + K[1, 1]) * image_size / crop_px
    c = (image_size - 1) / 2.0
    K_v = np.array([[f_v, 0.0, c], [0.0, f_v, c], [0.0, 0.0, 1.0]], dtype=np.float64)
    H = K_v @ R_v @ np.linalg.inv(K)
    return R_v, K_v, H


def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world points `X` (..., 3) into pixel coordinates `(..., 2)` and camera depths `(...,)`."""
    Xc = X @ R.T + t
    z = Xc[..., 2]
    uv = (Xc @ K.T)[..., :2] / np.clip(z, 1e-9, None)[..., None]
    return uv, z


def triangulate_point(Ps: list[np.ndarray], uvs: list[tuple[float, float]]) -> np.ndarray:
    """Linear DLT triangulation of a single 3D point from >= 2 calibrated views."""
    rows = []
    for P, (u, v) in zip(Ps, uvs):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def crop_to_original(
    uv: torch.Tensor,
    sigma: torch.Tensor,
    H: torch.Tensor,
    image_size: float = 256.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map normalized crop coordinates and uncertainties back to full-resolution source pixels.

    Args:
        uv: (V, T, N, 2) normalized crop coordinates in [-1, 1].
        sigma: (V, T, N) standard deviation in crop pixels.
        H: (V, T, 3, 3) source-pixel -> crop-pixel homography.
        image_size: crop resolution S.

    Returns:
        xy: (V, T, N, 2) 2D coordinates in source image pixels.
        sigma_orig: (V, T, N) standard deviation scaled by the local Jacobian of H^-1.
    """
    Hi = torch.linalg.inv(H.double())
    p = torch.cat(
        [uv.double() * (image_size / 2.0) + (image_size - 1.0) / 2.0, torch.ones_like(uv[..., :1]).double()],
        dim=-1,
    )
    q = torch.einsum("vtij,vtnj->vtni", Hi, p)
    w = q[..., 2:3]
    xy = q[..., :2] / w
    A = Hi[:, :, None, :2, :2] - xy[..., :, None] * Hi[:, :, None, 2:3, :2]
    J = A / w[..., None]
    scale = torch.sqrt(torch.abs(torch.linalg.det(J)))
    return xy.float(), (sigma.double() * scale).float()
