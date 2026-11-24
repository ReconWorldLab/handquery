"""Lightweight multi-view mesh & dense landmark visualization and Wavefront OBJ export.

Uses pure NumPy + OpenCV painter's-algorithm rasterization with camera-space Lambertian shading,
avoiding OpenGL / EGL / pyrender dependencies on headless servers and desktops alike.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from typing import Any, Sequence

import cv2
import numpy as np

HAND_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "right": (90, 175, 255),
    "left": (255, 145, 95),
}


def render_meshes_on_image(
    img_rgb: np.ndarray,
    meshes: Sequence[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """Render one or more 3D meshes `(verts_world, faces, rgb_color)` onto an RGB uint8 image.

    Triangles from all hands are depth-sorted jointly so inter-hand occlusions are rendered accurately.
    """
    H, W = img_rgb.shape[:2]
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    tris_2d_list = []
    depths_list = []
    colors_list = []

    for verts_w, faces, base_rgb in meshes:
        if verts_w is None or not np.isfinite(verts_w).all():
            continue
        Xc = np.asarray(verts_w, dtype=np.float64) @ R.T + t
        tri_3d = Xc[faces]  # (F, 3, 3)
        z_min = tri_3d[..., 2].min(axis=1)
        in_front = z_min > 0.02
        if not in_front.any():
            continue
        tri_3d = tri_3d[in_front]

        # Camera-space face normal & back-face culling (OpenCV camera looks along +Z)
        e1 = tri_3d[:, 1] - tri_3d[:, 0]
        e2 = tri_3d[:, 2] - tri_3d[:, 0]
        normals = np.cross(e1, e2)
        n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(n_norm, 1e-12, None)
        centers = tri_3d.mean(axis=1)
        view_dirs = -centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
        cos_theta = (normals * view_dirs).sum(axis=1)
        front_facing = cos_theta > -0.05
        if not front_facing.any():
            continue

        tri_3d = tri_3d[front_facing]
        cos_theta = np.clip(cos_theta[front_facing], 0.0, 1.0)

        proj = np.einsum("ij,fkj->fki", K, tri_3d)
        tri_2d = proj[..., :2] / proj[..., 2:3]

        # Bounding-box frustum cull
        in_img = (
            (tri_2d[..., 0].max(axis=1) >= 0)
            & (tri_2d[..., 0].min(axis=1) < W)
            & (tri_2d[..., 1].max(axis=1) >= 0)
            & (tri_2d[..., 1].min(axis=1) < H)
        )
        if not in_img.any():
            continue
        tri_2d = tri_2d[in_img]
        tri_3d = tri_3d[in_img]
        cos_theta = cos_theta[in_img]

        shade = 0.32 + 0.68 * cos_theta[:, None]
        base = np.asarray(base_rgb, dtype=np.float64)[None, :]
        shaded_rgb = np.clip(base * shade + 18.0 * (cos_theta[:, None] ** 4), 0, 255).astype(np.uint8)

        tris_2d_list.append(tri_2d)
        depths_list.append(tri_3d[..., 2].mean(axis=1))
        colors_list.append(shaded_rgb)

    if not tris_2d_list:
        return img_rgb.copy()

    all_tris = np.concatenate(tris_2d_list, axis=0)
    all_depths = np.concatenate(depths_list, axis=0)
    all_colors = np.concatenate(colors_list, axis=0)

    # Crop ROI around projected triangles for fast rasterization & anti-aliased 2x supersampling
    x0 = max(0, int(math.floor(all_tris[..., 0].min())) - 2)
    y0 = max(0, int(math.floor(all_tris[..., 1].min())) - 2)
    x1 = min(W, int(math.ceil(all_tris[..., 0].max())) + 3)
    y1 = min(H, int(math.ceil(all_tris[..., 1].max())) + 3)
    if x1 <= x0 or y1 <= y0:
        return img_rgb.copy()

    ss = 2
    rw, rh = (x1 - x0) * ss, (y1 - y0) * ss
    canvas = cv2.resize(img_rgb[y0:y1, x0:x1], (rw, rh), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((rh, rw), dtype=np.uint8)

    pts = np.round((all_tris - np.array([x0, y0], dtype=np.float64)) * ss).astype(np.int32)
    order = np.argsort(-all_depths)  # back-to-front painter's order
    for idx in order:
        c = (int(all_colors[idx, 0]), int(all_colors[idx, 1]), int(all_colors[idx, 2]))
        cv2.fillConvexPoly(canvas, pts[idx], c)
        cv2.fillConvexPoly(mask, pts[idx], 255)

    overlay_roi = cv2.resize(canvas, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    alpha_roi = (cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0) * alpha

    out = img_rgb.copy()
    base_roi = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(
        base_roi * (1.0 - alpha_roi[..., None]) + overlay_roi.astype(np.float32) * alpha_roi[..., None],
        0,
        255,
    ).astype(np.uint8)
    return out


def render_landmarks_on_image(
    img_rgb: np.ndarray,
    lm_2d: np.ndarray,
    vis: np.ndarray,
    color: tuple[int, int, int] = (90, 175, 255),
    vis_thresh: float = 0.5,
) -> np.ndarray:
    """Draw dense 2D vertex predictions onto an RGB uint8 image (visible points solid, occluded dim)."""
    out = img_rgb.copy()
    H, W = out.shape[:2]
    radius = max(1, int(round(min(H, W) / 360)))
    dim_color = tuple(max(0, int(c * 0.45)) for c in color)
    for k in range(len(lm_2d)):
        x, y = int(round(float(lm_2d[k, 0]))), int(round(float(lm_2d[k, 1])))
        if 0 <= x < W and 0 <= y < H:
            is_vis = float(vis[k]) >= vis_thresh
            cv2.circle(out, (x, y), radius if is_vis else max(1, radius - 1), color if is_vis else dim_color, -1, cv2.LINE_AA)
    return out


def write_multiview_video(
    out_path: str | os.PathLike,
    frame_source,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    hand_results: dict[str, dict[str, Any]],
    fps: float = 30.0,
    max_cols: int = 4,
    tile_width: int = 480,
) -> None:
    """Render a multi-view grid MP4 video overlaying fitted 3D MANO meshes (or 2D vertices if unfitted)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    C = K.shape[0]
    first_res = next(iter(hand_results.values()))
    n_frames = first_res["lm"].shape[1]

    cols = min(C, max_cols)
    rows = int(math.ceil(C / cols))
    writer = None
    ffmpeg_proc = None
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None

    th = None
    try:
        for f in range(n_frames):
            tiles = []
            for c in range(C):
                img = frame_source.get_rgb(c, f)
                if img is None:
                    img = np.zeros((tile_width, tile_width, 3), dtype=np.uint8)
                has_mesh = any("verts" in res for res in hand_results.values())
                if has_mesh:
                    meshes = []
                    for side, res in hand_results.items():
                        if "verts" in res and bool(res["observed"][f]):
                            meshes.append(
                                (
                                    res["verts"][f],
                                    res["faces"],
                                    HAND_COLORS_RGB.get(side, (90, 175, 255)),
                                )
                            )
                    canvas = render_meshes_on_image(img, meshes, K[c], R[c], t[c])
                else:
                    canvas = img.copy()
                    for side, res in hand_results.items():
                        if bool(res["valid"][c, f]):
                            canvas = render_landmarks_on_image(
                                canvas,
                                res["lm"][c, f],
                                res["vis"][c, f],
                                color=HAND_COLORS_RGB.get(side, (90, 175, 255)),
                            )
                scale = tile_width / canvas.shape[1]
                cur_h = int(round(canvas.shape[0] * scale))
                if th is None:
                    th = cur_h + (cur_h % 2)
                tile = cv2.resize(canvas, (tile_width, th), interpolation=cv2.INTER_AREA)
                tiles.append(tile)

            assert th is not None
            while len(tiles) < rows * cols:
                tiles.append(np.zeros((th, tile_width, 3), dtype=np.uint8))
            grid_rows = [np.concatenate(tiles[r * cols : (r + 1) * cols], axis=1) for r in range(rows)]
            grid_rgb = np.concatenate(grid_rows, axis=0)

            if writer is None and ffmpeg_proc is None:
                gh, gw = grid_rgb.shape[:2]
                if ffmpeg_bin:
                    ffmpeg_proc = subprocess.Popen(
                        [
                            ffmpeg_bin,
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-s",
                            f"{gw}x{gh}",
                            "-r",
                            str(float(fps)),
                            "-i",
                            "-",
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-crf",
                            "20",
                            os.fspath(out_path),
                        ],
                        stdin=subprocess.PIPE,
                    )
                else:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(os.fspath(out_path), fourcc, float(fps), (gw, gh))

            if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
                ffmpeg_proc.stdin.write(np.ascontiguousarray(grid_rgb).tobytes())
            elif writer is not None:
                writer.write(cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))
    finally:
        if ffmpeg_proc is not None:
            if ffmpeg_proc.stdin is not None:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        if writer is not None:
            writer.release()


def export_obj_sequence(
    out_dir: str | os.PathLike,
    verts: np.ndarray,
    faces: np.ndarray,
    observed: np.ndarray | None = None,
) -> int:
    """Write per-frame Wavefront `.obj` meshes (`frame_000000.obj`, ...) for observed frames."""
    os.makedirs(out_dir, exist_ok=True)
    faces_1 = np.asarray(faces, dtype=np.int64) + 1
    face_lines = "".join(f"f {a} {b} {c}\n" for a, b, c in faces_1)
    written = 0
    for t in range(len(verts)):
        if observed is not None and not bool(observed[t]):
            continue
        if not np.isfinite(verts[t]).all():
            continue
        vert_lines = "".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts[t])
        with open(os.path.join(out_dir, f"frame_{t:06d}.obj"), "w", encoding="utf-8") as f:
            f.write(vert_lines)
            f.write(face_lines)
        written += 1
    return written
