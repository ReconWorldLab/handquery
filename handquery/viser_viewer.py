"""Interactive 3D + multi-view image viewer powered by `viser`.

Supports both:
  1. `scene.json` manifests (built-in demos, custom rigs, and zero-shot `SHOW3D` / `GigaHands` exports), and
  2. Prepared benchmark `.npz` sequence files (`DexYCB`, `HO3D`, `OakInk`, `ARCTIC`, `InterHand-MV`).

Displays:
  - Estimated 3D MANO hand meshes (left & right) and optional 21-joint skeletons
  - Tracked 3D object meshes & motion (both rigid `.ply` objects and articulated multi-part objects)
  - Calibrated 3D camera frustums textured with synchronized original camera frames (click to jump view)
  - Live multi-view image grid in the sidebar with optional 2D prediction overlay toggle
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Any

import cv2
import numpy as np
import trimesh
import viser
import viser.transforms as vtf

from handquery.ply import load_ply
from handquery.vis import render_landmarks_on_image, render_meshes_on_image

HAND_BASE_RGB = {
    "left": (255, 214, 122),
    "right": (176, 142, 238),
}

BONE_PAIRS_21 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class SceneBundle:
    name: str
    fps: float
    up: str
    num_frames: int
    K_orig: np.ndarray                      # (C, 3, 3) full-res intrinsics
    K_disp: np.ndarray                      # (C, 3, 3) scaled intrinsics matching pre-cached display frames
    K_disp_frames: np.ndarray | None        # Optional (C, T, 3, 3) when per-frame crop affine is present (e.g. ARCTIC)
    affine_disp_frames: np.ndarray | None   # Optional (C, T, 3, 3) mapping original pixels -> display crop pixels
    R: np.ndarray                           # (C, 3, 3) world -> cam
    t: np.ndarray                           # (C, 3) world -> cam
    disp_w: int
    disp_h: int
    disp_scale: float
    jpeg_frames: list[list[bytes]]          # [C][T] compressed JPEGs for zero-seek scrubbing
    hands_fit: dict[str, dict[str, np.ndarray]]
    hands_pred2d: dict[str, dict[str, np.ndarray]]
    obj_parts: list[trimesh.Trimesh]
    obj_transforms: np.ndarray | None       # (T, P, 4, 4) float32
    obj_valid: np.ndarray | None            # (T,) bool
    move_labels: dict[int, str]
    look_at: tuple[float, float, float]
    cam_pos: tuple[float, float, float]
    grid_pos: tuple[float, float, float]


def _resolve_path(base_dir: str, rel_or_abs: str) -> str:
    p = os.path.expanduser(rel_or_abs)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


def _load_hands_from_result_dir(
    result_dir: str,
    allowed_sides: tuple[str, ...] = ("left", "right"),
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    hands_fit: dict[str, dict[str, np.ndarray]] = {}
    hands_pred2d: dict[str, dict[str, np.ndarray]] = {}
    has_prefixed = any(
        os.path.isfile(os.path.join(result_dir, fn))
        for fn in ("left_mano_fit.npz", "right_mano_fit.npz", "left_pred2d.npz", "right_pred2d.npz")
    )
    for side in ("left", "right"):
        for prefix in (f"{side}_", ""):
            if prefix == "" and (has_prefixed or side not in allowed_sides):
                continue
            fit_p = os.path.join(result_dir, f"{prefix}mano_fit.npz")
            p2d_p = os.path.join(result_dir, f"{prefix}pred2d.npz")
            if os.path.isfile(fit_p):
                with np.load(fit_p) as z:
                    hands_fit[side] = {k: z[k] for k in z.files}
            if os.path.isfile(p2d_p):
                with np.load(p2d_p) as z:
                    hands_pred2d[side] = {k: z[k] for k in z.files}
    return hands_fit, hands_pred2d


def _compute_default_view(
    hands_fit: dict[str, dict[str, np.ndarray]],
    R: np.ndarray,
    t: np.ndarray,
    up: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    pts = []
    for fit in hands_fit.values():
        obs = fit.get("observed")
        v = fit["verts"] if obs is None else fit["verts"][np.asarray(obs, dtype=bool)]
        v = v[np.isfinite(v[:, 0, 0])]
        if len(v) > 0:
            pts.append(v.reshape(-1, 3))
    if not pts:
        return (0.0, 0.0, 0.08), (0.48, -0.52, 0.42), (0.0, 0.0, 0.0)

    med = np.median(np.concatenate(pts, axis=0), axis=0).astype(np.float64)
    if np.linalg.norm(med) < 0.35 and up in ("+z", "-z"):
        return (0.0, 0.0, 0.08), (0.48, -0.52, 0.42), (0.0, 0.0, 0.0)

    cam_centers = np.einsum("cij,cj->ci", -np.transpose(R, (0, 2, 1)), t)
    rel = cam_centers[0] - med
    dist = float(np.linalg.norm(rel))
    dir_vec = rel / max(dist, 1e-6)
    pos = med + dir_vec * min(max(dist * 0.75, 0.45), 0.85)
    grid_z = float(med[2] - 0.12) if up == "+z" else float(med[2])
    return (
        (float(med[0]), float(med[1]), float(med[2])),
        (float(pos[0]), float(pos[1]), float(pos[2])),
        (float(med[0]), float(med[1]), grid_z),
    )


def _load_npz_bundle(
    npz_path: str,
    result_dir: str,
    preview_width: int = 540,
) -> SceneBundle:
    with np.load(npz_path, allow_pickle=True) as z:
        L = {k: z[k] for k in z.files}

    name = os.path.splitext(os.path.basename(npz_path))[0]
    side = str(L["side"]) if "side" in L else "right"
    K_orig = np.asarray(L["K"], dtype=np.float64)
    R = np.asarray(L["R"], dtype=np.float64)
    t = np.asarray(L["t"], dtype=np.float64)
    C = K_orig.shape[0]
    n_frames = int(L["verts"].shape[0]) if "verts" in L else int(L["boxes"].shape[1])

    offset = int(L["img_offset"]) if "img_offset" in L else 0
    fids = [int(x) for x in L["img_frame_ids"]] if "img_frame_ids" in L else [f + offset for f in range(n_frames)]
    patterns = []
    for pat_raw in L["img_pattern"]:
        pat = str(pat_raw)
        if "InterHand" in pat and pat.endswith("image{}.jpg"):
            pat = pat[: -len("image{}.jpg")] + "image{:04d}.jpg"
        patterns.append(pat)

    # Probe first available image to determine native stored dimensions
    orig_w, orig_h = 640, 480
    for c in range(C):
        for fid in fids:
            p = patterns[c].format(fid)
            if os.path.isfile(p):
                im = cv2.imread(p, cv2.IMREAD_COLOR)
                if im is not None:
                    orig_h, orig_w = im.shape[:2]
                    break
        else:
            continue
        break

    disp_w = preview_width
    disp_scale = preview_width / float(orig_w)
    disp_h = int(round(orig_h * disp_scale))
    _, blank_enc = cv2.imencode(".jpg", np.zeros((disp_h, disp_w, 3), dtype=np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
    blank_bytes = blank_enc.tobytes()

    jpeg_frames: list[list[bytes]] = [[] for _ in range(C)]
    for c in range(C):
        pat = patterns[c]
        for fid in fids:
            p = pat.format(fid)
            bgr = cv2.imread(p, cv2.IMREAD_COLOR) if os.path.isfile(p) else None
            if bgr is None:
                jpeg_frames[c].append(blank_bytes)
            else:
                small = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                _, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 86])
                jpeg_frames[c].append(enc.tobytes())

    K_disp = K_orig.copy()
    K_disp[:, 0, :] *= disp_scale
    K_disp[:, 1, :] *= disp_scale

    K_disp_frames = None
    affine_disp_frames = None
    if "img_affine" in L and L["img_affine"] is not None:
        A = np.asarray(L["img_affine"], dtype=np.float64)  # (C, T, 3, 3)
        S = np.array([[disp_scale, 0.0, 0.0], [0.0, disp_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        affine_disp_frames = np.einsum("ij,ctjk->ctik", S, A)
        K_disp_frames = np.einsum("ctij,cjk->ctik", affine_disp_frames, K_orig)

    hands_fit, hands_pred2d = _load_hands_from_result_dir(result_dir, allowed_sides=(side,))
    up = "+z"
    look_at, cam_pos, grid_pos = _compute_default_view(hands_fit, R, t, up)

    return SceneBundle(
        name=name,
        fps=30.0,
        up=up,
        num_frames=n_frames,
        K_orig=K_orig,
        K_disp=K_disp,
        K_disp_frames=K_disp_frames,
        affine_disp_frames=affine_disp_frames,
        R=R,
        t=t,
        disp_w=disp_w,
        disp_h=disp_h,
        disp_scale=disp_scale,
        jpeg_frames=jpeg_frames,
        hands_fit=hands_fit,
        hands_pred2d=hands_pred2d,
        obj_parts=[],
        obj_transforms=None,
        obj_valid=None,
        move_labels={},
        look_at=look_at,
        cam_pos=cam_pos,
        grid_pos=grid_pos,
    )


def load_scene_bundle(
    scene_path: str,
    result_dir: str,
    preview_width: int = 540,
) -> SceneBundle:
    """Load camera calibration, compressed preview frames, object mesh+motion, and HandQuery predictions."""
    scene_path = os.path.abspath(os.path.expanduser(scene_path))
    result_dir = os.path.abspath(os.path.expanduser(result_dir))
    if scene_path.lower().endswith(".npz"):
        return _load_npz_bundle(scene_path, result_dir, preview_width=preview_width)

    base_dir = os.path.dirname(scene_path)
    with open(scene_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    name = str(meta.get("name") or os.path.basename(base_dir))
    fps = float(meta.get("fps", 30.0))
    up = str(meta.get("up", "+z"))

    def _arr(val, dtype=np.float64):
        if isinstance(val, str):
            return np.asarray(np.load(_resolve_path(base_dir, val)), dtype=dtype)
        return np.asarray(val, dtype=dtype)

    K_orig = _arr(meta["K"], np.float64)
    R = _arr(meta["R"], np.float64)
    t = _arr(meta["t"], np.float64)
    C = K_orig.shape[0]

    jpeg_frames: list[list[bytes]] = [[] for _ in range(C)]
    disp_w = preview_width
    disp_h = 0
    disp_scale = 1.0

    if "videos" in meta:
        for c, vp in enumerate(meta["videos"]):
            cap = cv2.VideoCapture(_resolve_path(base_dir, vp))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {vp}")
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            disp_scale = preview_width / float(orig_w)
            disp_h = int(round(orig_h * disp_scale))
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                small = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                _, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 86])
                jpeg_frames[c].append(enc.tobytes())
            cap.release()
    elif "frames" in meta:
        for c, flist in enumerate(meta["frames"]):
            for fp in flist:
                bgr = cv2.imread(_resolve_path(base_dir, fp), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(f"Missing frame: {fp}")
                orig_h, orig_w = bgr.shape[:2]
                disp_scale = preview_width / float(orig_w)
                disp_h = int(round(orig_h * disp_scale))
                small = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                _, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 86])
                jpeg_frames[c].append(enc.tobytes())
    else:
        raise ValueError(f"{scene_path}: expected 'videos' or 'frames' in scene.json")

    num_frames = min(len(ch) for ch in jpeg_frames)
    K_disp = K_orig.copy()
    K_disp[:, 0, :] *= disp_scale
    K_disp[:, 1, :] *= disp_scale

    allowed = tuple(meta["hands"].keys()) if "hands" in meta else ("left", "right")
    hands_fit, hands_pred2d = _load_hands_from_result_dir(result_dir, allowed_sides=allowed)

    obj_parts: list[trimesh.Trimesh] = []
    obj_transforms = None
    obj_valid = None
    move_labels: dict[int, str] = {}

    if "object" in meta and meta["object"]:
        obj_cfg = meta["object"]
        mot_z = np.load(_resolve_path(base_dir, obj_cfg["motion"]), allow_pickle=True)
        obj_valid = np.asarray(mot_z["valid"], dtype=bool) if "valid" in mot_z.files else np.ones(num_frames, dtype=bool)
        if obj_cfg.get("articulated", False):
            for pm in obj_cfg["part_meshes"]:
                v, f, col = load_ply(_resolve_path(base_dir, pm))
                obj_parts.append(trimesh.Trimesh(vertices=v, faces=f, vertex_colors=col, process=False))
            obj_transforms = np.asarray(mot_z["part_T_world"], dtype=np.float32)
        else:
            v, f, col = load_ply(_resolve_path(base_dir, obj_cfg["mesh"]))
            obj_parts.append(trimesh.Trimesh(vertices=v, faces=f, vertex_colors=col, process=False))
            obj_transforms = np.asarray(mot_z["object_T_world"], dtype=np.float32)[:, None, :, :]

        if "moves_frame" in mot_z.files and "moves" in mot_z.files:
            m_frames = [int(x) for x in mot_z["moves_frame"]]
            m_names = [str(x) for x in mot_z["moves"]]
            for mf, mn in zip(m_frames, m_names):
                for dt in range(-6, 12):
                    if 0 <= mf + dt < num_frames:
                        move_labels[mf + dt] = mn

    look_at, cam_pos, grid_pos = _compute_default_view(hands_fit, R, t, up)

    return SceneBundle(
        name=name,
        fps=fps,
        up=up,
        num_frames=num_frames,
        K_orig=K_orig,
        K_disp=K_disp,
        K_disp_frames=None,
        affine_disp_frames=None,
        R=R,
        t=t,
        disp_w=disp_w,
        disp_h=disp_h,
        disp_scale=disp_scale,
        jpeg_frames=jpeg_frames,
        hands_fit=hands_fit,
        hands_pred2d=hands_pred2d,
        obj_parts=obj_parts,
        obj_transforms=obj_transforms,
        obj_valid=obj_valid,
        move_labels=move_labels,
        look_at=look_at,
        cam_pos=cam_pos,
        grid_pos=grid_pos,
    )


def _decode_rgb(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr[..., ::-1]


def _se3_to_wxyz_pos(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_mat = np.asarray(T[:3, :3], dtype=np.float64)
    U, _, Vt = np.linalg.svd(R_mat)
    R_orth = U @ Vt
    if np.linalg.det(R_orth) < 0:
        U[:, -1] *= -1
        R_orth = U @ Vt
    return vtf.SO3.from_matrix(R_orth).wxyz, np.asarray(T[:3, 3], dtype=np.float64)


def run_viser_server(
    bundles: list[SceneBundle],
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Start an interactive `viser` web server displaying 3D reconstructions and original camera views."""
    if not bundles:
        raise ValueError("At least one SceneBundle is required")

    server = viser.ViserServer(host=host, port=port, label="HandQuery Viewer")
    by_name = {b.name: b for b in bundles}
    state: dict[str, Any] = {
        "scene": bundles[0],
        "frame": 0,
        "playing": True,
        "hand_handles": {},
        "skel_handles": {},
        "obj_handles": [],
        "frustum_handles": [],
        "updating_slider": False,
    }

    with server.gui.add_folder("Scene & Playback", expand_by_default=True):
        scene_dd = server.gui.add_dropdown(
            "Sequence",
            options=[b.name for b in bundles],
            initial_value=bundles[0].name,
            visible=len(bundles) > 1,
        )
        play_btn = server.gui.add_button("Pause")
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, bundles[0].num_frames - 1),
            step=1,
            initial_value=0,
        )
        speed_dd = server.gui.add_dropdown(
            "Speed",
            options=["0.25x", "0.5x", "1.0x"],
            initial_value="1.0x",
        )
        info_md = server.gui.add_markdown("")

    with server.gui.add_folder("3D & Overlay Layers", expand_by_default=True):
        cb_hands = server.gui.add_checkbox("3D Hand Meshes", initial_value=True)
        cb_skel = server.gui.add_checkbox("3D 21-Joint Skeleton", initial_value=False)
        cb_object = server.gui.add_checkbox("3D Object Mesh", initial_value=True)
        cb_frustums = server.gui.add_checkbox("3D Camera Frustums", initial_value=True)
        frustum_scale = server.gui.add_slider("Frustum Scale", min=0.05, max=0.35, step=0.01, initial_value=0.14)
        cb_overlay2d = server.gui.add_checkbox("Overlay Predictions on Camera Images", initial_value=False)
        cb_grid = server.gui.add_checkbox("Tabletop Reference Grid", initial_value=True)

    with server.gui.add_folder("Original Multi-View Camera Feeds", expand_by_default=True):
        dummy_grid = np.zeros((240, 320, 3), dtype=np.uint8)
        gui_mv_image = server.gui.add_image(dummy_grid, label="Synchronized Views", format="jpeg", jpeg_quality=85)

    def _build_scene_graph(bundle: SceneBundle) -> None:
        server.scene.reset()
        server.scene.set_up_direction(bundle.up)

        state["grid_handle"] = server.scene.add_grid(
            "/table_grid",
            width=1.0,
            height=1.0,
            plane="xy",
            cell_size=0.05,
            section_size=0.25,
            cell_color=(195, 198, 205),
            section_color=(135, 140, 152),
            position=bundle.grid_pos,
            visible=cb_grid.value,
        )

        state["hand_handles"] = {}
        state["skel_handles"] = {}
        for side, fit in bundle.hands_fit.items():
            v0 = np.nan_to_num(fit["verts"][0]).astype(np.float32)
            faces = fit["faces"].astype(np.uint32)
            state["hand_handles"][side] = server.scene.add_mesh_simple(
                f"/hands/{side}_mesh",
                vertices=v0,
                faces=faces,
                color=HAND_BASE_RGB.get(side, (180, 200, 255)),
                side="double",
                visible=cb_hands.value,
            )
            j0 = np.nan_to_num(fit["joints21"][0]).astype(np.float32)
            segs = np.stack([[j0[a], j0[b]] for a, b in BONE_PAIRS_21], axis=0)
            state["skel_handles"][side] = server.scene.add_line_segments(
                f"/hands/{side}_skeleton",
                points=segs,
                colors=HAND_BASE_RGB.get(side, (180, 200, 255)),
                thickness=3.0,
                thickness_units="screen",
                visible=cb_skel.value,
            )

        state["obj_handles"] = []
        for k, part_mesh in enumerate(bundle.obj_parts):
            h = server.scene.add_mesh_trimesh(
                f"/object/part_{k:02d}",
                mesh=part_mesh,
                visible=cb_object.value,
            )
            state["obj_handles"].append(h)

        state["frustum_handles"] = []
        C = bundle.K_disp.shape[0]
        for c in range(C):
            Rc = bundle.R[c]
            tc = bundle.t[c]
            R_c2w = Rc.T
            pos_c2w = -R_c2w @ tc
            wxyz = vtf.SO3.from_matrix(R_c2w).wxyz
            K_c0 = bundle.K_disp_frames[c, 0] if bundle.K_disp_frames is not None else bundle.K_disp[c]
            fovy = float(2.0 * math.atan(bundle.disp_h / (2.0 * max(K_c0[1, 1], 1e-6))))
            aspect = float(bundle.disp_w) / float(bundle.disp_h)
            img0 = _decode_rgb(bundle.jpeg_frames[c][0])
            fh = server.scene.add_camera_frustum(
                f"/cameras/cam_{c:02d}",
                fov=fovy,
                aspect=aspect,
                scale=float(frustum_scale.value),
                thickness=1.8,
                thickness_units="screen",
                color=(55, 65, 85),
                image=img0,
                format="jpeg",
                jpeg_quality=82,
                wxyz=wxyz,
                position=pos_c2w,
                visible=cb_frustums.value,
            )

            def _make_click_cb(wxyz_c=wxyz, pos_c=pos_c2w, look_c=bundle.look_at):
                def _on_click(event: viser.SceneNodePointerEvent) -> None:
                    with event.client.atomic():
                        event.client.camera.wxyz = wxyz_c
                        event.client.camera.position = pos_c
                        event.client.camera.look_at = look_c

                return _on_click

            fh.on_click(_make_click_cb())
            state["frustum_handles"].append(fh)

        for client in server.get_clients().values():
            client.camera.position = bundle.cam_pos
            client.camera.look_at = bundle.look_at

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        b: SceneBundle = state["scene"]
        client.camera.position = b.cam_pos
        client.camera.look_at = b.look_at

    def _render_frame(f_idx: int) -> None:
        bundle: SceneBundle = state["scene"]
        f_idx = max(0, min(f_idx, bundle.num_frames - 1))
        state["frame"] = f_idx

        with server.atomic():
            for side, fit in bundle.hands_fit.items():
                obs = bool(fit["observed"][f_idx]) if "observed" in fit else True
                verts_f = fit["verts"][f_idx].astype(np.float32)
                j21_f = fit["joints21"][f_idx].astype(np.float32)
                obs = obs and bool(np.isfinite(verts_f[0, 0]))

                hh = state["hand_handles"].get(side)
                if hh is not None:
                    hh.visible = bool(cb_hands.value and obs)
                    if obs:
                        hh.vertices = verts_f

                sh = state["skel_handles"].get(side)
                if sh is not None:
                    sh.visible = bool(cb_skel.value and obs)
                    if obs:
                        sh.points = np.stack([[j21_f[a], j21_f[b]] for a, b in BONE_PAIRS_21], axis=0)

            if state["obj_handles"] and bundle.obj_transforms is not None:
                valid_o = bool(bundle.obj_valid[f_idx]) if bundle.obj_valid is not None else True
                show_o = bool(cb_object.value and valid_o)
                T_parts = bundle.obj_transforms[f_idx]
                for k, oh in enumerate(state["obj_handles"]):
                    oh.visible = show_o
                    if show_o:
                        wxyz_o, pos_o = _se3_to_wxyz_pos(T_parts[k])
                        oh.wxyz = wxyz_o
                        oh.position = pos_o

            C = bundle.K_disp.shape[0]
            cam_tiles = []
            for c in range(C):
                img = _decode_rgb(bundle.jpeg_frames[c][f_idx])
                if cb_overlay2d.value:
                    K_cf = bundle.K_disp_frames[c, f_idx] if bundle.K_disp_frames is not None else bundle.K_disp[c]
                    if bundle.hands_fit:
                        meshes = []
                        for side, fit in bundle.hands_fit.items():
                            if bool(fit["observed"][f_idx]) and np.isfinite(fit["verts"][f_idx, 0, 0]):
                                meshes.append((fit["verts"][f_idx], fit["faces"], HAND_BASE_RGB[side]))
                        img = render_meshes_on_image(img, meshes, K_cf, bundle.R[c], bundle.t[c], alpha=0.68)
                    else:
                        for side, p2d in bundle.hands_pred2d.items():
                            if bool(p2d["valid"][c, f_idx]):
                                lm_cf = p2d["lm"][c, f_idx]
                                if bundle.affine_disp_frames is not None:
                                    A_cf = bundle.affine_disp_frames[c, f_idx]
                                    lm_cf = lm_cf @ A_cf[:2, :2].T + A_cf[:2, 2]
                                else:
                                    lm_cf = lm_cf * bundle.disp_scale
                                img = render_landmarks_on_image(
                                    img,
                                    lm_cf,
                                    p2d["vis"][c, f_idx],
                                    color=HAND_BASE_RGB[side],
                                )
                if c < len(state["frustum_handles"]):
                    fh = state["frustum_handles"][c]
                    fh.visible = cb_frustums.value
                    if cb_frustums.value:
                        fh.image = img

                tile = img.copy()
                cv2.putText(
                    tile,
                    f"cam {c:02d}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cam_tiles.append(tile)

            cols = 2 if C > 1 else 1
            rows = int(math.ceil(C / cols))
            while len(cam_tiles) < rows * cols:
                cam_tiles.append(np.zeros_like(cam_tiles[0]))
            grid_img = np.vstack([np.hstack(cam_tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
            gui_mv_image.image = grid_img

            t_sec = f_idx / max(bundle.fps, 1e-6)
            tot_sec = (bundle.num_frames - 1) / max(bundle.fps, 1e-6)
            extra = f" | Move: **`{bundle.move_labels[f_idx]}`**" if f_idx in bundle.move_labels else ""
            info_md.content = (
                f"**{bundle.name}** — Frame `{f_idx}/{bundle.num_frames - 1}` "
                f"(`{t_sec:.2f}s / {tot_sec:.2f}s`){extra}\n\n"
                "*Tip: Click any 3D camera frustum to view from that camera.*"
            )

    @play_btn.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]
        play_btn.label = "Pause" if state["playing"] else "Play"

    @frame_slider.on_update
    def _(_) -> None:
        if state["updating_slider"]:
            return
        _render_frame(int(frame_slider.value))

    @scene_dd.on_update
    def _(_) -> None:
        new_bundle = by_name[scene_dd.value]
        state["scene"] = new_bundle
        state["updating_slider"] = True
        frame_slider.max = max(0, new_bundle.num_frames - 1)
        frame_slider.value = 0
        state["updating_slider"] = False
        _build_scene_graph(new_bundle)
        _render_frame(0)

    for cb in (cb_hands, cb_skel, cb_object, cb_frustums, cb_overlay2d):
        @cb.on_update
        def _(_) -> None:
            _render_frame(state["frame"])

    @cb_grid.on_update
    def _(_) -> None:
        gh = state.get("grid_handle")
        if gh is not None:
            gh.visible = cb_grid.value

    @frustum_scale.on_update
    def _(_) -> None:
        s = float(frustum_scale.value)
        for fh in state["frustum_handles"]:
            fh.scale = s

    _build_scene_graph(bundles[0])
    _render_frame(0)

    speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1.0x": 1.0}
    print(f"[viser] HandQuery interactive viewer running at http://localhost:{port}")
    try:
        while True:
            if state["playing"]:
                bundle = state["scene"]
                spd = speed_map.get(speed_dd.value, 1.0)
                nxt = (state["frame"] + 1) % bundle.num_frames
                state["updating_slider"] = True
                frame_slider.value = nxt
                state["updating_slider"] = False
                _render_frame(nxt)
                time.sleep(1.0 / max(bundle.fps * spd, 1.0))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[viser] Shutting down viewer.")
