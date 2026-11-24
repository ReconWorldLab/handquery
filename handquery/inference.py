"""Sliding-window multi-view HandQuery inference and end-to-end 4D reconstruction pipeline."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import os
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from handquery.fit import fit_sequence
from handquery.geometry import crop_to_original, triangulate_point, virtual_camera
from handquery.model import load_model
from handquery.model.network import HandQueryNet


class FrameSource:
    """Unified multi-view frame reader supporting RGB arrays, per-frame image paths, or video files."""

    def __init__(self, source: np.ndarray | Sequence[Any], cache_size: int = 64):
        self._caps: list[cv2.VideoCapture | None] = []
        self._cap_pos: list[int] = []
        self._cache: OrderedDict[tuple[int, int], np.ndarray | None] = OrderedDict()
        self._cache_size = cache_size

        if isinstance(source, np.ndarray):
            if source.ndim != 5 or source.shape[-1] != 3 or source.dtype != np.uint8:
                raise ValueError(
                    f"In-memory image array must be uint8 RGB of shape (C, T, H, W, 3), got {source.shape} ({source.dtype})"
                )
            self._kind = "array_rgb"
            self._array = source
            self.num_cams, self.num_frames = int(source.shape[0]), int(source.shape[1])
        elif isinstance(source, (list, tuple)) and len(source) > 0:
            first = source[0]
            if isinstance(first, (str, os.PathLike)):
                self._kind = "videos"
                self._video_paths = [os.path.expanduser(os.fspath(p)) for p in source]
                self.num_cams = len(self._video_paths)
                counts = []
                for vp in self._video_paths:
                    if not os.path.isfile(vp):
                        raise FileNotFoundError(f"Video file not found: {vp}")
                    cap = cv2.VideoCapture(vp)
                    if not cap.isOpened():
                        raise RuntimeError(f"Failed to open video file: {vp}")
                    counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                    self._caps.append(cap)
                    self._cap_pos.append(0)
                self.num_frames = min(counts) if counts and min(counts) > 0 else 0
            elif isinstance(first, (list, tuple)):
                self._kind = "paths"
                self._paths = [[os.path.expanduser(os.fspath(p)) if p else "" for p in cam_list] for cam_list in source]
                self.num_cams = len(self._paths)
                self.num_frames = len(self._paths[0])
                if any(len(cam_list) != self.num_frames for cam_list in self._paths):
                    raise ValueError("All camera frame-path lists must have the same length")
            elif isinstance(first, np.ndarray):
                arr = np.stack(source, axis=0)
                if arr.ndim != 5 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
                    raise ValueError(f"Stacked camera arrays must have shape (C, T, H, W, 3) uint8, got {arr.shape}")
                self._kind = "array_rgb"
                self._array = arr
                self.num_cams, self.num_frames = int(arr.shape[0]), int(arr.shape[1])
            else:
                raise TypeError(f"Unsupported frame source element type: {type(first)}")
        else:
            raise ValueError("Frame source must be a non-empty (C, T, H, W, 3) uint8 array, video list, or path list")

    def get_rgb(self, cam_idx: int, frame_idx: int) -> np.ndarray | None:
        """Return an RGB uint8 image `(H, W, 3)` for `(cam_idx, frame_idx)`, or `None` if missing."""
        if self._kind == "array_rgb":
            return self._array[cam_idx, frame_idx]
        bgr = self.get_bgr(cam_idx, frame_idx)
        return None if bgr is None else bgr[..., ::-1]

    def get_bgr(self, cam_idx: int, frame_idx: int) -> np.ndarray | None:
        """Return a BGR uint8 image `(H, W, 3)` for `(cam_idx, frame_idx)`, or `None` if missing."""
        if self._kind == "array_rgb":
            return self._array[cam_idx, frame_idx][..., ::-1]
        key = (cam_idx, frame_idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if self._kind == "paths":
            p = self._paths[cam_idx][frame_idx]
            img = cv2.imread(p, cv2.IMREAD_COLOR) if (p and os.path.isfile(p)) else None
        else:
            cap = self._caps[cam_idx]
            assert cap is not None
            if self._cap_pos[cam_idx] != frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, img = cap.read()
            if ok:
                self._cap_pos[cam_idx] = frame_idx + 1
            else:
                img = None
                self._cap_pos[cam_idx] = -1

        self._cache[key] = img
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img

    def close(self) -> None:
        for cap in self._caps:
            if cap is not None:
                cap.release()
        self._caps.clear()
        self._cache.clear()


def sliding_windows(num_frames: int, window_size: int = 48, stride: int = 24) -> list[np.ndarray]:
    """Generate overlapping frame index windows of length `min(window_size, num_frames)` covering `0..num_frames-1`."""
    if num_frames <= 0:
        return []
    span = min(window_size, num_frames)
    last_start = max(num_frames - span, 0)
    starts = list(range(0, last_start + 1, max(1, stride)))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [np.arange(s, s + span, dtype=np.int64) for s in starts]


def prepare_window(
    frame_source: FrameSource,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    boxes: np.ndarray,
    frames: np.ndarray,
    side: str,
    valid_seq: np.ndarray,
    image_size: int = 256,
    crop_margin: float = 1.4,
    cam_subset: Sequence[int] | None = None,
    img_affine: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Construct calibrated virtual-camera perspective crops for one temporal window."""
    T = len(frames)
    valid = valid_seq[frames]
    if not valid.any():
        return None

    n_cams = K.shape[0]
    cams = np.arange(n_cams)
    boxes_win = boxes[:, frames]  # (C, T, 4)
    has_box = ~np.isnan(boxes_win[..., 0]) & valid[None]
    cams = cams[has_box.any(axis=1)]
    if cam_subset is not None:
        allowed = set(np.asarray(cam_subset, dtype=int).tolist())
        cams = np.array([c for c in cams if int(c) in allowed], dtype=np.int64)
    V = len(cams)
    if V == 0:
        return None

    centres = []
    for ti in range(T):
        Ps, uvs = [], []
        for c in cams:
            b = boxes_win[c, ti]
            if not np.isnan(b[0]):
                Ps.append(K[c] @ np.hstack([R[c], t[c][:, None]]))
                uvs.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        if len(Ps) >= 2:
            centres.append(triangulate_point(Ps, uvs))
    ref = int(cams[0])
    R_ref = R[ref].astype(np.float64)
    origin = np.median(np.asarray(centres), axis=0) if centres else (-R_ref.T @ t[ref].astype(np.float64))

    images = np.zeros((V, T, image_size, image_size, 3), dtype=np.uint8)
    Hs = np.zeros((V, T, 3, 3), dtype=np.float32)
    K_crop = np.zeros((V, T, 3, 3), dtype=np.float32)
    T_crop = np.broadcast_to(np.eye(4, dtype=np.float32), (V, T, 4, 4)).copy()
    missing = np.zeros((V, T), dtype=bool)

    for vi, c in enumerate(cams):
        Kc = K[c].astype(np.float64)
        Rc = R[c].astype(np.float64)
        tc = t[c].astype(np.float64)
        b = boxes_win[c].copy()
        ok = ~np.isnan(b[:, 0])
        idx_ok = np.flatnonzero(ok)
        b = b[idx_ok[np.abs(frames[:, None] - frames[idx_ok][None]).argmin(axis=1)]]
        crop_px = float(crop_margin * np.max(np.maximum(b[ok, 2] - b[ok, 0], b[ok, 3] - b[ok, 1])))
        ctr = np.stack([(b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0], axis=1)

        for ti, f in enumerate(frames):
            R_v, K_v, H = virtual_camera(Kc, ctr[ti], crop_px, image_size)
            rep = ti > 0 and f == frames[ti - 1] and np.array_equal(ctr[ti], ctr[ti - 1])
            if rep:
                images[vi, ti] = images[vi, ti - 1]
                missing[vi, ti] = missing[vi, ti - 1]
            else:
                img_bgr = frame_source.get_bgr(int(c), int(f))
                missing[vi, ti] = img_bgr is None
                if img_bgr is not None:
                    Hi = H if img_affine is None else (H @ np.linalg.inv(img_affine[c, f]))
                    images[vi, ti] = cv2.warpPerspective(
                        img_bgr,
                        Hi,
                        (image_size, image_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )[..., ::-1]
            R_wv = R_v @ Rc
            K_crop[vi, ti] = K_v
            T_crop[vi, ti, :3, :3] = R_wv
            T_crop[vi, ti, :3, 3] = R_v @ tc
            Hs[vi, ti] = H

    return {
        "images": images,
        "K_crop": K_crop,
        "T_crop": T_crop,
        "origin": origin.astype(np.float32),
        "side": np.int64(0 if side.lower() == "right" else 1),
        "H": Hs,
        "cams": cams.astype(np.int64),
        "frames": frames.astype(np.int64),
        "missing": missing,
    }


def predict_multiview_sequence(
    model: HandQueryNet,
    images_or_source: FrameSource | np.ndarray | Sequence[Any],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    boxes: np.ndarray,
    side: str,
    window_size: int = 48,
    stride: int = 24,
    image_size: int = 256,
    crop_margin: float = 1.4,
    vert_ids: Sequence[int] | np.ndarray | None = None,
    valid_frames: np.ndarray | None = None,
    cam_subset: Sequence[int] | None = None,
    img_affine: np.ndarray | None = None,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Run sliding-window dense 2D vertex prediction and `1 / sigma^2` window fusion over a sequence.

    Returns:
        Dictionary containing:
          - `lm`: `(C, T_seq, N, 2)` fused 2D vertex positions in original image pixels
          - `sigma`: `(C, T_seq, N)` fused pixel uncertainties
          - `vis`: `(C, T_seq, N)` fused visibility probabilities in `[0, 1]`
          - `valid`: `(C, T_seq)` boolean mask of `(camera, frame)` pairs with valid observations
          - `vert_ids`: `(N,)` MANO vertex indices
    """
    owns_source = not isinstance(images_or_source, FrameSource)
    src = FrameSource(images_or_source) if owns_source else images_or_source
    try:
        K = np.asarray(K)
        R = np.asarray(R)
        t = np.asarray(t)
        boxes = np.asarray(boxes, dtype=np.float32)
        C, n_frames = boxes.shape[:2]
        if K.shape != (C, 3, 3) or R.shape != (C, 3, 3) or t.shape != (C, 3):
            raise ValueError(
                f"Expected K=(C,3,3), R=(C,3,3), t=(C,3) matching C={C} from boxes={boxes.shape}; "
                f"got K={K.shape}, R={R.shape}, t={t.shape}"
            )
        if src.num_cams != C:
            raise ValueError(f"Frame source has {src.num_cams} cameras, but boxes has {C}")
        if src.num_frames and src.num_frames < n_frames:
            raise ValueError(f"Frame source has only {src.num_frames} frames, fewer than boxes length {n_frames}")

        ids = np.arange(model.per_hand, dtype=np.int64) if vert_ids is None else np.asarray(vert_ids, dtype=np.int64)
        N = len(ids)
        dev = torch.device(device) if device is not None else next(model.parameters()).device

        if valid_frames is None:
            valid_seq = (~np.isnan(boxes[..., 0])).any(axis=0)
        else:
            valid_seq = np.asarray(valid_frames, dtype=bool)
            if valid_seq.shape != (n_frames,):
                raise ValueError(f"valid_frames must have shape ({n_frames},), got {valid_seq.shape}")

        wins = sliding_windows(n_frames, window_size=window_size, stride=stride)
        acc_w = np.zeros((C, n_frames, N), dtype=np.float64)
        acc_uv = np.zeros((C, n_frames, N, 2), dtype=np.float64)
        acc_s = np.zeros((C, n_frames, N), dtype=np.float64)
        acc_v = np.zeros((C, n_frames, N), dtype=np.float64)
        nwin = np.zeros((C, n_frames), dtype=np.float64)

        amp_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if dev.type == "cuda"
            else nullcontext()
        )

        for frames in wins:
            if not valid_seq[frames].any():
                continue
            nwin[:, frames] += 1.0
            smp = prepare_window(
                frame_source=src,
                K=K,
                R=R,
                t=t,
                boxes=boxes,
                frames=frames,
                side=side,
                valid_seq=valid_seq,
                image_size=image_size,
                crop_margin=crop_margin,
                cam_subset=cam_subset,
                img_affine=img_affine,
            )
            if smp is None:
                continue

            batch = {
                k: torch.as_tensor(smp[k])[None].to(dev)
                for k in ("images", "K_crop", "T_crop", "origin", "side", "H")
            }
            with amp_ctx:
                uv, sig, vis = model.predict(batch, vert_ids=ids)

            xy, so = crop_to_original(uv[0].float(), sig[0].float(), batch["H"][0], float(image_size))
            xy_np = xy.cpu().numpy()
            so_np = so.cpu().numpy()
            vp_np = vis[0].float().cpu().numpy()
            wcams = smp["cams"]
            miss = smp["missing"]

            w = 1.0 / (np.clip(so_np, 0.5, None) ** 2)
            if miss.any():
                w = np.where(miss[..., None], 0.0, w)
                vp_np = np.where(miss[..., None], 0.0, vp_np)

            for vi, c in enumerate(wcams):
                acc_w[c, frames] += w[vi]
                acc_uv[c, frames] += w[vi][..., None] * xy_np[vi]
                acc_s[c, frames] += w[vi] * so_np[vi]
                acc_v[c, frames] += vp_np[vi]

        cnt = acc_w > 0
        lm = np.where(cnt[..., None], acc_uv / np.maximum(acc_w, 1e-12)[..., None], 0.0).astype(np.float32)
        sigma = np.where(cnt, acc_s / np.maximum(acc_w, 1e-12), 50.0).astype(np.float32)
        vis_out = np.where(cnt, acc_v / np.maximum(nwin[..., None], 1.0), 0.0).astype(np.float32)
        valid_out = cnt.any(axis=-1) & valid_seq[None] & ~np.isnan(boxes[..., 0])

        return {
            "lm": lm,
            "sigma": sigma,
            "vis": vis_out,
            "valid": valid_out,
            "vert_ids": ids,
        }
    finally:
        if owns_source:
            src.close()


class HandQueryPipeline:
    """High-level multi-view 4D hand reconstruction pipeline."""

    def __init__(
        self,
        model: HandQueryNet,
        config: dict[str, Any],
        mano_root: str | os.PathLike | None = None,
        device: str | torch.device = "cuda",
    ):
        self.model = model
        self.config = config
        self.mano_root = None if mano_root is None else os.path.expanduser(os.fspath(mano_root))
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str | os.PathLike,
        mano_root: str | os.PathLike | None = None,
        device: str | torch.device | None = None,
    ) -> "HandQueryPipeline":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg = load_model(ckpt_path, device=device)
        return cls(model=model, config=cfg, mano_root=mano_root, device=device)

    def reconstruct(
        self,
        images_or_source: FrameSource | np.ndarray | Sequence[Any],
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        boxes: np.ndarray,
        side: str = "right",
        window_size: int | None = None,
        stride: int | None = None,
        fit_mano: bool = True,
        tips: Sequence[int] | str = "smplx",
        fit_iters: tuple[int, int, int] = (150, 300, 200),
        temporal_w: float = 0.0,
        beta_w: float = 0.0,
        valid_frames: np.ndarray | None = None,
        cam_subset: Sequence[int] | None = None,
        img_affine: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Predict dense 2D vertices across views and optionally fit a metric 4D MANO mesh sequence."""
        win = int(window_size or self.config.get("window_size", 48))
        st = int(stride if stride is not None else max(1, win // 2))
        img_size = int(self.config.get("image_size", 256))
        margin = float(self.config.get("crop_margin", 1.4))

        pred = predict_multiview_sequence(
            model=self.model,
            images_or_source=images_or_source,
            K=K,
            R=R,
            t=t,
            boxes=boxes,
            side=side,
            window_size=win,
            stride=st,
            image_size=img_size,
            crop_margin=margin,
            valid_frames=valid_frames,
            cam_subset=cam_subset,
            img_affine=img_affine,
            device=self.device,
        )
        result: dict[str, Any] = dict(pred)
        result["side"] = side.lower()
        result["observed"] = pred["valid"].astype(bool).sum(axis=0) >= 2

        if fit_mano:
            if self.mano_root is None:
                raise ValueError("`mano_root` is required when `fit_mano=True` (pass `--no-fit` for 2D-only inference)")
            C = np.asarray(K).shape[0]
            E = np.tile(np.eye(4, dtype=np.float32), (C, 1, 1))
            E[:, :3, :3] = np.asarray(R, dtype=np.float32)
            E[:, :3, 3] = np.asarray(t, dtype=np.float32)
            fit_res = fit_sequence(
                lm=pred["lm"],
                sig=pred["sigma"],
                vis=pred["vis"],
                valid=pred["valid"].astype(np.float32),
                Ks=list(np.asarray(K, dtype=np.float32)),
                Es=list(E),
                side=side,
                mano_root=self.mano_root,
                lm_idx=pred["vert_ids"],
                tips=tips,
                iters=fit_iters,
                temporal_w=temporal_w,
                beta_w=beta_w,
                device=self.device,
            )
            result.update(fit_res)
        return result
