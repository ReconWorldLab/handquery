#!/usr/bin/env python3
"""Command-line multi-view 4D hand reconstruction with HandQuery.

Supports two convenient input formats via `--input`:
  1. A JSON scene manifest (`scene.json`):
     {
       "fps": 30.0,
       "K": [[[fx, 0, cx], [0, fy, cy], [0, 0, 1]], ...],     // (C, 3, 3) pixels
       "R": [[...], ...],                                     // (C, 3, 3) world -> cam
       "t": [[tx, ty, tz], ...],                              // (C, 3) world -> cam (meters)
       "videos": ["cam0.mp4", "cam1.mp4", ...],               // OR "frames": [["c0_0.jpg", ...], ...]
       "hands": {
         "right": "boxes_right.npy",                          // (C, T, 4) [x0, y0, x1, y1] or nested list
         "left": "boxes_left.npy"                             // optional second hand
       }
     }

  2. A self-contained `.npz` sequence bundle (`sequence.npz`) containing:
     - `K`: (C, 3, 3), `R`: (C, 3, 3), `t`: (C, 3)
     - One of `images` ((C, T, H, W, 3) uint8 RGB), `videos` ((C,) paths), `frames` ((C, T) paths),
       or `img_pattern` ((C,) format strings with optional `img_frame_ids` / `img_offset` / `img_affine`)
     - `boxes` ((C, T, 4) with `side` = 'right'/'left') OR `boxes_right` / `boxes_left` ((C, T, 4))
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handquery import FrameSource, HandQueryPipeline  # noqa: E402
from handquery.vis import export_obj_sequence, write_multiview_video  # noqa: E402


def _resolve_rel(base_dir: str, path_or_val: str) -> str:
    expanded = os.path.expanduser(path_or_val)
    return expanded if os.path.isabs(expanded) else os.path.normpath(os.path.join(base_dir, expanded))


def _load_array_or_path(base_dir: str, val: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(val, str):
        p = _resolve_rel(base_dir, val)
        return np.asarray(np.load(p), dtype=dtype)
    return np.asarray(val, dtype=dtype)


def load_scene_input(input_path: str, side_override: str | None = None) -> dict[str, Any]:
    """Parse a `.json` scene manifest or `.npz` sequence bundle into normalized camera, frame, and box inputs."""
    input_path = os.path.expanduser(input_path)
    base_dir = os.path.dirname(os.path.abspath(input_path))
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        K = _load_array_or_path(base_dir, meta["K"], dtype=np.float64)
        R = _load_array_or_path(base_dir, meta["R"], dtype=np.float64)
        t = _load_array_or_path(base_dir, meta["t"], dtype=np.float64)
        fps = float(meta.get("fps", 30.0))
        img_affine = (
            _load_array_or_path(base_dir, meta["img_affine"], dtype=np.float64) if "img_affine" in meta else None
        )
        valid_frames = (
            _load_array_or_path(base_dir, meta["valid"], dtype=bool) if "valid" in meta else None
        )

        if "videos" in meta:
            source_raw = [_resolve_rel(base_dir, vp) for vp in meta["videos"]]
        elif "frames" in meta:
            source_raw = [[_resolve_rel(base_dir, fp) if fp else "" for fp in cam_f] for cam_f in meta["frames"]]
        elif "images" in meta:
            source_raw = _load_array_or_path(base_dir, meta["images"], dtype=np.uint8)
        else:
            raise ValueError("JSON manifest must specify one of 'videos', 'frames', or 'images'")

        hands: dict[str, np.ndarray] = {}
        if "hands" in meta:
            for s_name, b_val in meta["hands"].items():
                hands[s_name.lower()] = _load_array_or_path(base_dir, b_val, dtype=np.float32)
        elif "boxes" in meta:
            s_name = (side_override or meta.get("side", "right")).lower()
            hands[s_name] = _load_array_or_path(base_dir, meta["boxes"], dtype=np.float32)
        else:
            raise ValueError("JSON manifest must specify 'hands' ({'right': ..., 'left': ...}) or 'boxes'")

    elif ext == ".npz":
        z = np.load(input_path, allow_pickle=True)
        K = np.asarray(z["K"])
        R = np.asarray(z["R"])
        t = np.asarray(z["t"])
        fps = float(z["fps"]) if "fps" in z.files else 30.0
        img_affine = np.asarray(z["img_affine"]) if "img_affine" in z.files else None
        valid_frames = np.asarray(z["valid"], dtype=bool) if "valid" in z.files else None

        hands = {}
        for s_name in ("right", "left"):
            key = f"boxes_{s_name}"
            if key in z.files:
                hands[s_name] = np.asarray(z[key], dtype=np.float32)
        if not hands and "boxes" in z.files:
            s_default = str(z["side"]) if "side" in z.files else "right"
            s_name = (side_override or s_default).lower()
            hands[s_name] = np.asarray(z["boxes"], dtype=np.float32)
        if not hands:
            raise ValueError("NPZ input must contain 'boxes' or 'boxes_right' / 'boxes_left'")

        n_frames = next(iter(hands.values())).shape[1]
        if "images" in z.files:
            source_raw = np.asarray(z["images"], dtype=np.uint8)
        elif "videos" in z.files:
            source_raw = [_resolve_rel(base_dir, str(vp)) for vp in z["videos"]]
        elif "frames" in z.files:
            source_raw = [[_resolve_rel(base_dir, str(fp)) if fp else "" for fp in row] for row in z["frames"]]
        elif "img_pattern" in z.files:
            offset = int(z["img_offset"]) if "img_offset" in z.files else 0
            fids = (
                [int(x) for x in z["img_frame_ids"]]
                if "img_frame_ids" in z.files
                else [f + offset for f in range(n_frames)]
            )
            source_raw = []
            for pat_raw in z["img_pattern"]:
                pat = str(pat_raw)
                if "InterHand" in pat and pat.endswith("image{}.jpg"):
                    pat = pat[: -len("image{}.jpg")] + "image{:04d}.jpg"
                source_raw.append([_resolve_rel(base_dir, pat.format(fid)) for fid in fids])
        else:
            raise ValueError("NPZ input must contain one of 'images', 'videos', 'frames', or 'img_pattern'")
    else:
        raise ValueError(f"Unsupported input format {ext!r}; expected '.json' or '.npz'")

    if side_override and side_override != "both":
        if side_override not in hands:
            if len(hands) == 1:
                only_boxes = next(iter(hands.values()))
                hands = {side_override: only_boxes}
            else:
                raise ValueError(f"Requested --side {side_override}, but input only contains {sorted(hands)}")
        else:
            hands = {side_override: hands[side_override]}

    return {
        "source": FrameSource(source_raw),
        "K": K,
        "R": R,
        "t": t,
        "hands": hands,
        "fps": fps,
        "img_affine": img_affine,
        "valid_frames": valid_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HandQuery multi-view 4D hand reconstruction CLI.")
    parser.add_argument("--input", required=True, help="Path to scene.json manifest or sequence.npz bundle.")
    parser.add_argument(
        "--ckpt",
        default="checkpoints/handquery_vitb.pt",
        help="Path to HandQuery model checkpoint (default: checkpoints/handquery_vitb.pt).",
    )
    parser.add_argument(
        "--mano-root",
        default="assets",
        help="Directory containing mano/MANO_{RIGHT,LEFT}.pkl (required unless --no-fit is set).",
    )
    parser.add_argument("--out", required=True, help="Output directory for predictions and meshes.")
    parser.add_argument(
        "--side",
        choices=["auto", "right", "left", "both"],
        default="auto",
        help="Hand side(s) to reconstruct (default: auto from input).",
    )
    parser.add_argument("--window-size", type=int, default=None, help="Sliding-window clip length T (default: 48).")
    parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride (default: T // 2 = 24).")
    parser.add_argument("--cams", type=int, nargs="*", default=None, help="Optional subset of camera indices to use.")
    parser.add_argument("--no-fit", action="store_true", help="Skip 3D MANO fitting and output 2D dense predictions only.")
    parser.add_argument(
        "--tips",
        default="smplx",
        choices=["smplx", "dexycb", "ho3d", "oakink", "arctic", "interhand"],
        help="Fingertip vertex convention for 21-joint output (default: smplx).",
    )
    parser.add_argument("--iters", type=int, nargs=3, default=[150, 300, 200], help="3-stage MANO fitting iterations.")
    parser.add_argument("--temporal-w", type=float, default=0.0, help="Optional temporal smoothness weight in MANO fitting.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu).")
    parser.add_argument("--render", action="store_true", help="Render a multi-view overlay video (overlay.mp4).")
    parser.add_argument("--export-obj", action="store_true", help="Export per-frame 3D MANO meshes as Wavefront .obj files.")
    args = parser.parse_args()

    side_override = None if args.side == "auto" else args.side
    scene = load_scene_input(args.input, side_override=side_override)
    src: FrameSource = scene["source"]

    try:
        pipe = HandQueryPipeline.from_pretrained(
            ckpt_path=args.ckpt,
            mano_root=None if args.no_fit else args.mano_root,
            device=args.device,
        )
        os.makedirs(args.out, exist_ok=True)
        hand_results: dict[str, dict[str, Any]] = {}

        for side, boxes in scene["hands"].items():
            t0 = time.time()
            res = pipe.reconstruct(
                images_or_source=src,
                K=scene["K"],
                R=scene["R"],
                t=scene["t"],
                boxes=boxes,
                side=side,
                window_size=args.window_size,
                stride=args.stride,
                fit_mano=not args.no_fit,
                tips=args.tips,
                fit_iters=tuple(args.iters),
                temporal_w=args.temporal_w,
                valid_frames=scene["valid_frames"],
                cam_subset=args.cams,
                img_affine=scene["img_affine"],
            )
            elapsed = time.time() - t0
            hand_results[side] = res

            prefix = f"{side}_" if len(scene["hands"]) > 1 else ""
            np.savez_compressed(
                os.path.join(args.out, f"{prefix}pred2d.npz"),
                lm=res["lm"],
                sigma=res["sigma"],
                vis=res["vis"],
                valid=res["valid"],
                vert_ids=res["vert_ids"],
                observed=res["observed"],
            )

            if not args.no_fit:
                np.savez_compressed(
                    os.path.join(args.out, f"{prefix}mano_fit.npz"),
                    verts=res["verts"],
                    joints21=res["joints21"],
                    faces=res["faces"],
                    betas=res["betas"],
                    global_orient=res["global_orient"],
                    pca=res["pca"],
                    hand_pose=res["hand_pose"],
                    transl=res["transl"],
                    observed=res["observed"],
                    reproj_px=np.float32(res["reproj_px"]),
                    tri_ok_frac=np.float32(res["tri_ok_frac"]),
                )
                print(
                    f"[{side}] {boxes.shape[1]} frames, {boxes.shape[0]} views | "
                    f"reproj {res['reproj_px']:.2f} px | tri_ok {res['tri_ok_frac'] * 100:.1f}% | {elapsed:.1f}s"
                )
                if args.export_obj:
                    obj_dir = os.path.join(args.out, f"{prefix}meshes")
                    n_obj = export_obj_sequence(obj_dir, res["verts"], res["faces"], res["observed"])
                    print(f"[{side}] exported {n_obj} .obj meshes to {obj_dir}")
            else:
                print(
                    f"[{side}] {boxes.shape[1]} frames, {boxes.shape[0]} views | "
                    f"2D dense vertices saved ({elapsed:.1f}s)"
                )

        if args.render:
            vid_path = os.path.join(args.out, "overlay.mp4")
            write_multiview_video(
                out_path=vid_path,
                frame_source=src,
                K=scene["K"],
                R=scene["R"],
                t=scene["t"],
                hand_results=hand_results,
                fps=scene["fps"],
            )
            print(f"[render] saved multi-view overlay video to {vid_path}")
    finally:
        src.close()


if __name__ == "__main__":
    main()
