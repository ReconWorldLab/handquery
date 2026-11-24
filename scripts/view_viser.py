#!/usr/bin/env python3
"""Launch the interactive HandQuery `viser` 3D + multi-view image viewer.

Examples:
  # View a single reconstructed sequence (from scene.json or a prepared benchmark .npz):
  python scripts/view_viser.py --input examples/rubik_cube/scene.json --result outputs/rubik_cube
  python scripts/view_viser.py --input data/benchmarks/ho3d/GPMF1.npz --result outputs/eval/ho3d_mv/GPMF1

  # View multiple demo sequences in one viewer (switchable from the sidebar dropdown):
  python scripts/view_viser.py \
      --scenes examples/rubik_cube/scene.json:outputs/rubik_cube \
               examples/pen_spinning/scene.json:outputs/pen_spinning
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handquery.viser_viewer import load_scene_bundle, run_viser_server  # noqa: E402


def _split_scene_pair(spec: str) -> tuple[str, str]:
    """Split `<scene.json|seq.npz>:<result_dir>` cleanly on Windows drive-letter paths as well as POSIX paths."""
    for i in range(len(spec) - 1, 0, -1):
        if spec[i] == ":" and not (i == 1 and spec[0].isalpha()):
            left, right = spec[:i], spec[i + 1 :]
            if left.lower().endswith((".json", ".npz")):
                return left, right
    raise ValueError(f"Expected '<path/to/scene.json|seq.npz>:<path/to/result_dir>', got {spec!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive HandQuery 3D + Multi-View Camera Viewer (viser).")
    parser.add_argument("--input", default=None, help="Path to scene.json manifest or prepared benchmark .npz.")
    parser.add_argument("--result", default=None, help="Path to HandQuery output directory containing *mano_fit.npz / *pred2d.npz.")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="One or more '<scene.json|seq.npz>:<result_dir>' pairs to load into the sequence selector.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8080, help="HTTP/WebSocket port (default: 8080).")
    parser.add_argument("--preview-width", type=int, default=540, help="Per-camera display width in pixels (default: 540).")
    args = parser.parse_args()

    pairs: list[tuple[str, str]] = []
    if args.scenes:
        pairs.extend(_split_scene_pair(s) for s in args.scenes)
    if args.input and args.result:
        pairs.append((args.input, args.result))
    if not pairs:
        parser.error("Provide either `--input <scene.json|seq.npz> --result <out_dir>` or `--scenes <scene>:<out_dir> ...`")

    bundles = []
    for scene_path, res_dir in pairs:
        print(f"[viser] Loading {scene_path} + {res_dir} ...")
        bundles.append(load_scene_bundle(scene_path, res_dir, preview_width=args.preview_width))

    run_viser_server(bundles, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
