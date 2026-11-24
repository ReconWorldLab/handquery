#!/usr/bin/env python3
"""Prepare multi-view benchmark and zero-shot datasets from their raw downloaded directories.

Pass any combination of dataset paths you have downloaded:

  # Prepare all 5 quantitative benchmarks at once:
  python scripts/prepare_datasets.py \
      --dexycb /path/to/DexYCB \
      --ho3d /path/to/HO3D_v3 \
      --oakink /path/to/OakInk \
      --arctic /path/to/ARCTIC \
      --interhand-mv /path/to/POEM_Interhand_mv \
      --mano-root assets \
      --out data/benchmarks

  # Export zero-shot SHOW3D / GigaHands scenes into ready-to-infer scene.json bundles:
  python scripts/prepare_datasets.py \
      --show3d /path/to/SHOW3D \
      --gigahands /path/to/GigaHands \
      --mano-root assets \
      --out data/zeroshot
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handquery.datasets import (  # noqa: E402
    prepare_arctic,
    prepare_dexycb,
    prepare_gigahands,
    prepare_ho3d,
    prepare_interhand_mv,
    prepare_oakink,
    prepare_show3d,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare multi-view benchmarks and zero-shot datasets from raw directories (no pyrender/EGL required)."
    )
    parser.add_argument("--dexycb", default=None, help="Raw DexYCB root directory.")
    parser.add_argument("--ho3d", default=None, help="Raw HO3D v3 root directory.")
    parser.add_argument("--oakink", default=None, help="Raw OakInk-Image v2 root directory.")
    parser.add_argument("--arctic", default=None, help="Raw ARCTIC root directory.")
    parser.add_argument("--interhand-mv", default=None, help="Directory containing POEM-v2 Interhand_mv/*.tar shards.")
    parser.add_argument("--show3d", default=None, help="Raw SHOW3D v3 root directory.")
    parser.add_argument("--show3d-scenes", nargs="*", default=None, help="Optional '<subject>/<scene>' list for SHOW3D.")
    parser.add_argument("--gigahands", default=None, help="Raw GigaHands root directory.")
    parser.add_argument("--gigahands-scenes", nargs="*", default=None, help="Optional '<scene>[:<seq>]' list for GigaHands.")
    parser.add_argument("--gigahands-cams", type=int, default=8, help="Views per GigaHands sequence (default: 8).")
    parser.add_argument("--gigahands-frames", type=int, default=192, help="Frames per GigaHands clip (default: 192).")
    parser.add_argument("--mano-root", default="assets", help="Directory containing mano/MANO_{RIGHT,LEFT}.pkl.")
    parser.add_argument("--out", default="data/benchmarks", help="Output directory for prepared index and label files.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes (default: 8).")
    parser.add_argument("--all-splits", action="store_true", help="Also prepare train/val splits (default: evaluation splits only).")
    args = parser.parse_args()

    any_selected = any(
        [args.dexycb, args.ho3d, args.oakink, args.arctic, args.interhand_mv, args.show3d, args.gigahands]
    )
    if not any_selected:
        parser.error(
            "Provide at least one dataset path (`--dexycb`, `--ho3d`, `--oakink`, `--arctic`, "
            "`--interhand-mv`, `--show3d`, or `--gigahands`)."
        )

    if args.dexycb:
        prepare_dexycb(args.dexycb, args.mano_root, args.out, workers=args.workers, all_splits=args.all_splits)
    if args.ho3d:
        prepare_ho3d(args.ho3d, args.mano_root, args.out, all_splits=args.all_splits)
    if args.oakink:
        prepare_oakink(args.oakink, args.out, workers=args.workers, all_splits=args.all_splits)
    if args.arctic:
        prepare_arctic(args.arctic, args.mano_root, args.out, workers=args.workers, all_splits=args.all_splits)
    if args.interhand_mv:
        prepare_interhand_mv(args.interhand_mv, args.mano_root, args.out)
    if args.show3d:
        prepare_show3d(args.show3d, args.mano_root, args.out, scenes=args.show3d_scenes)
    if args.gigahands:
        prepare_gigahands(
            args.gigahands,
            args.mano_root,
            args.out,
            scenes=args.gigahands_scenes,
            num_cams=args.gigahands_cams,
            num_frames=args.gigahands_frames,
        )


if __name__ == "__main__":
    main()
