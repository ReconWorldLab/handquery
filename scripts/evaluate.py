#!/usr/bin/env python3
"""Multi-GPU / single-GPU evaluation across the 5 multi-view benchmarks (Table 1 protocol).

Example:
  python scripts/evaluate.py \
      --data-root data/benchmarks \
      --ckpt checkpoints/handquery_vitb.pt \
      --mano-root assets \
      --benchmarks dexycb_mv ho3d_mv oakink_mv arctic_mv interhand_mv \
      --gpus 0 1 2 3 \
      --out outputs/eval
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from handquery import FrameSource, HandQueryPipeline  # noqa: E402
from handquery.metrics import (  # noqa: E402
    ALL_METRIC_KEYS,
    eval_frames,
    load_subset,
    sequence_metrics,
    subset_mask,
    weighted_mean,
)

BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "dexycb_mv": {
        "title": "DexYCB-MV",
        "dataset": "dexycb",
        "protocol": "s0",
        "split": "test",
        "subset": "assets/subsets/dexycb_mv_test.json",
        "auc_mm": 20.0,
    },
    "ho3d_mv": {
        "title": "HO3D-MV",
        "dataset": "ho3d",
        "protocol": "poem",
        "split": "test",
        "subset": None,
        "auc_mm": 50.0,
    },
    "oakink_mv": {
        "title": "OakInk-MV",
        "dataset": "oakink",
        "protocol": "object",
        "split": "test",
        "subset": "assets/subsets/oakink_mv_test.json",
        "auc_mm": 20.0,
    },
    "arctic_mv": {
        "title": "ARCTIC-MV",
        "dataset": "arctic",
        "protocol": "p1",
        "split": "val",
        "subset": "assets/subsets/arctic_mv_val.json",
        "auc_mm": 20.0,
    },
    "interhand_mv": {
        "title": "InterHand-MV",
        "dataset": "interhand",
        "groups": [f"g{g:02d}" for g in range(17)],
        "split": "val",
        "subset": None,
        "auc_mm": 20.0,
    },
}


def _frame_paths_from_npz(L: dict[str, Any], n_frames: int) -> list[list[str]]:
    offset = int(L["img_offset"]) if "img_offset" in L else 0
    fids = [int(x) for x in L["img_frame_ids"]] if "img_frame_ids" in L else [f + offset for f in range(n_frames)]
    out = []
    for pat_raw in L["img_pattern"]:
        pat = str(pat_raw)
        if "InterHand" in pat and pat.endswith("image{}.jpg"):
            pat = pat[: -len("image{}.jpg")] + "image{:04d}.jpg"
        out.append([pat.format(fid) for fid in fids])
    return out


def _collect_jobs(data_root: str, benchmarks: list[str], limit: int = 0) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for bm in benchmarks:
        spec = BENCHMARK_SPECS[bm]
        ds = spec["dataset"]
        idx_file = os.path.join(data_root, f"index_{ds}.json")
        if not os.path.isfile(idx_file):
            raise FileNotFoundError(
                f"Missing {idx_file} for benchmark '{bm}'. Run `scripts/prepare_datasets.py` first."
            )
        all_entries = json.load(open(idx_file, encoding="utf-8"))
        subset = load_subset(os.path.join(REPO_ROOT, spec["subset"])) if spec.get("subset") else None

        if "groups" in spec:
            for g in spec["groups"]:
                matched = [e for e in all_entries if e["splits"].get(g) == spec["split"] and e["n_valid"] > 0]
                if limit:
                    matched = matched[:limit]
                for e in matched:
                    jobs.append(
                        {
                            "benchmark": bm,
                            "group": g,
                            "entry": e,
                            "auc_mm": spec["auc_mm"],
                            "subset_frames": None,
                        }
                    )
        else:
            matched = [e for e in all_entries if e["splits"].get(spec["protocol"]) == spec["split"] and e["n_valid"] > 0]
            filtered = []
            for e in matched:
                if subset is not None and f"{ds}/{e['name']}" not in subset:
                    continue
                filtered.append(e)
            if limit:
                filtered = filtered[:limit]
            for e in filtered:
                jobs.append(
                    {
                        "benchmark": bm,
                        "group": None,
                        "entry": e,
                        "auc_mm": spec["auc_mm"],
                        "subset_frames": subset[f"{ds}/{e['name']}"].tolist() if subset is not None else None,
                    }
                )
    return jobs


def _worker(gpu: int, shard: list[dict[str, Any]], args_dict: dict[str, Any], out_q: mp.Queue) -> None:
    torch.set_num_threads(2)
    if gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        dev = torch.device("cuda", gpu)
    else:
        dev = torch.device("cpu")

    pipe = HandQueryPipeline.from_pretrained(
        ckpt_path=args_dict["ckpt"],
        mano_root=args_dict["mano_root"],
        device=dev,
    )

    for job in shard:
        bm = job["benchmark"]
        grp = job["group"]
        e = job["entry"]
        sub_dir = f"{bm}_{grp}" if grp else bm
        seq_dir = os.path.join(args_dict["out"], sub_dir, e["name"].replace("/", "__"))
        metrics_json = os.path.join(seq_dir, "metrics.json")

        if args_dict["resume"] and os.path.isfile(metrics_json):
            out_q.put(json.load(open(metrics_json, encoding="utf-8")))
            continue

        with np.load(e["npz"], allow_pickle=True) as z:
            L = {k: z[k] for k in z.files}

        n_frames = int(L["verts"].shape[0])
        has = eval_frames(L["valid"], L["boxes"])
        if job["subset_frames"] is not None:
            sub_m = np.zeros(n_frames, dtype=bool)
            sub_m[np.asarray(job["subset_frames"], dtype=np.int64)] = True
            has &= sub_m
        if not has.any():
            continue

        side = str(L["side"])
        tips_conv = e.get("tips", "smplx")
        frame_paths = _frame_paths_from_npz(L, n_frames)

        t0 = time.time()
        src = FrameSource(frame_paths)
        try:
            res = pipe.reconstruct(
                images_or_source=src,
                K=L["K"],
                R=L["R"],
                t=L["t"],
                boxes=L["boxes"],
                side=side,
                window_size=args_dict["window_size"],
                stride=args_dict["stride"],
                fit_mano=True,
                tips=tips_conv,
                fit_iters=tuple(args_dict["iters"]),
                valid_frames=L["valid"].astype(bool),
                cam_subset=args_dict["cams"],
                img_affine=L.get("img_affine"),
            )
        finally:
            src.close()
        elapsed = time.time() - t0

        m = sequence_metrics(
            verts=res["verts"],
            j21=res["joints21"],
            gt_verts=L["verts"],
            gt_j21=L["joints"],
            mask=has,
            side=side,
            mano_root=args_dict["mano_root"],
            auc_max_mm=float(job["auc_mm"]),
        )
        m.update(
            {
                "benchmark": bm,
                "group": grp,
                "seq": e["name"],
                "side": side,
                "reproj_px": float(res["reproj_px"]),
                "elapsed_s": elapsed,
            }
        )

        os.makedirs(seq_dir, exist_ok=True)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        np.savez_compressed(
            os.path.join(seq_dir, "mano_fit.npz"),
            verts=res["verts"],
            joints21=res["joints21"],
            faces=res["faces"],
            betas=res["betas"],
            global_orient=res["global_orient"],
            pca=res["pca"],
            hand_pose=res["hand_pose"],
            transl=res["transl"],
            observed=res["observed"],
        )
        if args_dict["save_pred2d"]:
            np.savez_compressed(
                os.path.join(seq_dir, "pred2d.npz"),
                lm=res["lm"],
                sigma=res["sigma"],
                vis=res["vis"],
                valid=res["valid"],
                vert_ids=res["vert_ids"],
            )
        out_q.put(m)

    out_q.put(None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HandQuery across standard multi-view 3D hand benchmarks.")
    parser.add_argument("--data-root", default="data/benchmarks", help="Prepared benchmark root from prepare_datasets.py.")
    parser.add_argument("--ckpt", default="checkpoints/handquery_vitb.pt", help="Path to HandQuery checkpoint.")
    parser.add_argument("--mano-root", default="assets", help="Directory containing mano/MANO_{RIGHT,LEFT}.pkl.")
    parser.add_argument("--out", default="outputs/eval", help="Output directory for fits and summary.json.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        choices=list(BENCHMARK_SPECS.keys()),
        help="Benchmarks to evaluate (default: all prepared benchmarks found in --data-root).",
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU indices to use in parallel (default: 0).")
    parser.add_argument("--per-gpu", type=int, default=1, help="Worker processes per GPU (default: 1).")
    parser.add_argument("--window-size", type=int, default=48, help="Sliding-window clip length T (default: 48).")
    parser.add_argument("--stride", type=int, default=24, help="Sliding-window stride (default: 24).")
    parser.add_argument("--iters", type=int, nargs=3, default=[150, 300, 200], help="3-stage MANO fitting iterations.")
    parser.add_argument("--cams", type=int, nargs="*", default=None, help="Optional camera index subset (e.g. 0 2 4 6).")
    parser.add_argument("--limit", type=int, default=0, help="Optional max sequences per benchmark/group for quick checks.")
    parser.add_argument("--save-pred2d", action="store_true", help="Also save fused 2D vertex predictions per sequence.")
    parser.add_argument("--no-resume", action="store_true", help="Re-evaluate sequences even if metrics.json already exists.")
    args = parser.parse_args()

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    if args.benchmarks:
        benchmarks = args.benchmarks
    else:
        benchmarks = [
            bm
            for bm, spec in BENCHMARK_SPECS.items()
            if os.path.isfile(os.path.join(data_root, f"index_{spec['dataset']}.json"))
        ]
        if not benchmarks:
            raise FileNotFoundError(
                f"No `index_*.json` files found in {data_root}. Run `scripts/prepare_datasets.py` first."
            )

    jobs = _collect_jobs(data_root, benchmarks, limit=args.limit)
    if not jobs:
        raise RuntimeError("No evaluation sequences matched.")

    os.makedirs(args.out, exist_ok=True)
    args_dict = {
        "ckpt": os.path.abspath(os.path.expanduser(args.ckpt)),
        "mano_root": os.path.abspath(os.path.expanduser(args.mano_root)),
        "out": os.path.abspath(os.path.expanduser(args.out)),
        "window_size": args.window_size,
        "stride": args.stride,
        "iters": args.iters,
        "cams": args.cams,
        "save_pred2d": args.save_pred2d,
        "resume": not args.no_resume,
    }

    slots = [g for g in args.gpus for _ in range(max(1, args.per_gpu))]
    shards = [jobs[k :: len(slots)] for k in range(len(slots))]
    active_slots = [(slots[k], shards[k]) for k in range(len(slots)) if shards[k]]

    results: list[dict[str, Any]] = []
    t_start = time.time()

    if len(active_slots) == 1:
        q_local: Any = queue.Queue()
        _worker(active_slots[0][0], active_slots[0][1], args_dict, q_local)
        while True:
            item = q_local.get()
            if item is None:
                break
            results.append(item)
            print(
                f"[{len(results)}/{len(jobs)}] {item['benchmark']}"
                + (f"/{item['group']}" if item["group"] else "")
                + f" {item['seq']}: MPVPE {item['MPVPE_mm']:.2f} mm | PA-V {item['PAV_mm']:.2f} mm | reproj {item['reproj_px']:.2f} px",
                flush=True,
            )
    else:
        ctx = mp.get_context("spawn")
        q_mp = ctx.Queue()
        procs = [ctx.Process(target=_worker, args=(gpu, sh, args_dict, q_mp)) for gpu, sh in active_slots]
        for p in procs:
            p.start()
        done = 0
        while done < len(procs):
            try:
                item = q_mp.get(timeout=10)
            except queue.Empty:
                failed = [i for i, p in enumerate(procs) if p.exitcode not in (None, 0)]
                if failed:
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError(f"Evaluation worker(s) {failed} exited with error")
                continue
            if item is None:
                done += 1
                continue
            results.append(item)
            print(
                f"[{len(results)}/{len(jobs)} {time.time() - t_start:.0f}s] {item['benchmark']}"
                + (f"/{item['group']}" if item["group"] else "")
                + f" {item['seq']}: MPVPE {item['MPVPE_mm']:.2f} mm | PA-V {item['PAV_mm']:.2f} mm | reproj {item['reproj_px']:.2f} px",
                flush=True,
            )
        for p in procs:
            p.join()

    summary: dict[str, Any] = {
        "ckpt": args.ckpt,
        "window_size": args.window_size,
        "stride": args.stride,
        "iters": args.iters,
        "cams": args.cams,
        "benchmarks": {},
    }

    print("\n" + "=" * 102)
    print(
        f"{'Benchmark':<16} {'Frames':>8} | {'MPVPE':>7} {'RR-V':>7} {'PA-V':>7} {'AUC-V':>7} | "
        f"{'MPJPE*':>7} {'RR-J*':>7} {'PA-J*':>7}"
    )
    print("-" * 102)
    for bm in benchmarks:
        bm_rows = [r for r in results if r["benchmark"] == bm]
        if not bm_rows:
            continue
        spec = BENCHMARK_SPECS[bm]
        if "groups" in spec:
            grp_means = []
            for g in spec["groups"]:
                g_rows = [r for r in bm_rows if r["group"] == g]
                if g_rows:
                    grp_means.append(weighted_mean(g_rows, ALL_METRIC_KEYS))
            agg = {
                "groups": len(grp_means),
                "sequences": len(bm_rows),
                "frames_with_gt": int(sum(gm["frames_with_gt"] for gm in grp_means)),
            }
            for k in ALL_METRIC_KEYS:
                agg[k] = float(np.mean([gm[k] for gm in grp_means]))
        else:
            agg = weighted_mean(bm_rows, ALL_METRIC_KEYS)

        summary["benchmarks"][bm] = agg
        print(
            f"{spec['title']:<16} {agg['frames_with_gt']:>8,d} | "
            f"{agg['MPVPE_mm']:>7.2f} {agg['RRV_mm']:>7.2f} {agg['PAV_mm']:>7.2f} {agg['AUC_V']:>7.3f} | "
            f"{agg['MPJPE_poem_mm']:>7.2f} {agg['RRJ_poem_mm']:>7.2f} {agg['PAJ_poem_mm']:>7.2f}"
        )
    print("=" * 102)

    sum_path = os.path.join(args.out, "summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {sum_path}")


if __name__ == "__main__":
    main()
