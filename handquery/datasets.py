"""Dataset preparation converters for all 7 supported multi-view hand benchmarks & zero-shot datasets.

Supported datasets (directly from their official raw directory layouts, with zero OpenGL/pyrender dependency):
  - Quantitative Multi-View Benchmarks:
      1. `DexYCB`       (`prepare_dexycb`)
      2. `HO3D v3`      (`prepare_ho3d`)
      3. `OakInk v2`    (`prepare_oakink`)
      4. `ARCTIC`       (`prepare_arctic`)
      5. `InterHand-MV` (`prepare_interhand_mv`, from POEM-v2's `Interhand_mv` tar shards)
  - Zero-Shot Qualitative Multi-View Datasets (exported directly as ready-to-infer `scene.json` bundles):
      6. `SHOW3D`       (`prepare_show3d`)
      7. `GigaHands`    (`prepare_gigahands`)
"""

from __future__ import annotations

import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import tarfile
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from handquery.mano import FINGERTIP_VERTS, ManoModel, joints21

DEXYCB_SUBJECTS = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
]

HO3D_POEM_SPLITS = {
    "train": {"ABF1", "BB1", "GSF1", "MDF1", "SiBF1"},
    "test": {"GPMF1", "SB1"},
}

OAKINK_CAMS = ["north_east_color", "south_east_color", "north_west_color", "south_west_color"]


class _TupleSafeLoader(yaml.SafeLoader):
    pass


_TupleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))


def _load_yaml(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_TupleSafeLoader)


def _project_cam(K: np.ndarray, Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = Xc @ K.T
    z = Xc[..., 2]
    return p[..., :2] / np.clip(z[..., None], 1e-6, None), z


def _box_of(uv: np.ndarray, ok: np.ndarray) -> np.ndarray:
    if not np.any(ok):
        return np.full(4, np.nan, dtype=np.float32)
    pts = uv[ok]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def _rodrigues_batch(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    th = np.linalg.norm(aa, axis=1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    Kx = np.zeros((len(aa), 3, 3), dtype=np.float64)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, :, None], np.cos(th)[:, :, None]
    return np.eye(3)[None] + s * Kx + (1.0 - c) * (Kx @ Kx)


def _load_obj_vertices(obj_path: str | os.PathLike) -> np.ndarray:
    verts = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. DexYCB
# ---------------------------------------------------------------------------

_WORKER_STATE: dict[str, Any] = {}


def _dexycb_split(protocol: str, si: int, i: int) -> str:
    if protocol == "s0":
        if i % 5 != 4:
            return "train"
        return "val" if si in (0, 1) else "test"
    return {0: "train", 1: "train", 2: "train", 3: "train", 4: "train", 5: "train", 9: "train", 6: "val"}.get(
        si, "test"
    )


def _prep_dexycb_one(job: tuple) -> dict[str, Any]:
    root, mano_root, out_root, si, i, subject, seq = job
    if "mano" not in _WORKER_STATE:
        import torch

        torch.set_num_threads(1)
        _WORKER_STATE["mano"] = ManoModel(mano_root)
    mano: ManoModel = _WORKER_STATE["mano"]

    seq_dir = os.path.join(root, subject, seq)
    meta = _load_yaml(os.path.join(seq_dir, "meta.yml"))
    serials = [str(s) for s in meta["serials"]]
    n = int(meta["num_frames"])
    side = str(meta["mano_sides"][0])
    calib = meta["mano_calib"][0]
    betas = np.asarray(_load_yaml(os.path.join(root, "calibration", f"mano_{calib}", "mano.yml"))["betas"], np.float32)
    ext = _load_yaml(os.path.join(root, "calibration", f"extrinsics_{meta['extrinsics']}", "extrinsics.yml"))[
        "extrinsics"
    ]

    K_list, R_list, t_list = [], [], []
    for s in serials:
        c = _load_yaml(os.path.join(root, "calibration", "intrinsics", f"{s}_640x480.yml"))["color"]
        K_list.append([[c["fx"], 0.0, c["ppx"]], [0.0, c["fy"], c["ppy"]], [0.0, 0.0, 1.0]])
        T = np.eye(4, dtype=np.float64)
        T[:3] = np.asarray(ext[s], dtype=np.float64).reshape(3, 4)
        E = np.linalg.inv(T)
        R_list.append(E[:3, :3])
        t_list.append(E[:3, 3])
    K = np.asarray(K_list, dtype=np.float64)
    R = np.asarray(R_list, dtype=np.float64)
    t = np.asarray(t_list, dtype=np.float64)

    pose_m = np.zeros((n, 51), dtype=np.float32)
    cam_of = np.full(n, -1, dtype=np.int64)
    for f in range(n):
        for c_idx, s in enumerate(serials):
            lab = np.load(os.path.join(seq_dir, s, f"labels_{f:06d}.npz"))
            pm = lab["pose_m"][0]
            if not np.all(pm == 0) and not np.all(lab["joint_3d"] == -1):
                pose_m[f], cam_of[f] = pm, c_idx
                break

    valid = cam_of >= 0
    verts = np.full((n, 778, 3), np.nan, dtype=np.float32)
    j21 = np.full((n, 21, 3), np.nan, dtype=np.float32)
    if valid.any():
        v_cam, j16_cam = mano.forward_dexycb(side, betas, pose_m[valid])
        jj_cam = joints21(v_cam, j16_cam, FINGERTIP_VERTS["dexycb"][side])
        for k, f in enumerate(np.flatnonzero(valid)):
            c_idx = int(cam_of[f])
            verts[f] = (v_cam[k] - t[c_idx]) @ R[c_idx]
            j21[f] = (jj_cam[k] - t[c_idx]) @ R[c_idx]

    V = len(serials)
    boxes = np.full((V, n, 4), np.nan, dtype=np.float32)
    for c_idx in range(V):
        if valid.any():
            Xc = verts[valid].astype(np.float64) @ R[c_idx].T + t[c_idx]
            uv, z = _project_cam(K[c_idx], Xc)
            for k, f in enumerate(np.flatnonzero(valid)):
                boxes[c_idx, f] = _box_of(uv[k], z[k] > 0)

    name = f"{subject}/{seq}"
    out_npz = os.path.join(out_root, "dexycb", subject, f"{seq}.npz")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        verts=verts,
        joints=j21,
        valid=valid,
        K=K.astype(np.float32),
        R=R.astype(np.float32),
        t=t.astype(np.float32),
        boxes=boxes,
        betas=betas,
        side=side,
        serials=np.array(serials),
        img_pattern=np.array([os.path.join(seq_dir, s, "color_{:06d}.jpg") for s in serials]),
        img_size=np.array([640, 480]),
    )
    return {
        "dataset": "dexycb",
        "name": name,
        "npz": out_npz,
        "n_frames": n,
        "n_cams": V,
        "side": side,
        "splits": {"s0": _dexycb_split("s0", si, i), "s1": _dexycb_split("s1", si, i)},
        "n_valid": int(valid.sum()),
        "tips": "dexycb",
    }


def prepare_dexycb(
    raw_root: str | os.PathLike,
    mano_root: str | os.PathLike,
    out_root: str | os.PathLike,
    workers: int = 8,
    all_splits: bool = False,
) -> list[dict[str, Any]]:
    """Prepare DexYCB test (or all) sequences into `<out_root>/dexycb/` and `<out_root>/index_dexycb.json`."""
    raw_root = os.path.abspath(os.path.expanduser(os.fspath(raw_root)))
    mano_root = os.path.abspath(os.path.expanduser(os.fspath(mano_root)))
    out_root = os.path.abspath(os.path.expanduser(os.fspath(out_root)))

    jobs = []
    for si, subject in enumerate(DEXYCB_SUBJECTS):
        subj_dir = os.path.join(raw_root, subject)
        if not os.path.isdir(subj_dir):
            continue
        for i, seq in enumerate(sorted(os.listdir(subj_dir))):
            if not all_splits and _dexycb_split("s0", si, i) != "test":
                continue
            jobs.append((raw_root, mano_root, out_root, si, i, subject, seq))

    os.makedirs(out_root, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))), mp_context=mp.get_context("spawn")) as pool:
        entries = list(pool.map(_prep_dexycb_one, jobs, chunksize=2))
    with open(os.path.join(out_root, "index_dexycb.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"[DexYCB] Prepared {len(entries)} sequences -> {os.path.join(out_root, 'index_dexycb.json')}")
    return entries


# ---------------------------------------------------------------------------
# 2. HO3D v3
# ---------------------------------------------------------------------------

def _ho3d_groups(root: str) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for sub in ("train", "evaluation"):
        sub_dir = os.path.join(root, sub)
        if not os.path.isdir(sub_dir):
            continue
        for d in sorted(os.listdir(sub_dir)):
            m = re.match(r"^(.+?)(\d)$", d)
            if m and os.path.isdir(os.path.join(root, "calibration", m.group(1), "calibration")):
                out.setdefault(m.group(1), []).append((sub, d))
    return {k: sorted(v, key=lambda x: x[1]) for k, v in out.items() if len(v) >= 2}


def prepare_ho3d(
    raw_root: str | os.PathLike,
    mano_root: str | os.PathLike,
    out_root: str | os.PathLike,
    all_splits: bool = False,
) -> list[dict[str, Any]]:
    """Prepare HO3D v3 multi-camera test groups (`GPMF1`, `SB1`) with zero external object model dependencies."""
    raw_root = os.path.abspath(os.path.expanduser(os.fspath(raw_root)))
    mano_root = os.path.abspath(os.path.expanduser(os.fspath(mano_root)))
    out_root = os.path.abspath(os.path.expanduser(os.fspath(out_root)))

    mano = ManoModel(mano_root)
    flip = np.diag([1.0, -1.0, -1.0])
    groups = _ho3d_groups(raw_root)
    entries = []

    for base, cams in sorted(groups.items()):
        split = "train" if base in HO3D_POEM_SPLITS["train"] else "test" if base in HO3D_POEM_SPLITS["test"] else "other"
        if not all_splits and split != "test":
            continue

        cdir = os.path.join(raw_root, "calibration", base, "calibration")
        orders = [int(float(ln)) for ln in open(os.path.join(cdir, "cam_orders.txt")) if ln.strip()]
        n = min(len(glob.glob(os.path.join(raw_root, sub, d, "rgb", "*.jpg"))) for sub, d in cams)
        V = len(cams)
        K = np.zeros((V, 3, 3), dtype=np.float64)
        R = np.zeros((V, 3, 3), dtype=np.float64)
        t = np.zeros((V, 3), dtype=np.float64)
        c2w_list = []

        for c_idx, (sub, d) in enumerate(cams):
            T = np.loadtxt(os.path.join(cdir, f"trans_{orders[int(d[-1])]}.txt")).reshape(4, 4)
            c2w_list.append(T)
            E = np.linalg.inv(T)
            R[c_idx], t[c_idx] = E[:3, :3], E[:3, 3]
            for f in range(n):
                p = os.path.join(raw_root, sub, d, "meta", f"{f:04d}.pkl")
                if os.path.isfile(p):
                    m = pickle.load(open(p, "rb"), encoding="latin1")
                    if m is not None and m.get("camMat") is not None:
                        K[c_idx] = m["camMat"]
                        break

        verts = np.full((n, 778, 3), np.nan, dtype=np.float32)
        j21 = np.full((n, 21, 3), np.nan, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        train_cams = [(c_idx, sub, d) for c_idx, (sub, d) in enumerate(cams) if sub == "train"]

        for f in range(n):
            for c_idx, sub, d in train_cams:
                p = os.path.join(raw_root, sub, d, "meta", f"{f:04d}.pkl")
                if not os.path.isfile(p):
                    continue
                m = pickle.load(open(p, "rb"), encoding="latin1")
                if m is None or m.get("handPose") is None or m.get("handTrans") is None:
                    continue
                hp = np.asarray(m["handPose"], dtype=np.float32)
                v_gl, j16_gl = mano.forward("right", m["handBeta"], hp[:3], hp[3:], m["handTrans"])
                v_cv, j_cv = v_gl[0] @ flip, j16_gl[0] @ flip
                Tw = c2w_list[c_idx]
                verts[f] = v_cv @ Tw[:3, :3].T + Tw[:3, 3]
                j21[f] = joints21(verts[f], j_cv @ Tw[:3, :3].T + Tw[:3, 3], FINGERTIP_VERTS["ho3d"]["right"])
                valid[f] = True
                break

        boxes = np.full((V, n, 4), np.nan, dtype=np.float32)
        for c_idx in range(V):
            if valid.any():
                Xc = verts[valid].astype(np.float64) @ R[c_idx].T + t[c_idx]
                uv, z = _project_cam(K[c_idx], Xc)
                for k, f in enumerate(np.flatnonzero(valid)):
                    boxes[c_idx, f] = _box_of(uv[k], z[k] > 0)

        out_npz = os.path.join(out_root, "ho3d", f"{base}.npz")
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        np.savez_compressed(
            out_npz,
            verts=verts,
            joints=j21,
            valid=valid,
            K=K.astype(np.float32),
            R=R.astype(np.float32),
            t=t.astype(np.float32),
            boxes=boxes,
            betas=np.zeros(10, dtype=np.float32),
            side="right",
            serials=np.array([d for _, d in cams]),
            img_pattern=np.array([os.path.join(raw_root, sub, d, "rgb", "{:04d}.jpg") for sub, d in cams]),
            img_size=np.array([640, 480]),
        )
        entries.append(
            {
                "dataset": "ho3d",
                "name": base,
                "npz": out_npz,
                "n_frames": n,
                "n_cams": V,
                "side": "right",
                "splits": {"poem": split},
                "n_valid": int(valid.sum()),
                "tips": "ho3d",
            }
        )

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "index_ho3d.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"[HO3D] Prepared {len(entries)} multi-camera groups -> {os.path.join(out_root, 'index_ho3d.json')}")
    return entries


# ---------------------------------------------------------------------------
# 3. OakInk-Image v2
# ---------------------------------------------------------------------------

def _oakink_load_pkl(root: str, kind: str, entry: Sequence[Any]) -> Any:
    fn = "__".join(str(x) for x in entry).replace("/", "__") + ".pkl"
    with open(os.path.join(root, "anno", kind, fn), "rb") as f:
        return pickle.load(f)


def _prep_oakink_one(job: tuple) -> dict[str, Any]:
    root, out_root, seq, hand, entries, splits = job
    W, H = 848, 480
    frames = sorted({int(e[2]) for e in entries})
    f0, n = frames[0], frames[-1] - frames[0] + 1
    have = {(int(e[2]), int(e[3])): e for e in entries}

    K = np.zeros((4, 3, 3), dtype=np.float64)
    R = np.zeros((4, 3, 3), dtype=np.float64)
    t = np.zeros((4, 3), dtype=np.float64)
    cam_ok = np.zeros(4, dtype=bool)
    for c in range(4):
        e = next((have[(f, c)] for f in frames if (f, c) in have), None)
        if e is None:
            continue
        K[c] = np.asarray(_oakink_load_pkl(root, "cam_intr", e), dtype=np.float64)
        E = np.asarray(_oakink_load_pkl(root, "general_info", e)["cam_extr"], dtype=np.float64)
        R[c], t[c] = E[:3, :3], E[:3, 3]
        cam_ok[c] = True

    verts = np.full((n, 778, 3), np.nan, dtype=np.float32)
    joints = np.full((n, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for f in frames:
        c = next(c for c in range(4) if (f, c) in have)
        e = have[(f, c)]
        verts[f - f0] = ((np.asarray(_oakink_load_pkl(root, "hand_v", e), np.float64) - t[c]) @ R[c]).astype(np.float32)
        joints[f - f0] = ((np.asarray(_oakink_load_pkl(root, "hand_j", e), np.float64) - t[c]) @ R[c]).astype(np.float32)
        valid[f - f0] = True

    boxes = np.full((4, n, 4), np.nan, dtype=np.float32)
    for c in np.flatnonzero(cam_ok):
        for fi in np.flatnonzero(valid):
            Xc = verts[fi].astype(np.float64) @ R[c].T + t[c]
            uv, z = _project_cam(K[c], Xc)
            ins = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)
            if ins.mean() >= 0.3:
                boxes[c, fi] = _box_of(uv, ins)

    name = seq if hand == 0 else f"{seq}/h{hand}"
    img_dir = os.path.join(root, "image", "stream_release_v2", seq)
    out_npz = os.path.join(out_root, "oakink", name.replace("/", "__") + ".npz")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        verts=verts,
        joints=joints,
        valid=valid,
        K=K.astype(np.float32),
        R=R.astype(np.float32),
        t=t.astype(np.float32),
        boxes=boxes,
        betas=np.zeros(10, dtype=np.float32),
        side="right",
        serials=np.array(OAKINK_CAMS),
        img_pattern=np.array([os.path.join(img_dir, f"{cn}_{{}}.png") for cn in OAKINK_CAMS]),
        img_size=np.array([W, H]),
        img_frame_ids=np.arange(f0, f0 + n, dtype=np.int64),
    )
    seen = ~np.isnan(boxes[..., 0])
    return {
        "dataset": "oakink",
        "name": name,
        "npz": out_npz,
        "n_frames": n,
        "n_cams": 4,
        "side": "right",
        "splits": splits,
        "hand": int(hand),
        "n_valid": int((valid & seen.any(axis=0)).sum()),
        "tips": "oakink",
    }


def prepare_oakink(
    raw_root: str | os.PathLike,
    out_root: str | os.PathLike,
    workers: int = 8,
    all_splits: bool = False,
) -> list[dict[str, Any]]:
    """Prepare OakInk-Image v2 `object` (`split2`) test sequences."""
    raw_root = os.path.abspath(os.path.expanduser(os.fspath(raw_root)))
    out_root = os.path.abspath(os.path.expanduser(os.fspath(out_root)))
    anno_dir = os.path.join(raw_root, "anno")

    def _part(sp: str, p: str) -> set[tuple[str, int]]:
        fn = os.path.join(anno_dir, "split", sp, f"seq_{p}.json")
        if not os.path.isfile(fn):
            return set()
        return {(str(e[0]), int(e[1])) for e in json.load(open(fn, encoding="utf-8"))}

    s2_test = _part("split2", "test")
    s2_train = _part("split2", "train") if all_splits else set()

    by: dict[tuple[str, int], list] = collections.defaultdict(list)
    for e in json.load(open(os.path.join(anno_dir, "seq_all.json"), encoding="utf-8")):
        k = (str(e[0]), int(e[1]))
        if not all_splits and k not in s2_test:
            continue
        by[k].append(e)

    jobs = []
    for k, es in sorted(by.items()):
        s2 = "test" if k in s2_test else ("train" if k in s2_train else "other")
        jobs.append((raw_root, out_root, k[0], k[1], es, {"object": s2}))

    os.makedirs(out_root, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))), mp_context=mp.get_context("spawn")) as pool:
        entries = list(pool.map(_prep_oakink_one, jobs, chunksize=4))
    with open(os.path.join(out_root, "index_oakink.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"[OakInk] Prepared {len(entries)} hand sequences -> {os.path.join(out_root, 'index_oakink.json')}")
    return entries


# ---------------------------------------------------------------------------
# 4. ARCTIC
# ---------------------------------------------------------------------------

def _arctic_crop_affine(K: np.ndarray, obj_cam: np.ndarray, crop_res: int = 1000) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    uv, _ = _project_cam(K, obj_cam)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    m = float((hi - lo).max())
    cx, cy = (lo + hi) / 2.0
    scale = max((1.1 * m + 0.6 * m) / 200.0, 3.0) * 1.5
    d = scale * 200.0
    x0, y0, x1, y1 = (int(round(v)) for v in (cx - d / 2.0, cy - d / 2.0, cx + d / 2.0, cy + d / 2.0))
    sx, sy = crop_res / (x1 - x0), crop_res / (y1 - y0)
    A = np.array([[sx, 0.0, (0.5 - x0) * sx - 0.5], [0.0, sy, (0.5 - y0) * sy - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    return A, (x0, y0, x1, y1)


def _prep_arctic_one(job: tuple) -> list[dict[str, Any]]:
    root, mano_root, out_root, sid, seq, split = job
    if "mano" not in _WORKER_STATE:
        import torch

        torch.set_num_threads(1)
        _WORKER_STATE["mano"] = ManoModel(mano_root)
        _WORKER_STATE["arctic_obj"] = {}
    mano: ManoModel = _WORKER_STATE["mano"]
    obj_cache: dict[str, tuple[np.ndarray, np.ndarray]] = _WORKER_STATE["arctic_obj"]

    misc = json.load(open(os.path.join(root, "meta", "misc.json"), encoding="utf-8"))[sid]
    M = np.load(os.path.join(root, "raw_seqs", sid, f"{seq}.mano.npy"), allow_pickle=True).item()
    O = np.load(os.path.join(root, "raw_seqs", sid, f"{seq}.object.npy"), allow_pickle=True).astype(np.float64)
    n = O.shape[0]
    ioi = int(misc["ioi_offset"])
    views = list(range(1, 9))
    K = np.array([misc["intris_mat"][v - 1] for v in views], dtype=np.float64)
    E = np.array(misc["world2cam"], dtype=np.float64)
    R, t = E[:, :3, :3], E[:, :3, 3]
    sizes = np.array([misc["image_size"][v] for v in views], dtype=int)
    img_dir = os.path.join(root, "cropped_images", sid, seq)
    have = np.array([[os.path.exists(os.path.join(img_dir, str(v), f"{f + ioi:05d}.jpg")) for f in range(n)] for v in views])

    obj_name = seq.split("_")[0]
    if obj_name not in obj_cache:
        odir = os.path.join(root, "meta", "object_vtemplates", obj_name)
        v_obj = _load_obj_vertices(os.path.join(odir, "mesh.obj")) / 1000.0
        parts = np.asarray(json.load(open(os.path.join(odir, "parts.json"), encoding="utf-8")), dtype=bool)
        obj_cache[obj_name] = (v_obj, ~parts)
    v_tmpl, top_mask = obj_cache[obj_name]

    arti, rot, trans = O[:, 0], O[:, 1:4], O[:, 4:7] / 1000.0
    Ra = _rodrigues_batch(np.array([[0.0, 0.0, -1.0]]) * arti[:, None])
    Rg = _rodrigues_batch(rot)
    vt = np.where(top_mask[None, :, None], np.einsum("nij,vj->nvi", Ra, v_tmpl), v_tmpl[None])
    ov = np.einsum("nij,nvj->nvi", Rg, vt) + trans[:, None]

    hands = {}
    for side in ("right", "left"):
        h = M[side]
        v, j16 = mano.forward(side, h["shape"], h["rot"], h["pose"], h["trans"], flat_hand_mean=False)
        hands[side] = (v.astype(np.float64), joints21(v, j16, FINGERTIP_VERTS["arctic"][side]).astype(np.float64))

    V = len(views)
    A = np.tile(np.eye(3, dtype=np.float64), (V, n, 1, 1))
    boxes = {s: np.full((V, n, 4), np.nan, dtype=np.float32) for s in hands}
    for c in range(V):
        W, H = sizes[c]
        for f in range(n):
            if not have[c, f]:
                continue
            oc = ov[f] @ R[c].T + t[c]
            A[c, f], (x0, y0, x1, y1) = _arctic_crop_affine(K[c], oc)
            for s in hands:
                hc = hands[s][0][f] @ R[c].T + t[c]
                uv, z = _project_cam(K[c], hc)
                ins = (
                    (uv[:, 0] >= max(x0, 0))
                    & (uv[:, 0] < min(x1, W))
                    & (uv[:, 1] >= max(y0, 0))
                    & (uv[:, 1] < min(y1, H))
                    & (z > 0)
                )
                if ins.mean() >= 0.3:
                    boxes[s][c, f] = _box_of(uv, ins)

    out_entries = []
    valid = have.any(axis=0)
    for side, (v, j) in hands.items():
        out_npz = os.path.join(out_root, "arctic", sid, f"{seq}__{side}.npz")
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        verts = np.where(valid[:, None, None], v, np.nan).astype(np.float32)
        j_out = np.where(valid[:, None, None], j, np.nan).astype(np.float32)
        np.savez_compressed(
            out_npz,
            verts=verts,
            joints=j_out,
            valid=valid,
            K=K.astype(np.float32),
            R=R.astype(np.float32),
            t=t.astype(np.float32),
            boxes=boxes[side],
            betas=np.asarray(M[side]["shape"], dtype=np.float32),
            side=side,
            serials=np.array([str(v_id) for v_id in views]),
            img_pattern=np.array([os.path.join(img_dir, str(v_id), "{:05d}.jpg") for v_id in views]),
            img_size=sizes,
            img_offset=np.int64(ioi),
            img_affine=A.astype(np.float32),
        )
        seen = ~np.isnan(boxes[side][..., 0])
        out_entries.append(
            {
                "dataset": "arctic",
                "name": f"{sid}/{seq}/{side}",
                "npz": out_npz,
                "n_frames": n,
                "n_cams": V,
                "side": side,
                "splits": {"p1": split},
                "n_valid": int((valid & seen.any(axis=0)).sum()),
                "tips": "arctic",
            }
        )
    return out_entries


def prepare_arctic(
    raw_root: str | os.PathLike,
    mano_root: str | os.PathLike,
    out_root: str | os.PathLike,
    workers: int = 8,
    all_splits: bool = False,
) -> list[dict[str, Any]]:
    """Prepare ARCTIC `p1` validation (or train+val) sequences."""
    raw_root = os.path.abspath(os.path.expanduser(os.fspath(raw_root)))
    mano_root = os.path.abspath(os.path.expanduser(os.fspath(mano_root)))
    out_root = os.path.abspath(os.path.expanduser(os.fspath(out_root)))

    sp = json.load(open(os.path.join(raw_root, "splits_json", "protocol_p1.json"), encoding="utf-8"))
    split_of = {s: k for k in ("train", "val", "test") for s in sp.get(k, [])}
    jobs = []
    for sid in sorted(os.listdir(os.path.join(raw_root, "raw_seqs"))):
        sdir = os.path.join(raw_root, "raw_seqs", sid)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".mano.npy"):
                continue
            seq = fn[: -len(".mano.npy")]
            key = f"{sid}/{seq}"
            st = split_of.get(key)
            if st is None or st == "test":
                continue
            if not all_splits and st != "val":
                continue
            jobs.append((raw_root, mano_root, out_root, sid, seq, st))

    os.makedirs(out_root, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))), mp_context=mp.get_context("spawn")) as pool:
        nested = list(pool.map(_prep_arctic_one, jobs, chunksize=1))
    entries = sorted([e for es in nested for e in es], key=lambda e: e["name"])
    with open(os.path.join(out_root, "index_arctic.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"[ARCTIC] Prepared {len(entries)} hand sequences -> {os.path.join(out_root, 'index_arctic.json')}")
    return entries


# ---------------------------------------------------------------------------
# 5. InterHand-MV (from POEM-v2's 30-fps 17-group tar shards)
# ---------------------------------------------------------------------------

def prepare_interhand_mv(
    shards_dir: str | os.PathLike,
    mano_root: str | os.PathLike,
    out_root: str | os.PathLike,
) -> list[dict[str, Any]]:
    """Prepare InterHand-MV (30 fps, 4 sequences x 17 camera groups = 85,255 samples) from POEM-v2 shards."""
    shards_dir = os.path.abspath(os.path.expanduser(os.fspath(shards_dir)))
    mano_root = os.path.abspath(os.path.expanduser(os.fspath(mano_root)))
    out_root = os.path.abspath(os.path.expanduser(os.fspath(out_root)))

    shard_files = sorted(glob.glob(os.path.join(shards_dir, "*.tar")))
    if not shard_files:
        raise FileNotFoundError(f"No .tar shards found in {shards_dir}")

    samples: dict[tuple[str, int], list[tuple[str, str, list[str]]]] = collections.defaultdict(list)
    gt: dict[tuple[str, int], np.ndarray] = {}
    betas: dict[tuple[str, int], np.ndarray] = {}
    cam_params: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for shard in shard_files:
        with tarfile.open(shard, "r") as tf:
            for m in tf:
                if not m.name.endswith("label.pyd"):
                    continue
                key = m.name.split(".")[0]
                fobj = tf.extractfile(m)
                assert fobj is not None
                L = pickle.loads(fobj.read())
                r = re.match(r".*/Capture(\d+)/([^/]+)/cam(\d+)/image(\d+)\.jpg", str(L["image_path"][0]))
                assert r is not None
                seq = f"Capture{r.group(1)}/{r.group(2)}/right"
                fid = int(r.group(4))
                cams = [re.match(r".*/cam(\d+)/", str(p)).group(1) for p in L["image_path"]]
                samples[(seq, fid)].append((shard, key, cams))
                for v_idx, c_id in enumerate(cams):
                    if c_id not in cam_params:
                        K_c = np.asarray(L["cam_intr"][v_idx], dtype=np.float64)
                        E_c = np.asarray(L["cam_extr"][v_idx], dtype=np.float64)
                        R_c, t_c = E_c[:3, :3], E_c[:3, 3]
                        cam_params[c_id] = (K_c, R_c, t_c)
                if key.split("_")[1] == "00":
                    E0 = np.asarray(L["cam_extr"][0], dtype=np.float64)
                    Xw = (np.asarray(L["verts_3d"][0], dtype=np.float64) - E0[:3, 3]) @ E0[:3, :3]
                    gt[(seq, fid)] = Xw.astype(np.float32)
                    betas[(seq, fid)] = np.asarray(L["mano_shape"][0], dtype=np.float32)

    mano = ManoModel(mano_root)
    Jreg = mano.J_reg["right"].astype(np.float64)
    tips = FINGERTIP_VERTS["interhand"]["right"]

    by_seq: dict[str, list[int]] = collections.defaultdict(list)
    for seq, fid in samples:
        by_seq[seq].append(fid)

    groups = sorted({key.split("_")[1] for v in samples.values() for _, key, _ in v})
    need_imgs: dict[str, dict[str, str]] = collections.defaultdict(dict)
    entries = []

    for seq in sorted(by_seq):
        fids = sorted(by_seq[seq])
        avail = {f: {c for _, _, cs in samples[(seq, f)] for c in cs} for f in fids}
        V = np.stack([gt[(seq, f)] for f in fids]).astype(np.float64)
        Jt = joints21(V, np.einsum("jv,tvc->tjc", Jreg, V), tips)
        beta = np.mean([betas[(seq, f)] for f in fids], axis=0)
        cap_name, seq_short = seq.split("/")[0], seq.split("/")[1]

        for f in fids:
            for shard, key, cs in samples[(seq, f)]:
                for v_idx, c_id in enumerate(cs):
                    need_imgs[shard][f"{key}.image_{v_idx}.jpg"] = os.path.join(
                        out_root, "interhand", "images", "val", cap_name, seq_short, f"cam{c_id}", f"image{f:04d}.jpg"
                    )

        for gg in groups:
            lists = collections.Counter(
                tuple(cs) for f in fids for _, key, cs in samples[(seq, f)] if key.endswith("_" + gg)
            )
            cams = list(lists.most_common(1)[0][0])
            Cn, n = len(cams), len(fids)
            K = np.stack([cam_params[c][0] for c in cams], axis=0)
            R = np.stack([cam_params[c][1] for c in cams], axis=0)
            t = np.stack([cam_params[c][2] for c in cams], axis=0)
            have = np.array([[c in avail[f] for f in fids] for c in cams], dtype=bool)

            boxes = np.full((Cn, n, 4), np.nan, dtype=np.float32)
            for ci in range(Cn):
                for fi in range(n):
                    if not have[ci, fi]:
                        continue
                    Xc = V[fi] @ R[ci].T + t[ci]
                    uv, z = _project_cam(K[ci], Xc)
                    ins = (uv[:, 0] >= 0) & (uv[:, 0] < 334) & (uv[:, 1] >= 0) & (uv[:, 1] < 512) & (z > 0)
                    if ins.mean() >= 0.3:
                        boxes[ci, fi] = _box_of(uv, ins)

            g_tag = f"g{gg}"
            out_npz = os.path.join(out_root, "interhand", "val", cap_name, g_tag, f"{seq_short}__right.npz")
            os.makedirs(os.path.dirname(out_npz), exist_ok=True)
            np.savez_compressed(
                out_npz,
                verts=V.astype(np.float32),
                joints=Jt.astype(np.float32),
                valid=np.ones(n, dtype=bool),
                K=K.astype(np.float32),
                R=R.astype(np.float32),
                t=t.astype(np.float32),
                boxes=boxes,
                betas=beta.astype(np.float32),
                side="right",
                serials=np.array(cams),
                img_pattern=np.array(
                    [
                        os.path.join(
                            out_root, "interhand", "images", "val", cap_name, seq_short, f"cam{c}", "image{:04d}.jpg"
                        )
                        for c in cams
                    ]
                ),
                img_size=np.array([334, 512]),
                img_frame_ids=np.array(fids, dtype=np.int64),
            )
            seen = ~np.isnan(boxes[..., 0])
            entries.append(
                {
                    "dataset": "interhand",
                    "name": seq,
                    "npz": out_npz,
                    "n_frames": n,
                    "n_cams": Cn,
                    "side": "right",
                    "splits": {g_tag: "val"},
                    "n_valid": int((seen.sum(axis=0) >= 2).sum()),
                    "tips": "interhand",
                }
            )

    for shard, mpp in sorted(need_imgs.items()):
        with tarfile.open(shard, "r") as tf:
            for m in tf:
                dst_img = mpp.get(m.name)
                if dst_img is None:
                    continue
                if not (os.path.isfile(dst_img) and os.path.getsize(dst_img) == m.size):
                    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                    fobj = tf.extractfile(m)
                    assert fobj is not None
                    with open(dst_img, "wb") as fo:
                        fo.write(fobj.read())

    os.makedirs(out_root, exist_ok=True)
    entries = sorted(entries, key=lambda e: (e["name"], sorted(e["splits"].keys())[0]))
    with open(os.path.join(out_root, "index_interhand.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"[InterHand-MV] Prepared {len(entries)} group-sequences (17 groups) -> {os.path.join(out_root, 'index_interhand.json')}")
    return entries


# ---------------------------------------------------------------------------
# 6. SHOW3D (Zero-Shot -> scene.json bundle referencing native rig0..rig7.mp4)
# ---------------------------------------------------------------------------

def prepare_show3d(
    raw_root: str | os.PathLike,
    mano_root: str | os.PathLike,
    out_root: str | os.PathLike,
    scenes: Sequence[str] | None = None,
) -> list[str]:
    """Export SHOW3D v3 scenes as ready-to-run `scene.json` bundles (zero video copying)."""
    raw = Path(raw_root).expanduser().resolve()
    out = Path(out_root).expanduser().resolve() / "show3d"
    mano = ManoModel(mano_root, fix_left_shapedirs=True)

    if not scenes:
        scenes_dir = raw / "scenes"
        scenes = [
            f"{subj.name}/{sc.name}"
            for subj in sorted(scenes_dir.iterdir())
            if subj.is_dir()
            for sc in sorted(subj.iterdir())
            if sc.is_dir() and (raw / "hand_pose" / "v3" / "scenes" / subj.name / sc.name).is_dir()
        ]

    written_manifests = []
    for scene_id in scenes:
        subject, scene = scene_id.split("/")
        base = raw / "scenes" / subject / scene
        annot = raw / "hand_pose" / "v3" / "scenes" / subject / scene
        profile = json.loads((raw / "hand_pose" / "hand_profiles" / subject / "profile_mano.json").read_text())
        ume = json.loads((annot / "hand_pose_umetrack.json").read_text())
        poses = json.loads((annot / "hand_pose_mano.json").read_text())
        rec = json.loads((base / "metadata" / "recording_info.json").read_text())
        frames_info = json.loads((base / "metadata" / "frame_info.json").read_text())
        n = int(rec["num_frames"])

        cam_names = [f"rig{i}" for i in range(8)]
        K_list, R_list, t_list, sizes = [], [], [], []
        for cn in cam_names:
            cal = json.loads((base / "camera_calibration" / f"{cn}.json").read_text())
            E = np.linalg.inv(np.asarray(cal["T_WorldFromCamera"], dtype=np.float64))
            E[:3, 3] /= 1000.0
            K_list.append([[cal["fx"], 0.0, cal["cx"]], [0.0, cal["fy"], cal["cy"]], [0.0, 0.0, 1.0]])
            R_list.append(E[:3, :3])
            t_list.append(E[:3, 3])
            sizes.append((int(cal["ImageSizeX"]), int(cal["ImageSizeY"])))
        K = np.asarray(K_list, dtype=np.float32)
        R = np.asarray(R_list, dtype=np.float32)
        t = np.asarray(t_list, dtype=np.float32)

        dst = out / f"{subject}__{scene}"
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / "K.npy", K)
        np.save(dst / "R.npy", R)
        np.save(dst / "t.npy", t)

        hands_manifest = {}
        for slot, side in [("0", "left"), ("1", "right")]:
            valid = np.array(
                [
                    ume[str(f)]["hand_poses"].get(slot, {}).get("confidence", 0) > 0.1
                    and all(
                        ume[str(f)]["hand_poses"].get(slot, {}).get(k) is not None
                        for k in ("wrist_rotation", "wrist_translation", "landmarks_3d_mm")
                    )
                    for f in range(n)
                ],
                dtype=bool,
            )
            ids = np.flatnonzero(valid)
            v = np.full((n, 778, 3), np.nan, dtype=np.float32)
            for chunk in np.array_split(ids, max(1, (len(ids) + 127) // 128)):
                if len(chunk) == 0:
                    continue
                h_list = [poses["frames"][str(f)]["hand_poses"][slot] for f in chunk]
                g = np.array([h["global_transform"] for h in h_list], dtype=np.float32)
                hp = np.array([h["pose_parameters"] for h in h_list], dtype=np.float32)
                vv, _ = mano.forward(side, profile["betas"], g[:, :3], hp, g[:, 3:], flat_hand_mean=False)
                v[chunk] = vv

            boxes = np.full((8, n, 4), np.nan, dtype=np.float32)
            for ci, cn in enumerate(cam_names):
                W_c, H_c = sizes[ci]
                available = np.array([cn not in frames_info[f]["missing_cameras"] for f in range(n)])
                for f in ids:
                    if not available[f]:
                        continue
                    Xc = v[f].astype(np.float64) @ R[ci].T + t[ci]
                    uv, z = _project_cam(K[ci].astype(np.float64), Xc)
                    ins = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W_c) & (uv[:, 1] >= 0) & (uv[:, 1] < H_c)
                    if ins.mean() >= 0.9 and (z > 0).all():
                        lo, hi = uv.min(axis=0), uv.max(axis=0)
                        if np.min(hi - lo) >= 12.0:
                            boxes[ci, f] = np.r_[lo, hi]
            if np.isfinite(boxes[..., 0]).any():
                b_fn = f"boxes_{side}.npy"
                np.save(dst / b_fn, boxes)
                hands_manifest[side] = b_fn

        manifest = {
            "name": f"show3d_{subject}_{scene}",
            "fps": float(rec["fps"]),
            "K": "K.npy",
            "R": "R.npy",
            "t": "t.npy",
            "videos": [str(base / f"{cn}.mp4") for cn in cam_names],
            "hands": hands_manifest,
        }
        m_path = dst / "scene.json"
        m_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        written_manifests.append(str(m_path))

    print(f"[SHOW3D] Exported {len(written_manifests)} scene.json bundles -> {out}")
    return written_manifests


# ---------------------------------------------------------------------------
# 7. GigaHands (Zero-Shot -> undistorted multi-view scene.json bundle)
# ---------------------------------------------------------------------------

def _gigahands_cameras(optim_path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for line in optim_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        a = line.split()
        K = np.array([[float(a[3]), 0.0, float(a[5])], [0.0, float(a[4]), float(a[6])], [0.0, 0.0, 1.0]])
        q = np.array(a[12:16], dtype=np.float64)
        R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        out[a[11]] = {
            "K": K,
            "R": R,
            "t": np.array(a[16:19], dtype=np.float64),
            "dist": np.array(a[7:11], dtype=np.float64),
            "size": (int(a[1]), int(a[2])),
        }
    return out


def _gigahands_hand_boxes(verts: np.ndarray, valid: np.ndarray, cam: dict[str, Any]) -> np.ndarray:
    Xc = verts.astype(np.float64) @ cam["R"].T + cam["t"]
    uv, z = _project_cam(cam["K"], Xc)
    W, H = cam["size"]
    inside = (z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    boxes = np.full((len(verts), 4), np.nan, dtype=np.float32)
    for fi in np.flatnonzero(valid):
        if inside[fi].mean() < 0.9:
            continue
        lo, hi = uv[fi].min(axis=0), uv[fi].max(axis=0)
        if np.min(hi - lo) >= 12.0:
            boxes[fi] = np.r_[lo, hi]
    return boxes


def _write_undistorted_mp4(
    cap: cv2.VideoCapture,
    dst_mp4: Path,
    cam: dict[str, Any],
    start: int,
    length: int,
    fps: float = 30.0,
) -> None:
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    W, H = cam["size"]
    maps = cv2.initUndistortRectifyMap(cam["K"], cam["dist"], None, cam["K"], (W, H), cv2.CV_32FC1)
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None

    proc = None
    writer = None
    if ffmpeg_bin:
        proc = subprocess.Popen(
            [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{W}x{H}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                str(dst_mp4),
            ],
            stdin=subprocess.PIPE,
        )
    else:
        writer = cv2.VideoWriter(str(dst_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    try:
        for f in range(start + length):
            ok, im = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended early at frame {f} while writing {dst_mp4}")
            if f >= start:
                rect = cv2.remap(im, *maps, interpolation=cv2.INTER_LINEAR)
                if proc is not None and proc.stdin is not None:
                    proc.stdin.write(np.ascontiguousarray(rect).tobytes())
                elif writer is not None:
                    writer.write(rect)
    finally:
        cap.release()
        if proc is not None:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait()
        if writer is not None:
            writer.release()


def prepare_gigahands(
    raw_root: str | os.PathLike,
    mano_root: str | os.PathLike,
    out_root: str | os.PathLike,
    scenes: Sequence[str] | None = None,
    num_cams: int = 8,
    num_frames: int = 192,
) -> list[str]:
    """Export GigaHands sequences (undistorted, camera-spread selected) as ready-to-run `scene.json` bundles."""
    raw = Path(raw_root).expanduser().resolve()
    out = Path(out_root).expanduser().resolve() / "gigahands"
    mano = ManoModel(mano_root)

    csv_map: dict[tuple[str, int], dict[str, str]] = {}
    csv_file = raw / "multiview_camera_video_map.csv"
    if csv_file.is_file():
        with csv_file.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                csv_map[(row["scene"], int(row["sequence"]))] = {
                    k: v for k, v in row.items() if k not in ("scene", "sequence") and v
                }

    targets: list[tuple[str, int]] = []
    if scenes:
        for item in scenes:
            if ":" in item:
                sc, sq = item.split(":", 1)
                targets.append((sc, int(sq)))
            else:
                params_dir = raw / "hand_poses" / item / "params"
                seqs = sorted(int(p.stem) for p in params_dir.glob("*.json"))
                targets.append((item, seqs[len(seqs) // 2]))
    else:
        hp_root = raw / "hand_poses"
        for sc_dir in sorted(hp_root.iterdir()):
            if not sc_dir.is_dir() or not (sc_dir / "params").is_dir():
                continue
            seqs = sorted(int(p.stem) for p in (sc_dir / "params").glob("*.json"))
            if seqs:
                targets.append((sc_dir.name, seqs[len(seqs) // 2]))

    written_manifests = []
    for scene, seq in targets:
        labels = raw / "hand_poses" / scene
        kp_dir = labels / "keypoints_3d" / f"{seq:03d}"
        chosen = {
            s: json.loads((kp_dir / f"chosen_frames_{s}.json").read_text())
            if (kp_dir / f"chosen_frames_{s}.json").is_file()
            else []
            for s in ("left", "right")
        }
        union = sorted(set(chosen["left"]) | set(chosen["right"]))
        if not union:
            continue
        params = json.loads((labels / "params" / f"{seq:03d}.json").read_text())
        cams = _gigahands_cameras(labels / "optim_params.txt")
        ignore = set((labels / "ignore_camera.txt").read_text().split()) if (labels / "ignore_camera.txt").is_file() else set()

        vid_map = csv_map.get((scene, seq), {})
        if not vid_map:
            v_scene = raw / "multiview_rgb_vids" / scene
            for c_name in cams:
                cand = v_scene / c_name / f"{seq:03d}.mp4"
                if cand.is_file():
                    vid_map[c_name] = f"{c_name}/{seq:03d}.mp4"

        names = [c for c in sorted(vid_map) if c in cams and c not in ignore]
        caps, counts, times = [], [], []
        valid_names = []
        for c in names:
            vf = raw / "multiview_rgb_vids" / scene / vid_map[c]
            if not vf.is_file():
                continue
            cap = cv2.VideoCapture(str(vf))
            if not cap.isOpened():
                continue
            cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ts_file = vf.with_suffix(".txt")
            if ts_file.is_file():
                ts_arr = np.array(
                    [int(x.split("_")[1]) for x in ts_file.read_text(encoding="utf-8").splitlines() if x.strip()],
                    dtype=np.int64,
                )
            else:
                ts_arr = np.arange(cnt, dtype=np.int64) * 33333
            valid_names.append(c)
            caps.append(cap)
            counts.append(cnt)
            times.append(ts_arr)

        ref_cnt = int(np.median(counts))
        keep_len = [i for i, cnt in enumerate(counts) if cnt >= ref_cnt and len(times[i]) >= cnt]
        for i, cap in enumerate(caps):
            if i not in keep_len:
                cap.release()
        names = [valid_names[i] for i in keep_len]
        caps = [caps[i] for i in keep_len]
        counts = [counts[i] for i in keep_len]
        times = [times[i] for i in keep_len]

        stop = min(min(counts) - 5, max(union) + 1)
        length = min(num_frames, stop)
        start = max(0, (stop - length) // 2)
        frames = np.arange(start, start + length)
        ts_clip = np.stack([t_arr[frames] for t_arr in times])
        frame_row = {f: i for i, f in enumerate(union)}

        hands = {}
        for side in ("left", "right"):
            if side not in params or not params[side].get("poses") or not chosen[side]:
                continue
            p_arr = np.asarray(params[side]["poses"], dtype=np.float32)
            rh = np.asarray(params[side]["Rh"], dtype=np.float32)
            th = np.asarray(params[side]["Th"], dtype=np.float32)
            v0, _ = mano.forward(side, params[side]["shapes"], p_arr[:, :3], p_arr[:, 3:], np.zeros_like(rh), flat_hand_mean=False)
            rot = Rotation.from_rotvec(rh).as_matrix().astype(np.float32)
            full_v = np.einsum("tij,tnj->tni", rot, v0) + th[:, None]
            v_clip = np.full((length, 778, 3), np.nan, dtype=np.float32)
            for idx_i, f in enumerate(frames):
                if int(f) in frame_row:
                    v_clip[idx_i] = full_v[frame_row[int(f)]]
            hands[side] = (v_clip, np.isin(frames, chosen[side]))

        if num_cams and len(names) > num_cams:
            dev_us = np.abs(ts_clip - np.median(ts_clip, axis=0)).max(axis=1)
            cover = np.array(
                [
                    min(
                        np.isfinite(_gigahands_hand_boxes(v_c, val_c, cams[c])[:, 0]).sum() / max(int(val_c.sum()), 1)
                        for v_c, val_c in hands.values()
                    )
                    for c in names
                ]
            )
            elig = [i for i in range(len(names)) if dev_us[i] <= 33334 and cover[i] >= 0.5]
            if len(elig) < num_cams:
                elig = [i for i in range(len(names)) if cover[i] >= 0.5] or list(range(len(names)))
            X_mean = np.nanmean(
                np.concatenate([v_c[val_c] for v_c, val_c in hands.values() if val_c.any()]).reshape(-1, 3),
                axis=0,
            )
            dirs = np.stack([-cams[c]["R"].T @ cams[c]["t"] - X_mean for c in names])
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            pick = [max(elig, key=lambda i: (cover[i], -i))]
            while len(pick) < min(num_cams, len(elig)):
                rest = [i for i in elig if i not in pick]
                pick.append(rest[int(np.argmax([np.arccos(np.clip(dirs[pick] @ dirs[i], -1.0, 1.0)).min() for i in rest]))])
            pick = sorted(pick)
            for i, cap in enumerate(caps):
                if i not in pick:
                    cap.release()
            names = [names[i] for i in pick]
            caps = [caps[i] for i in pick]

        dst = out / f"{scene}__{seq:03d}"
        dst.mkdir(parents=True, exist_ok=True)
        rel_videos = []
        for vi, (c_name, cap) in enumerate(zip(names, caps)):
            rel_v = f"videos/cam{vi:02d}.mp4"
            _write_undistorted_mp4(cap, dst / rel_v, cams[c_name], start=start, length=length, fps=30.0)
            rel_videos.append(rel_v)

        K = np.stack([cams[c]["K"] for c in names]).astype(np.float32)
        R = np.stack([cams[c]["R"] for c in names]).astype(np.float32)
        t = np.stack([cams[c]["t"] for c in names]).astype(np.float32)
        np.save(dst / "K.npy", K)
        np.save(dst / "R.npy", R)
        np.save(dst / "t.npy", t)

        hands_manifest = {}
        for side, (v_c, val_c) in hands.items():
            boxes = np.stack([_gigahands_hand_boxes(v_c, val_c, cams[c]) for c in names], axis=0)
            if np.isfinite(boxes[..., 0]).any():
                b_fn = f"boxes_{side}.npy"
                np.save(dst / b_fn, boxes)
                hands_manifest[side] = b_fn

        manifest = {
            "name": f"gigahands_{scene}_{seq:03d}",
            "fps": 30.0,
            "K": "K.npy",
            "R": "R.npy",
            "t": "t.npy",
            "videos": rel_videos,
            "hands": hands_manifest,
        }
        m_path = dst / "scene.json"
        m_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        written_manifests.append(str(m_path))

    print(f"[GigaHands] Exported {len(written_manifests)} scene.json bundles -> {out}")
    return written_manifests
