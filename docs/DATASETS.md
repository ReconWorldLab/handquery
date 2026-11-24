# Multi-View Benchmark Evaluation & External Datasets

This guide explains how to download, prepare, evaluate, and interactively visualize **all 7 external multi-view hand datasets** supported out-of-the-box by HandQuery:

- **5 Quantitative Multi-View Benchmarks** (reproducing Table 1):
  1. [DexYCB-MV](#1-dexycb-mv) (`dexycb_mv`, 8 cameras)
  2. [HO3D-MV](#2-ho3d-mv) (`ho3d_mv`, 5 cameras)
  3. [OakInk-MV](#3-oakink-mv) (`oakink_mv`, 4 cameras)
  4. [ARCTIC-MV](#4-arctic-mv) (`arctic_mv`, 8 cameras)
  5. [InterHand-MV](#5-interhand-mv) (`interhand_mv`, 8 cameras × 17 groups)
- **2 Zero-Shot Multi-View Datasets** (exported directly to ready-to-infer `scene.json` bundles):
  6. [SHOW3D](#6-show3d-zero-shot) (8-camera rig)
  7. [GigaHands](#7-gigahands-zero-shot) (dense ~40-camera dome, undistorted & view-selected)

All dataset converters in [`handquery/datasets.py`](../handquery/datasets.py) run in pure Python + NumPy/OpenCV/MANO with **zero OpenGL, `pyrender`, or `EGL` dependencies**, and prepare **evaluation splits only** by default.

---

## Quick Start (2 Commands After Downloading)

### Step 1: Prepare Any Downloaded Datasets

Pass whichever raw dataset directories you have downloaded to [`scripts/prepare_datasets.py`](../scripts/prepare_datasets.py):

```bash
python scripts/prepare_datasets.py \
    --dexycb /path/to/DexYCB \
    --ho3d /path/to/HO3D_v3 \
    --oakink /path/to/OakInk \
    --arctic /path/to/ARCTIC \
    --interhand-mv /path/to/POEM_Interhand_mv \
    --mano-root assets \
    --out data/benchmarks
```

### Step 2: Run Multi-GPU Quantitative Evaluation

[`scripts/evaluate.py`](../scripts/evaluate.py) automatically evaluates all prepared benchmarks found in `--data-root` (or a specific `--benchmarks` subset) across one or more GPUs, applies the built-in paper frame-subset lists under [`assets/subsets/`](../assets/subsets/), and prints the full 7-metric summary table:

```bash
python scripts/evaluate.py \
    --data-root data/benchmarks \
    --ckpt checkpoints/handquery_vitb.pt \
    --mano-root assets \
    --gpus 0 1 2 3 \
    --out outputs/eval
```

> **Resume Support:** Per-sequence `metrics.json` and `mano_fit.npz` files are saved incrementally under `outputs/eval/`. Interrupted runs resume automatically unless `--no-resume` is passed.

---

## Expected Quantitative Results (`handquery_vitb.pt`)

All errors are reported in **millimeters (mm)** (`AUC-V` uses a `50 mm` upper threshold on `HO3D-MV` and `20 mm` on the other four benchmarks; `MPJPE*`, `RR-J*`, and `PA-J*` denote POEM's standardized 21-joint regressor applied identically to predicted and ground-truth MANO meshes):

| Benchmark | Views | Scored Frames | MPVPE ↓ | RR-V ↓ | PA-V ↓ | AUC-V ↑ | MPJPE\* ↓ | RR-J\* ↓ | PA-J\* ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DexYCB-MV** | 8 | 4,951 | **4.38** | 5.09 | **2.77** | 0.784 | 4.26 | 5.12 | 2.72 |
| **HO3D-MV** | 5 | 2,706 | **5.63** | 6.81 | **3.45** | 0.887 | 5.30 | 6.52 | 2.96 |
| **OakInk-MV** | 4 | 4,978 | **5.26** | 5.86 | **3.37** | 0.737 | 4.97 | 5.67 | 3.00 |
| **ARCTIC-MV** | 8 | 17,393 | **4.48** | 6.38 | **3.41** | 0.776 | 3.94 | 6.09 | 2.89 |
| **InterHand-MV** | 8 (×17 groups) | 85,255 | **3.68** | 4.46 | **2.59** | 0.817 | 3.29 | 4.43 | 2.09 |

---

## Dataset Download & Expected Directory Layouts

Due to dataset licensing terms, raw archives must be downloaded from each benchmark's official release page. Below are the **minimal files required for evaluation** and the expected directory structures.

### 1. DexYCB-MV

- **Official Website:** <https://dex-ycb.github.io/>
- **Minimal Download:** `dex-ycb-20210415.tar.gz` (depth images are not needed — you can unpack RGB + calibration + labels only via `tar -xzf dex-ycb-20210415.tar.gz --exclude='*aligned_depth_to_color_*'`).
- **Expected Layout (`--dexycb /path/to/DexYCB`):**
  ```text
  DexYCB/
  ├── calibration/
  │   ├── extrinsics_*/extrinsics.yml
  │   ├── intrinsics/*_640x480.yml
  │   └── mano_*/mano.yml
  ├── 20200709-subject-01/
  ├── ...
  └── 20201022-subject-10/
      └── <seq_id>/<cam_serial>/{color_XXXXXX.jpg, labels_XXXXXX.npz}
  ```

### 2. HO3D-MV

- **Official Website:** <https://www.tugraz.at/index.php?id=40231> (HO-3D v3)
- **Minimal Download:** `HO3D_v3.zip` only (segmentation masks and YCB object models are **not** required).
- **Expected Layout (`--ho3d /path/to/HO3D_v3`):**
  ```text
  HO3D_v3/
  ├── calibration/
  │   ├── GPMF1/calibration/{cam_orders.txt, trans_0.txt, ...}
  │   └── SB1/calibration/{cam_orders.txt, trans_0.txt, ...}
  ├── train/
  │   ├── GPMF10..GPMF14/{rgb/*.jpg, meta/*.pkl}
  │   └── SB10..SB14/{rgb/*.jpg, meta/*.pkl}
  └── evaluation/
      └── SB11, SB13/{rgb/*.jpg, meta/*.pkl}
  ```

### 3. OakInk-MV

- **Official Website:** <https://oakink.net/> (OakInk-Image v2)
- **Minimal Download:** `anno_v2.1.zip` (or `anno/`) and `oakink_image_v2.zip` (`image/stream_release_v2/`; `obj.zip` is **not** required for evaluation).
- **Expected Layout (`--oakink /path/to/OakInk`):**
  ```text
  OakInk/
  ├── anno/
  │   ├── seq_all.json
  │   ├── split/split2/seq_test.json
  │   ├── cam_intr/*.pkl
  │   ├── general_info/*.pkl
  │   ├── hand_v/*.pkl
  │   └── hand_j/*.pkl
  └── image/
      └── stream_release_v2/<seq>/{north_east,south_east,north_west,south_west}_color_<frame>.png
  ```

### 4. ARCTIC-MV

- **Official Website:** <https://arctic.is.tue.mpg.de/>
- **Minimal Download:** `meta.zip`, `raw_seqs.zip`, `splits_json.zip`, and `cropped_images_zips/` (for `protocol_p1` `val` evaluation only, unpacking subject `s05` cropped image zips is sufficient; unpack all subjects if `--all-splits` is desired).
- **Expected Layout (`--arctic /path/to/ARCTIC`):**
  ```text
  ARCTIC/
  ├── meta/
  │   ├── misc.json
  │   └── object_vtemplates/<object>/{mesh.obj, parts.json}
  ├── splits_json/protocol_p1.json
  ├── raw_seqs/<sid>/<seq>.{mano.npy, object.npy}
  └── cropped_images/<sid>/<seq>/<1..8>/<frame>.jpg
  ```

### 5. InterHand-MV

- **Official Source:** POEM-v2's packaged 30-fps 17-camera-group `Interhand_mv` WebDataset tar shards (<https://github.com/LinyuXie/POEM-v2>).
- **Minimal Download:** Place the `.tar` shards of `Interhand_mv` in a single folder. Camera intrinsics, extrinsics, 30-fps validation images, and ground-truth meshes are self-contained inside the shards (the separate multi-hundred-GB raw InterHand2.6M archive is **not** needed).
- **Expected Layout (`--interhand-mv /path/to/POEM_Interhand_mv`):**
  ```text
  POEM_Interhand_mv/
  └── *.tar
  ```

---

### 6. SHOW3D (Zero-Shot)

- **Official Website:** <https://github.com/facebookresearch/show3d> (SHOW3D v3)
- **Expected Layout (`--show3d /path/to/SHOW3D`):**
  ```text
  SHOW3D/
  ├── scenes/<subject>/<scene>/
  │   ├── rig0.mp4 .. rig7.mp4
  │   ├── camera_calibration/rig0.json .. rig7.json
  │   └── metadata/{recording_info.json, frame_info.json}
  └── hand_pose/
      ├── hand_profiles/<subject>/profile_mano.json
      └── v3/scenes/<subject>/<scene>/{hand_pose_umetrack.json, hand_pose_mano.json}
  ```
- **Prepare & Run Inference (Zero Video Copying):**
  ```bash
  python scripts/prepare_datasets.py \
      --show3d /path/to/SHOW3D \
      --show3d-scenes <subject>/<scene> \
      --mano-root assets \
      --out data/zeroshot

  python scripts/infer.py \
      --input data/zeroshot/show3d/<subject>__<scene>/scene.json \
      --ckpt checkpoints/handquery_vitb.pt \
      --mano-root assets \
      --out outputs/show3d_demo
  ```

### 7. GigaHands (Zero-Shot)

- **Official Website:** <https://gigahands.github.io/>
- **Expected Layout (`--gigahands /path/to/GigaHands`):**
  ```text
  GigaHands/
  ├── multiview_camera_video_map.csv          # optional if videos follow <cam>/<seq>.mp4
  ├── multiview_rgb_vids/<scene>/<cam>/<seq>.mp4
  └── hand_poses/<scene>/
      ├── optim_params.txt
      ├── params/<seq>.json
      └── keypoints_3d/<seq>/chosen_frames_{left,right}.json
  ```
- **Prepare (Undistort + Select Spread Views) & Run Inference:**
  ```bash
  python scripts/prepare_datasets.py \
      --gigahands /path/to/GigaHands \
      --gigahands-scenes <scene>:<seq> \
      --gigahands-cams 8 \
      --gigahands-frames 192 \
      --mano-root assets \
      --out data/zeroshot

  python scripts/infer.py \
      --input data/zeroshot/gigahands/<scene>__<seq>/scene.json \
      --ckpt checkpoints/handquery_vitb.pt \
      --mano-root assets \
      --out outputs/gigahands_demo
  ```

---

## Viewing Reconstructed External Datasets in `viser`

[`scripts/view_viser.py`](../scripts/view_viser.py) natively loads **both** `scene.json` bundles (`SHOW3D`, `GigaHands`, custom rigs) **and** prepared benchmark `.npz` sequences (`DexYCB`, `HO3D`, `OakInk`, `ARCTIC`, `InterHand-MV`):

```bash
# View a prepared benchmark sequence (e.g. HO3D GPMF1):
python scripts/view_viser.py \
    --input data/benchmarks/ho3d/GPMF1.npz \
    --result outputs/eval/ho3d_mv/GPMF1 \
    --port 8080

# View a zero-shot SHOW3D or GigaHands sequence:
python scripts/view_viser.py \
    --input data/zeroshot/show3d/<subject>__<scene>/scene.json \
    --result outputs/show3d_demo \
    --port 8080
```
