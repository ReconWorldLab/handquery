"""MANO parametric hand model helpers: chumpy-free loading, 45-DoF PCA decoding, and 21-joint assembly.

Expected MANO directory layout under `mano_root`:
    <mano_root>/mano/MANO_RIGHT.pkl
    <mano_root>/mano/MANO_LEFT.pkl
(or directly `<mano_root>/MANO_RIGHT.pkl` and `<mano_root>/MANO_LEFT.pkl`, both are supported automatically).
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch

# Standard fingertip vertex indices (thumb, index, middle, ring, little) across common conventions.
FINGERTIP_VERTS: dict[str, dict[str, list[int]]] = {
    "smplx": {"right": [744, 320, 443, 554, 671], "left": [744, 320, 443, 554, 671]},
    "dexycb": {"right": [745, 317, 444, 556, 673], "left": [745, 317, 445, 556, 673]},
    "ho3d": {"right": [744, 333, 444, 555, 672], "left": [744, 333, 444, 555, 672]},
    "oakink": {"right": [745, 317, 444, 556, 673], "left": [745, 317, 444, 556, 673]},
    "arctic": {"right": [744, 320, 443, 554, 671], "left": [744, 320, 443, 554, 671]},
    "interhand": {"right": [745, 318, 445, 556, 673], "left": [745, 318, 445, 556, 673]},
}

# smplx MANO regressed joints: 0 wrist, 1-3 index, 4-6 middle, 7-9 little, 10-12 ring, 13-15 thumb.
# Standard 21-joint order: wrist, then thumb, index, middle, ring, little (root -> tip); -1 marks tip vertex.
_J21_ORDER = [0, 13, 14, 15, -1, 1, 2, 3, -1, 4, 5, 6, -1, 10, 11, 12, -1, 7, 8, 9, -1]


def resolve_mano_paths(mano_root: str | os.PathLike, side: str) -> tuple[str, str]:
    """Return `(smplx_model_root, pkl_file_path)` whether `mano_root` points to `assets` or `assets/mano`."""
    root = os.path.expanduser(os.fspath(mano_root))
    fname = f"MANO_{side.upper()}.pkl"
    sub_pkl = os.path.join(root, "mano", fname)
    direct_pkl = os.path.join(root, fname)
    if os.path.isfile(sub_pkl):
        return root, sub_pkl
    if os.path.isfile(direct_pkl):
        return os.path.dirname(os.path.abspath(root)), direct_pkl
    raise FileNotFoundError(
        f"Could not find {fname} under '{root}/mano/{fname}' or '{root}/{fname}'. "
        "Please download MANO_RIGHT.pkl and MANO_LEFT.pkl from https://mano.is.tue.mpg.de/."
    )


def load_mano_pkl(pkl_path: str) -> dict:
    """Load a MANO `.pkl` file without requiring `chumpy` to be installed."""
    with open(pkl_path, "rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except ModuleNotFoundError as exc:
            if "chumpy" not in str(exc):
                raise
    import sys
    import types

    ch = types.ModuleType("chumpy")
    ch_ch = types.ModuleType("chumpy.ch")

    class _Ch:
        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})

        def __array__(self, dtype=None):
            arr = np.asarray(self.r)
            return arr.astype(dtype) if dtype is not None else arr

    ch.Ch = _Ch
    ch_ch.Ch = _Ch
    ch.array = np.array
    sys.modules.setdefault("chumpy", ch)
    sys.modules.setdefault("chumpy.ch", ch_ch)
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    return {k: (np.asarray(v) if isinstance(v, _Ch) else v) for k, v in raw.items()}


def joints21(verts, joints16, tips: list[int] | tuple[int, ...]):
    """Assemble 21 3D joints `(..., 21, 3)` from `(..., 778, 3)` vertices and `(..., 16, 3)` regressed joints."""
    out, k = [], 0
    for j in _J21_ORDER:
        if j >= 0:
            out.append(joints16[..., j, :])
        else:
            out.append(verts[..., tips[k], :])
            k += 1
    stack = torch.stack if torch.is_tensor(verts) else np.stack
    return stack(out, -2)


class ManoModel:
    """CPU/GPU batched MANO forward helper for both hands (used by dataset preparation and evaluation)."""

    def __init__(self, mano_root: str | os.PathLike, fix_left_shapedirs: bool = False):
        import smplx

        self.layers = {}
        self.comps = {}
        self.mean = {}
        self.J_reg = {}
        for side in ("right", "left"):
            smplx_root, pkl_path = resolve_mano_paths(mano_root, side)
            self.layers[side] = smplx.create(
                smplx_root,
                model_type="mano",
                is_rhand=(side == "right"),
                use_pca=False,
                flat_hand_mean=True,
                batch_size=1,
            )
            d = load_mano_pkl(pkl_path)
            self.comps[side] = torch.as_tensor(np.asarray(d["hands_components"][:45], dtype=np.float32))
            self.mean[side] = torch.as_tensor(np.asarray(d["hands_mean"], dtype=np.float32))
            J = d["J_regressor"]
            self.J_reg[side] = np.asarray(J.toarray() if hasattr(J, "toarray") else J, dtype=np.float32)

        if fix_left_shapedirs:
            left_sd = self.layers["left"].shapedirs
            right_sd = self.layers["right"].shapedirs
            if torch.sum(torch.abs(left_sd[:, 0, :] - right_sd[:, 0, :])) < 1.0:
                with torch.no_grad():
                    left_sd[:, 0, :] *= -1.0

        self.faces = np.asarray(self.layers["right"].faces, dtype=np.int64)

    def pca_to_axis_angle(self, side: str, coeffs) -> torch.Tensor:
        c = torch.as_tensor(coeffs, dtype=torch.float32).reshape(-1, 45)
        return c @ self.comps[side] + self.mean[side]

    @torch.no_grad()
    def forward(
        self,
        side: str,
        betas,
        global_orient,
        hand_pose,
        transl,
        flat_hand_mean: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched axis-angle MANO forward returning `(verts (N, 778, 3), joints16 (N, 16, 3))` in meters."""
        side = side.lower()
        go = torch.as_tensor(global_orient, dtype=torch.float32).reshape(-1, 3)
        n = go.shape[0]
        b = torch.as_tensor(betas, dtype=torch.float32).reshape(-1, 10).expand(n, 10)
        hp = torch.as_tensor(hand_pose, dtype=torch.float32).reshape(n, 45)
        if not flat_hand_mean:
            hp = hp + self.mean[side]
        tr = torch.as_tensor(transl, dtype=torch.float32).reshape(n, 3)
        out = self.layers[side](betas=b, global_orient=go, hand_pose=hp, transl=tr)
        return out.vertices.numpy(), out.joints[:, :16].numpy()

    def forward_dexycb(self, side: str, betas, pose_m) -> tuple[np.ndarray, np.ndarray]:
        """Decode DexYCB `pose_m` `(N, 51) = [global_orient (3) | 45 PCA | transl (3)]`."""
        pm = np.asarray(pose_m, dtype=np.float32).reshape(-1, 51)
        return self.forward(side, betas, pm[:, :3], self.pca_to_axis_angle(side, pm[:, 3:48]), pm[:, 48:])


class ManoPCALayer:
    """Differentiable MANO layer driven by 45 PCA pose coefficients without GPU->CPU kinematic syncs.

    Matches `smplx` MANO (`use_pca=False, flat_hand_mean=True`) op-for-op while walking the kinematic
    tree with Python integers so forward + backward passes can be captured inside a CUDA graph.
    """

    def __init__(self, mano_root: str | os.PathLike, side: str, device: str | torch.device = "cuda"):
        import smplx

        side = side.lower()
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")
        smplx_root, pkl_path = resolve_mano_paths(mano_root, side)
        self.layer = smplx.create(
            smplx_root,
            model_type="mano",
            is_rhand=(side == "right"),
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)
        d = load_mano_pkl(pkl_path)
        self.comps = torch.as_tensor(np.asarray(d["hands_components"][:45], dtype=np.float32), device=device)
        self.mean = torch.as_tensor(np.asarray(d["hands_mean"], dtype=np.float32), device=device)
        self.parents = [int(p) for p in self.layer.parents.cpu()]
        self.faces = np.asarray(self.layer.faces, dtype=np.int64)

    def pca_to_axis_angle(self, coeffs: torch.Tensor) -> torch.Tensor:
        return coeffs @ self.comps + self.mean

    def __call__(
        self,
        glob: torch.Tensor,
        coeffs: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from smplx.lbs import batch_rodrigues, transform_mat

        L = self.layer
        betas = betas.expand(glob.shape[0], -1)
        full_pose = torch.cat([glob, coeffs @ self.comps + self.mean], dim=1)
        full_pose = full_pose + L.pose_mean
        B = full_pose.shape[0]
        v_shaped = L.v_template + torch.einsum("bl,mkl->bmk", [betas, L.shapedirs])
        J = torch.einsum("bik,ji->bjk", [v_shaped, L.J_regressor])
        ident = torch.eye(3, dtype=betas.dtype, device=betas.device)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view([B, -1, 3, 3])
        pose_offsets = torch.matmul((rot_mats[:, 1:, :, :] - ident).view([B, -1]), L.posedirs).view(B, -1, 3)
        v_posed = pose_offsets + v_shaped
        joints = torch.unsqueeze(J, dim=-1)
        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, L.parents[1:]]
        tm = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(
            -1, joints.shape[1], 4, 4
        )
        chain = [tm[:, 0]]
        for i in range(1, len(self.parents)):
            chain.append(torch.matmul(chain[self.parents[i]], tm[:, i]))
        transforms = torch.stack(chain, dim=1)
        A = transforms - torch.nn.functional.pad(
            torch.matmul(transforms, torch.nn.functional.pad(joints, [0, 0, 0, 1])),
            [3, 0, 0, 0, 0, 0, 0, 0],
        )
        W = L.lbs_weights.unsqueeze(dim=0).expand([B, -1, -1])
        T = torch.matmul(W, A.view(B, J.shape[1], 16)).view(B, -1, 4, 4)
        v_homo = torch.cat(
            [v_posed, torch.ones([B, v_posed.shape[1], 1], dtype=betas.dtype, device=betas.device)], dim=2
        )
        verts = torch.matmul(T, torch.unsqueeze(v_homo, dim=-1))[:, :, :3, 0]
        return verts + transl.unsqueeze(dim=1), (transforms[:, :, :3, 3] + transl.unsqueeze(dim=1))[:, :16]
