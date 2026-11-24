"""Dependency-free Wavefront/Stanford PLY triangle mesh loader (positions, faces, and RGB vertex colors)."""

from __future__ import annotations

import os
import struct

import numpy as np

_PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_ply(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a binary-little-endian or ASCII `.ply` mesh.

    Returns:
        verts: `(V, 3)` float32 vertex coordinates.
        faces: `(F, 3)` int64 triangle vertex indices.
        colors: `(V, 3)` uint8 RGB vertex colors, or `None` if the PLY has no color properties.
    """
    with open(path, "rb") as f:
        raw = f.read()

    end_marker = b"end_header\n"
    idx = raw.find(end_marker)
    if idx < 0:
        end_marker = b"end_header\r\n"
        idx = raw.find(end_marker)
    if idx < 0:
        raise ValueError(f"Invalid PLY file (missing end_header): {path}")

    header_lines = raw[:idx].decode("ascii", errors="ignore").splitlines()
    pos = idx + len(end_marker)

    fmt = "ascii"
    elements: list[tuple[str, int, list[tuple]]] = []
    for line in header_lines:
        parts = line.strip().split()
        if not parts or parts[0] == "comment":
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if parts[1] == "list":
                elements[-1][2].append(("list", parts[2], parts[3], parts[4]))
            else:
                elements[-1][2].append(("scalar", parts[1], parts[2]))

    if fmt == "ascii":
        body_lines = raw[pos:].decode("ascii", errors="ignore").splitlines()
        line_it = iter(ln.strip() for ln in body_lines if ln.strip())
        verts, colors, faces = None, None, []
        for el_name, count, props in elements:
            if el_name == "vertex":
                pnames = [p[2] for p in props]
                rows = [[float(x) for x in next(line_it).split()] for _ in range(count)]
                arr = np.asarray(rows, dtype=np.float64)
                verts = arr[:, [pnames.index("x"), pnames.index("y"), pnames.index("z")]].astype(np.float32)
                if all(c in pnames for c in ("red", "green", "blue")):
                    colors = arr[:, [pnames.index("red"), pnames.index("green"), pnames.index("blue")]].astype(np.uint8)
            elif el_name == "face":
                for _ in range(count):
                    vals = [int(x) for x in next(line_it).split()]
                    n_idx = vals[0]
                    poly = vals[1 : 1 + n_idx]
                    for k in range(1, len(poly) - 1):
                        faces.append([poly[0], poly[k], poly[k + 1]])
        assert verts is not None
        return verts, np.asarray(faces, dtype=np.int64), colors

    if fmt != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format {fmt!r} in {path}")

    verts = None
    colors = None
    faces_arr = None

    for el_name, count, props in elements:
        if el_name == "vertex":
            dtype = np.dtype([(p[2], _PLY_SCALAR_DTYPES[p[1]]) for p in props])
            vdata = np.frombuffer(raw, dtype=dtype, count=count, offset=pos)
            pos += count * dtype.itemsize
            verts = np.stack([vdata["x"], vdata["y"], vdata["z"]], axis=-1).astype(np.float32)
            names = set(dtype.names or ())
            if {"red", "green", "blue"}.issubset(names):
                colors = np.stack([vdata["red"], vdata["green"], vdata["blue"]], axis=-1).astype(np.uint8)
        elif el_name == "face":
            assert len(props) == 1 and props[0][0] == "list"
            _, cnt_type, idx_type, _ = props[0]
            c_dt = np.dtype(_PLY_SCALAR_DTYPES[cnt_type])
            i_dt = np.dtype(_PLY_SCALAR_DTYPES[idx_type])
            tri_stride = c_dt.itemsize + 3 * i_dt.itemsize
            # Fast vectorized path when all faces are triangles
            if pos + count * tri_stride <= len(raw):
                tri_dt = np.dtype([("n", c_dt), ("idx", i_dt, (3,))])
                fdata = np.frombuffer(raw, dtype=tri_dt, count=count, offset=pos)
                if np.all(fdata["n"] == 3):
                    faces_arr = fdata["idx"].astype(np.int64)
                    pos += count * tri_stride
                    continue
            # General polygon fallback
            tris = []
            c_fmt = "<" + c_dt.char
            for _ in range(count):
                n_v = struct.unpack_from(c_fmt, raw, pos)[0]
                pos += c_dt.itemsize
                poly = np.frombuffer(raw, dtype=i_dt, count=n_v, offset=pos).astype(np.int64)
                pos += n_v * i_dt.itemsize
                for k in range(1, n_v - 1):
                    tris.append([int(poly[0]), int(poly[k]), int(poly[k + 1])])
            faces_arr = np.asarray(tris, dtype=np.int64)
        else:
            dtype = np.dtype([(p[2], _PLY_SCALAR_DTYPES[p[1]]) for p in props if p[0] == "scalar"])
            pos += count * dtype.itemsize

    if verts is None or faces_arr is None:
        raise ValueError(f"PLY file missing vertex or face elements: {path}")
    return verts, faces_arr, colors
