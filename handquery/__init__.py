"""HandQuery: Dense Vertex Queries for Multi-View 4D Hand Reconstruction."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "FINGERTIP_VERTS",
    "FrameSource",
    "HandQueryNet",
    "HandQueryPipeline",
    "ManoModel",
    "ManoPCALayer",
    "build_model",
    "fit_sequence",
    "joints21",
    "load_model",
    "load_ply",
    "predict_multiview_sequence",
]


def __getattr__(name: str):
    if name == "load_ply":
        from handquery.ply import load_ply

        return load_ply
    if name in ("FINGERTIP_VERTS", "ManoModel", "ManoPCALayer", "joints21"):
        from handquery import mano

        return getattr(mano, name)
    if name == "fit_sequence":
        from handquery.fit import fit_sequence

        return fit_sequence
    if name in ("FrameSource", "HandQueryPipeline", "predict_multiview_sequence"):
        from handquery import inference

        return getattr(inference, name)
    if name in ("build_model", "load_model"):
        from handquery import model

        return getattr(model, name)
    if name == "HandQueryNet":
        from handquery.model.network import HandQueryNet

        return HandQueryNet
    raise AttributeError(f"module 'handquery' has no attribute {name!r}")
