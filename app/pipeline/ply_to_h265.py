"""PLY sequence → H.265 encoded MP4 streams for web/UE5 streaming."""

import math
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from app.utils.ply_reader import load_gaussian_ply
from app.utils.morton import sort_3d_morton_order


# --- Quantization ---

def quantize_position(pos: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray):
    """Quantize (N,3) float32 position → uint16 → split into high/low uint8."""
    norm = (pos - bounds_min) / (bounds_max - bounds_min)
    u16 = (norm * 65535).clip(0, 65535).astype(np.uint16)
    high = (u16 >> 8).astype(np.uint8)
    low = (u16 & 0xFF).astype(np.uint8)
    return high, low


def dequantize_position(high: np.ndarray, low: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray):
    """Reconstruct float32 position from high/low uint8 (for verification)."""
    u16 = high.astype(np.uint16) << 8 | low.astype(np.uint16)
    norm = u16.astype(np.float32) / 65535.0
    return bounds_min + norm * (bounds_max - bounds_min)


def quantize_uint8(data: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray):
    """Quantize float data to uint8 using min-max normalization."""
    norm = (data - bounds_min) / (bounds_max - bounds_min)
    return (norm * 255).clip(0, 255).astype(np.uint8)


def dequantize_uint8(data: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray):
    """Reconstruct float from uint8 (for verification)."""
    norm = data.astype(np.float32) / 255.0
    return bounds_min + norm * (bounds_max - bounds_min)


# --- Activation & Bounds ---

def activate_gaussians(gaussians: dict) -> dict:
    """Apply activations to raw PLY attributes (no coordinate transform).

    Coordinate space: COLMAP (raw PLY values). The COLMAP-to-rendering
    transform (axis swaps, scaling) is deferred to the playback shader.

    Quaternion order: (W, X, Y, Z) — matches PLY convention where
    rot_0=W, rot_1=X, rot_2=Y, rot_3=Z.

    Returns dict with:
      position: (N, 3) float32 — raw COLMAP coords (x, y, z)
      rotation: (N, 4) float32 — normalized quaternion (W, X, Y, Z)
      scale_opacity: (N, 4) float32 — [exp(s0), exp(s1), exp(s2), sigmoid(opa)]
      sh_dc: (N, 3) float32 — DC color coefficients
    """
    rot = gaussians["rotation"].copy()
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)

    scale = np.exp(gaussians["scale"])
    opacity = 1.0 / (1.0 + np.exp(-gaussians["opacity"]))
    so = np.column_stack([scale, opacity])

    return {
        "position": gaussians["position"],
        "rotation": rot,
        "scale_opacity": so,
        "sh_dc": gaussians["sh_dc"],
    }


def compute_global_bounds(ply_paths: list[Path], sample_ratio: float = 0.1,
                          points_per_frame: int = 50000, seed: int = 42) -> dict:
    """Sample frames to compute global min/max per attribute with percentile clipping.

    Rotation uses fixed bounds [-1, 1] (quaternion components are always in
    this range after normalization), not percentile-based, to avoid clipping.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    n_sample = max(3, int(len(ply_paths) * sample_ratio))
    sampled = rng.sample(list(ply_paths), min(n_sample, len(ply_paths)))

    all_vals = {"position": [], "scale_opacity": [], "sh_dc": []}

    for path in sampled:
        gaussians = load_gaussian_ply(str(path))
        activated = activate_gaussians(gaussians)
        n = len(activated["position"])
        idx = np_rng.choice(n, min(points_per_frame, n), replace=False)

        for key in all_vals:
            all_vals[key].append(activated[key][idx])

    bounds = {}
    for key in all_vals:
        merged = np.concatenate(all_vals[key])
        bounds[key] = {
            "min": np.percentile(merged, 0.1, axis=0).astype(np.float32),
            "max": np.percentile(merged, 99.9, axis=0).astype(np.float32),
        }

    # Rotation: fixed bounds [-1, 1] (quaternion components always in this range)
    bounds["rotation"] = {
        "min": np.full(4, -1.0, dtype=np.float32),
        "max": np.full(4, 1.0, dtype=np.float32),
    }

    return bounds
