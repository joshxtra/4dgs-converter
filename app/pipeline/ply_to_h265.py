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


# --- Channel Tiling ---

def _tile_horizontal(arrays: list[np.ndarray], grid_h: int, grid_w: int) -> np.ndarray:
    """Tile multiple (N,) uint8 arrays horizontally into one row of grid_w-wide columns."""
    cols = [arr.reshape(grid_h, grid_w) for arr in arrays]
    return np.concatenate(cols, axis=1)


def tile_stream_position(pos_high: np.ndarray, pos_low: np.ndarray,
                         grid_h: int, grid_w: int) -> np.ndarray:
    """Tile position high+low → (2*grid_h, 3*grid_w) grayscale.

    Layout:
      Row 0: [posHi_X | posHi_Y | posHi_Z]
      Row 1: [posLo_X | posLo_Y | posLo_Z]
    """
    row_hi = _tile_horizontal([pos_high[:, i] for i in range(3)], grid_h, grid_w)
    row_lo = _tile_horizontal([pos_low[:, i] for i in range(3)], grid_h, grid_w)
    return np.concatenate([row_hi, row_lo], axis=0)


def tile_stream_motion(rot: np.ndarray, so: np.ndarray,
                       grid_h: int, grid_w: int) -> np.ndarray:
    """Tile rotation XYZW + scaleOpacity XY → (2*grid_h, 3*grid_w) grayscale.

    Layout:
      Row 0: [rot_X | rot_Y | rot_Z]
      Row 1: [rot_W | so_X  | so_Y ]
    """
    row0 = _tile_horizontal([rot[:, 0], rot[:, 1], rot[:, 2]], grid_h, grid_w)
    row1 = _tile_horizontal([rot[:, 3], so[:, 0], so[:, 1]], grid_h, grid_w)
    return np.concatenate([row0, row1], axis=0)


def tile_stream_appearance(so_zw: np.ndarray, sh_dc: np.ndarray,
                           grid_h: int, grid_w: int) -> np.ndarray:
    """Tile scaleOpacity ZW + shDC RGB → (grid_h, 5*grid_w) grayscale.

    Layout:
      Row 0: [so_Z | so_W | shDC_R | shDC_G | shDC_B]
    """
    return _tile_horizontal(
        [so_zw[:, 0], so_zw[:, 1], sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]],
        grid_h, grid_w,
    )


# --- ffmpeg H.265 Encoding ---

def start_encoder(width: int, height: int, fps: int, crf: int,
                  output_path: str) -> subprocess.Popen:
    """Start ffmpeg H.265 encoder reading raw grayscale frames from stdin.

    Input: grayscale (pix_fmt gray) raw frames via stdin.
    Output: YUV420p H.265 Main Profile. ffmpeg converts gray→yuv420p by
    placing gray data in Y plane and filling U/V with 128 (neutral chroma).
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-x265-params", f"keyint={fps}:min-keyint={fps}:log-level=error",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def write_frame(proc: subprocess.Popen, frame: np.ndarray):
    """Write one grayscale frame (H, W) uint8 to the encoder."""
    proc.stdin.write(frame.tobytes())


def finish_encoder(proc: subprocess.Popen):
    """Close stdin and wait for ffmpeg to finish. Returns stderr output."""
    proc.stdin.close()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {stderr.decode()}")
    return stderr.decode()


# --- Manifest Generation ---

def write_manifest(output_dir: str, sequence_name: str, frame_count: int,
                   fps: int, grid_w: int, grid_h: int,
                   gaussian_count: int, bounds: dict):
    """Write manifest.json with quantization bounds and stream metadata."""
    def _to_list(arr):
        return [round(float(x), 6) for x in arr]

    manifest = {
        "version": 1,
        "format": "4dgs-h265",
        "sequenceName": sequence_name,
        "frameCount": frame_count,
        "targetFPS": fps,
        "duration": round(frame_count / fps, 2),
        "shDegree": 0,
        "gridWidth": grid_w,
        "gridHeight": grid_h,
        "gaussianCount": gaussian_count,
        "requiredCodec": "hev1.1.6.L150.B0",
        "coordinateSpace": "colmap",
        "quaternionOrder": "wxyz",
        "quantization": {
            "position": {
                "precision": "uint16",
                "min": _to_list(bounds["position"]["min"]),
                "max": _to_list(bounds["position"]["max"]),
            },
            "rotation": {
                "precision": "uint8",
                "min": _to_list(bounds["rotation"]["min"]),
                "max": _to_list(bounds["rotation"]["max"]),
            },
            "scaleOpacity": {
                "precision": "uint8",
                "min": _to_list(bounds["scale_opacity"]["min"]),
                "max": _to_list(bounds["scale_opacity"]["max"]),
            },
            "shDC": {
                "precision": "uint8",
                "min": _to_list(bounds["sh_dc"]["min"]),
                "max": _to_list(bounds["sh_dc"]["max"]),
            },
        },
        "streams": {
            "position": {
                "file": "stream_position.mp4",
                "width": grid_w * 3,
                "height": grid_h * 2,
                "channels": 6,
            },
            "motion": {
                "file": "stream_motion.mp4",
                "width": grid_w * 3,
                "height": grid_h * 2,
                "channels": 6,
            },
            "appearance": {
                "file": "stream_appearance.mp4",
                "width": grid_w * 5,
                "height": grid_h,
                "channels": 5,
            },
        },
    }

    out_path = Path(output_dir) / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
