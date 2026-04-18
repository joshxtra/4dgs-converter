"""Minimal minibatch k-means on PyTorch CUDA.

Matches the signature of `ply_to_gsd_v2._vq_encode` so it can be swapped in
when CUDA is available. Returns (codebook float32 [k,d], indices uint8 [n]).
"""

from typing import Optional

import numpy as np


_torch = None
_device = None


def _get_torch():
    """Lazy import so non-GPU paths stay import-free."""
    global _torch, _device
    if _torch is None:
        import torch
        _torch = torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _torch, _device


def gpu_kmeans(
    data: np.ndarray,
    k: int = 256,
    n_iter: int = 10,
    n_sample: int = 50000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run k-means on the GPU.

    Args:
        data: (N, D) float32 numpy array. Must fit into GPU memory.
        k: Number of clusters (max 256 for uint8 indices).
        n_iter: Lloyd iterations on the sample for codebook fitting.
        n_sample: Subsample size for fitting (matches sklearn _vq_encode).
        seed: Random seed for reproducibility.

    Returns:
        (codebook float32 [k,d], indices uint8 [n])
    """
    if k > 256:
        raise ValueError(f"k={k} exceeds uint8 index range (max 256)")

    torch, device = _get_torch()
    n = data.shape[0]
    d = data.shape[1]

    # Upload full data once
    x = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).to(device)

    # Sample for codebook fit
    gen = torch.Generator(device=device).manual_seed(seed)
    sample_n = min(n_sample, n)
    sample_idx = torch.randperm(n, generator=gen, device=device)[:sample_n]
    sample = x[sample_idx]

    # Init centroids: k random points from the sample (k-means++ is overkill here)
    init_idx = torch.randperm(sample_n, generator=gen, device=device)[:k]
    centroids = sample[init_idx].clone()

    # Lloyd iterations on the sample
    for _ in range(n_iter):
        # (sample_n, k) squared distances via broadcasting in chunks
        assign = _nearest(sample, centroids)
        # Recompute centroids
        new_centroids = centroids.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centroids[j] = sample[mask].mean(dim=0)
        centroids = new_centroids

    # Assign all points to final centroids (chunked to bound memory)
    indices = _nearest(x, centroids).to(torch.uint8).cpu().numpy()
    codebook = centroids.cpu().numpy().astype(np.float32)
    return codebook, indices


def _nearest(points, centroids, chunk: int = 1_000_000):
    """Return argmin over centroids for each point, in chunks to save VRAM."""
    torch, _ = _get_torch()
    n = points.shape[0]
    out = torch.empty(n, dtype=torch.int64, device=points.device)
    c2 = (centroids * centroids).sum(dim=1)  # (k,)
    for i in range(0, n, chunk):
        p = points[i:i + chunk]
        # ||p - c||^2 = ||p||^2 + ||c||^2 - 2 p·c ; ||p||^2 is constant per row
        d = -2.0 * p @ centroids.T + c2
        out[i:i + chunk] = d.argmin(dim=1)
    return out
