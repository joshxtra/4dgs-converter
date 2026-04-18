"""Shared helpers for worker-pool sizing."""

import os


def default_workers() -> tuple[int, str]:
    """Pick a sensible default worker count.

    Returns (n_workers, reason_string). Targets cpu_count - 1 but caps at
    ~2 GB/worker when psutil is available so we don't OOM on big scenes.
    """
    cpu = os.cpu_count() or 4
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / 1e9
        mem_cap = max(1, int(ram_gb // 2))
        n = max(1, min(cpu - 1, mem_cap))
        return n, f"cpu={cpu}, ram={ram_gb:.1f}GB, cap={mem_cap}"
    except ImportError:
        n = max(1, cpu - 1)
        return n, f"cpu={cpu} (psutil missing, no memory cap)"
