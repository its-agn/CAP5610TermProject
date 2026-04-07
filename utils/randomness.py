"""Shared reproducibility helpers."""

import random
import numpy as np

DEFAULT_SEED = 0

def set_random_seed(seed: int = DEFAULT_SEED, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and Torch (if available) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
