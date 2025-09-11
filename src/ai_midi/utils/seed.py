"""Reproducible seeding utilities.

Seeds Python, NumPy, and PyTorch. Also sets deterministic flags where possible.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed: int
        The base seed to use across libraries.
    deterministic: bool
        If True, set backend deterministic flags (may reduce performance).
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

