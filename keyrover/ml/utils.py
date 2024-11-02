from typing import Literal

import numpy as np
import torch


def get_device() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def identity(*x: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
    if len(x) == 1:
        return x[0]
    return x


__all__ = ["get_device", "identity"]
