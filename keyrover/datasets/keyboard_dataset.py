from typing import Sequence

import torch
from torchvision.transforms import v2 as transforms

from .abstract import KeyboardDatasetBase


class KeyboardDataset(KeyboardDatasetBase):
    def __init__(self, paths: Sequence[torch.Tensor], _transforms=()):
        super().__init__()
        self._images = paths
        self._transforms = transforms.Compose(_transforms)

    def __getitem__(self, idx) -> torch.Tensor:
        img = self._images[idx]
        img = self._transforms(img)
        return img


__all__ = ["KeyboardDataset"]
