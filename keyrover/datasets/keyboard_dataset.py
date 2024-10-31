from typing import Sequence

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class KeyboardDataset(Dataset):
    def __init__(self, paths: Sequence[torch.Tensor], _transforms=()):
        self._images = paths
        self._transforms = transforms.Compose(_transforms)

    def __getitem__(self, idx) -> torch.Tensor:
        img = self._images[idx]
        img = self._transforms(img)
        return img

    def __len__(self) -> int:
        return len(self._images)


__all__ = ["KeyboardDataset"]
