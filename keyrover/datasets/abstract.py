from typing import Sequence
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from keyrover.ml import identity


class KeyboardDatasetBase(Dataset):
    def __init__(self):
        self._images = []
        self._targets = []

        self._transforms = identity
        self._augmentations = transforms.Compose([transforms.ToDtype(torch.float32, scale=True)])
        self._target_augmentation = identity

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img = self._images[idx]
        target = self._targets[idx]

        img, target = self._transforms(img, target)
        img = self._augmentations(img)
        target = self._target_augmentation(target)

        return img, target

    def set_transforms(self, val: Sequence[transforms.Transform]) -> None:
        self._transforms = transforms.Compose(val)

    def set_augmentations(self, val: Sequence[transforms.Transform]) -> None:
        self._augmentations = transforms.Compose(val)

    def random_img(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self[random.randint(0, len(self) - 1)]
