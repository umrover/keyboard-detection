from typing import Sequence
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from keyrover.vision import identity


class KeyboardDataset(Dataset):
    def __init__(self, images, targets):
        self._images = images
        self._targets = targets

        self._transforms = identity
        self._input_augmentations = identity
        self._target_augmentations = identity

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._images[item]
        target = self._targets[item]

        image, target = self._transforms(image, target)
        image = self._input_augmentations(image)
        target = self._target_augmentations(target)

        return image, target

    def set_transforms(self, val: Sequence[transforms.Transform]) -> None:
        self._transforms = transforms.Compose(val)

    def set_input_augmentations(self, val: Sequence[transforms.Transform]) -> None:
        self._input_augmentations = transforms.Compose(val)

    def set_target_augmentations(self, val: Sequence[transforms.Transform]) -> None:
        self._target_augmentations = transforms.Compose(val)

    def random_img(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self[random.randint(0, len(self) - 1)]
