from typing import Sequence
import random

from tqdm.notebook import tqdm
from multiprocessing import Pool

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from keyrover.ml import identity
from keyrover import RAW_MASKS
from .internal import get_mask_path


class BinaryKeyboardSegmentationDataset(Dataset):
    _to_image = transforms.ToImage()

    def __init__(self, paths: Sequence[str], size: tuple[float, float] = None, masks_path: str = RAW_MASKS):
        self._resize = identity if size is None else transforms.Resize(size)
        self._masks_path = masks_path

        with Pool() as p:
            images = list(tqdm(p.imap(self._get_img, paths), total=len(paths)))

        self._images, self._masks = zip(*images)

        self._transforms = identity
        self._augmentations = transforms.Compose([transforms.ToDtype(torch.float32, scale=True)])

    def set_transforms(self, val: Sequence[transforms.Transform]) -> None:
        self._transforms = transforms.Compose(val)

    def set_augmentations(self, val: Sequence[transforms.Transform]) -> None:
        self._augmentations = transforms.Compose(val)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self._transforms(self._images[idx], self._masks[idx])
        img = self._augmentations(img)
        mask = mask[0, :, :] > 1

        return img, mask

    def __len__(self) -> int:
        return len(self._images)

    def _get_img(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        img = self._resize(self._to_image(Image.open(path)))
        mask = self._resize(self._to_image(Image.open(get_mask_path(path, self._masks_path)).convert("L")))
        return img, mask

    def random_img(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self[random.randint(0, len(self) - 1)]


__all__ = ["BinaryKeyboardSegmentationDataset"]
