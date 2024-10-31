from typing import Sequence
from tqdm.notebook import tqdm
from multiprocessing import Pool

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from .internal import get_mask_path, identity


class BinaryKeyboardSegmentationDataset(Dataset):
    _to_image = transforms.ToImage()

    def __init__(self, paths: Sequence[str], size: tuple[float, float] = None, masks_path: str = "blender/masks"):
        self._resize = identity if size is None else transforms.Resize(size)
        self._masks_path = masks_path

        with Pool() as p:
            images = list(tqdm(p.imap(self._get_img, paths), total=len(paths)))

        self._images, self._masks = zip(*images)
        # self._images = np.array(self._images, dtype="float32")
        # self._masks = np.array(self._masks, dtype="int32")

        self._transforms = identity
        self._augmentations = transforms.Compose([transforms.ToDtype(torch.float32, scale=True)])

    def set_transforms(self, val: Sequence[transforms.Transform]) -> None:
        self._transforms = transforms.Compose(val)

    def set_augmentations(self, val: Sequence[transforms.Transform]) -> None:
        self._augmentations = transforms.Compose(val)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        _img, _mask = self._transforms(self._images[idx], self._masks[idx])
        _img = self._augmentations(_img)
        _mask = _mask[0, :, :] > 1

        return _img, _mask

    def __len__(self) -> int:
        return len(self._images)

    def _get_img(self, _img: str) -> tuple[torch.Tensor, torch.Tensor]:
        img = self._resize(self._to_image(Image.open(_img)))
        mask = self._resize(self._to_image(Image.open(get_mask_path(_img, self._masks_path)).convert("L")))
        return img, mask


__all__ = ["BinaryKeyboardSegmentationDataset"]
