from typing import Sequence
import random

from tqdm.notebook import tqdm
from multiprocessing import Pool

from PIL import Image

import torch
from torchvision.transforms import v2 as transforms

from keyrover.ml import identity
from keyrover import RAW_MASKS
from .internal import get_mask_path
from .abstract import KeyboardDatasetBase


class BinaryKeyboardSegmentationDataset(KeyboardDatasetBase):
    _to_image = transforms.ToImage()

    def __init__(self, paths: Sequence[str], size: tuple[float, float] = None, masks_path: str = RAW_MASKS):
        super().__init__()

        self._resize = identity if size is None else transforms.Resize(size)
        self._masks_path = masks_path

        with Pool() as p:
            images = list(tqdm(p.imap(self._get_img, paths), total=len(paths)))

        self._images, self._targets = zip(*images)
        self._target_augmentation = lambda target: target[0, :, :] > 1

    def _get_img(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        img = self._resize(self._to_image(Image.open(path)))
        mask = self._resize(self._to_image(Image.open(get_mask_path(path, self._masks_path)).convert("L")))
        return img, mask


__all__ = ["BinaryKeyboardSegmentationDataset"]
