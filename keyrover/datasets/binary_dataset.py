from typing import Sequence
import os

from tqdm.notebook import tqdm
from multiprocessing import Pool

from PIL import Image

import torch
from torchvision.transforms import v2 as transforms

from keyrover.ml import identity
from keyrover import RAW_MASKS
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
        img = Image.open(path)
        img = self._resize(self._to_image(img))

        mask_path = f"{RAW_MASKS}/{os.path.basename(path).removesuffix('.jpg')}.png"
        target = Image.open(mask_path).convert("L")
        target = self._resize(self._to_image(target))
        return img, target


__all__ = ["BinaryKeyboardSegmentationDataset"]
