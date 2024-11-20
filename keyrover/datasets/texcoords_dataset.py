from typing import Sequence

import os

from tqdm.notebook import tqdm
from multiprocessing import Pool

from PIL import Image

import torch
from torchvision.transforms import v2 as transforms

from keyrover.ml import identity
from keyrover import RAW_TEXCOORDS
from .abstract import KeyboardDatasetBase


class KeyboardTexCoordsDataset(KeyboardDatasetBase):
    _to_image = transforms.ToImage()

    def __init__(self, paths: Sequence[str], size: tuple[float, float] = None, norm: bool = True):
        super().__init__()
        self._resize = identity if size is None else transforms.Resize(size)
        self._norm = norm

        with Pool() as p:
            images = list(tqdm(p.imap(self._get_img, paths), total=len(paths)))

        self._images, self._targets = zip(*images)

    def _get_img(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(path)
        img = self._resize(self._to_image(img))

        mask_path = f"{RAW_TEXCOORDS}/{os.path.basename(path).removesuffix('.jpg')}.png"
        target = Image.open(mask_path).convert("RGB")
        target = self._resize(self._to_image(target))[:2].float()
        if self._norm:
            target = (target / 128) - 1
        return img, target


__all__ = ["KeyboardTexCoordsDataset"]
