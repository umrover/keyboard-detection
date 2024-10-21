import os

from PIL import Image

from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


def identity(*x):
    return x


class KeyboardDataset(Dataset):
    def __init__(self, _imgs, _transforms=()):
        self._images = _imgs
        self._transforms = transforms.Compose(_transforms)

    def __getitem__(self, idx):
        _img = self._images[idx]
        _img = self._transforms(_img)
        return _img

    def __len__(self):
        return len(self._images)


class BinaryKeyboardSegmentationDataset(Dataset):
    _to_image = transforms.ToImage()

    def __init__(self, _imgs, size=None, device="cpu"):
        self._device = device

        self._resize = identity if size is None else transforms.Resize(size)
        self._images = list(tqdm(map(self._get_img, _imgs), total=len(_imgs)))
        self._masks = list(tqdm(map(self._get_mask, _imgs), total=len(_imgs)))

        self._transforms = identity
        self._augmentations = transforms.Compose([transforms.ToDtype(torch.float32, scale=True)])

    def set_transforms(self, val):
        self._transforms = transforms.Compose(val)

    def set_augmentations(self, val):
        self._augmentations = transforms.Compose(val)

    def __getitem__(self, idx):
        _img, _mask = self._transforms(self._images[idx], self._masks[idx])
        _img = self._augmentations(_img)
        _mask = _mask[0, :, :] > 1

        return _img, _mask

    def __len__(self):
        return len(self._images)

    def _get_img(self, _img: str):
        return self._resize(self._to_image(Image.open(_img))).to(self._device)

    def _get_mask(self, _img: str):
        _img = Image.open(f"blender/masks/keyboard_{os.path.basename(_img)[:3]}.png").convert("RGB")
        return self._resize(self._to_image(_img)).to(self._device)


__all__ = ["KeyboardDataset", "BinaryKeyboardSegmentationDataset"]
