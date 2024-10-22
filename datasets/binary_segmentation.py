from typing import Sequence
from tqdm.notebook import tqdm

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from .internal import get_mask_path, identity


class BinaryKeyboardSegmentationDataset(Dataset):
    _to_image = transforms.ToImage()

    def __init__(self, paths: Sequence[str], size: tuple[float, float] | None = None):
        self._resize = identity if size is None else transforms.Resize(size)
        self._images = list(tqdm(map(self._get_img, paths), total=len(paths)))
        self._masks = list(tqdm(map(self._get_mask, paths), total=len(paths)))

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

    def _get_img(self, _img: str) -> torch.Tensor:
        return self._resize(self._to_image(Image.open(_img)))

    def _get_mask(self, _img: str) -> torch.Tensor:
        _img = Image.open(get_mask_path(_img)).convert("RGB")
        return self._resize(self._to_image(_img))


__all__ = ["BinaryKeyboardSegmentationDataset"]
