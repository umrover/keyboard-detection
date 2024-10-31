import os.path
import pickle
from typing import Iterable, Sequence, Any

from PIL import Image
import cv2 as cv
import torch

from tqdm.notebook import tqdm

from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from .util import reorder_image_axes
from keyrover.ml import identity
from keyrover import imshow, RAW_DATASET


class KeyboardBBoxDataset(Dataset):
    def __init__(self, paths: Iterable[str]):

        self._images = []
        self._targets = []

        with open(f"{RAW_DATASET}/regions.pkl", "rb") as file:
            targets = pickle.load(file)

        f = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])

        to_xyxy = transforms.ConvertBoundingBoxFormat("XYXY")

        for path in tqdm(paths):
            img = Image.open(path)
            self._images.append(f(img))
            self._targets.append(to_xyxy(targets[os.path.basename(path)]))

        self._transforms = identity
        self._augmentations = identity

    def set_transforms(self, val: Sequence[transforms.Transform]) -> None:
        self._transforms = transforms.Compose(val)

    def set_augmentations(self, val: Sequence[transforms.Transform]) -> None:
        self._augmentations = transforms.Compose(val)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        img = self._images[idx]
        target = self._targets[idx]

        img, target = self._transforms(img, target)
        img = self._augmentations(img)

        return img, target

    def __len__(self) -> int:
        return len(self._images)

    def show(self, idx: int):
        img, target = self[idx]
        img = reorder_image_axes(img.numpy()).copy()

        for quad in target["boxes"]:
            x1, y1, x2, y2 = quad.numpy()
            cv.rectangle(img, (x1, y1), (x2, y2), (1.0, 0, 0), 2)

        poly = 0
        factor = int(255 / len(target["masks"]))

        for i, mask in enumerate(target["masks"]):
            poly += i * factor * mask

        imshow(img, poly)
