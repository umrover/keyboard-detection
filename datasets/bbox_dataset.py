import pickle
from typing import Iterable, Sequence, Any

from PIL import Image
import cv2 as cv
import torch

from tqdm.notebook import tqdm

from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors

from datasets.internal import get_frame_from_path, identity
from utils import imshow, reorder_image_axes


class KeyboardBBoxDataset(Dataset):
    def __init__(self, paths: Iterable[str], _transforms: Sequence[transforms.Transform] = (identity,)):
        self._images = []
        self._targets = []

        with open("blender/masks/regions.pkl", "rb") as file:
            targets = pickle.load(file)

        for path in tqdm(paths):
            img = Image.open(path)
            img = tv_tensors.Image(img)
            self._images.append(img)

            i = get_frame_from_path(path)
            self._targets.append(targets[i])

        self._transforms = transforms.Compose(_transforms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        img = self._images[idx]
        img = self._transforms(img)
        return img[0], self._targets[idx]

    def __len__(self) -> int:
        return len(self._images)

    def show(self, idx: int):
        img, target = self[idx]
        img = reorder_image_axes(img.numpy())

        for quad in target["boxes"]:
            x, y, w, h = quad.numpy()
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        imshow(img)
