from typing import Final, Sequence, Iterable, Literal, Any, overload

import numpy as np
import torch

from PIL import Image
import cv2


def describe(arr) -> None:
    print(
        f"""{arr.__class__} ({arr.dtype}, shape={arr.shape})
        Min: {arr.min()}
        Max: {arr.max()}
        Mean: {arr.mean()}""")


ImageType = np.ndarray | Image.Image | torch.Tensor


def to_numpy(img: ImageType, convert_bool: bool = False) -> np.ndarray:
    if isinstance(img, (map, filter, zip)):
        return np.array(tuple(img))

    if isinstance(img, (Image.Image, tuple, list)):
        img = np.array(img)

    elif isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    elif not isinstance(img, np.ndarray):
        raise TypeError("Can't convert unknown type image to numpy array")

    if convert_bool and img.dtype == bool:
        img = img.astype("uint8")
    return img


def to_pillow(img: ImageType) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(to_numpy(img))


def to_int(vec: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(map(int, vec))


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]
