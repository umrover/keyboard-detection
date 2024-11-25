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


def to_int(vec: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(map(int, vec))


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]

ImageType = np.ndarray | Image.Image | torch.Tensor
