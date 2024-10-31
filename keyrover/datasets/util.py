from typing import Sequence

import glob
import random

import numpy as np
import torch

from keyrover import SEGMENTATION_DATASET


def get_dataset_paths(shuffle: bool = True, base: str = SEGMENTATION_DATASET) -> list[str]:
    image_paths = glob.glob(f"{base}/**")

    if shuffle:
        random.shuffle(image_paths)

    return image_paths


def get_dataset_norm_params(version: int) -> tuple[np.ndarray, np.ndarray]:
    if version == 1:
        raise NotImplemented()

    elif version == 2:
        return (np.array([0.2517634, 0.26364404, 0.27402246]),
                np.array([0.241223, 0.24496816, 0.25682035]))

    elif version == 3:
        return (np.array([0.26772413, 0.28418145, 0.28728417]),
                np.array([0.24711585, 0.24890053, 0.25881228]))

    raise ValueError(f"Unknown dataset version {version}")


def split_train_test_valid(arr: Sequence, train_size: float = 0.8, valid_size: float = 0.16):
    assert 0 <= train_size <= 1
    assert 0 <= valid_size <= train_size

    valid_size = int(len(arr) * valid_size)
    train_size = int(len(arr) * train_size)

    return arr[valid_size:train_size], arr[train_size:], arr[:valid_size]


def reorder_image_axes(img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if img.shape[-1] in {1, 3, 4}:
        return img

    if len(img.shape) != 3:
        return img

    if isinstance(img, torch.Tensor):
        return img.permute(1, 2, 0)

    elif isinstance(img, np.ndarray):
        return img.transpose(1, 2, 0)

    raise ValueError("Only np.ndarray or torch.Tensor are supported")


def zip_collate_fn(batch):
    return tuple(zip(*batch))


__all__ = ["get_dataset_paths", "get_dataset_norm_params", "split_train_test_valid", "reorder_image_axes",
           "zip_collate_fn"]
