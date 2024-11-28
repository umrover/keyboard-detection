from typing import Sequence

import glob
import random
import pickle

import numpy as np

from keyrover import DATASETS


def get_dataset_paths(version: str, shuffle: bool = True, extension: str = "jpg") -> list[str]:
    image_paths = glob.glob(f"{version}/**.{extension}", root_dir=f"{DATASETS}/keyboard")

    if shuffle:
        random.shuffle(image_paths)

    return image_paths


def get_dataset_norm_params(version: str) -> tuple[np.ndarray, np.ndarray]:
    with open("datasets/segmentation/normalization.bin", "rb") as f:
        norm_parameters = pickle.load(f)
    return norm_parameters[version]


def calculate_dataset_norm_params(_data):
    d = np.array([np.array(_img) for _img in _data])
    m = d.mean(axis=(0, -1, -2))
    s = d.std(axis=(0, -1, -2))
    return m, s


def split_train_test_valid(arr: Sequence, train_size: float = 0.8, valid_size: float = 0.16):
    assert 0 <= train_size <= 1
    assert 0 <= valid_size <= train_size

    valid_size = int(len(arr) * valid_size)
    train_size = int(len(arr) * train_size)

    return arr[valid_size:train_size], arr[train_size:], arr[:valid_size]


def zip_collate_fn(batch):
    return tuple(zip(*batch))


__all__ = ["get_dataset_paths", "get_dataset_norm_params", "calculate_dataset_norm_params", "split_train_test_valid",
           "zip_collate_fn"]
