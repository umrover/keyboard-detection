import torch
import numpy as np

import matplotlib.pyplot as plt

from functools import reduce

from .datasets.util import reorder_image_axes
from .effects import img_to_numpy, IMAGE_TYPE


def normalize(img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    img -= img.min()
    img /= img.max()
    return img


def _imshow(img: np.ndarray | torch.Tensor, ax, **kwargs) -> None:
    if isinstance(img, np.ndarray):
        img = normalize(img.astype("float"))
    elif isinstance(img, torch.Tensor):
        img = normalize(img.float())

    reorder_image_axes(img)

    ax.axis("off")
    ax.imshow(img, interpolation="nearest", **kwargs)


def imshow(img: np.ndarray | torch.Tensor, _mask: np.ndarray | torch.Tensor | None = None, _ax=None, **kwargs) -> None:
    if _mask is not None and _ax is not None:
        raise ValueError("Can't specify both mask and axis!")

    if _mask is None:
        plt.figure()
        return _imshow(img, plt.gca() if _ax is None else _ax)

    if isinstance(img, torch.Tensor) and img.device != "cpu":
        img = img.cpu()
    if isinstance(_mask, torch.Tensor) and _mask.device != "cpu":
        _mask = _mask.cpu()

    img = reorder_image_axes(img)

    _, (ax1, ax2) = plt.subplots(ncols=2, **kwargs)
    _imshow(img, ax1)
    _imshow(_mask, ax2)


def show_images(imgs):
    a, b = get_best_grid(len(imgs))
    _, axes = plt.subplots(a, b)
    axes = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes):
        axes[i].axis("off")
        if i < len(imgs):
            imshow(imgs[i], _ax=axes[i])


def imhist(img: IMAGE_TYPE, ax=None, **kwargs):
    img = img_to_numpy(img)
    try:
        r, g, b = img.T
    except ValueError:
        r, g, b, _ = img.T

    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    if ax is None:
        ax = plt.gca()

    ax.axis("off")
    ax.hist(r, color="#fa3c3c", **kwargs)
    ax.hist(g, color="#74db95", **kwargs)
    ax.hist(b, color="#42b3f5", **kwargs)


def factors(_n):
    return sorted(list(set(reduce(list.__add__, ([i, _n // i] for i in range(1, int(_n ** 0.5) + 1) if _n % i == 0)))))


def get_best_non_empty_grid(_n):
    f = factors(_n)
    if len(f) % 2 == 0:
        return f[len(f) // 2 - 1: len(f) // 2 + 1]
    return f[len(f) // 2], f[len(f) // 2]


def get_best_grid(_n):
    best_score = float("infinity")
    best_grid = None

    i = _n
    while True:
        a, b = get_best_non_empty_grid(i)
        diff = b - a
        empty = i - _n

        if (score := diff + empty) < best_score:
            best_score = score
            best_grid = a, b

        if empty > best_score:
            break
        i += 1

    return best_grid


__all__ = ["imshow", "imhist", "show_images"]
