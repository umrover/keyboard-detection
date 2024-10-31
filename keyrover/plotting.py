import torch
import numpy as np

import matplotlib.pyplot as plt

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


def imshow(img: np.ndarray | torch.Tensor, _mask: np.ndarray | torch.Tensor | None = None, **kwargs) -> None:
    if _mask is None:
        plt.figure()
        return _imshow(img, plt.gca())

    if isinstance(img, torch.Tensor) and img.device != "cpu":
        img = img.cpu()
    if isinstance(_mask, torch.Tensor) and _mask.device != "cpu":
        _mask = _mask.cpu()

    img = reorder_image_axes(img)

    _, (ax1, ax2) = plt.subplots(ncols=2, **kwargs)
    _imshow(img, ax1)
    _imshow(_mask, ax2)


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


__all__ = ["imshow", "imhist"]
