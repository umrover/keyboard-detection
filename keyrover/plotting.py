import torch
import numpy as np

import matplotlib.pyplot as plt
import cv2

from functools import reduce

from .datasets.util import reorder_image_axes
from .effects import img_to_numpy, IMAGE_TYPE


def normalize(img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    img -= img.min()
    img /= img.max()
    return img


def _imshow(img: np.ndarray | torch.Tensor, ax) -> None:
    if isinstance(img, np.ndarray):
        img = normalize(img.astype("float"))
    elif isinstance(img, torch.Tensor):
        img = normalize(img.float())

    reorder_image_axes(img)

    ax.axis("off")
    ax.imshow(img, interpolation="nearest")


def imshow(img: np.ndarray | torch.Tensor, _mask: np.ndarray | torch.Tensor | None = None, _ax=None, **kwargs) -> None:
    if _mask is not None and _ax is not None:
        raise ValueError("Can't specify both mask and axis!")

    if _mask is None:
        if _ax is None:
            plt.figure()
            _ax = plt.gca()
        return _imshow(img, _ax)

    if isinstance(img, torch.Tensor) and img.device != "cpu":
        img = img.cpu()
    if isinstance(_mask, torch.Tensor) and _mask.device != "cpu":
        _mask = _mask.cpu()

    img = reorder_image_axes(img)

    _, (ax1, ax2) = plt.subplots(ncols=2, **kwargs)
    _imshow(img, ax1)
    _imshow(_mask, ax2)


def show_images(imgs, **kwargs):
    if len(imgs) == 1:
        return imshow(imgs[0])

    a, b = get_best_grid(len(imgs))
    _, axes = plt.subplots(a, b, **kwargs)

    axes = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes):
        axes[i].axis("off")
        if i < len(imgs):
            imshow(imgs[i], _ax=axes[i])

    plt.tight_layout()


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


def plot_text_box(img: np.ndarray,
                  p1: tuple[int, int], p2: tuple[int, int], text: str,
                  color: tuple[int, int, int] = (230, 55, 107),
                  scale: int = 5,
                  font=cv2.FONT_HERSHEY_TRIPLEX,
                  size: float = 0.3,
                  thickness: float = 0.5) -> np.ndarray:

    x1, y1 = p1
    size *= scale
    thickness = int(scale * thickness)

    cv2.rectangle(img, p1, p2, color, thickness=scale * 2)

    (w, h), _ = cv2.getTextSize(text, font, size, thickness)
    img = cv2.rectangle(img, (x1, y1 - h - scale * 2), (x1 + w + scale * 3, y1), color, -1)
    img = cv2.putText(img, text, (x1 + scale, y1 - scale), font, size, (255, 255, 255), thickness, cv2.LINE_AA)

    return img


def plot_results(results, scale: int = 4, plot=True) -> np.ndarray:
    size = results.orig_img.shape
    img = cv2.resize(results.orig_img, (scale * size[1], scale * size[0]))

    for box in results.boxes:
        x1, y1, x2, y2 = map(lambda v: int(scale * v), box.xyxy[0])
        plot_text_box(img, (x1, y1), (x2, y2), f"{int(100 * box.conf)}%", scale=scale)

    if plot:
        plt.figure(figsize=(10, 10))
        imshow(img, _ax=plt.gca())
    else:
        return img


__all__ = ["imshow", "imhist", "show_images", "plot_text_box", "plot_results", "plt", "get_best_grid"]
