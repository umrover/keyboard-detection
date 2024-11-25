from typing import Sequence, Hashable

import numpy as np


def image_color(image: np.ndarray, ignore_black: bool = True, reduce="median") -> tuple[int, int, int] | None:
    image = np.vstack(image)

    if ignore_black:
        # noinspection PyUnresolvedReferences
        image = image[(image != 0).any(axis=-1)]

    if image.size == 0:
        return None
    return getattr(np, reduce)(image, axis=0)


class Palette:
    def __init__(self, colors: Sequence, names: Sequence) -> None:
        self.colors = np.array(colors)

        self.colors_to_name = {}
        for c, name in zip(colors, names):
            self.colors_to_name[c] = name

    def __getitem__(self, i: Hashable):
        return self.colors_to_name[i]


__all__ = ["image_color", "Palette"]
