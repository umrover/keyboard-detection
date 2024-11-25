from typing import Sequence, Hashable

import numpy as np


class Palette:
    def __init__(self, colors: Sequence, names: Sequence) -> None:
        self.colors = np.array(colors)

        self.colors_to_name = {}
        for c, name in zip(colors, names):
            self.colors_to_name[c] = name

    def __getitem__(self, i: Hashable):
        return self.colors_to_name[i]


__all__ = ["image_color", "Palette"]
