from typing import Sequence

from PIL import Image

import math
import random
import colorsys

from goopylib import color


# TODO move to goopylib
class Color(color.Color):
    def __iter__(self):
        for val in (self.red, self.green, self.blue):
            yield val


# TODO move to goopylib
class ColorRGB(color.ColorRGB):
    def __iter__(self):
        for val in (self.red, self.green, self.blue):
            yield val


# TODO move to goopylib
class ColorHSV(color.ColorHSV):
    def __iter__(self):
        for val in (self.hue, self.saturation, self.value):
            yield val

    def to_rgb(self):
        rgb = super().to_rgb()
        return ColorRGB(rgb.red, rgb.green, rgb.blue)

    def copy(self):
        return ColorHSV(self.hue, self.saturation, self.value)


# TODO move to goopylib
def random_hsv(h_lims: tuple[int, int] = (0, 360),
               s_lims: tuple[float, float] = (0, 1),
               v_lims: tuple[float, float] = (0, 1)) -> ColorHSV:
    """
    Returns:
        a random HSV color
    """
    return ColorHSV(random.randint(*h_lims),
                    random.uniform(*s_lims),
                    random.uniform(*v_lims))


# Adapted from Pablo Guerrero's answer
# https://stackoverflow.com/questions/32530345/pil-generating-vertical-gradient-image

class Point:
    def __init__(self, x: float, y: float) -> None:
        self._x = x
        self._y = y

    def rot_x(self, degrees: float) -> float:
        radians = math.radians(degrees)
        return self._x * math.cos(radians) + self._y * math.sin(radians)

    x = property(lambda self: self._x)
    y = property(lambda self: self._y)


class Rect:
    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        minx, maxx = (x1, x2) if x1 < x2 else (x2, x1)
        miny, maxy = (y1, y2) if y1 < y2 else (y2, y1)
        self._min = Point(minx, miny)
        self._max = Point(maxx, maxy)

    def min_max_rot_x(self, degrees: float) -> tuple[float, float]:
        first = True
        for x in (self._min.x, self._max.x):
            for y in (self._min.y, self._max.y):
                p = Point(x, y)
                rot_d = p.rot_x(degrees)

                if first:
                    min_d = rot_d
                    max_d = rot_d
                    first = False
                else:
                    min_d = min(min_d, rot_d)
                    max_d = max(max_d, rot_d)

        return min_d, max_d

    min = property(lambda self: self._min)
    max = property(lambda self: self._max)
    width = property(lambda self: self.max.x - self.min.x)
    height = property(lambda self: self.max.y - self.min.y)


def interpolate_color(minval: float, maxval: float, val: float, color_palette: Sequence[Color]) -> ColorRGB:
    """
    Computes intermediate RGB color of a value in the range of minval
    to maxval (inclusive) based on a color_palette representing the range.
    """
    max_index = len(color_palette) - 1
    delta = maxval - minval
    if delta == 0:
        delta = 1

    v = (val - minval) / delta * max_index
    i1 = int(v)
    i2 = min(i1 + 1, max_index)
    (r1, g1, b1) = color_palette[i1]
    (r2, g2, b2) = color_palette[i2]
    f = v - i1
    return ColorRGB(int(r1 + f * (r2 - r1)),
                    int(g1 + f * (g2 - g1)),
                    int(b1 + f * (b2 - b1)))


def degrees_gradient(im: Image.Image,
                     rect: Rect,
                     color_palette: Sequence[Color],
                     degrees: float) -> None:

    minval, maxval = 1, len(color_palette)
    delta = maxval - minval
    min_d, max_d = rect.min_max_rot_x(degrees)
    range_d = max_d - min_d
    for x in range(rect.min.x, rect.max.x + 1):
        for y in range(rect.min.y, rect.max.y + 1):
            p = Point(x, y)
            f = (p.rot_x(degrees) - min_d) / range_d
            val = minval + f * delta
            color = interpolate_color(minval, maxval, val, color_palette)
            im.putpixel((x, y), tuple(color))


def gradient_image(size: tuple[int, int], color_palette: Sequence[Color], degrees: float) -> Image.Image:
    region = Rect(0, 0, *size)
    width, height = region.max.x + 1, region.max.y + 1
    image = Image.new("RGB", (width, height), (0, 0, 0))
    degrees_gradient(image, region, color_palette, -degrees)
    return image


def random_grey(lims: tuple[int, int] = (0, 255),
                alpha_lims: tuple[float, float] | None = None) -> ColorRGB:
    c = random.randint(*lims)

    if alpha_lims is None:
        return ColorRGB(c, c, c)
    return ColorRGB(c, c, c, random.uniform(*alpha_lims))


__all__ = ["colorsys", "gradient_image", "random_hsv", "random_grey"]
