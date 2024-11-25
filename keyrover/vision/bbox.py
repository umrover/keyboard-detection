from typing import overload

from keyrover.util import Vec2


class BBox:
    @overload
    def __init__(self, p1: Vec2, p2: Vec2):
        ...

    @overload
    def __init__(self, centre: Vec2, width: float, height: float):
        ...

    def __init__(self, *args):
        if len(args) == 2:
            self._p1, self._p2 = args

            self._width = abs(self._p2[0] - self._p1[0])
            self._height = abs(self._p2[1] - self._p1[1])

            self._centre = (self._p1[0] + self._p2[0]) / 2, (self._p1[1] + self._p2[1]) / 2

        if len(args) == 3:
            self._centre, self._width, self._height = args
            self._p1 = (self._centre[0] - self._width / 2, self._centre[1] - self._height / 2)
            self._p2 = (self._centre[0] + self._width / 2, self._centre[1] + self._height / 2)

    width: float = property(lambda self: self._width)
    height: float = property(lambda self: self._height)

    area: float = property(lambda self: self._width * self._height)

    p1: Vec2 = property(lambda self: self._p1)
    p2: Vec2 = property(lambda self: self._p2)
    centre: Vec2 = property(lambda self: self._centre)


__all__ = ["BBox"]
