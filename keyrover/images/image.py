from __future__ import annotations

from keyrover import *
from keyrover.vision import BBox


class KeyboardImage:
    default_folder = None

    @overload
    def __init__(self, path: str):
        ...

    @overload
    def __init__(self, image: ImageType):
        ...

    def __init__(self, arg):
        if isinstance(arg, str):
            self._path = arg

            if not os.path.isfile(self._path):
                self._path = f"{DATASETS}/{self.default_folder}/{self._path}"

            self._image = cv2.imread(self._path)
            self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)

        else:
            self._path = None
            self._image = arg

        self._shape = self._image.shape[0], self._image.shape[1]

    def __getitem__(self, *args) -> np.ndarray:
        return self._image[*args]

    def __eq__(self, other):
        return self._image == other

    def __ne__(self, other):
        return self._image != other

    def __le__(self, other):
        return self._image <= other

    def __lt__(self, other):
        return self._image < other

    def __ge__(self, other):
        return self._image >= other

    def __gt__(self, other):
        return self._image > other

    def __neg__(self):
        return ~self._image

    @overload
    def crop(self, x: int, y: int, w: int, h: int) -> KeyboardImage:
        ...

    @overload
    def crop(self, bbox: BBox) -> KeyboardImage:
        ...

    def crop(self, *args) -> KeyboardImage:
        if len(args) == 1:
            bbox = args[0]
            return self.crop(*bbox.p1, bbox.width, bbox.height)

        x, y, w, h = map(int, args)
        return KeyboardImage(self._image[y:y + h, x:x + w])

    def show(self) -> None:
        imshow(self._image)

    def binarize(self) -> np.ndarray:
        return cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY)

    path = property(lambda self: self._path)
    image = property(lambda self: self._image)

    shape = property(lambda self: self._shape)
    width = property(lambda self: self._shape[1])
    height = property(lambda self: self._shape[0])


__all__ = ["KeyboardImage"]
