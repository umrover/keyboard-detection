from keyrover import *
from keyrover.vision.bbox import *
from keyrover.math import median_filter
from keyrover.image import to_palette

from .image import KeyboardImage


class Key(LabeledBBox):
    _id_to_label: Final[dict[int, str]] = {
        1:  "Q", 2: "W", 3: "E", 4: "R", 5: "T", 6: "Y", 7: "U", 8: "I", 9: "O", 10: "P",
        11: "[", 12: "]",

        13: "space",

        14: "up", 15: "left", 16: "down", 17: "right",

        18: "del", 19: "end", 20: "pg dn", 21: "insert", 22: "home", 23: "pg up", 24: "print", 25: "lock", 26: "pause",

        27: "back", 28: "rshift", 29: "rctrl",
        30: "menu", 31: "fn", 32: "ralt", 33: "lalt", 34: "windows", 35: "lctrl", 36: "lshift", 85: "caps", 37: "ltab",
        38: "esc",

        47: "f1", 48: "f2", 49: "f3", 50: "f4", 43: "f5", 44: "f6", 45: "f7", 46: "f8",
        39: "f9", 40: "f10", 41: "f11", 42: "f12",

        51: "`",

        52: "1", 53: "2", 54: "3", 55: "4", 56: "5", 57: "6", 58: "7", 59: "8", 60: "9", 61: "0",

        62: "-", 63: "+", 71: "<", 72: ">", 73: "?", 87: "|",

        64: "Z", 65: "X", 66: "C", 67: "V", 68: "B", 69: "N", 70: "M",
        74: "A", 75: "S", 76: "D", 77: "F", 78: "G", 79: "H", 80: "J", 81: "K", 82: "L", 83: ";", 84: "\"",
        86: "enter",
    }

    def __init__(self, bbox: BBox, id_: int):
        super().__init__(bbox.p1, bbox.p2, self.to_label(id_))
        self._id: Final = id_

    @staticmethod
    def to_label(id_: int) -> str:
        return Key._id_to_label.get(id_, str(id_))


class KeyMaskImage(KeyboardImage):
    default_folder = "masks"
    palette = None

    def __init__(self, path: str) -> None:
        super().__init__(path)

        self._image = to_palette(self._image, self.palette)

        self._bboxes: Final = self._extract_rects()
        self._keys: Final = self._extract_keys()

    def _extract_rects(self) -> list[BBox]:
        bboxes = []

        for contour in cv2.findContours(self.binarize(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
            cx, cy, width, height = cv2.boundingRect(contour)
            bboxes.append(BBox((cx, cy), width, height))

        areas = map(lambda bbox: bbox.area, bboxes)
        return median_filter(bboxes, statistic=tuple(areas))

    def _extract_keys(self) -> list[Key]:
        colors = map(self._extract_color, self._bboxes)
        colors = filter(lambda c: c is not None, colors)

        key_ids = map(self.color_to_id, colors)
        return [Key(bbox, id_) for bbox, id_ in zip(self._bboxes, key_ids)]

    #  @override
    def show(self, show_labels: bool = True, scale: int = 3) -> None:
        image = self.image.copy()
        image = cv2.resize(image, (self.width * scale, self.height * scale))

        for key in self._keys:
            key = key.scale(scale)
            image = draw_textbox(image, key, draw_text=show_labels)

        imshow(image)

    #  @override
    def binarize(self) -> np.ndarray:
        image = super().binarize()
        return (image > 1).astype("uint8")

    def _extract_color(self, bbox: BBox, ignore_black: bool = True, reduce="median") -> tuple[int, int, int] | None:
        image = self.crop(bbox)

        if ignore_black:
            # noinspection PyUnresolvedReferences
            image = image[(image != 0).any(axis=-1)]

        if image.size == 0:
            return None
        return getattr(np, reduce)(image, axis=0)

    @staticmethod
    def id_to_color(i: int) -> tuple[int, int, int]:
        r = i % 12
        g = (i // 12) % 12
        b = (i // 144) % 12

        return r * 21, b * 21, g * 21

    @staticmethod
    def color_to_id(color: tuple[int, int, int]) -> int:
        r, b, g = color
        r //= 21
        g //= 21
        b //= 21

        return int(b * 144 + g * 12 + r)


KeyMaskImage.palette = np.array([KeyMaskImage.id_to_color(i) for i in range(100)], dtype="uint8")

__all__ = ["KeyMaskImage"]