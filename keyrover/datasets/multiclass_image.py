import numpy as np

from keyrover.color import image_color
from keyrover.vision import extract_rects
from keyrover.math import median_filter
from keyrover.image import to_palette
from keyrover import *


id_to_key = {
    1: "Q", 2: "W", 3: "E", 4: "R", 5: "T", 6: "Y", 7: "U", 8: "I", 9: "O", 10: "P",
    11: "[", 12: "]",

    13: "space",

    14: "up", 15: "left", 16: "down", 17: "right",

    18: "delete", 19: "end", 20: "pg down", 21: "insert", 22: "home", 23: "pg up", 24: "print", 25: "lock", 26: "pause",

    27: "backspace", 28: "rshift", 29: "rcontrol",
    30: "menu", 31: "fn", 32: "ralt", 33: "lalt", 34: "windows", 35: "lcontrol", 36: "lshift", 85: "caps lock", 37: "ltab",
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


class MulticlassKeyboardImage:
    def __init__(self, path: str) -> None:
        self.path = path
        self.filename = os.path.basename(path).removesuffix(".png")

        self.mask_path = f"{RAW_MASKS}/{self.filename}.png"
        self.image_path = f"{SEGMENTATION_DATASET}/{self.filename}.jpg"

        self.image = Image.open(self.image_path).convert("RGB")

        self.mask = Image.open(self.mask_path).convert("RGB")
        self.binary_mask = binarize_mask(self.mask)
        self.mask = to_palette(self.mask, palette)

        self.rects = []
        self.classes = []
        self.labels = []

    def extract_rects(self):
        if len(self.rects) > 0:
            return

        self.rects = extract_rects(self.binary_mask)
        areas = np.array([rect[-1] * rect[-2] for rect in self.rects])
        self.rects = median_filter(self.rects, statistic=areas)

    def extract_keys(self):
        colors = (c for rect in self.rects if (c := self.get_region_color(*rect)) is not None)
        self.classes = list(map(self.color_to_key, colors))
        self.labels = [id_to_key.get(i, str(i)) for i in self.classes]

    def crop(self, x: int, y: int, w: int, h: int, attr: str = "mask") -> np.ndarray:
        img = getattr(self, attr)
        img = img_to_numpy(img)
        return img[y:y+h, x:x+w]

    def get_region_color(self, x: int, y: int, w: int, h: int) -> tuple[int, ...] | None:
        return image_color(self.crop(x, y, w, h))

    def show(self) -> None:
        imshow(self.image, self.mask)

    def show_rects(self) -> None:
        img = np.array(self.image)
        for rect in self.rects:
            cv2.rectangle(img, rect, (255,), 1)

        imshow(img)

    def show_keys(self) -> None:
        img = np.array(self.image)

        for (x, y, w, h), label in zip(self.rects, self.labels):
            img = draw_textbox(img, (x, y), (x + w, y + h), text=label, scale=1, size=0.3)

        imshow(img)

    def show_crops(self, attr: str = "image"):
        crops = [self.crop(*rect, attr=attr) for rect in self.rects]
        show_images(crops)

    @staticmethod
    def key_to_color(i: int) -> tuple[int, int, int]:
        r = i % 12
        g = (i // 12) % 12
        b = (i // 144) % 12

        return r * 21, b * 21, g * 21

    @staticmethod
    def color_to_key(color: tuple[int, int, int]) -> int:
        r, b, g = color
        r //= 21
        g //= 21
        b //= 21

        return b * 144 + g * 12 + r

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.mask.shape[0], self.mask.shape[1]


OUTLIER_THRESHOLD = -0.5

palette = np.array([MulticlassKeyboardImage.key_to_color(i) for i in range(100)])

__all__ = ["MulticlassKeyboardImage", "id_to_key"]
