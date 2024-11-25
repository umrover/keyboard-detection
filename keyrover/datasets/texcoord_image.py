from keyrover import *
from keyrover.color import image_color

from .multiclass_image import MulticlassKeyboardImage


class TexCoordKeyboardImage(MulticlassKeyboardImage):
    def __init__(self, path: str) -> None:
        super().__init__(path)

        self.texcoord_path = f"{TEXCOORDS_DATASET}/{self.filename}.png"
        self.texcoords = Image.open(self.texcoord_path).convert("RGB")
        self.texcoords = img_to_numpy(self.texcoords)

        self.key_texcoords: dict[str, list[float]] = {}

    def show(self) -> None:
        imshow(self.image, self.texcoords)

    def extract_key_texcoords(self, reduce: str = "mean") -> None:
        self.key_texcoords = {"U": [], "V": []}

        for label, rect in zip(self.labels, self.rects):
            crop = self.crop(*rect, attr="texcoords")
            r, g, _ = image_color(crop, reduce=reduce)

            self.key_texcoords["U"].append(r)
            self.key_texcoords["V"].append(g)


__all__ = ["TexCoordKeyboardImage"]
