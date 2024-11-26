from keyrover.color import image_color

from .image import KeyboardImage
from .key_mask import KeyMaskImage


TextureCoordinate = tuple[float, float]


class TexcoordImage(KeyboardImage):
    default_folder = "texcoords"

    def __init__(self, path: str) -> None:
        super().__init__(path)

        self._mask = KeyMaskImage(path)
        self._texcoords = self._extract_texcoords()

    def _extract_texcoords(self, reduce: str = "mean") -> dict[str, TextureCoordinate]:
        texcoords = {}
        for key in self._mask:
            crop = self.crop(key)
            r, g, _ = image_color(crop, reduce=reduce)
            texcoords[key.label] = (r, g)

        return texcoords

    texcoords: dict[str, TextureCoordinate] = property(lambda self: self._texcoords)


__all__ = ["TexcoordImage"]
