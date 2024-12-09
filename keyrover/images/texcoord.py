from keyrover.color import image_color

from .image import KeyboardImage, NormalizationType
from .key_mask import KeyMaskImage


TextureCoordinate = tuple[float, float]


class TexcoordImage(KeyboardImage):
    default_folder = "texcoords"

    def __init__(self, path: str, reduce: str = "median") -> None:
        super().__init__(path.replace("jpg", "png"))

        if isinstance(self, NormalizedTexcoordImage):
            self.normalize(self._normalization)

        self._mask = KeyMaskImage(path)
        self._texcoords = self._extract_texcoords(reduce=reduce)

    def _extract_texcoords(self, reduce: str) -> dict[str, TextureCoordinate]:
        texcoords = {}

        for key in self._mask:
            crop = self.crop(key)
            r, g, _ = image_color(crop, reduce=reduce)
            texcoords[key.label] = (r, g)

        return texcoords

    def scatter(self, color="#ccc", **kwargs):
        from keyrover.plotting import scatter
        return scatter(self._texcoords.values(), color=color, **kwargs)

    texcoords: dict[str, TextureCoordinate] = property(lambda self: self._texcoords)


class NormalizedTexcoordImage(TexcoordImage):
    _normalization = "minmax"

    def __init__(self, *args, kind: NormalizationType = "minmax", **kwargs):
        self._normalization = kind
        super().__init__(*args, **kwargs)  # normalization is handled by TexcoordImage constructor


__all__ = ["TexcoordImage", "NormalizedTexcoordImage",
           "TextureCoordinate", "NormalizationType"]
