import os


def get_mask_path(path: str, base: str = "blender/masks") -> str:
    return f"{base}/{os.path.basename(path)}"


__all__ = ["get_mask_path"]
