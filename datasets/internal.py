import os

import torch


def identity(*x: torch.Tensor) -> tuple:
    return x


def get_frame_from_path(path: str) -> int:
    filename = os.path.basename(path).split('.')[0]

    if "masks" in path or "renders" in path:
        frame = filename[-3:]
    else:
        frame = filename[:3]

    return int(frame)


def get_mask_path(path: str) -> str:
    return f"blender/masks/keyboard_{get_frame_from_path(path)}.png"