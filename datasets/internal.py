import os

import torch


def identity(*x: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
    if len(x) == 1:
        return x[0]
    return x


def get_frame_from_path(path: str) -> int:
    filename = os.path.basename(path).split('.')[0]

    if "masks" in path or "renders" in path:
        frame = filename[-3:]
    else:
        frame = filename[:3]

    return int(frame)


def get_mask_path(path: str, base: str = "blender/masks") -> str:
    return f"{base}/{os.path.basename(path)}"
    # return f"datasets/segmentation/masks/keyboard_{get_frame_from_path(path):03d}.png"
