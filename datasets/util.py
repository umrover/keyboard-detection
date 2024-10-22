import os


def identity(*x):
    return x


def get_frame_from_path(path):
    return os.path.basename(path)[:3]


def get_mask_path(path):
    return f"blender/masks/keyboard_{get_frame_from_path(path)}.png"
