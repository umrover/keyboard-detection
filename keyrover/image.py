import torch
import numpy as np

from PIL import Image
import cv2

from scipy.spatial import cKDTree


ImageType = np.ndarray | Image.Image | torch.Tensor


def img_to_numpy(img: ImageType, convert_bool: bool = False) -> np.ndarray:
    if isinstance(img, Image.Image):
        img = np.array(img)
    elif isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    elif not isinstance(img, np.ndarray):
        raise TypeError("Can't convert unknown type image to numpy array")

    if convert_bool and img.dtype == bool:
        img = img.astype("uint8")
    return img


def img_to_PIL(img: ImageType) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, Image.Image):
        return img
    raise TypeError("Can't convert unknown type image to PIL Image")


def reorder_image_axes(img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if len(img.shape) == 2:
        return img

    if len(img.shape) != 3:
        raise ValueError(f"reorder_image_axes() requires images with 2 or 3 dimensions, not {img.shape}")

    if img.shape[-1] in {1, 3, 4}:
        return img

    if isinstance(img, torch.Tensor):
        return img.permute(1, 2, 0)

    elif isinstance(img, np.ndarray):
        return img.transpose(1, 2, 0)

    raise ValueError(f"Only np.ndarray or torch.Tensor are supported, not {type(img)}")


def binarize_mask(image: Image.Image) -> np.ndarray:
    binary = np.array(image.convert("L"))
    return (binary > 1).astype("uint8")


def to_palette(image: ImageType, palette: np.ndarray) -> np.ndarray:
    image = img_to_numpy(image)
    indices = cKDTree(palette).query(image, k=1)[1]
    return palette[indices]


__all__ = ["img_to_PIL", "img_to_numpy", "reorder_image_axes", "binarize_mask",
           "ImageType", "Image", "cv2"]
