import random
import numpy as np

from PIL import Image, ImageEnhance
import cv2

IMAGE_TYPE = np.ndarray | Image.Image


def img_to_numpy(img: IMAGE_TYPE) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.array(img)
    raise TypeError("Can't convert unknown type image to numpy array")


def img_to_PIL(img: IMAGE_TYPE) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, Image.Image):
        return img
    raise TypeError("Can't convert unknown type image to PIL Image")


def add_soft_shadow(img: IMAGE_TYPE) -> Image.Image:
    img = img_to_PIL(img)

    opacity = random.randint(50, 200)
    thickness = random.randint(150, 400)
    blur = max(30, int(random.gauss(40, 7.5))) * 2 + 1

    overlay = np.zeros((*img.size, 4), dtype="uint8")
    cv2.line(overlay, (0, 0), img.size, (0, 0, 0, opacity), thickness)
    overlay = apply_random_affine(overlay)

    overlay = cv2.GaussianBlur(overlay, (blur, blur), blur)
    overlay = Image.fromarray(overlay)

    return Image.alpha_composite(img, overlay)


def add_hard_overlay(img: IMAGE_TYPE) -> Image.Image:
    img = img_to_PIL(img)

    thickness = random.randint(100, 400)
    blur = max(5, int(random.gauss(25, 10)))

    if random.random() > 0.5:
        exp = max(0.5, random.gauss(2.5, 1))
    else:
        exp = max(0.5, random.gauss(0.75, 0.5))

    alpha_mask = np.zeros((*img.size, 1), dtype="uint8")
    cv2.line(alpha_mask, (0, 0), img.size, (255,), thickness)
    alpha_mask = apply_random_affine(alpha_mask)

    alpha_mask = cv2.blur(alpha_mask, (blur, blur))
    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 4, axis=2)
    alpha_mask = alpha_mask.astype("float32") / 255

    bright_img = np.array(ImageEnhance.Brightness(img).enhance(exp))
    img = alpha_mask * bright_img + (1 - alpha_mask) * img

    return Image.fromarray(img.astype("uint8"))


def add_chromatic_aberration(img: IMAGE_TYPE, strength: float = 0.01) -> np.ndarray:
    img = img_to_numpy(img)

    try:
        r, g, b, _ = cv2.split(img)
    except ValueError:
        r, g, b = cv2.split(img)

    height, width = img.shape[:2]

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    center_x = width // 2
    center_y = height // 2
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    r_shifted = cv2.warpAffine(r, np.float32([[1, 0, strength * distance.max()], [0, 1, 0]]), (width, height))
    b_shifted = cv2.warpAffine(b, np.float32([[1, 0, -strength * distance.max()], [0, 1, 0]]), (width, height))

    return cv2.merge((r_shifted, g, b_shifted))


def apply_random_affine(img: IMAGE_TYPE, scale_lims: tuple[float, float] = (0.25, 1.5)) -> np.ndarray:
    img = img_to_numpy(img)

    if len(img.shape) == 3:
        w, h, _ = img.shape
    elif len(img.shape) == 2:
        w, h = img.shape
    else:
        raise ValueError("Image shape must have 2 or 3 dimensions")

    angle = np.random.uniform(0, 360)
    scale = np.random.uniform(*scale_lims)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)

    tx = np.random.uniform(-w / 2, w / 2)
    ty = np.random.uniform(-h / 2, h / 2)
    T = np.float32([[1, 0, tx], [0, 1, ty]])

    img = cv2.warpAffine(img, M, (w, h))
    img = cv2.warpAffine(img, T, (w, h))

    return img


def apply_motion_blur(img: IMAGE_TYPE, theta: int, ksize: int) -> np.ndarray:
    img = img_to_numpy(img)

    blur_kernel = get_motion_blur_kernel(theta=theta, thickness=1, ksize=ksize)
    img = cv2.filter2D(img, ddepth=-1, kernel=blur_kernel)
    return img


def apply_vignette(img: IMAGE_TYPE, sigma: int) -> np.ndarray:
    img = img_to_numpy(img)
    vignette_kernel = get_vignette_kernel(sigma, img.shape)
    return (vignette_kernel * img).astype("uint8")


# See https://medium.com/@itberrios6/how-to-apply-motion-blur-to-images-75b745e3ef17
def get_motion_blur_kernel(theta: int, thickness: int = 1, ksize: int = 21) -> np.ndarray:
    """ Obtains Motion Blur Kernel

    @param theta - direction of blur
    @param thickness - thickness of blur kernel line
    @param ksize - size of blur kernel
    """
    c = ksize // 2

    theta = np.radians(theta)
    x = int(np.cos(theta) * 100)
    y = int(np.sin(theta) * 100)

    _kernel = np.zeros((ksize, ksize))
    _kernel = cv2.line(_kernel, (c + x, c + y), (c, c), (1,), thickness)
    return _kernel / _kernel.sum()


def get_vignette_kernel(sigma: int, size: tuple[int, ...]) -> np.ndarray:
    _kernel_x = cv2.getGaussianKernel(size[0], sigma)
    _kernel_y = cv2.getGaussianKernel(size[1], sigma)
    _kernel = _kernel_y * _kernel_x.T

    _kernel /= _kernel.max()
    return np.repeat(_kernel[:, :, np.newaxis], 3, axis=2)


__all__ = ["add_soft_shadow", "add_hard_overlay", "add_chromatic_aberration",
           "get_motion_blur_kernel", "get_vignette_kernel", "img_to_PIL", "img_to_numpy", "IMAGE_TYPE",
           "apply_motion_blur", "apply_vignette", "apply_random_affine"]