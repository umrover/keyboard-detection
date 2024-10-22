import random
import numpy as np

from PIL import Image, ImageEnhance
import cv2 as cv


def add_soft_shadow(img):
    opacity = random.randint(50, 200)
    thickness = random.randint(150, 400)
    blur = max(30, int(random.gauss(40, 7.5))) * 2 + 1

    overlay = np.zeros((*img.size, 4), dtype="uint8")
    cv.line(overlay, (0, 0), img.size, (0, 0, 0, opacity), thickness)
    overlay = apply_random_affine(overlay)

    overlay = cv.GaussianBlur(overlay, (blur, blur), blur)
    overlay = Image.fromarray(overlay)

    return Image.alpha_composite(img, overlay)


def add_hard_overlay(img):
    thickness = random.randint(100, 400)
    blur = max(5, int(random.gauss(25, 10)))

    if random.random() > 0.5:
        exp = max(0.2, random.gauss(2.5, 1))
    else:
        exp = max(0.4, random.gauss(0.5, 0.5))

    alpha_mask = np.zeros((*img.size, 1), dtype="uint8")
    cv.line(alpha_mask, (0, 0), img.size, (255,), thickness)
    alpha_mask = apply_random_affine(alpha_mask)

    alpha_mask = cv.blur(alpha_mask, (blur, blur))
    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 4, axis=2)
    alpha_mask = alpha_mask / 255

    bright_img = np.array(ImageEnhance.Brightness(img).enhance(exp))
    img = alpha_mask * bright_img + (1 - alpha_mask) * img

    return Image.fromarray(img.astype("uint8"))


def apply_random_affine(img, scale_lims=(0.25, 1.5)):
    w, h, _ = img.shape

    angle = np.random.uniform(0, 360)
    scale = np.random.uniform(*scale_lims)
    M = cv.getRotationMatrix2D((w / 2, h / 2), angle, scale)

    tx = np.random.uniform(-w / 2, w / 2)
    ty = np.random.uniform(-h / 2, h / 2)
    T = np.float32([[1, 0, tx], [0, 1, ty]])

    img = cv.warpAffine(img, M, (w, h))
    img = cv.warpAffine(img, T, (w, h))

    return img


# See https://medium.com/@itberrios6/how-to-apply-motion-blur-to-images-75b745e3ef17
def get_motion_blur_kernel(theta, thickness=1, ksize=21):
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
    _kernel = cv.line(_kernel, (c + x, c + y), (c, c), (1,), thickness)
    return _kernel / _kernel.sum()


def get_vignette_kernel(sigma, size):
    _kernel_x = cv.getGaussianKernel(size[0], sigma)
    _kernel_y = cv.getGaussianKernel(size[1], sigma)
    _kernel = _kernel_y * _kernel_x.T

    _kernel /= _kernel.max()
    return np.repeat(_kernel[:, :, np.newaxis], 3, axis=2)
