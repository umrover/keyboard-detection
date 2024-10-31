import os
import glob
import shutil
import random

from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageEnhance

from keyrover.effects import *
from keyrover.paths import RESIZED_BACKGROUNDS, RAW_RENDERS, SEGMENTATION_DATASET

_overlay_funcs = (add_hard_overlay, add_soft_shadow)


def _get_img_with_random_background(path: str, backgrounds: str = RESIZED_BACKGROUNDS) -> Image.Image:
    image = Image.open(path)
    background = random.choice(glob.glob(f"{backgrounds}/*.jpg"))

    img = Image.open(background)
    img = img.convert("RGB")
    img = img.resize(image.size)
    img.paste(image, mask=image)

    return img


def _apply_random_motion_blur(img: IMAGE_TYPE) -> np.ndarray:
    theta = random.randint(0, 360)
    ksize = random.randint(4, 16)
    return apply_motion_blur(img, theta=theta, ksize=ksize)


def _apply_random_vignette(img: IMAGE_TYPE) -> np.ndarray:
    sigma = random.randint(300, 600)
    return apply_vignette(img, sigma=sigma)


def _apply_random_image_enhance(img: IMAGE_TYPE, min_exposure: float = 0.4, min_contrast: float = 0.2) -> Image.Image:
    img = img_to_PIL(img)

    contrast, exposure, sharpness, saturation = np.random.normal(loc=1.0, scale=0.5, size=4)
    exposure = max(exposure, min_exposure)
    contrast = max(contrast, min_contrast)

    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(exposure)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Color(img).enhance(saturation)

    return img


def _create_random_highlights(img: IMAGE_TYPE) -> Image.Image:
    img = img_to_PIL(img)
    img = img.convert("RGBA")

    num_shadows = np.random.geometric(0.5, (1,))[0] - 1
    for i in range(num_shadows):
        f = random.choice(_overlay_funcs)
        img = f(img)

    return img.convert("RGB")


def _add_chromatic_abberation(img: IMAGE_TYPE) -> np.ndarray:
    strength = random.uniform(0, 0.005)
    img = add_chromatic_aberration(img, strength=strength)
    return img


def create_random_image(path):
    output = _get_img_with_random_background(path)

    output = _add_chromatic_abberation(output)
    output = _create_random_highlights(output)
    output = _apply_random_motion_blur(output)
    # output = _apply_random_image_enhance(output, min_exposure=1, min_contrast=0.5)
    output = Image.fromarray(output)

    output_path = f"{SEGMENTATION_DATASET}/{os.path.basename(path)}"
    output.save(output_path)


if __name__ == "__main__":
    DATASET_VERSION = 3

    shutil.rmtree(SEGMENTATION_DATASET)
    os.mkdir(SEGMENTATION_DATASET)

    with Pool() as p:
        paths = glob.glob(f"{RAW_RENDERS}/*.png")
        r = list(tqdm(p.imap(create_random_image, paths), total=len(paths)))
