import os
import random
import shutil

from PIL import Image, ImageEnhance
import numpy as np
import cv2 as cv

from tqdm import tqdm
from multiprocessing import Pool

from image_processing import get_motion_blur_kernel, get_vignette_kernel, add_hard_overlay, add_soft_shadow


_overlay_funcs = (add_hard_overlay, add_soft_shadow)


def _get_img_with_random_background(path):
    image = Image.open(f"datasets/segmentation/renders/{path}")

    img = Image.open(f"assets/backgrounds/{random.choice(os.listdir('assets/backgrounds'))}").convert("RGB")
    img = img.resize(image.size)
    img.paste(image, mask=image)
    
    return img


def _apply_motion_blur(img):
    blur_kernel = get_motion_blur_kernel(theta=random.randint(0, 360), thickness=1, ksize=random.randint(4, 16))
    img = cv.filter2D(np.array(img), ddepth=-1, kernel=blur_kernel)
    return Image.fromarray(img)


def _apply_vignette(img):
    vignette_kernel = get_vignette_kernel(random.randint(300, 600), img.size)
    return (vignette_kernel * img).astype("uint8")


def _apply_random_image_enhance(img):
    contrast, exposure, sharpness, saturation = np.random.normal(loc=1.0, scale=0.5, size=4)
    exposure = max(exposure, 0.4)
    contrast = max(contrast, 0.2)

    img = Image.fromarray(img)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(exposure)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Color(img).enhance(saturation)

    return img


def _create_random_highlights(img):
    img = img.convert("RGBA")

    for i in range(np.random.geometric(0.5, (1,))[0] - 1):
        f = random.choice(_overlay_funcs)
        img = f(img)

    return img.convert("RGB")
    

def create_random_image_v1(n):
    path = random.choice(os.listdir("datasets/segmentation/renders"))

    output = _get_img_with_random_background(path)
    output = _apply_motion_blur(output)
    output = _apply_vignette(output)
    output = _apply_random_image_enhance(output)

    output_path = f"assets/keyboards_v1/{path[-7:-4]}_{n}.png"
    output.save(output_path)


def create_random_image_v2(n):
    path = random.choice(os.listdir("datasets/segmentation/renders"))

    output = _get_img_with_random_background(path)
    output = _create_random_highlights(output)
    output = _apply_motion_blur(output)
    output = _apply_vignette(output)
    output = _apply_random_image_enhance(output)

    output_path = f"assets/keyboards_v2/{path[-7:-4]}_{n}.png"
    output.save(output_path)


if __name__ == "__main__":
    N = 1000
    DATASET_VERSION = 2

    shutil.rmtree(f"assets/keyboards_v{DATASET_VERSION}")
    os.mkdir(f"assets/keyboards_v{DATASET_VERSION}")

    with Pool() as p:
        r = list(tqdm(p.imap(create_random_image_v2, range(N)), total=N))
