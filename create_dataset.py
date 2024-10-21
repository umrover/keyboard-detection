import os
import random
import shutil

from PIL import Image, ImageEnhance
import numpy as np
import cv2 as cv

from tqdm import tqdm
from multiprocessing import Pool

from image_processing import get_motion_blur_kernel, get_vignette_kernel


def create_random_image(n):
    path = random.choice(os.listdir("blender/renders"))
    image = Image.open(f"blender/renders/{path}")

    background = Image.open(f"assets/backgrounds/{random.choice(os.listdir('assets/backgrounds'))}").convert("RGB")
    background = background.resize(image.size)
    background.paste(image, mask=image)

    blur_kernel = get_motion_blur_kernel(theta=random.randint(0, 360), thickness=1, ksize=random.randint(4, 16))
    output = cv.filter2D(np.array(background), ddepth=-1, kernel=blur_kernel)

    vignette_kernel = get_vignette_kernel(random.randint(300, 600), image.size)
    output = (vignette_kernel * output).astype("uint8")

    contrast, exposure, sharpness, saturation = np.random.normal(loc=1.0, scale=0.5, size=4)
    exposure = max(exposure, 0.4)
    contrast = max(contrast, 0.2)

    output = Image.fromarray(output)
    output = ImageEnhance.Contrast(output).enhance(contrast)
    output = ImageEnhance.Brightness(output).enhance(exposure)
    output = ImageEnhance.Sharpness(output).enhance(sharpness)
    output = ImageEnhance.Color(output).enhance(saturation)

    output_path = f"assets/keyboards/{path[-7:-4]}_{n}.png"
    output.save(output_path)


if __name__ == "__main__":
    N = 1000

    shutil.rmtree("assets/keyboards")
    os.mkdir("assets/keyboards")

    with Pool() as p:
        r = list(tqdm(p.imap(create_random_image, range(N)), total=N))
