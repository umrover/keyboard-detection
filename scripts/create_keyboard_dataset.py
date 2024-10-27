import os
import glob
import shutil

from tqdm import tqdm
from multiprocessing import Pool

from image_processing import *
from PIL import ImageOps


_overlay_funcs = (add_hard_overlay, add_soft_shadow)


def _get_img_with_random_background(path):
    image = Image.open(path)

    img = Image.open(random.choice(glob.glob("datasets/bg-20k-resized/*.jpg"))).convert("RGB")
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


def _apply_random_image_enhance(img, min_exposure=0.4, min_contrast=0.2):
    contrast, exposure, sharpness, saturation = np.random.normal(loc=1.0, scale=0.5, size=4)
    exposure = max(exposure, min_exposure)
    contrast = max(contrast, min_contrast)

    if not isinstance(img, Image.Image):
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


def _add_chromatic_abberation(img):
    strength = random.uniform(0, 0.005)
    img = np.array(img)
    img = add_chromatic_aberration(img, strength=strength)
    return Image.fromarray(img)


def create_random_image_v1(n):
    global OUTPUT_PATH
    OUTPUT_PATH = "datasets/segmentation/v1"

    path = random.choice(os.listdir("datasets/segmentation/renders"))

    output = _get_img_with_random_background(path)
    output = _apply_motion_blur(output)
    output = _apply_vignette(output)
    output = _apply_random_image_enhance(output)

    output_path = f"{OUTPUT_PATH}/{path[-7:-4]}_{n}.png"
    output.save(output_path)


def create_random_image_v2(n):
    global OUTPUT_PATH
    OUTPUT_PATH = "datasets/segmentation/v2"

    path = random.choice(os.listdir("datasets/segmentation/renders"))

    output = _get_img_with_random_background(path)
    output = _create_random_highlights(output)
    output = _apply_motion_blur(output)
    output = _apply_vignette(output)
    output = _apply_random_image_enhance(output)

    output_path = f"{OUTPUT_PATH}/{path[-7:-4]}_{n}.png"
    output.save(output_path)


def create_image_dataset_v1(N):
    global IMAGES_PATH

    shutil.rmtree(f"assets/keyboards_v{DATASET_VERSION}")
    os.mkdir(f"assets/keyboards_v{DATASET_VERSION}")

    IMAGES_PATH = "datasets/segmentation/renders"

    with Pool() as p:
        r = list(tqdm(p.imap(create_random_image_v2, range(N)), total=N))


def create_random_image_v3(p, flip=False):
    output = _get_img_with_random_background(p)
    if flip:
        output = ImageOps.mirror(output)
    output = _add_chromatic_abberation(output)
    output = _create_random_highlights(output)
    output = _apply_motion_blur(output)
    # output = _apply_random_image_enhance(output, min_exposure=1, min_contrast=0.5)

    output_path = f"datasets/segmentation/v3/{os.path.basename(p)}"
    output.save(output_path)


if __name__ == "__main__":
    DATASET_VERSION = 3

    BACKGROUNDS = glob.glob("datasets/bg-20k-resized/")
    IMAGES_PATH = "blender/renders/*.png"
    OUTPUT_PATH = "datasets/segmentation/v3"

    shutil.rmtree(OUTPUT_PATH)
    os.mkdir(OUTPUT_PATH)

    with Pool() as p:
        paths = glob.glob(IMAGES_PATH)
        r = list(tqdm(p.imap(create_random_image_v3, paths), total=len(paths)))
