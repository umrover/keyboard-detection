import glob
import shutil
import random

from tqdm import tqdm
from multiprocessing import Pool

from keyrover.effects import *
from keyrover import *

_overlay_funcs = (add_hard_overlay, add_soft_shadow)


def _get_img_with_random_background(path: str, backgrounds: str = RESIZED_BACKGROUNDS) -> Image.Image:
    image = Image.open(path)
    background = random.choice(glob.glob(f"{backgrounds}/*.jpg"))

    img = Image.open(background)
    img = img.convert("RGB")
    img = img.resize(image.size)
    img.paste(image, mask=image)

    return img


def _apply_random_motion_blur(img: ImageType) -> np.ndarray:
    theta = random.randint(0, 360)
    ksize = random.randint(4, 16)
    return apply_motion_blur(img, theta=theta, ksize=ksize)


def _apply_random_vignette(img: ImageType) -> np.ndarray:
    sigma = random.randint(300, 600)
    return apply_vignette(img, sigma=sigma)


def _apply_random_image_enhance(img: ImageType, min_exposure: float = 0.4, min_contrast: float = 0.2) -> Image.Image:
    img = to_pillow(img)

    contrast, exposure, sharpness, saturation = np.random.normal(loc=1.0, scale=0.5, size=4)
    exposure = max(exposure, min_exposure)
    contrast = max(contrast, min_contrast)

    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(exposure)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Color(img).enhance(saturation)

    return img


def _create_random_highlights(img: ImageType) -> Image.Image:
    img = to_pillow(img)
    img = img.convert("RGBA")

    num_shadows = np.random.geometric(0.5, (1,))[0] - 1
    for i in range(num_shadows):
        f = random.choice(_overlay_funcs)
        img = f(img)

    return img.convert("RGB")


def _add_chromatic_abberation(img: ImageType) -> np.ndarray:
    strength = random.uniform(0, 0.005)
    img = add_chromatic_aberration(img, strength=strength)
    return img


def _add_gaussian_noise(img: ImageType, sigma: float = 0.1) -> np.ndarray:
    img = to_numpy(img)
    gauss = np.random.normal(0, sigma, img.shape)
    img = img + gauss
    return normalize(img).astype("uint8")


def _add_salt_and_pepper_noise(img: ImageType, amount: float = 0.004, ratio: float = 0.5) -> np.ndarray:
    img = to_numpy(img)

    # Salt mode
    num_salt = int(np.ceil(amount * img.size * ratio))
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    img[coords] = 1

    # Pepper mode
    num_pepper = int(np.ceil(amount * img.size * (1 - ratio)))
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    img[coords] = 0
    return normalize(img).astype("uint8")


def _add_poisson_noise(img: ImageType) -> np.ndarray:
    img = to_numpy(img)

    vals = len(np.unique(img))
    vals = 3 ** np.ceil(np.log2(vals))
    img = np.random.poisson(img * vals) / vals
    return normalize(img).astype("uint8")


def _add_speckle_noise(img: ImageType) -> np.ndarray:
    img = to_numpy(img)
    img = img + img * np.random.randn(*img.shape)
    return normalize(img).astype("uint8")


def noisy(img: ImageType, noise_typ: str, **kwargs) -> np.ndarray:
    if noise_typ == "gauss":
        return _add_gaussian_noise(img, **kwargs)

    elif noise_typ == "s&p":
        return _add_salt_and_pepper_noise(img, **kwargs)

    elif noise_typ == "poisson":
        return _add_poisson_noise(img)

    elif noise_typ == "speckle":
        return _add_speckle_noise(img)


def normalize(img, lim: int = 255) -> np.ndarray:
    img -= img.min()
    img /= img.max() / lim
    return img


def create_random_image(path):
    output = _get_img_with_random_background(path)

    output = _add_chromatic_abberation(output)
    output = _create_random_highlights(output)
    output = _apply_random_motion_blur(output)
    output = _apply_random_image_enhance(output, min_exposure=0.9, min_contrast=0.5)
    output = _add_gaussian_noise(output, 0.1)
    output = _add_poisson_noise(output)

    output = Image.fromarray(output.astype("uint8"))

    output_path = f"{KEYBOARD_DATASET}/{os.path.basename(path).removesuffix('.png')}.jpg"
    output.save(output_path, quality=random.randint(25, 80))


if __name__ == "__main__":
    DATASET_VERSION = 4

    shutil.rmtree(KEYBOARD_DATASET)
    os.mkdir(KEYBOARD_DATASET)

    paths = glob.glob(f"{RAW_RENDERS}/*.png")

    if len(paths) > 100:
        with Pool() as p:
            r = list(tqdm(p.imap(create_random_image, paths), total=len(paths)))
    else:
        for path in tqdm(paths):
            create_random_image(path)
