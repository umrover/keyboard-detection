import glob
import pickle
from tqdm import tqdm

from multiprocessing import Pool

from keyrover.datasets import TexCoordKeyboardImage
from keyrover import *

U = []
V = []


def get_key_texcoords(path: str) -> tuple[list[float], list[float], list[int]]:
    img = TexCoordKeyboardImage(path)

    img.extract_rects()
    img.extract_keys()
    img.extract_key_texcoords()

    return (img.key_texcoords["U"],
            img.key_texcoords["V"],
            img.classes)


if __name__ == "__main__":
    paths = glob.glob(f"{RAW_MASKS}/*.png")
    with Pool() as p:
        data = list(tqdm(p.imap(get_key_texcoords, paths), total=len(paths)))

    with open(f"{RAW_TEXCOORDS}/key_texcoords.bin", "wb") as file:
        pickle.dump(data, file)
