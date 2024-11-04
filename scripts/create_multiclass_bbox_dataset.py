import glob
import pickle

from tqdm import tqdm
from multiprocessing import Pool

import numpy as np

import torch
from torchvision.tv_tensors import BoundingBoxes

from keyrover.datasets.multiclass_dataset import *
from keyrover.vision import extract_rects
from keyrover.math import median_filter
from keyrover.color import image_median_color
from keyrover.image import to_palette
from keyrover import *


def get_masks(path: str) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path)
    mask = np.array(image.convert("RGB"))
    binary = binarize_mask(image)

    return binary, mask


def filtered_key_regions(mask: np.ndarray) -> list[np.ndarray]:
    rects = extract_rects(mask)
    areas = np.array([rect[-1] * rect[-2] for rect in rects])
    return median_filter(rects, statistic=areas)


def get_region_color(mask: np.ndarray, rect: np.ndarray) -> tuple[int, ...] | None:
    x, y, w, h = rect
    crop = mask[y:y + h, x:x + w]

    return image_median_color(crop)


def get_bbox_data(path: str) -> dict:
    binary, mask = get_masks(path)

    palette = np.array([key_to_color(i) for i in range(100)])
    mask = to_palette(mask, palette)

    rects = filtered_key_regions(binary)
    colors = (c for rect in rects if (c := get_region_color(mask, rect)) is not None)
    labels = list(map(color_to_key, colors))

    boxes = BoundingBoxes(rects, format="XYWH", canvas_size=(binary.shape[0], binary.shape[1]))

    return {"boxes":    boxes,
            "labels":   torch.tensor(labels, dtype=torch.int64),
            "area":     torch.tensor([w * h for _, _, w, h in boxes], dtype=torch.int64),
            "iscrowd":  torch.tensor([False] * len(boxes), dtype=torch.bool)}


if __name__ == "__main__":
    paths = glob.glob(f"{RAW_MASKS}/*.png")
    with Pool() as p:
        target_data = list(tqdm(p.imap(get_bbox_data, paths), total=len(paths)))

    data = {}
    for i, (path, target) in enumerate(zip(paths, target_data)):
        filename = os.path.basename(path)
        target["image_id"] = i
        data[filename] = target

    with open(f"{RAW_DATASET}/multiclass-regions.pkl", "wb") as file:
        pickle.dump(data, file)
