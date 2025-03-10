import os
import glob
import pickle
import pprint

from tqdm import tqdm

from PIL import Image
import numpy as np

import torch
from torchvision.tv_tensors import BoundingBoxes

from keyrover.vision import extract_rects
from keyrover import RAW_MASKS, RAW_DATASET


if __name__ == "__main__":
    data = {}
    paths = glob.glob(f"{RAW_MASKS}/*.png")

    OUTLIER_THRESHOLD = -0.5

    filename = ""
    for i, path in enumerate(tqdm(paths)):
        image = Image.open(path).convert("L")

        mask = (np.array(image) > 1).astype("uint8")

        rects = extract_rects(mask)
        areas = np.array(list(map(lambda rect: rect[-1] * rect[-2], rects)))

        # outlier filtering
        d = areas - np.median(areas)
        mdev = np.median(areas)
        s = d / mdev if mdev else np.zeros(len(d))

        rects = zip(s, rects)
        rects = list(filter(lambda rect: rect[0] > OUTLIER_THRESHOLD, rects))

        if len(rects) == 0:
            continue

        _, rects = zip(*rects)

        boxes = BoundingBoxes(rects, format="XYWH", canvas_size=(mask.shape[0], mask.shape[1]))

        target = {"boxes":    boxes,
                  "labels":   torch.tensor([1] * len(boxes), dtype=torch.int64),
                  "image_id": i,
                  "area":     torch.tensor([w * h for _, _, w, h in boxes], dtype=torch.int64),
                  "iscrowd":  torch.tensor([False] * len(boxes), dtype=torch.bool)}

        filename = os.path.basename(path)
        data[filename] = target

    pprint.pprint(data[filename])
    with open(f"{RAW_DATASET}/regions.pkl", "wb") as file:
        pickle.dump(data, file)
