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

    colors = {}

    filename = ""
    for i, path in enumerate(tqdm(paths)):
        image = Image.open(path).convert("RGB")

        binary = (np.array(image.convert("L")) > 1).astype("uint8")
        classes = np.array(image)

        rects = extract_rects(binary)
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

        labels = []

        for x, y, w, h in rects:
            crop = classes[y:y + h, x:x + w]
            binary_crop = binary[y:y + h, x:x + w]

            crop = np.vstack(crop)
            crop = crop[binary_crop.flatten() == 1]

            color = tuple(np.quantile(crop, 0.75, axis=0))
            if color not in colors:
                colors[color] = len(colors) + 1

            labels.append(colors[color])

        boxes = BoundingBoxes(rects, format="XYWH", canvas_size=(binary.shape[0], binary.shape[1]))

        target = {"boxes":    boxes,
                  "labels":   torch.tensor(labels, dtype=torch.int64),
                  "image_id": i,
                  "area":     torch.tensor([w * h for _, _, w, h in boxes], dtype=torch.int64),
                  "iscrowd":  torch.tensor([False] * len(boxes), dtype=torch.bool)}

        filename = os.path.basename(path)
        data[filename] = target

    pprint.pprint(data[filename])
    with open(f"{RAW_DATASET}/multiclass-regions.pkl", "wb") as file:
        pickle.dump(data, file)
