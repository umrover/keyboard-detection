import glob
import pickle
import pprint

from PIL import Image
import numpy as np
import cv2 as cv

import torch
from torchvision.tv_tensors import BoundingBoxes, Mask
from datasets.internal import get_frame_from_path

from image_processing import extract_polygons, extract_rects

if __name__ == "__main__":
    data = {}

    PATH = "datasets/segmentation/masks"

    for i, path in enumerate(glob.glob(f"{PATH}/**.png")):
        image = Image.open(path).convert("RGB")

        mask = (np.array(image)[:, :, 0] > 1).astype("uint8")
        boxes = BoundingBoxes(extract_rects(mask), format="XYWH", canvas_size=mask.shape)

        masks = []
        for poly in extract_polygons(mask):
            mask = np.zeros(mask.shape, dtype="uint8")
            cv.drawContours(mask, [poly], 0, color=(255,), thickness=cv.FILLED)
            masks.append(mask)

        target = {"boxes":    boxes,
                  "labels":   torch.tensor([1] * len(boxes), dtype=torch.int64),
                  "image_id": i,
                  "area":     torch.tensor([w * h for _, _, w, h in boxes], dtype=torch.int64),
                  "iscrowd":  torch.tensor([False] * len(boxes), dtype=torch.bool),
                  "masks":    Mask(np.array(masks))}

        data[get_frame_from_path(path)] = target

    pprint.pprint(data[1])
    with open(f"{PATH}/regions.pkl", "wb") as file:
        pickle.dump(data, file)
