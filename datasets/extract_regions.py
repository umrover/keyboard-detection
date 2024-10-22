import glob
import pickle

from PIL import Image
import numpy as np
import cv2

import torch
from torchvision.tv_tensors import BoundingBoxes


def extract_rects(img):
    quads = []
    for c in cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        q = cv2.boundingRect(c)
        quads.append(q)

    return BoundingBoxes(quads, format="XYWH", canvas_size=img.shape)


if __name__ == "__main__":
    data = []

    PATH = "blender/masks"

    for i, path in enumerate(glob.glob(f"{PATH}/**")):
        image = Image.open(path).convert("RGB")

        mask = (np.array(image)[:, :, 0] > 1).astype("uint8")
        boxes = extract_rects(mask)

        target = {"boxes":    boxes,
                  "labels":   torch.Tensor([1] * len(boxes)),
                  "image_id": i,
                  "area":     torch.Tensor([w * h for _, _, w, h in boxes]),
                  "iscrowd":  torch.Tensor([False] * len(boxes))}

        data.append(target)

    print(data[0])
    with open(f"{PATH}/regions.pkl", "wb") as file:
        pickle.dump(data, file)
