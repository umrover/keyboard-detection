import os
import pickle
import shutil
import random

from tqdm import tqdm

from keyrover import RAW_DATASET, SEGMENTATION_DATASET, YOLO_DATASET


if __name__ == "__main__":

    train_split = 0.90

    shutil.rmtree(f"{YOLO_DATASET}/train")
    shutil.rmtree(f"{YOLO_DATASET}/valid")
    os.makedirs(f"{YOLO_DATASET}/train/labels")
    os.makedirs(f"{YOLO_DATASET}/train/images")
    os.makedirs(f"{YOLO_DATASET}/valid/labels")
    os.makedirs(f"{YOLO_DATASET}/valid/images")

    with open(f"{RAW_DATASET}/regions.pkl", "rb") as file:
        data = pickle.load(file)

    for i, (frame, target) in enumerate(tqdm(data.items())):
        boxes = target["boxes"]
        size = boxes.canvas_size

        boxes = ""
        for box, label in zip(target["boxes"], target["labels"]):
            box = box.float()
            box[0] += box[2] / 2  # Centre X & Y pixels
            box[1] += box[3] / 2

            box[0] /= size[1]
            box[1] /= size[0]  # Normalise box coordinates
            box[2] /= size[1]
            box[3] /= size[0]

            # label - 1 because class IDs for Yolo start at 0
            boxes += f"{label - 1} {box[0]} {box[1]} {box[2]} {box[3]}\n"

        folder = "valid" if random.random() > train_split else "train"

        shutil.copy(f"{SEGMENTATION_DATASET}/{frame}", f"{YOLO_DATASET}/{folder}/images/{i}.png")
        with open(f"{YOLO_DATASET}/{folder}/labels/{i}.txt", "w") as file:
            file.write(boxes)
