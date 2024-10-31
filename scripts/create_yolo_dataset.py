import os
import pickle
import shutil
import random


if __name__ == "__main__":
    MASKS = "blender/masks"
    IMAGES = "datasets/segmentation/v3"
    OUTPUT_PATH = "datasets/yolo/"

    train_split = 0.90

    shutil.rmtree(f"{OUTPUT_PATH}/train")
    shutil.rmtree(f"{OUTPUT_PATH}/valid")
    os.makedirs(f"{OUTPUT_PATH}/train/labels")
    os.makedirs(f"{OUTPUT_PATH}/train/images")
    os.makedirs(f"{OUTPUT_PATH}/valid/labels")
    os.makedirs(f"{OUTPUT_PATH}/valid/images")

    with open(f"{MASKS}/regions.pkl", "rb") as file:
        data = pickle.load(file)

    for i, (frame, target) in enumerate(data.items()):
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

        shutil.copy(f"{IMAGES}/{frame}", f"{OUTPUT_PATH}/{folder}/images/{i}.png")
        with open(f"{OUTPUT_PATH}/{folder}/labels/{i}.txt", "w") as file:
            file.write(boxes)
