import glob
import pickle

from tqdm import tqdm
from multiprocessing import Pool

from torchvision.tv_tensors import BoundingBoxes

from keyrover.datasets import MulticlassKeyboardImage
from keyrover import *


def get_bbox_data(path: str) -> dict:
    img = MulticlassKeyboardImage(path)
    img.extract_rects()
    img.extract_keys()

    boxes = BoundingBoxes(img.rects, format="XYWH", canvas_size=img.canvas_size)

    return {"boxes":    boxes,
            "labels":   torch.tensor(img.classes, dtype=torch.int64),
            "area":     torch.tensor([w * h for _, _, w, h in boxes], dtype=torch.int64),
            "iscrowd":  torch.tensor([False] * len(boxes), dtype=torch.bool)}


if __name__ == "__main__":
    paths = glob.glob(f"{MASKS_DATASET}/*.png")
    with Pool() as p:
        target_data = list(tqdm(p.imap(get_bbox_data, paths), total=len(paths)))

    data = {}
    for i, (path, target) in enumerate(zip(paths, target_data)):
        filename = os.path.basename(path)
        target["image_id"] = i
        data[filename] = target

    with open(f"{RAW_DATASET}/multiclass-regions.pkl", "wb") as file:
        pickle.dump(data, file)
