from typing import Final


DATASETS: Final = "datasets"

RAW_BACKGROUNDS: Final[str] = f"{DATASETS}/bg-20k"
RESIZED_BACKGROUNDS: Final[str] = f"{DATASETS}/bg-20k-resized"

RAW_DATASET: Final[str] = f"{DATASETS}/raw"
RAW_RENDERS: Final[str] = f"{RAW_DATASET}/renders"
RAW_MASKS: Final[str] = f"{RAW_DATASET}/masks"

YOLO_DATASET: Final[str] = f"{DATASETS}/yolo"
SEGMENTATION_DATASET: Final[str] = f"{DATASETS}/segmentation/v3"
TEST_DATASET: Final[str] = f"{DATASETS}/test"