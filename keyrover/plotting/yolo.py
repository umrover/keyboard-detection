import matplotlib.pyplot as plt

from ultralytics.engine.results import Boxes, Results

from keyrover import *
from keyrover.vision.bbox import *
from .image import imshow


def draw_textbox(img: ImageType, bbox: LabeledBBox, color: Vec3 = (230, 55, 107), thickness: int = 3,
                 font=cv2.FONT_HERSHEY_SIMPLEX, font_size: float = 1.5, draw_text: bool = True) -> np.array:
    """
    Plots a box with text above it
    """

    img = to_numpy(img)
    cv2.rectangle(img, bbox.p1.astype("int"), bbox.p2.astype("int"), color=color, thickness=thickness)

    if draw_text:
        img = cv2.putText(img, bbox.label, bbox.p1.astype("int"),
                          font, font_size, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def plot_predictions(img: np.ndarray,
                     boxes: list[Boxes],
                     labels: Iterable,
                     scale: int = 4,
                     plot: bool = True,
                     draw_text: bool = True,
                     fig_kwargs: dict[str, Any] = {},
                     **kwargs) -> np.ndarray | None:
    """
    Plots a YOLO boxes object with labels
    """
    if fig_kwargs is None:
        fig_kwargs = {}
    size = img.shape
    img = cv2.resize(img, (scale * size[1], scale * size[0]))

    for cls, box in zip(labels, boxes):
        x1, y1, x2, y2 = map(lambda v: int(scale * v), box.xyxy[0])
        draw_textbox(img, (x1, y1), (x2, y2), str(cls), draw_text=draw_text, **kwargs)

    if plot:
        plt.figure(**fig_kwargs)
        imshow(img, ax=plt.gca())
    else:
        return img


def plot_yolo(results: Results, **kwargs) -> np.ndarray | None:
    return plot_predictions(results.orig_img, results.boxes,
                            [f"{box.cls} {box.conf}%" for box in results.boxes], **kwargs)
