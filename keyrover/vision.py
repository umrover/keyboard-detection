import cv2
import numpy as np

import ultralytics.engine.results

from .util import to_numpy, ImageType


def extract_rotated_rects(img: ImageType) -> list[cv2.typing.MatLike]:
    img = to_numpy(img, convert_bool=True)

    quads = []
    for c in cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        q = cv2.minAreaRect(c)
        q = cv2.boxPoints(q)
        q = np.intp(q)
        quads.append(q)

    return quads


def extract_quads(img: ImageType) -> list[cv2.typing.MatLike]:
    img = to_numpy(img, convert_bool=True)

    qs = []

    for c in cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        e = 0.05 * cv2.arcLength(c, True)

        while len(cv2.approxPolyDP(c, e, True)) < 4:
            e -= 0.01
        while len(q := cv2.approxPolyDP(c, e, True)) > 4:
            e += 0.01

        qs.append(q)

    return qs


def extract_polygons(img: ImageType, epsilon=0.01) -> list[cv2.typing.MatLike]:
    img = to_numpy(img, convert_bool=True)

    poly = []
    for c in cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        e = epsilon * cv2.arcLength(c, True)
        q = cv2.approxPolyDP(c, e, True)
        poly.append(q)

    return poly


def crop_rect(img: ImageType, rect: np.ndarray) -> np.ndarray:
    if isinstance(rect, ultralytics.engine.results.Boxes):
        rect = rect.xyxy[0]

    if len(rect) == 4 and len(rect.shape) == 1:  # (x1, y1, x2, y2) format
        x1, y1, x2, y2 = rect
        return crop_rect(img, np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype="int"))

    img = to_numpy(img, convert_bool=True)

    width = int(((rect[0, 0] - rect[1, 0]) ** 2 + (rect[0, 1] - rect[1, 1]) ** 2) ** 0.5)
    height = int(((rect[2, 0] - rect[1, 0]) ** 2 + (rect[2, 1] - rect[1, 1]) ** 2) ** 0.5)

    src_pts = rect.astype("float32")
    dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_crop = cv2.warpPerspective(img, M, (width, height))

    return img_crop


__all__ = ["extract_quads", "extract_polygons", "extract_rotated_rects", "crop_rect"]
