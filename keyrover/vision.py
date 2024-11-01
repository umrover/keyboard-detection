import cv2
import numpy as np


def extract_rects(img: np.ndarray) -> list[cv2.typing.Rect]:
    quads = []
    for c in cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        q = cv2.boundingRect(c)
        quads.append(q)

    return quads


def extract_rotated_rects(img: np.ndarray) -> list[cv2.typing.MatLike]:
    quads = []
    for c in cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        q = cv2.minAreaRect(c)
        q = cv2.boxPoints(q)
        quads.append(q)

    return quads


def extract_quads(img: np.ndarray) -> list[cv2.typing.MatLike]:
    qs = []

    for c in cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        e = 0.05 * cv2.arcLength(c, True)

        while len(cv2.approxPolyDP(c, e, True)) < 4:
            e -= 0.01
        while len(q := cv2.approxPolyDP(c, e, True)) > 4:
            e += 0.01

        qs.append(q)

    return qs


def extract_polygons(img: np.ndarray, epsilon=0.01) -> list[cv2.typing.MatLike]:
    poly = []
    for c in cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        e = epsilon * cv2.arcLength(c, True)
        q = cv2.approxPolyDP(c, e, True)
        poly.append(q)

    return poly


def crop_rect(img: np.ndarray, rect: np.ndarray) -> np.ndarray:
    width, height = int(rect[1][0]), int(rect[1][1])

    src_pts = rect.astype("float32")
    dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_crop = cv2.warpPerspective(img, M, (width, height))

    return img_crop


__all__ = ["extract_rects", "extract_quads", "extract_polygons", "extract_rotated_rects", "crop_rect"]
