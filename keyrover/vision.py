import cv2
import numpy as np


def extract_rects(img: np.ndarray) -> list[cv2.typing.Rect]:
    quads = []
    for c in cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        q = cv2.boundingRect(c)
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


__all__ = ["extract_rects", "extract_quads", "extract_polygons"]
