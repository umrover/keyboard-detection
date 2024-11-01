def quad_area(x1, y1, x2, y2, x3, y3, x4, y4):
    return 0.5 * abs((x1 * y2 - x2 * y1)
                     + (x2 * y3 - x3 * y2)
                     + (x3 * y4 - x4 * y3)
                     + (x4 * y1 - x1 * y4))


def aspect_ratio(x1, y1, x2, y2, x3, y3, x4, y4):
    w = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    h = ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 0.5
    return w / h
