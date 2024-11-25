def describe(arr) -> None:
    print(
        f"""{arr.__class__} ({arr.dtype}, shape={arr.shape})
        Min: {arr.min()}
        Max: {arr.max()}
        Mean: {arr.mean()}""")


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]
