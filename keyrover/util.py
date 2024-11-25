def describe(arr) -> None:
    print(
        f"""{arr.__class__} ({arr.dtype}, shape={arr.shape})
        Min: {arr.min()}
        Max: {arr.max()}
        Mean: {arr.mean()}""")
