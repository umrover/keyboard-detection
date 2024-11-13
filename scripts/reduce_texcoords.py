import pickle

import numpy as np

from keyrover.datasets import id_to_key
from keyrover import *


def reduce_texcoords(data) -> tuple[tuple[list, dict[int, float]], tuple[list, dict[int, float]]]:
    u_all = []
    v_all = []

    u_by_class = {}
    v_by_class = {}

    for u, v, classes in data:
        u = np.array(u) / 127.5 - 1  # scale to [-1, 1]
        v = np.array(v) / 127.5 - 1  # scale to [-1, 1]

        u_all += list(u)
        v_all += list(v)

        for x, y, cls in zip(u, v, classes):
            if cls not in id_to_key:
                print(f"WARNING: skipping class {cls} not found in id_to_key")
                continue

            if cls not in u_by_class:
                u_by_class[cls] = []
                v_by_class[cls] = []

            u_by_class[cls].append(x)
            v_by_class[cls].append(y)

    u_means = {}
    v_means = {}

    for (cls, u), (_, v) in zip(u_by_class.items(), v_by_class.items()):
        cls = int(cls)
        u_means[cls] = np.mean(u)
        v_means[cls] = np.mean(v)

    return (u_all, u_means), (v_all, v_means)


if __name__ == "__main__":
    with open(f"{RAW_TEXCOORDS}/key_texcoords.bin", "rb") as file:
        texcoords = pickle.load(file)

    (U, U_means), (V, V_means) = reduce_texcoords(texcoords)

    with open(f"{RAW_TEXCOORDS}/key_texcoords_means.bin", "wb") as file:
        pickle.dump(U_means, file)
        pickle.dump(V_means, file)

    with open(f"{RAW_TEXCOORDS}/key_texcoords_all.bin", "wb") as file:
        pickle.dump(U, file)
        pickle.dump(V, file)
