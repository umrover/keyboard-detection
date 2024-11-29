import pickle

from .keyboard_dataset import KeyboardTensorDataset


class KeyboardCornersDataset(KeyboardTensorDataset):
    corners = None

    @staticmethod
    def corners_from_filename(filename):
        if KeyboardCornersDataset.corners is None:
            with open("datasets/corners/corners.pkl", "rb") as f:
                KeyboardCornersDataset.corners = pickle.load(f)

        frame = int(filename.split("_")[1])
        return KeyboardCornersDataset.corners[frame - 1]

    _target = corners_from_filename


__all__ = ["KeyboardCornersDataset"]
