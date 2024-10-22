from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class KeyboardDataset(Dataset):
    def __init__(self, _imgs, _transforms=()):
        self._images = _imgs
        self._transforms = transforms.Compose(_transforms)

    def __getitem__(self, idx):
        _img = self._images[idx]
        _img = self._transforms(_img)
        return _img

    def __len__(self):
        return len(self._images)


__all__ = ["KeyboardDataset"]
