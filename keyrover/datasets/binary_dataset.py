from typing import Sequence

from multiprocessing import Pool
from tqdm import tqdm

from torchvision.transforms import v2 as transforms

from keyrover.util import to_tensor
from keyrover.images.binary_key_mask import KeyBinaryMaskImage

from .abstract import KeyboardDataset


class KeyboardBinaryMaskDataset(KeyboardDataset):
    def __init__(self, filenames: Sequence[str], size: tuple[int, int], **kwargs):
        resize = transforms.Resize(size)

        with Pool() as p:
            images = tqdm(p.imap(KeyboardImage, filenames), total=len(filenames))
            images = [(to_tensor(image)) for image in images]

            masks = tqdm(p.imap(KeyBinaryMaskImage, filenames), total=len(filenames))
            masks = [(to_tensor(mask)) for mask in masks]

        super().__init__(images=images, targets=masks, **kwargs)


__all__ = ["KeyboardBinaryMaskDataset"]


from keyrover.images import KeyboardImage
