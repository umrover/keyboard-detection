from typing import Sequence

import numpy as np


def median_filter(objects: Sequence, statistic: np.ndarray, threshold: float = -0.5):
    """
    Filters objects using the statistic corresponding to the object
    """
    deviation = statistic - np.median(statistic)
    median = np.median(statistic)
    statistic = deviation / median if median else np.zeros(len(deviation))

    return [obj for s, obj in zip(statistic, objects) if s > threshold]


__all__ = ['median_filter']
