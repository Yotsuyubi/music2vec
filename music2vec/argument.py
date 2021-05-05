import torch as th
import numpy as np


class Crop(object):
    """Crop
    Crop samples
    Args:
        length (int): length for crop sample.
        start (int, optional): index for start crop.
                              Default: `0`
    Returns:
        numpy.ndarray with shape (length).
    """
    def __init__(self,
                length: int,
                *,
                start: int=0):
        self.length: int = length
        self.start: int = start

    def __call__(self, data):
        if self.start > len(data):
            raise IndexError('`start` out of range.')
        if self.start < 0 or self.length < 0:
            raise ValueError('`start` and `length` must be positive int.')
        return data[self.start:self.start+self.length]


class RandomCrop(object):
    """RandomCrop
    Crop samples randamly
    Args:
        length (int): length for crop sample.
    Returns:
        numpy.ndarray with shape (length).
    """
    def __init__(self,
                length=None):
        self.length: int = length if length is not None else 0

    def __call__(self, data):
        start = np.random.randint(0, np.abs(len(data)-self.length))
        if self.length < 0:
            raise ValueError('`length` must be positive int.')
        if len(data)-self.length < 0:
            raise ValueError('`length` too large.')
        return Crop(length=self.length, start=start)(data)