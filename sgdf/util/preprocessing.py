import numpy as np


def to_mask(im):
    assert len(im.shape) == 3
    im = np.mean(im, 2)
    im = (im > np.average(im))
    assert im.dtype == np.bool
    return im
