import numpy as np
import matplotlib.image
import os


def imread(path):
    if not os.path.exists(path):
        for extension in (".jpg", ".png"):
            if os.path.exists(path + extension):
                path = path + extension
                break
        else:
            raise ValueError("No such file: %s" % repr(path))
    im = matplotlib.image.imread(path)
    assert len(im.shape) == 3
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.
    if im.shape[2] == 4:
        assert np.alltrue(im[:, :, 3] == 1.0)
        im = im[:, :, :3]
    assert im.shape[2] == 3
    assert im.dtype == np.float32
    return im


def imsave(path, im):
    return matplotlib.image.imsave(path, im)
