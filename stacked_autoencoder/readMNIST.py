import struct
import numpy as np


def read(imagefile, labelfile):
    """
    Python function for reading in MNIST data. Run in the local directory.
    Inputs are name of imagefile and labelfile
    """
    with open(labelfile, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromfile(flbl, dtype=np.int8)

    with open(imagefile, 'rb') as fimg:

        magic = np.fromfile(fimg, dtype=np.dtype('>i4'), count=1)

        num_images = np.fromfile(fimg, dtype=np.dtype('>i4'), count=1)
        num_rows = np.fromfile(fimg, dtype=np.dtype('>i4'), count=1)
        num_cols = np.fromfile(fimg, dtype=np.dtype('>i4'), count=1)
        images = np.fromfile(fimg, dtype=np.ubyte)
        images = images.reshape((num_images[0], num_rows[0] * num_cols[0])).transpose()
        images = images.astype(np.float64) / 255

        fimg.close()

    return images, label
