import numpy as np


class ImageBuffer:
    def __init__(self, dim1, dim2, dim3=0, buffer_size=8):
        self._buffer_size = buffer_size

        if dim3 !=0:
            self._buffer = np.zeros((buffer_size, dim1, dim2, dim3), dtype=np.float32)
        else:
            self._buffer = np.zeros((buffer_size, dim1, dim2), dtype=np.float32)
        self._new_images = np.zeros((1, buffer_size), dtype=bool)
        self._ptr = 0

    def new_buffer(self):
        a = np.zeros((1, self._buffer_size), dtype=bool)
        self._new_images = np.append(self._new_images, a, axis=0)
        return self._new_images.shape[0] - 1

    def insert(self, image):
        if image is None:
            return
        self._buffer[self._ptr] = image
        for stream in range(self._new_images.shape[0]):
            self._new_images[stream, self._ptr] = True
        self._ptr += 1
        if self._ptr >= self._buffer_size:
            self._ptr = 0
        self._first = True

    def get_latest(self, stream=0):
        pos = self._get_pos()
        isnew = self._new_images[stream, pos]
        if isnew:
            self._new_images[stream, pos] = False
        return isnew, self._buffer[pos]

    def _get_pos(self):
        ptr = self._ptr - 1
        if ptr < 0:
            ptr += self._buffer_size
        return ptr
