import os
import threading
import multiprocessing as mp
import math
import random

import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class ImageDim():
    def __init__(self, width, height, channels):
        self._width = width
        self._height = height
        self._channels = channels

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_channels(self):
        return self._channels

    def get_HW(self):
        return (self._height, self._width)

    def get_WH(self):
        return (self._width, self._height)

    def get_CHW(self):
        return (self._channels, self._height, self._width)

    def get_WHC(self):
        return (self._width, self._height, self._channels)

    def get_BCHW(self, batch_size=1):
        return (batch_size, self._channels, self._height, self._width)


class DataLoader:
    def __init__(self, dataset, batch_size=16, workers=4, shuffle=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self._pool = mp.Pool(self._workers)
        self._results = None
        self._data = None
        self._batches = math.ceil(len(dataset) / batch_size)

    def __len__(self):
        return self._batches

    def __iter__(self):
        self._data = [i for i in range(len(self._dataset))]
        if self._shuffle:
            random.shuffle(self._data)
        self._queue_next()
        return self

    def __next__(self):
        if self._results is None:
            raise StopIteration
        else:
            results = list(self._results)
            if isinstance(results[0], np.ndarray):
                results = torch.from_numpy(np.array(results))
            else:
                res = []
                for i in range(len(results[0])):
                    data = []
                    for j in range(len(results)):
                        data.append(results[j][i])
                    res.append(torch.from_numpy(np.array(data)))
                results = res

            self._queue_next()
            return results

    def _queue_next(self):
        if len(self._data) == 0:
            self._results = None
        else:
            upper = min(self._batch_size, len(self._data))
            numbers = self._data[:upper]
            self._results = self._pool.imap(
                self._load_data,
                [(self._dataset, i) for i in numbers]
            )
            del self._data[:upper]

    @staticmethod
    def _load_data(args):
        return args[0][args[1]]


class TrainingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_dims, mask_dims, colors):
        self._img_dir = img_dir
        self._mask_dir = mask_dir
        self._img_dims = img_dims
        self._mask_dims = mask_dims
        self._colors = colors
        try:
            self._images = os.listdir(img_dir)
        except:
            self._images = None

    def __len__(self):
        if self._images is None:
            return 0
        else:
            return len(self._images)

    def __getitem__(self, index):
        return self._load_image(index)

    def _load_image(self, index):
        image_path = f"{self._img_dir}/{self._images[index]}"
        mask_path = f"{self._mask_dir}/{self._images[index].split('.')[0]}.png"

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB).astype(np.float32)

        image = cv2.resize(image, self._img_dims.get_WH(), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self._mask_dims.get_WH(), interpolation=cv2.INTER_NEAREST)

        for i in range(len(self._colors)):
            mask[np.all(mask == self._colors[i], axis=-1)] = i
        mask = mask[:, :, 0].astype(np.int_)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (0, 1))

        if np.amax(mask) > len(self._colors):
            print(f"{mask_path} has values outside number of classes. (Is it a png? Are colors blended?)")

        return image, mask


class ImageDataset(Dataset):
    def __init__(self, img_dir, img_dims):
        self._img_dir = img_dir
        self._img_dims = img_dims
        try:
            self._images = os.listdir(img_dir)
        except:
            self._images = None

    def __len__(self):
        if self._images is None:
            return 0
        else:
            return len(self._images)

    def __getitem__(self, index):
        return self._load_image(index)

    def _load_image(self, index):
        image_path = f"{self._img_dir}/{self._images[index]}"
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        image = cv2.resize(image, self._img_dims.get_WH(), interpolation=cv2.INTER_NEAREST)
        image = np.transpose(image, (2, 0, 1))
        return image


def load_colors(path):
    colors = []
    file = open(path, 'r')
    for line in file:
        if '\n' in line:
            line = line.replace('\n', '')
        if ' ' in line:
            line = line.replace(' ', '')
        numbers = line.split(',')
        for i in range(0, len(numbers)):
            numbers[i] = int(numbers[i])
        colors.append(tuple(numbers))
    return colors
