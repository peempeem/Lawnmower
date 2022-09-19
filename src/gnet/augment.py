import os
from random import random
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import cv2
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian


def clear_dir(path, rmdir=False):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(path + "/" + name)
        if rmdir:
            os.rmdir(path)


def read_RGB(img_name):
	img = cv2.imread(img_name)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_RGB(img_name, img):
	img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
	cv2.imwrite(img_name, img)


class ImageAugmenter:
	def __init__(self):
		self._img_dir = None
		self._mask_dir = None
		self._img_out_dir = None
		self._mask_out_dir = None

	def augment_images(self, img_dir, mask_dir, img_out_dir, mask_out_dir, workers=12):
		self._img_dir = img_dir
		self._mask_dir = mask_dir
		self._img_out_dir = img_out_dir
		self._mask_out_dir = mask_out_dir

		clear_dir(img_out_dir, rmdir=True)
		clear_dir(mask_out_dir, rmdir=True)
		os.mkdir(img_out_dir)
		os.mkdir(mask_out_dir)

		images = os.listdir(img_dir)

		with mp.Pool(workers) as pool:
			results = tqdm(pool.imap(self._augment_image, images), total=len(images))
			for _ in results:
				pass

	def _augment_image(self, image_name):
		image_name, image_type = image_name.split('.')[:2]

		image_path = f"{self._img_dir}/{image_name}.{image_type}"
		mask_path = f"{self._mask_dir}/{image_name}.png"

		image = read_RGB(image_path)
		mask = read_RGB(mask_path)

		rand = random()
		hrand = rand - 0.5

		img_out = f"{self._img_out_dir}/{image_name}"
		mask_out = f"{self._mask_out_dir}/{image_name}"

		image_rotated = rotate(image, angle=(90 * hrand), mode='wrap', preserve_range=True)
		mask_rotated = rotate(mask, angle=(90 * hrand), mode='wrap', preserve_range=True, order=0)
		write_RGB(f"{img_out}_rotated.jpg", image_rotated)
		write_RGB(f"{mask_out}_rotated.png", mask_rotated)

		height, width = image.shape[:2]
		transform = AffineTransform(translation=(width * hrand, height * hrand))
		image_shifted = warp(image, transform, mode='wrap', preserve_range=True)
		mask_shifted = warp(mask, transform, mode='wrap', preserve_range=True, order=0)
		write_RGB(f"{img_out}_shifted.jpg", image_shifted)
		write_RGB(f"{mask_out}_shifted.png", mask_shifted)

		image_fliph = cv2.flip(image, 1)
		mask_fliph = cv2.flip(mask, 1)
		write_RGB(f"{img_out}_fliph.jpg", image_fliph)
		write_RGB(f"{mask_out}_fliph.png", mask_fliph)

		image_flipv = cv2.flip(image, 0)
		mask_flipv = cv2.flip(mask, 0)
		write_RGB(f"{img_out}_flipv.jpg", image_flipv)
		write_RGB(f"{mask_out}_flipv.png", mask_flipv)

		image_noisy = random_noise(image, var=(rand / 50)) * 255
		write_RGB(f"{img_out}_noisy.jpg", image_noisy)
		write_RGB(f"{mask_out}_noisy.png", mask)

		image_blurred = gaussian(image, sigma=5, multichannel=True) * 255
		write_RGB(f"{img_out}_blurry.jpg", image_blurred)
		write_RGB(f"{mask_out}_blurry.png", mask)

		return True


if __name__ == "__main__":
	img_dir = "../ImageDB/raw/images"
	mask_dir = "../ImageDB/raw/masks"
	img_out_dir = "../ImageDB/aug/images"
	mask_out_dir = "../ImageDB/aug/masks"

	ImageAugmenter().augment_images(img_dir, mask_dir, img_out_dir, mask_out_dir)
