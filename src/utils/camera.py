import time
import threading
import cv2
import os
import math

from utils.sleeptools import Rate
from utils.imagebuffer import ImageBuffer


def gstreamer_pipeline(
    capture_width=1640,
    capture_height=1232,
    output_width=1280,
    output_height=720,
    framerate=29.9,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            output_width,
            output_height,
        )
    )


class Camera:
    def __init__(self, width, height, device_id=0, fps=30, buffer_size=8, timeout=15):
        self._width = width
        self._height = height
        self._device_id = device_id
        self._fps = fps
        self._timeout = timeout

        self._cam_height = 0.15 # meters
        self._cam_angle = -25  # degrees
        self._fov = 160        # degrees

        self._running = False
        self._lastcap = None

        self._update_thread = None
        self._imgbuf = ImageBuffer(height, width, 3, buffer_size)
        self._lock = threading.Lock()

        self._cap = cv2.VideoCapture(gstreamer_pipeline(output_width=width, output_height=height), cv2.CAP_GSTREAMER)

    def get_info(self):
        info = {
            'width': self._width,
            'height': self._height,
            'fov': self._fov,
            'cam_angle': self._cam_angle,
            'cam_height': self._cam_height,
            'distortion_fn': self.distortion_fn,
        }
        return info

    def distortion_fn(self, x):
        a = -0.397208
        b = -0.204189
        c = 1.6014
        d = 0
        y = a * pow(x, 3) + b * pow(x, 2) + c * x + d
        if y < 0:
            y = 0
        elif y > 1:
            y = 1
        return x

    def start(self):
        if not self._running:
            self._running = True
            self._lastcap = time.perf_counter()
            self._update_thread = threading.Thread(target=self._update)
            self._update_thread.start()

    def pause(self):
        if self._running:
            self._running = False
            self._update_thread.join()

    def close(self):
        self.pause()
        self._cap.release()

    def isRunning(self):
        return self._running

    def capture(self, stream=0):
        self._lastcap = time.perf_counter()
        with self._lock:
            cap = self._imgbuf.get_latest(stream)
        return cap

    def new_stream(self):
        with self._lock:
            res = self._imgbuf.new_buffer()
        return res

    def _update(self):
        rate = Rate(self._fps)
        while self._running:
            start = rate.get_start()
            if start - self._lastcap >= self._timeout:
                self._running = False
            ret, img = self._cap.read()
            with self._lock:
                self._imgbuf.insert(img)
            rate.sleep()


class ImageSaver:
    def __init__(self, path, base_name="image", clean=False):
        self._path = path
        self._base_name = base_name

        if clean:
            clear_dir(path)
        if not os.path.isdir(path):
            os.mkdir_path()
        for root, dirs, files in os.walk(path, topdown=False):
            self._image_count = len(files)

    def save_image(self, image):
        p = f"{self._path}/{self._base_name}"
        while os.path.isfile(f"{p}{self._image_count}.jpg"):
            self._image_count += 1
        cv2.imwrite(f"{p}{self._image_count}.jpg", image)


def clear_dir(path, rmdir=False):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(path + "/" + name)
        if rmdir:
            os.rmdir(path)
