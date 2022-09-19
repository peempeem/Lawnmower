import time
import threading
import cv2

from sleeptools import Rate
from imagebuffer import ImageBuffer


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    output_width=1280,
    output_height=720,
    framerate=60,
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

        self._running = False
        self._lastcap = None

        self._update_thread = None
        self._imgbuf = ImageBuffer(height, width, 3, buffer_size)
        self._lock = threading.Lock()

        self._cap = cv2.VideoCapture(gstreamer_pipeline(output_width=width, output_height=height), cv2.CAP_GSTREAMER)

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
            start = rate.set_start()
            if start - self._lastcap >= self._timeout:
                self._running = False
            ret, img = self._cap.read()
            with self._lock:
                self._imgbuf.insert(img)
            rate.sleep()
