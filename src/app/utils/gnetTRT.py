import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import time
import cv2
import threading
import multiprocessing as mp
import queue

from imagebuffer import ImageBuffer
from sleeptools import Rate


class GNetTRT:
    def __init__(self, verbose=False):
        self._verbose = verbose

        self._width = 320
        self._height = 240
        self._classes = 4

        self._out_width = 17
        self._out_height = 13

        self._engine = None
        self._in_cpu = None
        self._out_cpu = None
        self._in_gpu = None
        self._out_gpu = None

        self._logger = trt.Logger(trt.Logger.WARNING)

    def build_engine(self, model_path):
        if self._verbose:
            self._print(f"Building engine from \"{model_path}\"")
        batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(self._logger)
        config = builder.create_builder_config()
        network = builder.create_network(batch)
        parser = trt.OnnxParser(network, self._logger)
        with open(model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        config.max_workspace_size = 1 << 10
        builder.int8_mode = True
        self._engine = builder.build_engine(network, config)
        # host cpu memory
        host_in_size = trt.volume(self._engine.get_binding_shape(0))
        host_out_size = trt.volume(self._engine.get_binding_shape(1))
        host_in_dtype = trt.nptype(self._engine.get_binding_dtype(0))
        host_out_dtype = trt.nptype(self._engine.get_binding_dtype(1))
        self._in_cpu = cuda.pagelocked_empty(host_in_size, host_in_dtype)
        self._out_cpu = cuda.pagelocked_empty(host_out_size, host_out_dtype)
        # allocate gpu memory
        self._in_gpu = cuda.mem_alloc(self._in_cpu.nbytes)
        self._out_gpu = cuda.mem_alloc(self._out_cpu.nbytes)
        if self._verbose:
            self._print("Engine is built")

    def create_context(self):
        return self._engine.create_execution_context()

    def prepare_image(self, img):
        img = img.astype(np.float32) / 255
        img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=img.dtype)
        return img

    def inference(self, inputs, context):
        cuda.memcpy_htod(self._in_gpu, inputs)
        context.execute(1, [int(self._in_gpu), int(self._out_gpu)])
        cuda.memcpy_dtoh(self._out_cpu, self._out_gpu)
        result = self._out_cpu.reshape(self._classes, self._out_height, self._out_width)
        result = np.argmax(np.transpose(result, (1, 2, 0)), axis=2)
        return result

    def output_dims(self):
        return (self._out_height, self._out_width)

    def _print(self, msg):
        print(f"[GNET] {msg}")


class InferencePipeline:
    def __init__(self, camera, model_path, fps=30):
        self._camera = camera
        self._model_path = model_path
        self._fps = fps

        self._model = GNetTRT(verbose=True)
        self._inf_dims = self._model.output_dims()
        self._inference_buffer = ImageBuffer(self._inf_dims[0], self._inf_dims[1])

        self._stream = camera.new_stream()

        self._inf_thread = None
        self._stop = threading.Event()
        self._stop.set()

    def start(self):
        if self._stop.is_set():
            self._stop.clear()
            self._inf_thread = threading.Thread(target=self._update)
            self._inf_thread.start()

    def stop(self):
        if not self._stop.is_set():
            self._stop.set()
            self._inf_thread.join()

    def get_inference(self):
        return self._inference_buffer.get_latest()

    def get_inf_dims(self):
        return self._inf_dims

    def _update(self):
        img_q = mp.Queue()
        inf_q = mp.Queue()
        producer = Inferencer(0, self._model, self._model_path, self._fps, img_q, inf_q)
        producer.start()
        rate = Rate(self._fps * 2)

        while not self._stop.is_set():
            if producer.request_img.is_set():
                isnew, frame = self._camera.capture(stream=self._stream)
                if isnew:
                    try:
                        img_q.put_nowait(frame)
                        producer.request_img.clear()
                    except queue.Full as e:
                        pass
            try:
                inference = inf_q.get_nowait()
                self._inference_buffer.insert(inference)
            except queue.Empty as e:
                pass
            rate.sleep()

        try:
            while True:
                img_q.get_nowait()
        except queue.Empty as e:
            pass

        producer.stop.set()
        producer.join()


class Inferencer(mp.Process):
    def __init__(self, gpuID, model, model_path, fps, img_q, inf_q, timeout=5):
        mp.Process.__init__(self)

        self._gpuID = gpuID
        self._model = model
        self._model_path = model_path
        self._fps = fps
        self._img_q = img_q
        self._inf_q = inf_q
        self._timeout = timeout

        self.request_img = mp.Event()
        self.stop = mp.Event()

    def run(self):
        cuda.init()
        device = cuda.Device(self._gpuID)
        ctx = device.make_context()
        self._model.build_engine(self._model_path)
        exe_ctx = self._model.create_context()

        last = time.perf_counter()

        rate = Rate(self._fps)
        while not self.stop.is_set():
            try:
                img = self._img_q.get_nowait()
                last = time.perf_counter()
                new_img = True
            except queue.Empty as e:
                self.request_img.set()
                new_img = False
                if time.perf_counter() - last >= self._timeout:
                    self.stop.set()
            if new_img:
                img = self._model.prepare_image(img)
                output = self._model.inference(img, exe_ctx)
                try:
                    self._inf_q.put_nowait(output)
                except queue.Full as e:
                    pass
            rate.sleep()

        try:
            while True:
                self._inf_q.get_nowait()
        except queue.Empty as e:
            pass
        ctx.detach()
