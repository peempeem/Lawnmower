import time
import cv2
import numpy as np
import sys
import os

sys.path.insert(1, './utils')

from sleeptools import *
from datalink import *
from camera import *
from gnetTRT import *
from mapping import *
from robotcontroller import *
from helpfuncs import *

image_params = {'width': 480, 'height': 360, 'channels': 3}
encode_param = [cv2.IMWRITE_JPEG_QUALITY, 100]
model_path = './files/gnet.onnx'
class_path = './files/classes.txt'
color_path = './files/colors.txt'
classes = load_classes(class_path)
colors = load_colors(color_path)

saved_images = './images'

port = 42069

def main():
    link = DataLink('server', True, port=port)
    camera = Camera(image_params['width'], image_params['height'])
    rc = RobotController('/dev/ttyACM0', 115200)
    img_saver = ImageSaver(saved_images, clean=True)
    pipe = InferencePipeline(camera, model_path)

    camera_info = camera.get_info()

    status = {
        'power': True,
        'running': True,
        'sleeping': False,
        'rc': False,
        'img_logging': False
    }
    rc_data = {'forward': 0, 'steer': 0}
    last_rc = time.perf_counter()
    rc_timeout = 0.5

    main_rate = Rate(60)
    status_rate = Rate(3)
    stream_fps = Rate(15)
    img_save_rate = Rate(0.75)
    map_send_rate = Rate(2)
    map = Map(len(classes), size=5, scale=0.5)
    projections = None

    link.start()
    camera.start()
    rc.start()
    pipe.start()

    try:
        while status['running']:
            if link.data_available():
                msg = link.get()['data']
                type = msg['type']
                if type == 'cmd':
                    cmd = msg['data'].lower()
                    if cmd == 'quit' or cmd == 'exit':
                        status['running'] = False
                        status['power'] = False
                    elif cmd == 'rec':
                        status['img_logging'] = True
                    elif cmd == 'stop':
                        status['rc'] = False
                        status['img_logging'] = False
                elif type == 'rc':
                    status['rc'] = True
                    rc_data = msg['data']
                    last_rc = time.perf_counter()

            if status['rc']:
                if time.perf_counter() - last_rc > rc_timeout:
                    rc.set_motor_speed(0, 0, 0)
                    status['rc'] = False
                else:
                    left_pow = rc_data['forward'] + rc_data['steer']
                    right_pow = rc_data['forward'] - rc_data['steer']
                    abs_left = abs(left_pow)
                    abs_right = abs(right_pow)
                    if abs_left > abs_right:
                        if abs_left > 1:
                            right_pow *= 1 / abs_left
                            if left_pow > 0:
                                left_pow = 1
                            else:
                                left_pow = right_pow
                                right_pow = -1
                    else:
                        if abs_right > 1:
                            left_pow *= 1 / abs_right
                            if right_pow > 0:
                                right_pow = 1
                            else:
                                right_pow = left_pow
                                left_pow = -1
                    rc.set_motor_speed(left_pow, right_pow, 0)

            isnew, image = camera.capture()
            if isnew:
                if stream_fps.ready():
                    #image = draw_distortion(image, camera_info)
                    _, out = cv2.imencode('.jpg', image, encode_param)
                    msg = {'type': 'image_stream', 'data': out}
                    link.send(msg)
                if status['img_logging'] and img_save_rate.ready():
                    img_saver.save_image(frame)
                    print('Saving an image')

            isnew, inference = pipe.get_inference()
            if isnew:
                projections = project_global(inference, camera_info, rc.pos)
                map.map_projections(projections)
                if map_send_rate.ready():
                    '''image = map_to_image(map.get_data(), colors, image_params['width'])
                    _, out = cv2.imencode('.jpg', image, encode_param)
                    msg = {'type': 'image_stream', 'data': out}
                    link.send(msg)'''
                    msg = {'type': 'map_stream', 'data': map.get_data()}
                    link.send(msg)

            if status_rate.ready():
                msg = {'type': 'status', 'data': status}
                link.send(msg)

            main_rate.sleep()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, exc_obj, exc_tb)
    pipe.stop()
    rc.stop()
    link.stop()
    camera.close()

if __name__ == '__main__':
    main()
