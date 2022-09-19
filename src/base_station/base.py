import time
import cv2
import sys

sys.path.insert(1, './utils')

from datalink import DataLink
from console import Console
from sleeptools import Rate
from unetfunctions import UNETFunctions


class_path = "files/classes.txt"
color_path = "files/colors.txt"

ip = "192.168.2.73"
port = 42069

if __name__ == "__main__":
    link = DataLink("client", False, host=ip, port=port)
    link.start()
    con = Console()
    con.set_rc_profile(0.2, 0.4)
    unetf = UNETFunctions(class_path, color_path)

    img = None
    inference = None
    main_rate = Rate(100)
    print_rate = Rate(5)
    rc_send_rate = Rate(15)

    running = True
    sleeping = False

    print_rate.set_start()
    rc_send_rate.set_start()
    try:
        while running:
            main_rate.set_start()

            if con.data_available():
                cmd = con.get()
                msg = {'type': "cmd", 'data': cmd}
                print(f"Sending \"{cmd}\"")
                cmd = cmd.lower()
                if cmd == "rc":
                    print("Entering RC mode. Hit escape to exit.")
                    con.set_rc_mode(True)
                link.send(msg)

            if con.is_rc_mode() and rc_send_rate.ready():
                rc_send_rate.set_start()
                msg = {'type': "rc", 'data': con.get_rc()}
                link.send(msg)

            while link.data_available():
                msg = link.get()['data']
                type = msg['type']
                if type == "status":
                    status = msg['data']
                    if not status['power']:
                        running = False
                    if status['sleeping']:
                        sleeping = True
                        cv2.destroyAllWindows()
                    else:
                        sleeping = False
                elif type == "image_stream":
                    img = cv2.imdecode(msg['data'], cv2.IMREAD_COLOR)
                    height, width, _ = img.shape
                    height *= 2
                    width *= 2
                elif type == "inference_stream":
                    inference = msg['data']

            if img is not None and not sleeping:
                img = cv2.resize(img, (width, height))
                if inference is not None:
                    img = unetf.draw_dots(img, inference)
                cv2.imshow('', img)
                cv2.waitKey(1)

            if print_rate.ready():
                print_rate.set_start()
                print(link.latency(string=True))

            main_rate.sleep()
    except (Exception, KeyboardInterrupt) as e:
        print(e)

    con.stop()
    link.stop()
    cv2.destroyAllWindows()
