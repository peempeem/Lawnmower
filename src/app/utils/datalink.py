import time
import random
import threading
import pickle
import zmq

from sleeptools import Rate


class DataLink():
    PRIORITY_HIGH = 2
    PRIORITY_MED = 1
    PRIORITY_LOW = 0

    MSG_ID_LEN = 10
    PING_ID = "<[[__PING__]]>"
    REPING_ID = "<[[_REPING_]]>"

    SLEEP_FAST = 200
    SLEEP_SLOW = 10

    def __init__(self, name, static, host="localhost", port=42069):
        self._name = name

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR)
        if static:
            self._socket.bind(f"tcp://*:{str(port)}")
        else:
            self._socket.connect(f"tcp://{host}:{str(port)}")

        rand_chars = [random.randint(33, 126) for _ in range(self.MSG_ID_LEN)]
        self._link_id = "".join([chr(c) for c in rand_chars])

        self._ping_time = 0.5
        self._ping_timeout = 3
        self._latency = float('inf')

        self._send_msgs = []
        self._receive_msgs = []
        self._running = False

        self._sleep_rate = Rate(self.SLEEP_FAST)

        self._backend = threading.Thread(target=self._backend)
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def latency(self, string=False):
        if string:
            if self._latency is float('inf'):
                return "??? ms"
            return "{:.2f} ms".format(self._latency * 1000)
        return self._latency

    def start(self):
        if not self._running:
            self._running = True
            self._start = time.perf_counter()
            self._backend.start()

    def stop(self):
        if self._running:
            self._running = False
            self._backend.join()

    def set_sleep(self, sleep):
        if sleep:
            self._sleep_rate = Rate(self.SLEEP_SLOW)
        else:
            self._sleep_rate = Rate(self.SLEEP_FAST)

    def send(self, data, msg_id=None, priority=0):
        if self._running:
            if msg_id is None:
                msg_id_gen = [random.randint(33, 126) for _ in range(self.MSG_ID_LEN)]
                msg_id = "".join([chr(c) for c in msg_id_gen])

            msg = {'sender': self._name,
                   'msg_id': msg_id,
                   'priority': priority,
                   'data': data}
            with self._send_lock:
                self._send_msgs.append(msg)
            return msg
        return None

    def data_available(self):
        return len(self._receive_msgs) != 0

    def get(self):
        if len(self._receive_msgs) != 0:
            with self._recv_lock:
                msg = self._receive_msgs[0]
                self._receive_msgs.pop(0)
            return msg
        return None

    def _backend(self):
        pinging = False
        ping_start = 0
        ping_count = 0
        while self._running:
            t = time.perf_counter()

            # RECEIVING
            while self._running:
                try:
                    msg = pickle.loads(self._socket.recv(flags=zmq.NOBLOCK))
                    id = msg['msg_id']
                    if id == self.PING_ID:
                        self.send(0, msg_id=self.REPING_ID, priority=self.PRIORITY_HIGH)
                    elif id == self.REPING_ID:
                        self._latency = t - ping_start
                        pinging = False
                        ping_count += 1
                    else:
                        with self._recv_lock:
                            self._receive_msgs.append(msg)
                except zmq.error.Again:
                    break

            # SENDING
            if pinging:
                if t - ping_start > self._ping_timeout:
                    pinging = False
                    ping_count += 1
                    self._latency = float('inf')
            elif t - ping_start > self._ping_time:
                pinging = True
                ping_start = t
                self.send(ping_count, msg_id=self.PING_ID, priority=self.PRIORITY_HIGH)

            while len(self._send_msgs) != 0:
                priority = self.PRIORITY_LOW
                with self._send_lock:
                    for i in range(len(self._send_msgs)):
                        m = self._send_msgs[i]
                        if m['priority'] >= priority:
                            priority = m['priority']
                            msg = self._send_msgs[i]

                msg = pickle.dumps(msg, protocol=4)
                try:
                    self._socket.send(msg, flags=zmq.NOBLOCK)
                    self._send_msgs.pop(i)
                except zmq.error.Again:
                    break

            self._sleep_rate.sleep()


if __name__ == "__main__":
    import console

    d1 = DataLink("Link1", True)
    d1.start()
    d2 = DataLink("Link2", False)
    d2.start()
    con = console.ConsoleInput()

    try:
        str = ""
        while str != "quit":
            if con.data_available():
                str = con.get()
                d1.send(str)
            print(d2.latency(string=True))
            while d2.data_available():
                print(f"Link 2 received \"{d2.get()['data']}\" from link 1. {d2.latency(string=True)}")
            time.sleep(0.01)
    except KeyboardInterrupt as e:
        pass
    con.stop()
    d1.stop()
    d2.stop()
