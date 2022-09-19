import keyboard
import threading
import time
from datetime import datetime

from sleeptools import Rate


class Console():
        def __init__(self):
            self._current_str = []
            self._console_input = []

            self._rc_mode = False
            self._rc_thread = None
            self._new_rc_data = False
            self._rc_TTM = self._rc_TTR = 0
            self.set_rc_profile(0.25, 0.5)
            self._forward = self._steer = 0

            self._enabled = False

            keyboard.on_press(self._keydown_callback, suppress=False)

        def enable(self):
            self._enabled = True

        def disable(self):
            self._enabled = False

        def is_enabled(self):
            return self._enabled

        def data_available(self):
            if not self._rc_mode:
                return len(self._console_input) > 0
            else:
                return self._new_rc_data

        def get(self):
            if self.data_available():
                if self._rc_mode:
                    return {'forward': self._forward, 'steer': self._steer}
                else:
                    str = self._console_input[0]
                    self._console_input.pop(0)
                    return str
            return None

        def get_current(self):
            return "".join(self._current_str)

        def stop(self):
            if self._rc_mode:
                self.set_rc_mode(False)

        def set_rc_profile(self, time_to_max, time_to_reset):
            self._rc_TTM = time_to_max
            self._rc_TTR = time_to_reset

        def set_rc_mode(self, is_set):
            if self._rc_mode == is_set:
                return
            self._rc_mode = is_set
            if is_set:
                self._forward = self._steer = 0
                self._rc_thread = threading.Thread(target=self._update_rc)
                self._rc_thread.start()
            else:
                self._rc_thread.join()

        def is_rc_mode(self):
            return self._rc_mode

        def _keydown_callback(self, event):
            key = event.name
            if not self._rc_mode and key is not None:
                if self._enabled:
                    self._update_console(key)
            elif key == 'esc' and self._rc_mode:
                self.set_rc_mode(False)

        def _update_console(self, key):
            if key == 'space':
                key = ' '
            if len(key) == 1:
                self._current_str.append(key)
            else:
                if key == 'enter':
                    str = "".join(self._current_str)
                    self._console_input.append(str)
                    self._current_str = []
                elif key == 'backspace' and len(self._current_str) > 0:
                    self._current_str.pop()

        def _update_rc(self):
            rate = Rate(100)
            left = right = forwards = backwards = False
            forward = steer = 0
            while(self._rc_mode):
                left = keyboard.is_pressed('a') | keyboard.is_pressed('left')
                right = keyboard.is_pressed('d') | keyboard.is_pressed('right')
                forwards = keyboard.is_pressed('w') | keyboard.is_pressed('up')
                backwards = keyboard.is_pressed('s') | keyboard.is_pressed('down')

                change_active = rate.get_inverse_rate() / self._rc_TTM
                change_reset = rate.get_inverse_rate() / self._rc_TTR

                if left and not right:
                    steer -= change_active
                    if steer < -1:
                        steer = -1
                elif right and not left:
                    steer += change_active
                    if steer > 1:
                        steer = 1
                else:
                    if steer < 0:
                        if steer + change_reset > 0:
                            steer = 0
                        else:
                            steer += change_reset
                    elif steer > 0:
                        if steer - change_reset < 0:
                            steer = 0
                        else:
                            steer -= change_reset

                if forwards and not backwards:
                    forward += change_active
                    if forward > 1:
                        forward = 1
                elif backwards and not forwards:
                    forward -= change_active
                    if forward < -1:
                        forward = -1
                else:
                    if forward < 0:
                        if forward + change_reset > 0:
                            forward = 0
                        else:
                            forward += change_reset
                    elif forward > 0:
                        if forward - change_reset < 0:
                            forward = 0
                        else:
                            forward -= change_reset

                self._steer = steer
                self._forward = forward

                self._new_rc_data = True

                rate.sleep()


class ConsoleMemoryManager:
    def __init__(self, max_char_count=2**10):
        self._max_char_count = max_char_count
        self._current_size = 0
        self._strings = []

    def print(self, string, end='\n', timestamp=True):
        if timestamp:
            string = string.join([f"[ {datetime.now().strftime('%X')} ] >>> ", end])
        else:
            string += end
        if len(string) < self._max_char_count:
            self._current_size += len(string)
            self._strings.append(string)
            while self._current_size > self._max_char_count:
                self._current_size -= len(self._strings[0])
                self._strings.pop(0)

    def clear(self):
        self._strings = []

    def get(self):
        return "".join(self._strings)


if __name__ == '__main__':
    con = Console()
    cmm = ConsoleMemoryManager()
    rate = Rate(100)
    try:
        while True:
            if con.data_available() and not con.is_rc_mode():
                cmm.print(con.get(), end='')
                print(cmm.get())
            rate.sleep()
    except KeyboardInterrupt:
        pass
    con.stop()
