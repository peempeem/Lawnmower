import keyboard
import threading
import time

from sleeptools import Rate


class Console():
        def __init__(self):
            self._current_str = []
            self._console_input = []

            self._rc_mode = False
            self._rc_thread = None
            self._rc_TTM = self._rc_TTR = 0
            self.set_rc_profile(0.25, 0.5)

            self._forward = self._steer = 0

            keyboard.on_press(self._keydown_callback, suppress=False)

        def data_available(self):
            return len(self._console_input) > 0

        def get(self):
            if self.data_available():
                str = self._console_input[0]
                self._console_input.pop(0)
                return str
            return None

        def get_rc(self):
            return {'forward': self._forward, 'steer': self._steer}

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
                rate.set_start()
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

                rate.sleep()


if __name__ == "__main__":
    import sys
    con = Console()
    con.set_rc_mode(True)

    try:
        while True:
            if con.data_available():
                print(con.get())
            else:
                print(con.get_rc())
            time.sleep(0.01)
    except KeyboardInterrupt as e:
        pass

    con.stop()
