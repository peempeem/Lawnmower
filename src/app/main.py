import sys
sys.path.insert(1, './utils')

from kivy.config import Config
Config.set('kivy', 'exit_on_escape', '0')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')


from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.clock import Clock

import cv2
import numpy as np

from kivyclasses import *
from colortools import *
from console import *
from messages import *
from sleeptools import *
from datalink import *
from mapping import *
from helpfuncs import *


class MainWindow(App):
    def build(self):
        min_window = (1280, 720)
        Window.minimum_width, Window.minimum_height = min_window
        Window.size = min_window

        self.window = BoxLayout(orientation='vertical')

        white = Color(255, 255, 255)
        mostly_white = Color(240, 240, 240)
        light_gray = Color(200, 200, 200)
        gray = Color(178, 190, 195)
        leafy_green = Color(109, 163, 39)
        dark_green = Color(44, 74, 4)
        sunset_orange = Color(207, 94, 37)
        sunset_purple = Color(181, 66, 201)
        gunmetal = Color(45, 52, 54)
        dim_gray = Color(99, 110, 114)
        dark_gray = Color(70, 70, 70)

        header_cc = ColorComponent(gray, light_gray, POS_TOP_CENTER, POS_BOT_CENTER)
        headerdiv_cc = ColorComponent(sunset_orange, sunset_purple, POS_LEFT, POS_RIGHT, txtsize=3)
        shelf_cc = ColorComponent(dark_gray, dim_gray)
        vdiv_cc = ColorComponent(dim_gray, dark_gray)
        window_cc = ColorComponent(dim_gray, dark_gray, POS_TOP_CENTER, POS_BOT_CENTER)

        settings = IconButton(
            icon_normal="./assets/settings.png",
            icon_pressed="./assets/settings_pressed.png"
        )

        self.header = Header(
            size_hint=(1, 0.1),
            color_component=header_cc,
            title_image="./assets/title.png",
            buttons=[settings]
        )

        self.header_div = Divider(
            size_hint=(1, 0.01),
            color_component=headerdiv_cc,
            rotate_speed=10
        )

        self.content_window = BoxLayout(orientation='horizontal')

        home = IconButton(
            icon_normal="./assets/home.png",
            icon_pressed="./assets/home_pressed.png"
        )

        console = IconButton(
            icon_normal="./assets/console.png",
            icon_pressed="./assets/console_pressed.png"
        )

        shelf_div1 = Divider(
            size_hint=(None, 0.005),
            color_component=headerdiv_cc
        )

        shelf_div2 = Divider(
            size_hint=(None, 0.005),
            color_component=headerdiv_cc
        )

        self.widget_shelf = WidgetShelf(
            [home, shelf_div1, console, shelf_div2],
            color_component=shelf_cc,
            size_hint=(0.175, 1)
        )

        self.shelf_content_div = Divider(
            width=10,
            size_hint=(None, 1),
            color_component=vdiv_cc
        )

        self.main_content = FloatLayout()

        self.console_pane = ConsolePane(
            size_hint=(0.5, 0.9),
            pos_hint={'x': 0.05},
            color_component=window_cc,
            bar_color_component=headerdiv_cc
        )

        self.settings_pane = SettingsPane(
            size_hint=(0.95, 0.95),
            pos_hint={'right': 1},
            color_component=window_cc,
            bar_color_component=headerdiv_cc
        )

        self.image_viewer = ImagePane(
            size_hint=(0.5, 1),
            pos_hint={'right': 1}
        )

        self.map_viewer = ImagePane(
            size_hint=(0.5, 1),
            pos_hint={'right': 0.5}
        )



        self.window.add_widget(self.header)
        self.window.add_widget(self.header_div)

        self.window.add_widget(self.content_window)

        self.content_window.add_widget(self.widget_shelf)
        self.content_window.add_widget(self.shelf_content_div)
        self.content_window.add_widget(self.main_content)

        self.main_content.add_widget(self.image_viewer)
        self.main_content.add_widget(self.map_viewer)
        self.main_content.add_widget(self.console_pane)
        self.main_content.add_widget(self.settings_pane)



        settings.set_callback(self.settings_pressed)
        home.set_callback(self.home_pressed)
        console.set_callback(self.console_pressed)


        Clock.schedule_interval(self._background_tasks, 1/100)
        Clock.schedule_interval(self._update_viewer, 1/60)
        self._con = Console()
        self._cmm = ConsoleMemoryManager(max_char_count=2**11)
        self._rc_mode = False

        self._robot_connected = False
        self._high_latency = False
        self._link = None
        self._link_rc_rate = Rate(25)

        self._img = None
        self._map = None
        self._new_img = False
        self._new_map = False
        self._colors = load_colors("./files/colors.txt")

        return self.window

    def settings_pressed(self):
        if self.settings_pane.is_open():
            self.settings_pane.close()
        else:
            self.settings_pane.open()

    def home_pressed(self):
        self.console_pane.close()
        self.settings_pane.close()

    def console_pressed(self):
        if self.console_pane.is_open():
            self.console_pane.close()
        else:
            self.console_pane.open()

    def _connect(self, ip, port):
        if self._link is not None:
            self._link.stop()
        self._link = DataLink("client", False, host=ip, port=port)
        self._link.start()

    def _update_viewer(self, dt):
        if self._new_img:
            self.image_viewer.set_image(self._img)
            self._new_img = False
        if self._new_map:
            img = map_to_image(self._map, self._colors)
            self.map_viewer.set_image(img)
            self._new_map = False

    def _background_tasks(self, dt):
        """ LINK STUFF """
        self._robot_connected = self._link is not None and self._link.latency() is not float('inf')
        if self._robot_connected:
            self._high_latency = self._link.latency() > 0.5

        if self._robot_connected:
            while self._link.data_available():
                msg = self._link.get()['data']
                if msg['type'] == 'status':
                    pass
                elif msg['type'] == 'image_stream':
                    self._img = np.float32(cv2.imdecode(msg['data'], cv2.IMREAD_COLOR)) / 255
                    self._new_img = True
                elif msg['type'] == 'map_stream':
                    self._map = msg['data']
                    self._new_map = True

        """ CONSOLE STUFF """
        if self.console_pane.is_open() and not self._con.is_enabled():
            self._con.enable()
        elif not self.console_pane.is_open() and self._con.is_enabled():
            self._con.disable()

        if self._rc_mode and not self._con.is_rc_mode():
            self._rc_mode = False
            self.print("Leaving RC mode.")

        if self._con.data_available():
            if self._con.is_rc_mode():
                self._rc_mode = True
                if self._link_rc_rate.ready():
                    if self._robot_connected:
                        self._link.send(rc_msg(self._con.get()))
            else:
                cmd = self._con.get().lower()
                self.buffer_print("")
                self.print(cmd, timestamp=True)
                if cmd == "":
                    pass
                elif cmd == "rc":
                    self.print("Entering RC mode, hit [esc] (escape key) to exit.")
                    if not self._robot_connected:
                        self.warn("Robot is not connected.")
                    elif self._high_latency:
                        self.warn(f"High latency may result in unwanted actions. Current latency: {self._link.latency(string=True)}")

                    self._con.set_rc_mode(True)
                    self._link_rc_rate.set_start()
                elif cmd[:5] == "send ":
                    self.print(f"Sending \"{cmd[5:]}\"")
                    if not self._robot_connected:
                        self.warn("Robot is not connected.")
                    else:
                        self._link.send(cmd_msg(cmd[5:]))
                elif cmd[:8] == "connect ":
                    try:
                        parts = cmd[8:].split('@')
                        ip = parts[0]
                        port = int(parts[1])
                        self.print(f"Connecting to {ip}@{port}")
                        self._connect(ip, port)
                    except (IndexError, ValueError) as e:
                        self.warn("Error parsing connection command.")
                elif cmd == "ping":
                    if not self._robot_connected:
                        self.warn("Robot is not connected.")
                    else:
                        self.print(self._link.latency(string=True))
                elif cmd == "clear":
                    self._cmm.clear()
                    self.update_console()
                elif cmd == "quit":
                    self.print("Exiting application...")
                    if self._robot_connected:
                        self.warn("Robot is still connected. This command will not stop the robot.")
                        self.warn("To quit while connected, enter \"quit -f\"")
                    else:
                        self._stop()
                elif cmd == "quit -f":
                    self.print("Exiting application...")
                    self._stop()
                elif cmd == "help":
                    self.buffer_print("\"rc\": enter remote control mode, escape to quit.")
                    self.buffer_print("\"send ...\": send a command directly to the robot.")
                    self.buffer_print("\"connect ...ip...@...port...\": connect to the robot.")
                    self.buffer_print("\"ping\": get latest robot ping.")
                    self.buffer_print("\"clear\": clear the console.")
                    self.print("\"quit\": exit the application when the robot is not connected. Use \"-f\" to force quit.")
                else:
                    self.warn(f"Unrecognized command \"{cmd}\". Type \"help\" for a list of commands.")

        self.console_pane.set_input_text(self._con.get_current())

    def buffer_print(self, text, end='\n', timestamp=False):
        if type(text) is not str:
            text = str(text)
        self._cmm.print(text, end=end, timestamp=timestamp)

    def print(self, text, end='\n', timestamp=False):
        self.buffer_print(text, end=end, timestamp=timestamp)
        self.update_console()

    def warn(self, text, end='\n', timestamp=False):
        text = "[WARN] " + text
        self.print(text, end=end, timestamp=timestamp)

    def update_console(self):
        self.console_pane.set_window_text(self._cmm.get())

    def _stop(self):
        self._con.stop()
        if self._link is not None:
            self._link.stop()
        self.stop()


if __name__ == '__main__':
    MainWindow().run()
