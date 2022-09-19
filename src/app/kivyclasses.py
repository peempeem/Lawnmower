from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.animation import Animation

import copy
import cv2
import numpy as np


class ImagePane(FloatLayout):
    def __init__(self, size_hint=(1, 1), pos_hint={}, color_component=None):
        super(ImagePane, self).__init__()
        self.size_hint = size_hint
        self.pos_hint = pos_hint

        self._image = CV2Image(pos_hint={'center_x': 0.5, 'center_y':0.5})
        self.add_widget(self._image)

    def set_image(self, img):
        self._image.set_image(img)


class CV2Image(Image):
    def __init__(self, size_hint=(1, 1), pos_hint={}):
        super(CV2Image, self).__init__()
        self.size_hint = size_hint
        self.pos_hint = pos_hint

    def set_image(self, img):
        new_size = [int(self.size[0]), int(img.shape[0] * (self.size[0] / img.shape[1]))]
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
        img = cv2.flip(img, 0)
        img = np.uint8(img * 255)
        buf = img.tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture


class SettingsPane(FloatLayout):
    def __init__(self, size_hint=(1, 1), pos_hint={}, color_component=None, bar_size_hint=(1, 0.025), bar_color_component=None):
        super(SettingsPane, self).__init__()
        self.size_hint = size_hint
        self.pos_hint = pos_hint

        self._open = False
        self._hovering = False

        self.opacity = 0
        self._anim = Animation()
        self._anim_time = 0.2
        self._up_pos = copy.copy(self.pos)
        self._first = True

        if color_component is not None:
            with self.canvas:
                self._rect = Rectangle(pos=self.pos, size=self.size, texture=color_component.texture)
        else:
            self._rect = Rectangle(pos=self.pos, size=self.size)

        self._bar = Divider(size_hint=bar_size_hint, pos_hint={'x':0, 'top':1}, color_component=bar_color_component)
        self.add_widget(self._bar)

        self.bind(size=self.update, pos=self.update)

    '''def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos) and self._open:
            self.close()
        return super(SettingsPane, self).on_touch_down(touch)'''

    def is_open(self):
        return self._open

    def open(self):
        if not self._open:
            if self._first:
                self.pos=(self.pos[0], self.pos[1] - self.size[1] / 2)
                self._first = False
            self._open = True
            self._anim.stop_all(self)
            self._anim.stop_all(self._bar)
            self._anim = Animation(opacity=1, pos=self._up_pos, duration=self._anim_time)
            self._anim.start(self)
            self._anim.start(self._bar)

    def close(self):
        if self._open:
            self._open = False
            self._last_pos = copy.copy(self.pos)
            self._anim.stop_all(self)
            self._anim.stop_all(self._bar)
            self._anim = Animation(opacity=0, pos=(self.pos[0], self.pos[1] - self.size[1] / 2), duration=self._anim_time)
            self._anim.start(self)
            self._anim.start(self._bar)

    def update(self, *args):
        self._rect.size = self.size
        self._rect.pos = self.pos


class WrapLabel(Label):
    def __init__(self, text="", width=100, height=100, size_hint=(1, 1), pos_hint={}, bold=False, font_size='17sp', halign='left', valign='top'):
        super(WrapLabel, self).__init__()
        self.text = text
        self.width = width
        self.height = height
        self.size_hint = size_hint
        self.pos_hint = pos_hint
        self.bold = bold
        self.font_size = font_size
        self.halign = halign
        self.valign = valign

        self.bind(size=self.wrapping_fn)

    def wrapping_fn(self, *args):
        self.text_size = self.size


class ConsolePane(FloatLayout):
    def __init__(self, size_hint=(1, 1), pos_hint={}, color_component=None, bar_size_hint=(1, 0.025), bar_color_component=None):
        super(ConsolePane, self).__init__()
        self.size_hint = size_hint
        self.pos_hint = pos_hint

        self._open = False
        self._hovering = False

        self.opacity = 0
        self._anim = Animation()
        self._anim_time = 0.2
        self._up_pos = copy.copy(self.pos)
        self._first = True

        if color_component is not None:
            with self.canvas:
                self._rect = Rectangle(pos=self.pos, size=self.size, texture=color_component.texture)
        else:
            self._rect = Rectangle(pos=self.pos, size=self.size)

        self._bar = Divider(size_hint=bar_size_hint, pos_hint={'x':0, 'top':1}, color_component=bar_color_component)
        self.add_widget(self._bar)

        self._scroll_window = ScrollView(size_hint=(0.9, 0.85), pos_hint={'x':0.05, 'y':0.1})
        self._scroll_window.scroll_y = 0
        self.add_widget(self._scroll_window)
        self._txt_window = WrapLabel(text="", size_hint=(1, None), height=1000, valign='bottom')
        self._scroll_window.add_widget(self._txt_window)
        self._input_window = WrapLabel(text=">>> ", size_hint=(0.9, 0.1), pos_hint={'x':0.05, 'y':0})
        self.add_widget(self._input_window)

        self.bind(size=self.update, pos=self.update)

    '''def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos) and self._open:
            self.close()
        return super(ConsolePane, self).on_touch_down(touch)'''

    def set_window_text(self, text):
        self._txt_window.text = text

    def set_input_text(self, text):
        text = f">>> {text}"
        if text != self._input_window.text:
            self._input_window.text = text

    def is_open(self):
        return self._open

    def open(self):
        if not self._open:
            if self._first:
                self.pos=(self.pos[0], self.pos[1] - self.size[1] / 2)
                self._first = False
            self._open = True
            self._anim.stop_all(self)
            self._anim.stop_all(self._bar)
            self._anim = Animation(opacity=1, pos=self._up_pos, duration=self._anim_time)
            self._anim.start(self)
            self._anim.start(self._bar)

    def close(self):
        if self._open:
            self._open = False
            self._last_pos = copy.copy(self.pos)
            self._anim.stop_all(self)
            self._anim.stop_all(self._bar)
            self._anim = Animation(opacity=0, pos=(self.pos[0], self.pos[1] - self.size[1] / 2), duration=self._anim_time)
            self._anim.start(self)
            self._anim.start(self._bar)

    def update(self, *args):
        self._rect.size = self.size
        self._rect.pos = self.pos


class WidgetShelf(StackLayout):
    def __init__(self, widgets, size_hint=(1, 1), color_component=None):
        super(WidgetShelf, self).__init__()
        self.widgets = widgets

        self.orientation = 'lr-tb'
        self.size_hint = size_hint

        if color_component is not None:
            with self.canvas:
                self._rect = Rectangle(pos=self.pos, size=self.size, texture=color_component.texture)
        else:
            self._rect = Rectangle(pos=self.pos, size=self.size)

        for widget in self.widgets:
            if widget.size_hint[1] == 1:
                widget.size_hint = (None, None)
            widget.width = self.width
            self.add_widget(widget)

        self.bind(size=self.update, pos=self.update)

    def update(self, *args):
        self._rect.size = self.size
        self._rect.pos = self.pos
        for widget in self.widgets:
            widget.width = self.width


class IconButton(ButtonBehavior, Widget):
    def __init__(self, icon_normal, icon_pressed):
        super(IconButton, self).__init__()
        self._icon_normal = Image(source=icon_normal)
        self._icon_pressed = Image(source=icon_pressed)
        self._icon_pressed.opacity = 0

        self.always_release = True
        self._callback = None
        self.pressed = False

        self.blink_time = 0.3
        self.return_time = 0.2
        self.react_time = 0.1
        self._hovering = False

        self._hover_anim = Animation(opacity=0.6, duration=self.blink_time) \
            + Animation(opacity=1, duration=self.blink_time) \
            + Animation(duration=self.return_time)
        self._hover_anim.repeat = True
        self._return_anim = Animation(opacity=1, duration=self.return_time)

        self.add_widget(self._icon_normal)
        self.add_widget(self._icon_pressed)
        self.bind(size=self.update, pos=self.update)
        Window.bind(mouse_pos=self.on_mouse_pos)

    def on_mouse_pos(self, *args):
        if not self.get_root_window():
            return
        pos = args[1]
        if self.collide_point(*pos):
            Window.set_system_cursor('hand')
            if not self._hovering:
                self._hovering = True
                if not self.pressed:
                    self._hover_anim.start(self._icon_normal)
        else:
            if self._hovering:
                self._hovering = False
                Window.set_system_cursor('arrow')
                if not self.pressed:
                    self._hover_anim.stop_all(self._icon_normal)
                    self._return_anim.start(self._icon_normal)

    def set_callback(self, func):
        self._callback = func

    def on_press(self):
        self.pressed = True
        if self._hovering:
            self._hover_anim.stop_all(self._icon_normal)
        fadein = Animation(opacity=1, duration=self.react_time)
        fadeout = Animation(opacity=0, duration=self.react_time)
        fadein.start(self._icon_pressed)
        fadeout.start(self._icon_normal)
        if self._callback is not None:
            self._callback()

    def on_release(self):
        self.pressed = False
        fadein = Animation(opacity=1, duration=self.react_time)
        fadeout = Animation(opacity=0, duration=self.react_time)
        if self._hovering:
            fadein += self._hover_anim
        fadein.start(self._icon_normal)
        fadeout.start(self._icon_pressed)

    def update(self, *args):
        self._icon_normal.size = self.size
        self._icon_normal.pos = self.pos
        self._icon_pressed.size = self.size
        self._icon_pressed.pos = self.pos


class Header(Widget):
    def __init__(self, size_hint=(1, 1), color_component=None, title_image=None, buttons=[]):
        super(Header, self).__init__()
        self.size_hint = size_hint
        self.buttons = buttons

        if color_component is not None:
            with self.canvas:
                self._rect = Rectangle(pos=self.pos, size=self.size, texture=color_component.texture)
        else:
            self._rect = Rectangle(pos=self.pos, size=self.size)

        if title_image is not None:
            self._title_image = Image(source=title_image)
            self.add_widget(self._title_image)

        for button in buttons:
            self.add_widget(button)

        self.bind(size=self.update, pos=self.update)

    def update(self, *args):
        self._rect.size = self.size
        self._rect.pos = self.pos
        if self._title_image is not None:
            width = self._title_image.image_ratio * self.size[1] / 1.5
            self._title_image.size = [min(width, self.size[0] / 2), self.size[1] / 1.5]
            self._title_image.pos = [self.pos[0] + 20, self.pos[1] + self.size[1] / 6]
        for i in range(len(self.buttons)):
            self.buttons[i].size = [self.size[1], self.size[1]]
            self.buttons[i].pos = [self.size[0] - (i + 1) * self.size[1], self.pos[1]]


class Divider(Widget):
    def __init__(self, width=100, height=100, size_hint=(1, 1), pos_hint={}, color_component=None, rotate_speed=None):
        super(Divider, self).__init__()
        self.width = width
        self.height = height
        self.size_hint = size_hint
        self.pos_hint = pos_hint

        self.color_component = color_component
        self._rotate_speed = rotate_speed

        if color_component is not None:
            with self.canvas:
                self._rect = Rectangle(pos=self.pos, size=self.size, texture=color_component.texture)
            if rotate_speed is not None:
                Clock.schedule_interval(self._color_callback, 1/10)
        else:
            self._rect = Rectangle(pos=self.pos, size=self.size)

        self.bind(size=self.update, pos=self.update)

    def _color_callback(self, dt):
        self.color_component.rotate(self._rotate_speed)
        self._rect.texture = self.color_component.texture

    def update(self, *args):
        self._rect.size = self.size
        self._rect.pos = self.pos
