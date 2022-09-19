import numpy as np
import math
import cv2

from helpfuncs import *


def project(inference, camera_info, cutoff=5):
    height, width = inference.shape[:2]
    center = [(width - 1) / 2, (height - 1) / 2]
    c_dist = distance([0, 0], center)
    r_fov = camera_info['fov'] / 2

    projections = []
    for y in range(height):
        for x in range(width):
            y_diff = center[1] - y
            x_diff =  x - center[0]
            dist = distance([0, 0], [x_diff, y_diff])
            if dist != 0:
                scale = dist / c_dist
                new_scale = camera_info['distortion_fn'](scale) / scale
                v_angle = (y_diff * new_scale / c_dist) * r_fov + camera_info['cam_angle']
                h_angle = (x_diff * new_scale / c_dist) * r_fov
            else:
                v_angle = camera_info['cam_angle']
                h_angle = 0
            if v_angle >= 0:
                continue
            p_y = math.tan(math.radians(90 + v_angle)) * camera_info['cam_height']
            hyp = hypotenuse(p_y, camera_info['cam_height'])
            p_x = math.tan(math.radians(h_angle)) * hyp
            if distance([0, 0], [p_x, p_y]) > cutoff:
                continue
            projections.append({'class': int(inference[y, x]), 'pos': (p_x, p_y)})
    return projections


def project_global(inference, camera_info, pos, cutoff=5):
    projections = project(inference, camera_info, cutoff)
    angle = math.radians(pos.heading.get())
    for projection in projections:
        (x, y) = projection['pos']
        xx = x * math.cos(angle) + y * math.sin(angle) + pos.x
        yy = -x * math.sin(angle) + y * math.cos(angle) + pos.y
        projection['pos'] = (xx, yy)
        #print(xx, yy, pos.heading.get())
    return projections


def draw_inference(image, inference, colors):
    height, width = image.shape[:2]
    h_max, w_max = inference.shape
    h_scale = height / h_max
    w_scale = width / w_max

    for h in range(h_max):
        for w in range(w_max):
            c = int(inference[h, w])
            if c > 0:
                color = colors[c]
                image = cv2.circle(image,
                    (int((w + 0.5) * w_scale), int((h + 0.5) * h_scale)),
                    color=color, radius=1, thickness=2)
    return image

def draw_projection(image, projections, colors, center=(0.5, 0.75), scale=5):
    height, width = image.shape[:2]
    center = (center[0] * width, center[1] * height)
    scale = height / scale

    for p in projections:
        color = colors[p['class']]
        x, y = p['pos']
        x = x * scale + center[0]
        y = -y * scale + center[1]
        image = cv2.circle(image, (int(x), int(y)), color=color, radius=1, thickness=2)
    return image

def draw_distortion(image, camera_info, density=35):
    height, width = image.shape[:2]
    center = [(density - 1) / 2, (density - 1) / 2]
    c_dist = distance([0, 0], center)

    for y in range(density):
        for x in range(density):
            y_diff = center[1] - y
            x_diff = x - center[0]
            dist = distance([0, 0], [x_diff, y_diff])
            if dist != 0:
                scale = dist / c_dist
                new_scale = camera_info['distortion_fn'](scale) / scale
                v_angle = y_diff * new_scale / c_dist
                h_angle = x_diff * new_scale / c_dist
            else:
                v_angle = h_angle = 0
            o_dist = distance([0, 0], [width / 2, height / 2])
            py = int(v_angle * o_dist + height / 2)
            px = int(h_angle * o_dist + width / 2)
            image = cv2.circle(image, (px, py),
                color=(255, 255, 255), radius=1, thickness=2)
    return image

def map_to_image(map_data, colors):
    h, w = map_data.shape
    image = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            image[y, x] = colors[map_data[y, x]]
    return image / 255


class Map:
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3

    def __init__(self, num_classes, size=10, scale=0.2):
        self._num_classes = num_classes
        self._scale = scale

        m_size = math.ceil(size / self._scale)
        self._map = np.zeros((m_size, m_size, num_classes), dtype=np.float32)
        self._origin = [int(m_size / 2), int(m_size / 2)]

    def extend_map(self, direction, units):
        height, width = self._map.shape[:2]

        if direction == self.LEFT:
            self._map = np.column_stack([np.zeros((height, units, self._num_classes), dtype=self._map.dtype), self._map])
            self._origin[0] += units
        elif direction == self.RIGHT:
            self._map = np.column_stack([self._map, np.zeros((height, units, self._num_classes), dtype=self._map.dtype)])
        elif direction == self.TOP:
            self._map = np.vstack([np.zeros((units, width, self._num_classes), dtype=self._map.dtype), self._map])
            self._origin[1] += units
        elif direction == self.BOTTOM:
            self._map = np.vstack([self._map, np.zeros((units, width, self._num_classes), dtype=self._map.dtype)])

    def get_data(self):
        return np.argmax(self._map, axis=2)

    def get_origin(self):
        return self._origin

    def map_projections(self, projections, default_extend=1):
        for projection in projections:
            height, width = self._map.shape[:2]
            (x, y) = projection['pos']
            cap_dist = distance([0, 0], projection['pos'])

            m_x = round(x / self._scale) + self._origin[0]
            m_y = -round(y / self._scale) + self._origin[1]
            if m_x < 0:
                self.extend_map(self.LEFT, max(abs(m_x), default_extend))
            elif m_x >= width:
                self.extend_map(self.RIGHT, max(m_x + 1 - width, default_extend))
            if m_y < 0:
                self.extend_map(self.TOP, max(abs(m_y), default_extend))
            elif m_y >= height:
                self.extend_map(self.BOTTOM, max(m_y + 1 - height, default_extend))
            m_x = round(x / self._scale) + self._origin[0]
            m_y = -round(y / self._scale) + self._origin[1]

            self._map[m_y, m_x, projection['class']] += math.exp(-cap_dist)


if __name__ == '__main__':
    import time
    map = Map(4, size=5, scale=1)
    projections = [{'class': 1, 'pos': (i, i)} for i in range(10)]
    print(map.get_map())

    map.map_projections(projections)

    projections = [{'class': 3, 'pos': (i, i)} for i in range(9*17)]
    start = time.time()
    map.map_projections(projections)
    map.map_projections(projections)
    print(time.time() - start)
    print(map.get_map())
