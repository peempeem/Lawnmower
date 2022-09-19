import numpy as np
import math
import cv2
import time


class UNETFunctions:
    def __init__(self, class_path, color_path):
        self._classes = load_classes(class_path)
        self._colors = load_colors(color_path)

    def draw_dots(self, image, inference):
        height, width, _ = image.shape
        h_max, w_max = inference.shape
        h_scale = height / h_max
        w_scale = width / w_max

        for h in range(h_max):
            for w in range(w_max):
                c = int(inference[h, w])
                if c > 0:
                    color = self._colors[c]
                    image = cv2.circle(image,
                        (int((w + 0.5) * w_scale), int((h + 0.5) * h_scale)),
                        color=color, radius=1, thickness=2)
        return image

    def find_border(self, inference, c_type):
        idx = self._classes.index(c_type)
        inf = inference.copy()
        inf = inf.astype(np.uint8)
        inf[inf != idx] = 0
        inf[inf != 0] = 255
        contours, hierarchy = cv2.findContours(inf, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
        largest = 0
        idx = -1
        contour = None
        for i in range(len(contours)):
            if contours[i].shape[0] > largest:
                largest = contours[i].shape[0]
                idx = i
        if idx != -1:
            contour = contours[idx]

            '''M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)'''
        return contour

    '''def remove_horizon(self, inference, angle_dep):
        angle_dep += self._cinfo['angle_dep']
        height, width = inference.shape[0:2]
        vscale = height / self._cinfo['vfov']
        y = int(((self._cinfo['vfov'] / 2) - angle_dep) * vscale + 1)
        if y > height:
            y = height
        if y >= 0:
            inference[0:y] = 0
        return inference

    def draw_contour(self, image, contour, c_type, origin=(.5, .8), h=3):
        if contour.shape[0] == 0:
            return image
        height, width = image.shape[0:2]
        scale = height / h
        orig = (origin[0] * (width - 1), origin[1] * (height - 1))
        c = contour.copy()
        for i in range(c.shape[0]):
            c[i][0][0] = int(c[i][0][0] * scale + orig[0])
            c[i][0][1] = int(-c[i][0][1] * scale + orig[1])
        out = cv2.drawContours(image, [c.astype(np.int32)], -1,
            self._colors[self._classes.index(c_type)], 3)
        return out

    def project_contour(self, contour, img_dims, angle_dep):
        angle_dep += self._cinfo['angle_dep']
        height, width = img_dims[0:2]
        vscale = height / self._cinfo['vfov']
        proj = []
        for i in range(len(contour)):
            proj.append([self._px2coord(contour[i, 0], width, height, angle_dep)])
        proj = np.asarray(proj, dtype=np.float32)
        return proj

    def _px2coord(self, px, width, height, angle_dep):
        vscale = height / self._cinfo['vfov']
        hscale = width / self._cinfo['hfov']

        diff = (((height - 1) / 2) - px[1]) / vscale
        deg = angle_dep - diff
        print(deg)
        ycoord = self._cinfo['height'] / math.tan(math.radians(deg))

        deg = (px[0] - ((width - 1) / 2)) / hscale
        hyp = math.sqrt(pow(ycoord, 2) + pow(self._cinfo['height'], 2))
        xcoord = math.tan(math.radians(deg)) * hyp
        return [xcoord, ycoord]'''


def load_classes(path):
    classes = []
    file = open(path, 'r')
    for line in file:
        if '\n' in line:
            line = line.replace('\n', '')
        classes.append(line)
    return classes


def load_colors(path, bgr=True):
    colors = []
    file = open(path, 'r')
    for line in file:
        if '\n' in line:
            line = line.replace('\n', '')
        if ' ' in line:
            line = line.replace(' ', '')
        numbers = line.split(',')
        for i in range(0, len(numbers)):
            numbers[i] = int(numbers[i])
        if bgr:
            numbers[0], numbers[2] = numbers[2], numbers[0]
        colors.append(tuple(numbers))
    return colors
