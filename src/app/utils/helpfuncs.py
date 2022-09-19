import math


def distance(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def hypotenuse(b1, b2):
    return math.sqrt(pow(b1, 2) + pow(b2, 2))


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
