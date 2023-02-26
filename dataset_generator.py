import sys

import matplotlib.pyplot as plt
import random
import math
import os
import cv2

output_path = "input_data"
output_dir = "test_image"
line_list = ["-", ":", "-.", "--", "-", "-", "-"]


def main():
    print("Start dataset generator")

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    target_path = os.path.join(output_path, output_dir)
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for i in range(128):
        vertex = random.randint(10, 100)
        size = random.randint(100, 200)
        line_width = random.randint(16, 48)
        line_type = random.choice(line_list)

        message = "Image {} \t vertex: {} \t size: {} \t line width: {} \t line type: \"{}\"".format(i+1, vertex, size, str(line_width).zfill(2), line_type)
        sys.stdout.write("\r" + message)

        x, y = make_polygon(vertex, size)

        plt.figure("Polygon", figsize=(4, 4), layout="tight")
        plt.axis('equal')
        plt.xlim(-256, 256)
        plt.ylim(-256, 256)
        plt.axis("off")
        plt.plot(x, y, linestyle=line_type, linewidth=line_width, color=get_color())

        name = "image_{}.png".format(i+1)
        image_path = os.path.join(target_path, name)
        plt.savefig(image_path, dpi=16, format="png")
        plt.close()

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(image_path, image)


def make_polygon(vertex, size):
    alfa = (2*math.pi) / vertex
    x = []
    y = []
    for i in range(vertex):
        x.append(size * math.cos(alfa * i))
        y.append(size * math.sin(alfa * i))
    x.append(size * math.cos(0))
    y.append(size * math.sin(0))

    return x, y


def get_color():
    r = random.random()
    g = random.random()
    b = random.random()
    return [r, g, b]


main()