import os
import sys

import numpy as np
import cv2

source_path = "generator/best"
output_dir = "resize"

target_size = 1024

target_path = os.path.join(source_path, output_dir)
image_list = os.listdir(source_path)

for image_name in image_list:
    if image_name.endswith(".png") or image_name.endswith(".jpg"):
        message = "Resizing ... " + image_name
        sys.stdout.write("\r" + message)

        image_path = os.path.join(source_path, image_name)
        source_image = cv2.imread(image_path)

        x0, y0, _ = source_image.shape
        x_scale = int(target_size/x0)
        y_scale = int(target_size/y0)
        x = x0 * x_scale
        y = x0 * y_scale

        output_image = np.ones([x, y, 3])
        for x_px in range(x):
            for y_px in range(y):
                output_image[x_px, y_px] = source_image[int(x_px / x_scale), int(y_px / y_scale)]

        output_path = os.path.join(target_path, image_name)
        cv2.imwrite(output_path, output_image)


