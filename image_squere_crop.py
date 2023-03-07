import cv2
import os
import numpy as np

width = 64
height = width

output_path = "input_data/image_racoon2"
input_path = "input_data/source_1"

image_count = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in os.listdir(input_path):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    image = cv2.imread(os.path.join(input_path, filename))

    h0 = image.shape[0]
    w0 = image.shape[1]
    h02 = int(h0/2)
    w02 = int(w0/2)

    if h0 > w0:
        a0, a1 = h02-w02, h02+w02
        b0, b1 = 0, w0-1
    else:
        a0, a1 = 0, h0-1
        b0, b1 = w02-h02, w02+h02

    sq_image = image[a0:a1, b0:b1]

    resized_image = cv2.resize(sq_image, (width, height), interpolation=cv2.INTER_AREA)

    if not resized_image.shape[2] == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

    out_file_name = "img_" + str(image_count).zfill(3) + ".jpg"
    image_count += 1

    cv2.imwrite(os.path.join(output_path, out_file_name), resized_image)