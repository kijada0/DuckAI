import cv2
import os
import numpy as np

width = 64
height = width

output_path = "input_data/image"
input_path = "input_data/source_image"

image_count = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)
for filename in os.listdir(input_path):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    image = cv2.imread(os.path.join(input_path, filename))

    resized_image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    h, w = resized_image.shape[:2]

    if h > w:
        cropped_image = resized_image[(h-w)//2:(h-w)//2+w, :, :]
    else:
        cropped_image = resized_image[:, (w-h)//2:(w-h)//2+h, :]

    if cropped_image.shape[2] not in [1, 3, 4]:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

    out_file_name = "img_" + str(image_count).zfill(3) + ".jpg"
    image_count += 1

    cv2.imwrite(os.path.join(output_path, out_file_name), cropped_image)