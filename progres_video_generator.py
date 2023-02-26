import os
import datetime
import sys

import cv2
import numpy as np
import glob

data_path = "output/output_zielu"
output_dir = "generator/video"

def main():
    print("Start progress gif generator")

    frames = []
    frame_counter = 0

    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%Y%m%d_%H%M%S_%f")
    video_path = "progress_video_{}.mp4".format(timestamp)
    print("Output file name:", video_path)
    video_path = os.path.join(output_dir, video_path)

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))

    element_list = os.listdir(data_path)
    element_path_list = []
    for element in element_list:
        element_path_list.append(os.path.join(data_path, element))
    element_path_list.sort(key=os.path.getctime)

    for element_path in element_path_list:
        if os.path.isdir(element_path):
            image_list = os.listdir(element_path)

            image_path_list = []
            for image_name in image_list:
                if image_name.endswith(".png"):
                    image_path_list.append(os.path.join(element_path, image_name))
            image_path_list.sort(key=os.path.getmtime)

            for image_path in image_path_list:
                message = "Loading ... " + image_path
                sys.stdout.write("\r" + message)

                image = cv2.imread(image_path)
                video.write(image)

                frame_counter += 1

    print("")
    video.release()

main()