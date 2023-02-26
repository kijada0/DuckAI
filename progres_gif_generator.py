import os
import datetime
import sys

from PIL import Image

data_path = "output/output_zielu"
output_dir = "generator/gif"

def main():
    print("Start progress gif generator")

    frames = []
    frame_counter = 0

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

                image = Image.open(image_path)
                frames.append(image)

                frame_counter += 1

    print("")

    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%Y%m%d_%H%M%S_%f")
    gif_name = "progress_gif_{}.gif".format(timestamp)
    print("Output file name:", gif_name)
    gif_path = os.path.join(output_dir, gif_name)

    print("Saving", frame_counter, "frames ...")
    gif = frames[0]
    gif.save(gif_path, format="GIF", append_images=frames, save_all=True, duration=250, loop=0)

main()