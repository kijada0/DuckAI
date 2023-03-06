import os
import sys
import time

import numpy as np
import cv2
import pygame

output_path = "input_data/dataset01"
source_path = "input_data/datasource01"

def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    source_file = os.listdir(source_path)
    for file_name in source_file:
        if (file_name.endswith(".jpg") or file_name.endswith(".png")):
            source_file_path = os.path.join(source_path, file_name)
            image = cv2.imread(source_file_path)

            image = resize_to_window(image, [640, 480])

            img = pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "BGR")
            screen.fill((0, 0, 0))
            screen.blit(img, (0, 0))
            pygame.display.flip()

            status = True
            skip = False
            click_count = 0
            pos_list = []
            while status:
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            print(event.pos)
                            pos_list.append(event.pos)
                            click_count += 1
                        if event.button == 3:
                            if click_count > 0:
                                click_count -= 1
                                pos_list.pop()

                    if event.type == pygame.KEYDOWN:
                        if event.unicode == "s":
                            skip = True
                            status = False
                            break

                    if event.type == pygame.QUIT:
                        exit()

                if click_count > 2:
                    break

            print(pos_list)


    pygame.quit()


def resize_to_window(image, size):
    w0, h0, _ = image.shape
    if w0 < h0:
        w1 = size[0]
        h1 = int(w1 * (w0/h0))
    else:
        h1 = size[1]
        w1 = int(h1 * (h0/w0))

    return cv2.resize(image, (w1, h1))




main()




