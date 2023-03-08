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
    screen = pygame.display.set_mode((720, 540))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    source_file = os.listdir(source_path)
    source_file_count = len(source_file)
    file_index = 0
    file_counter = 1

    while file_index < source_file_count:
        file_name = source_file[file_index]
        print("\nProcessing:", file_name)
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            source_file_path = os.path.join(source_path, file_name)
            image0 = cv2.imread(source_file_path)
            image = resize_to_window(image0, [640, 480])

            img = pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "BGR")
            screen.fill((0, 0, 0))
            screen.blit(img, (0, 0))
            pygame.display.flip()

            status = True
            skip = False
            rect = False
            view = True
            click_count = 0
            pos_list = []
            while status:
                # mouse = pygame.mouse.get_pos()
                # mes = "mouse\t x:{} \t y:{} \t".format(mouse[0], mouse[1])
                # sys.stdout.write("\r" + mes)

                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            #print(event.pos)
                            pos_list.append(event.pos)
                            click_count += 1
                            pygame.draw.circle(screen, (255, 0, 0), event.pos, 5)
                            pygame.display.flip()

                        if event.button == 3:
                            if click_count > 0:
                                click_count -= 1
                                pos_list.pop()

                    if event.type == pygame.KEYDOWN:
                        if event.unicode == "s":
                            file_index += 1
                            skip = True
                            status = False
                            break
                        if event.unicode == "u":
                            skip = True
                            status = False
                            break
                        if event.unicode == "w":
                            if view:
                                view = False
                                print("View Off")
                            else:
                                view = True
                                print("View On")

                        if event.unicode == " ":
                            if click_count > 1:
                                file_index += 1
                                status = False
                                break

                    if event.type == pygame.QUIT:
                        exit()

                if click_count > 1 and not rect:
                    x0, y0, a = calculate_squarer(pos_list[0], pos_list[1])
                    print("Drawing rectangle: ", x0, y0, x0+a, y0+a)
                    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(x0, y0, a, a), 2)
                    pygame.display.flip()
                    rect = True

            print(pos_list)
            if not skip and click_count > 1:
                x0, y0, a = calculate_squarer(pos_list[0], pos_list[1])
                print("Saving file...", image.shape, "\t", x0, y0, x0+a, y0+a, a)
                area = image[x0:x0+a, y0:y0+a]
                print("Area shape: ", area.shape)
                if view:
                    cv2.imshow("Image", area)
                    if cv2.waitKey(0):
                        cv2.destroyWindow("Image")
                name = "image_" + str(file_counter) + ".png"
                file_counter += 1
                save_image(area, 64, name)

    pygame.quit()


def calculate_squarer(a, b):
    x0, y0 = a
    x1, y1 = b
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    w = x1 - x0
    h = y1 - y0
    if w > h:
        size = w
    else:
        size = h

    return x0, y0, size


def save_image(image, size, name):
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

    resized_image = cv2.resize(sq_image, (size, size), interpolation=cv2.INTER_AREA)
    if not resized_image.shape[2] == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

    print("Saving: ", name)
    cv2.imwrite(os.path.join(output_path, name), resized_image)


def resize_to_window(image, size):
    w0, h0, _ = image.shape
    if w0 < h0:
        w1 = size[0]
        h1 = int(w1 * (w0/h0))
    else:
        h1 = size[1]
        w1 = int(h1 * (h0/w0))

    if w1 > size[0]:
        w1 = size[0]
        h1 = int(w1 * (w0/h0))

    if h1 > size[1]:
        h1 = size[1]
        w1 = int(h1 * (h0/w0))

    #print(size[0], w0, w1, "\t\t", size[1], h0, h1)
    return cv2.resize(image, (w1, h1))




main()




