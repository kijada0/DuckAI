import os
import datetime
import sys

import tensorflow as tf
from tensorflow import keras
from keras import layers

output_path = "generator"
model_path = "model"
output_to_new_folder = True


def main():
    target_dir = create_output_directory(output_path)
    if output_to_new_folder:
        time_now = datetime.datetime.now()
        timestamp = time_now.strftime("%Y-%m-%d_%H-%M-%S")
        new_dir_name = "generator_{}".format(timestamp)
        target_dir = create_output_directory(os.path.join(target_dir, new_dir_name))

    generator, generator_optimizer = init_generator(model_path)

    for i in range(50):
        generate_image(generator, target_dir)


def generate_image(generator, path):
    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%Y%m%d_%H%M%S_%f")
    image_name = "image_{}.png".format(timestamp)

    message = "Image generation ... \t " + image_name
    sys.stdout.write("\r" + message)

    random_latent_vector = tf.random.normal(shape=(1, 128))
    fake = generator(random_latent_vector)
    image = keras.preprocessing.image.array_to_img(fake[0])

    image_path = os.path.join(path, image_name)
    image.save(image_path)


def init_generator(path):
    print("\nGenerator initialization ...")
    latent_dimension = 128
    model = keras.Sequential()

    model.add(layers.Input(shape=(latent_dimension,)))
    model.add(layers.Dense(8*8*128))
    model.add(layers.Reshape((8, 8, 128)))

    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2D(3, kernel_size=4, padding="same", activation="sigmoid"))

    generator_path = os.path.join(path, "generator.h5")
    if os.path.exists(generator_path):
        print("Loading weights ...")
        model.load_weights(generator_path)

    print(model.summary())
    optimizer = keras.optimizers.Adam(1e-4)

    return model, optimizer


def create_output_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)

    return path


main()
