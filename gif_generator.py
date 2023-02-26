import os
import datetime
import sys

import tensorflow as tf
from tensorflow import keras
from keras import layers

output_path = "generator"
gif_dir = "gif"
model_path = "model"

def main():
    target_dir = create_output_directory(output_path)
    target_dir = create_output_directory(os.path.join(target_dir, gif_dir))
    generator, generator_optimizer = init_generator(model_path)

    frames = []
    for i in range(1000):
        frames.append(generate_image(generator, i))

    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%Y%m%d_%H%M%S_%f")
    gif_name = "gif_{}.gif".format(timestamp)
    print(gif_name)
    gif_path = os.path.join(target_dir, gif_name)

    gif = frames[0]
    gif.save(gif_path, format="GIF", append_images=frames, save_all=True, duration=100, loop=0)


def generate_image(generator, i):
    message = "Image generation no. " + str(i)
    sys.stdout.write("\r" + message)

    random_latent_vector = tf.random.normal(shape=(1, 128))
    fake = generator(random_latent_vector)
    image = keras.preprocessing.image.array_to_img(fake[0])
    return image


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
