import os
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

source_path = "input_data/image"
output_path = "output"
model_path = "model"

def main():
    # check_gpu()
    epoch = 1

    target_dir = create_output_directory(output_path, "out")
    model_dir = create_output_directory(model_path, "model")
    dataset = load_dataset(source_path)

    discriminator, discriminator_optimizer = init_discriminator(model_path)
    generator, generator_optimizer = init_generator(model_path)
    loss_function = keras.losses.BinaryCrossentropy()

    while True:
        time_epoch = datetime.datetime.now()
        train_network(discriminator, discriminator_optimizer, generator, generator_optimizer, loss_function, dataset, epoch, target_dir, model_path)
        save_network(discriminator, generator, model_dir, epoch)
        save_results(generator, output_path)

        time_delta = round((datetime.datetime.now() - time_epoch).total_seconds(), 1)
        print("Epoch duration:", time_delta)
        epoch += 1


def train_network(discriminator, discriminator_optimizer, generator, generator_optimizer, loss_function, dataset, epoch, target_dir, model_dir):
    print("\nTraining ...")
    latent_dimension = 128
    batch_size = len(dataset)
    counter = 0

    for real in dataset:
        counter += 1
        time0 = datetime.datetime.now()

        random_latent_vector = tf.random.normal(shape=(batch_size, latent_dimension))
        fake = generator(random_latent_vector)

        if counter % 4 == 0:
            image = keras.preprocessing.image.array_to_img(fake[0])
            image_dir = f"{target_dir}/image_{epoch}_{counter}.png"
            image.save(image_dir)

            discriminator_path = os.path.join(model_dir, "discriminator.h5")
            discriminator.save_weights(discriminator_path)

            generator_path = os.path.join(model_dir, "generator.h5")
            generator.save_weights(generator_path)

        # Train Discriminator
        with tf.GradientTape() as discriminator_tape:
            losses_discriminator_real = loss_function(tf.ones((1, 1)), discriminator(real))
            losses_discriminator_fake = loss_function(tf.zeros(1, 1), discriminator(fake))
            losses_discriminator = (losses_discriminator_real + losses_discriminator_fake) / 2

        grads = discriminator_tape.gradient(losses_discriminator, discriminator.trainable_weights)
        discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

        # Train Generator
        with tf.GradientTape() as generator_tape:
            fake = generator(random_latent_vector)
            output = discriminator(fake)
            losses_generator = loss_function(tf.ones(batch_size, 1), output)

        grads = generator_tape.gradient(losses_generator, generator.trainable_weights)
        generator_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

        time_delta = round((datetime.datetime.now() - time0).total_seconds(), 1)
        message = "Progress: {}/{} \t Processing time: {}".format(counter, batch_size, time_delta)
        sys.stdout.write("\r" + message)

    print()


def save_network(discriminator, generator, path, epoch):
    print("Saving network ...")

    model_save_path = os.path.join(path, "model_"+str(epoch))
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    discriminator_path = os.path.join(model_save_path, "discriminator.h5")
    discriminator.save_weights(discriminator_path)

    generator_path = os.path.join(model_save_path, "generator.h5")
    generator.save_weights(generator_path)


def save_results(generator, path):
    print("Image generation ...\t", end="")
    random_latent_vector = tf.random.normal(shape=(1, 128))
    fake = generator(random_latent_vector)
    image = keras.preprocessing.image.array_to_img(fake[0])

    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%Y%m%d_%H%M%S_%f")
    image_name = "image_{}.png".format(timestamp)
    print(image_name)
    image_path = os.path.join(path, image_name)

    image.save(image_path)


def init_discriminator(path):
    print("\nDiscriminator initialization ...")
    model = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 3)),

            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),

            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    discriminator_path = os.path.join(path, "discriminator.h5")
    if os.path.exists(discriminator_path):
        print("Loading weights ...")
        model.load_weights(discriminator_path)

    print(model.summary())
    optimizer = keras.optimizers.Adam(1e-4)
    return model, optimizer


def init_generator(path):
    print("\nGenerator initialization ...")
    latent_dimension = 128
    model = keras.Sequential(
        [
            layers.Input(shape=(latent_dimension,)),
            layers.Dense(8 * 8 * 128),
            layers.Reshape((8, 8, 128)),

            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),

            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ]
    )

    generator_path = os.path.join(path, "generator.h5")
    if os.path.exists(generator_path):
        print("Loading weights ...")
        model.load_weights(generator_path)

    print(model.summary())
    optimizer = keras.optimizers.Adam(1e-4)

    return model, optimizer


def check_gpu():
    print("\nChecking available GPU ...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("Using GPU:", tf.test.gpu_device_name())
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("GPU not found")


def create_output_directory(path, dir_name):
    if not os.path.exists(path):
        os.mkdir(path)

    time_now = datetime.datetime.now()
    new_dir_name = str(dir_name) + time_now.strftime("_%Y-%m-%d_%H-%M-%S")

    new_dir_full_path = os.path.join(path, new_dir_name)
    os.mkdir(new_dir_full_path)

    print("New output directory:", new_dir_name)
    return new_dir_full_path


def load_dataset(path):
    print("\nLoading dataset ...")
    dataset = []
    file_list = os.listdir(path)
    map_func = lambda x: x / 255.0
    for file in file_list:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(path, file)
            image = plt.imread(image_path)
            image = np.array(image)
            image = map_func(image)
            image = np.expand_dims(image, axis=0)
            dataset.append(image)

    print("Dataset size:", len(dataset))
    dataset = np.array(dataset)
    return dataset


main()
