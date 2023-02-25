import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset = keras.preprocessing.image_dataset_from_directory(
    directory="image", label_mode=None, image_size=(64, 64), batch_size=132, shuffle=True
)
print(dataset)
dataset.map(lambda x: x/255.0)

exit()

print("Dataset size:", len(list(dataset)))

num_elements = 0
for element in dataset:
    num_elements += 1

print(num_elements)

discriminator = keras.Sequential(
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

print(discriminator.summary())

latent_dimension = 128
generator = keras.Sequential(
    [
        layers.Input(shape=(latent_dimension,)),
        layers.Dense(8*8*128),
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
generator.summary()

optimizer_generator = keras.optimizers.Adam(1e-4)
optimizer_discriminator = keras.optimizers.Adam(1e-4)
losses_function = keras.losses.BinaryCrossentropy()

# discriminator = keras.load_model("discriminator_trained.h5")
# generator = keras.load_model("generator_trained.h5")

for epoch in range(4096):
    for index, real in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        random_latent_vector = tf.random.normal(shape=(batch_size, latent_dimension))
        fake = generator(random_latent_vector)

        print(index)
        if index % 1 == 0:
            image = keras.preprocessing.image.array_to_img(fake[0])
            image.save(f"output/img_{epoch}_{index}.png")


        # Train Discriminator
        with tf.GradientTape() as discriminator_tape:
            losses_discriminator_real = losses_function(tf.ones((batch_size, 1)), discriminator(real))
            losses_discriminator_fake = losses_function(tf.zeros(batch_size, 1), discriminator(fake))
            losses_discriminator = (losses_discriminator_real + losses_discriminator_fake)/2

        grads = discriminator_tape.gradient(losses_discriminator, discriminator.trainable_weights)
        optimizer_discriminator.apply_gradients(zip(grads, discriminator.trainable_weights))

        # Train Generator
        with tf.GradientTape() as generator_tape:
            fake = generator(random_latent_vector)
            output = discriminator(fake)
            losses_generator = losses_function(tf.ones(batch_size, 1), output)

        grads = generator_tape.gradient(losses_generator, generator.trainable_weights)
        optimizer_generator.apply_gradients(zip(grads, generator.trainable_weights))
