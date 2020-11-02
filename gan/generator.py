import keras
from keras import layers


def init_generator(latent_dim, channels):

    generator_input = keras.Input(shape = (latent_dim,))

    half_latent_dim = int(latent_dim / 2)
    x = layers.Dense(128 * half_latent_dim * half_latent_dim)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((half_latent_dim, half_latent_dim, 128))(x)

    x = layers.Conv2D(256, 5, padding = "same")(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(256, 4, strides = 2,  padding = "same")(x)
    x = layers.LeakyReLU()(x)


    x = layers.Conv2D(256, 5, padding = "same")(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 5, padding = "same")(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(channels, 7, activation = "tanh", padding = "same")(x)

    generator = keras.models.Model(generator_input, x)

    return generator

