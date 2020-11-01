import keras
from keras import layers


def init_generator(latent_dim, channels):

    generator_input = keras.Input(shape = (latent_dim,))

    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)

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
