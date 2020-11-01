import keras
from keras import layers

def init_discriminator(width, height, channels):

    discriminator_input = layers.Input(shape = (height, width, channels))

    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)


    x = layers.Conv2D(128, 4, strides = 2)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides = 2)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides = 2)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1, activation = "sigmoid")(x)

    discriminator = keras.models.Model(discriminator_input, x)

    discriminator_optimizer = keras.optimizers.RMSprop(
            lr = 0.0008,
            clipvalue = 1.0,
            decay = 1e-8
            )

    discriminator.compile(optimizer = discriminator_optimizer,
            loss = "binary_crossentropy")

    return discriminator

