import tensorflow as tf
import keras
from keras.preprocessing import image

def limit_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        print(e)

import os
from gan.discriminator import init_discriminator
from gan.generator import init_generator
import numpy as np

class GAN:

    def __init__(self, images_arr, latent_dim, width, height, channels):

        limit_gpu()
        self.images_arr = images_arr
        self.generator = init_generator(latent_dim = latent_dim, channels = channels)
        self.discriminator = init_discriminator(width = width, height = height, channels = channels)

        self.generator.summary()
        print("*" * 10)

        self.discriminator.summary()
        self.images_arr = images_arr
        self.latent_dim = latent_dim

        self.discriminator.trianable = False
        gan_input = keras.Input(shape = (latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))

        self.gan = keras.models.Model(gan_input, gan_output)
        gan_optimizer = keras.optimizers.RMSprop(lr = 0.0004, clipvalue = 1.0, decay = 1e-8)
        self.gan.compile(optimizer = gan_optimizer, loss = "binary_crossentropy")


    def train(self, iterations, batch_size, save_dir):

        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)

        start = 0

        for step in range(iterations):
            print("Step : {}/{}".format(step, iterations))
            random_latent_vectors = np.random.normal(size = (batch_size, self.latent_dim))
            generated_images = self.generator.predict(random_latent_vectors)

            stop = start + batch_size
            real_images = self.images_arr[start:stop]
            combined_images = np.concatenate([generated_images, real_images])

            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            labels += 0.05 * np.random.random(labels.shape)

            d_loss = self.discriminator.train_on_batch(combined_images, labels)
            random_latent_vectors = np.random.normal(size = (batch_size, self.latent_dim))

            misleading_targets = np.zeros((batch_size, 1))
            a_loss = self.gan.train_on_batch(random_latent_vectors, misleading_targets)

            start += batch_size


            if start > len(self.images_arr) - batch_size:
                start = 0

            if (step + 1)% 100 == 0:
                self.gan.save_weights("gan.h5")
                print(f"discriminator loss : {d_loss}")
                print(f"adversarial loss : {a_loss}")

                img = image.array_to_img(generated_images[0] * 255., scale = False)
                img.save(os.path.join(save_dir, "generated_" + str(step) + ".png"))

                img = image.array_to_img(real_images[0] * 255., scale = False)
                img.save(os.path.join(save_dir, "real_" + str(step) + ".png"))
