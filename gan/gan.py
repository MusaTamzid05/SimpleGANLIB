import os
from gan.discriminator import init_discriminator
from gan.generator import init_generator

class GAN:

    def __init__(self, images_arr, latent_dim, width, height, channels):
        self.images_arr = images_arr
        self.generator = init_generator(latent_dim = latent_dim, channels = channels)
        self.discriminator = init_discriminator(width = width, height = height, channels = channels)

        self.generator.summary()
        print("*" * 10)

        self.discriminator.summary()
        self.images_arr = images_arr


    def train(self, iterations, batch_size, save_dir):

        if os.path.isdir(save_dir):
            ok.mkdir(save_dir)


