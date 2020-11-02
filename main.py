from gan.generator import init_generator
from gan.discriminator import init_discriminator
from gan.gan import GAN
import keras

def main():

    height = 32
    width = 32
    latent_dim = 32
    channels = 3

    #generator = init_generator(latent_dim = 32, channels = 3)
    #generator.summary()

    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    x_train = x_train[y_train.flatten() == 6]
    print(x_train.shape)
    x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype("float32") / 255.

    gan = GAN(x_train, latent_dim = 32, width = 32, height = 32, channels = 3)
    gan.train(iterations = 1000, batch_size = 20, save_dir = "results")

    #discriminator = init_discriminator(width = 32, height = 32, channels = 3)
    #discriminator.summary()


if __name__ == "__main__":
    main()
