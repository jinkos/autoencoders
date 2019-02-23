import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import matplotlib.pyplot as plt

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE(object):

    def __init__(self):

        # MNIST dataset
        (x_train, y_train), (x_test, self.y_test) = mnist.load_data()

        image_size = x_train.shape[1]
        self.original_dim = image_size * image_size
        x_train = np.reshape(x_train, [-1, self.original_dim])
        x_test = np.reshape(x_test, [-1, self.original_dim])
        self.x_train = x_train.astype('float32') / 255
        self.x_test = x_test.astype('float32') / 255

        # network parameters
        self.input_shape = (image_size, image_size)
        self.intermediate_dim = 512
        self.latent_dim = 10
        self.inputs = Input(shape=self.input_shape, name='encoder_input')

    def build_models(self):

        def build_encoder(keep_prob):

            activation = lrelu


            # VAE model = encoder + decoder
            # build encoder model
            X = tf.reshape(self.inputs, shape=[-1, 28, 28, 1])
            x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.contrib.layers.flatten(x)
            self.z_mean = Dense(self.latent_dim, name='z_mean')(x)
            self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

            # instantiate encoder model
            encoder = Model(self.inputs, [self.z_mean, self.z_log_var, z], name='encoder')
            return encoder
            # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        self.encoder = build_encoder()

        def build_decoder(sampled_z, keep_prob):

            activation = lrelu

            latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
            x = tf.layers.dense(latent_inputs, units=inputs_decoder, activation=lrelu)
            x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
            x = tf.reshape(x, reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same',
                                           activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=tf.nn.relu)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
            img = tf.reshape(x, shape=[-1, 28, 28])
            return img

        def build_decoder():

            # build decoder model
            latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
            x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
            outputs = Dense(self.original_dim, activation='sigmoid')(x)

            # instantiate decoder model
            decoder = Model(latent_inputs, outputs, name='decoder')
            return decoder

        self.decoder = build_decoder()

        self.encoder.summary()
        self.decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = Model(self.inputs, self.outputs, name='vae_mlp')

    def build_losses(self):

        # VAE loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = binary_crossentropy(self.inputs,
                                                  self.outputs)

        reconstruction_loss *= self.original_dim
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam', loss = None)
        self.vae.summary()

def plot_results(vae, batch_size=128):

    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(vae.x_test,
                                   batch_size=batch_size)

    print("z_mean.shape", z_mean.shape)
    print(z_mean[:10])
    print(np.mean(z_mean), np.std(z_mean))
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=vae.y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

def cool_stats(vae, batch_size=128):

    print("vae.y_test", vae.y_test.dtype, vae.y_test.shape)

    z_mean, _, _ = vae.encoder.predict(vae.x_test,
                                   batch_size=batch_size)

    print("z_mean", z_mean.dtype, z_mean.shape, np.mean(z_mean, axis=0), np.std(z_mean, axis=0))

    sub_index = vae.y_test == 1
    print("sub_index", sub_index.dtype, sub_index.shape)

    z_one = z_mean[vae.y_test == 4]
    print("z_one", z_one.dtype, z_one.shape, np.mean(z_one, axis=0), np.std(z_one, axis=0))
    z_one_mean = np.mean(z_one, axis=0)
    print(z_one_mean.shape)
    z_one_mean = np.expand_dims(z_one_mean,axis=0)
    print(z_one_mean.shape)

    a = np.arange(10)
    b = np.zeros((10, 10))
    b[a, a] = 1
    z_one_mean = b

    x_decoded = vae.decoder.predict(z_one_mean)
    x_decoded = np.reshape(x_decoded,(-1,28,28))
    print(x_decoded.shape)
    for a in range(10):
        plt.imshow(x_decoded[a])
    plt.show()



if __name__ == "__main__":

    weights = True
    epochs = 50
    batch_size = 128

    vae = VAE()
    vae.build_models()
    vae.build_losses()

    if weights:
        vae.vae.load_weights('vae_mlp_mnist.h5')
    else:
        # train the autoencoder
        vae.vae.fit(vae.x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(vae.x_test, None))
        vae.vae.save_weights('vae_mlp_mnist.h5')

    cool_stats(vae, batch_size)

    #plot_results(vae, batch_size)