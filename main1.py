import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

import ha_utils.graphic_an_ting as GnT

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

    def __init__(self, use_conv=False, latent_dim=10):

        self.use_conv = use_conv

        # MNIST dataset
        (x_train, y_train), (x_test, self.y_test) = mnist.load_data()

        image_size = x_train.shape[1]
        self.original_dim = image_size * image_size
        if self.use_conv:
            x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
            x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        else:
            x_train = np.reshape(x_train, [-1, self.original_dim])
            x_test = np.reshape(x_test, [-1, self.original_dim])

        self.x_train = x_train.astype('float32') / 255
        self.x_test = x_test.astype('float32') / 255

        # network parameters
        self.input_shape1 = (image_size * image_size,)
        self.input_shape2 = (image_size, image_size, 1)
        self.intermediate_dim = 512
        self.latent_dim = latent_dim
        self.inputs1 = Input(shape=self.input_shape1, name='encoder_input')
        self.inputs2 = Input(shape=self.input_shape2, name='encoder_input')

    def build_models(self):

        def build_encoder1():

            x = Dense(self.intermediate_dim, activation='relu')(self.inputs1)
            self.z_mean = Dense(self.latent_dim, name='z_mean')(x)
            self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

            z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

            encoder = Model(self.inputs1, [self.z_mean, self.z_log_var, z], name='encoder')
            return encoder

        def build_encoder2():

            x = Conv2D(16, (3, 3), activation='relu', padding='same')(self.inputs2)
            #x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            #x = Dropout(0.20)(x)
            #x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            #x = Dropout(0.25)(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Flatten()(x)

            self.z_mean = Dense(self.latent_dim, name='z_mean')(x)
            self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

            # instantiate encoder model
            encoder = Model(self.inputs2, [self.z_mean, self.z_log_var, z], name='encoder')
            return encoder
            # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        def build_decoder2():

            # build decoder model
            latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
            x = Dense(7*7*16, activation=tf.nn.relu)(latent_inputs)
            x = Reshape(target_shape=(7, 7, 16))(x)
            #x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            #x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

            # instantiate decoder model
            decoder = Model(latent_inputs, outputs, name='decoder')
            return decoder

        def build_decoder1():

            # build decoder model
            latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
            x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
            outputs = Dense(self.original_dim, activation='sigmoid')(x)

            # instantiate decoder model
            decoder = Model(latent_inputs, outputs, name='decoder')
            return decoder

        if self.use_conv:
            self.encoder = build_encoder2()
            self.decoder = build_decoder2()
            # instantiate VAE model
            self.outputs = self.decoder(self.encoder(self.inputs2)[2])
            self.vae = Model(self.inputs2, self.outputs, name='vae_mlp')
        else:
            self.encoder = build_encoder1()
            self.decoder = build_decoder1()
            # instantiate VAE model
            self.outputs = self.decoder(self.encoder(self.inputs1)[2])
            self.vae = Model(self.inputs1, self.outputs, name='vae_mlp')

        self.encoder.summary()
        self.decoder.summary()

    def build_losses(self):

        if self.use_conv:
            # VAE loss = mse_loss or xent_loss + kl_loss
            inputs = K.flatten(self.inputs2)
            outputs = K.flatten(self.outputs)
            reconstruction_loss = binary_crossentropy(inputs,

                                                      outputs)
        else:
            # VAE loss = mse_loss or xent_loss + kl_loss
            reconstruction_loss = binary_crossentropy(self.inputs1,
                                                      self.outputs)

        reconstruction_loss *= self.original_dim
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        #self.vae.compile(optimizer=sgd)
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

def predict_on_test_set(vae, batch_size=128):

    # predict on entire test set
    z_mean, _, _ = vae.encoder.predict(vae.x_test,
                                   batch_size=batch_size)

    return z_mean

def cool_stats(z_mean, digit, do_print=False):

    z_digit = z_mean[vae.y_test == digit]
    if do_print:
        print("z_digit", z_digit.shape, np.mean(z_digit, axis=0))

    z_digit_mean = np.mean(z_digit, axis=0)
    z_digit_mean = np.expand_dims(z_digit_mean,axis=0)
    if do_print:
        print("z_digit", z_digit_mean.shape)

    x_decoded = vae.decoder.predict(z_digit_mean)
    x_decoded = np.reshape(x_decoded,(-1,28,28))
    print(x_decoded.shape)
    plt.imshow(x_decoded[0])
    plt.show()


def RGBify(img):
    if len(img.shape) == 3:
        img = img[:, :, ::-1]

    return img


# displays 2x2 grid of images and prints some stats
def display_imgs(w, h, image_list, image_titles, plt_title):

    new_image_list = []
    for img in image_list:
        img = np.squeeze(img)
        img = RGBify(img)
        new_image_list.append(img)
    image_list = new_image_list

    fig = plt.figure("VAE")
    fig.set_size_inches(12, 9)
    fig.suptitle(plt_title)

    axes=[]
    for i, (img, title) in enumerate(zip(image_list,image_titles)):

        ax = fig.add_subplot(w, h, i+1)
        ax.axis('off')
        ax.set_title(title)
        axes.append(ax)
        plt.imshow(img)

    plt.ion()
    plt.show()
    plt.pause(0.001)


def kmeans(epoch, vae, z_mean, n_clusters, do_show=True):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(z_mean)

    image_list = []
    image_titles = []
    hit_list = []
    np_accuracy = np.zeros((n_clusters,))
    for i in range(n_clusters):
        cluster = np.expand_dims(kmeans.cluster_centers_[i], axis=0)
        hits = np.sum(kmeans.labels_ == i)

        all_digits = vae.y_test
        digits = all_digits[kmeans.labels_ == i]
        digits, counts = np.unique(digits, return_counts=True)
        counts = list(zip(counts, digits))
        counts.sort(reverse=True)
        image_titles.append("N={}, {}={}%, {}={}%".format(hits, counts[0][1], int(round(counts[0][0]/np.sum(all_digits==counts[0][1])*100)),
                            counts[1][1], int(round(counts[1][0]/np.sum(all_digits==counts[1][1])*100))))
        np_accuracy[i] = counts[0][0]/np.sum(all_digits==counts[0][1])
        x_decoded = vae.decoder.predict(cluster)
        x_decoded = np.reshape(x_decoded, (-1, 28, 28))
        image_list.append(x_decoded)
        hit_list.append(np_accuracy[i])

    av_acc = np.mean(np_accuracy)
    min_acc = np.min(np_accuracy)
    plt_title = "{} av / min accuracy = {} / {}".format(epoch, av_acc, min_acc)

    zip_list = list(zip(hit_list, image_list, image_titles))
    zip_list.sort(reverse=True)
    hit_list, image_list, image_titles = zip(*zip_list)

    if do_show:
        display_imgs(4, 3, image_list, image_titles, plt_title)
        print(plt_title)

from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, vae):
        self.vae = vae

    def on_epoch_end(self, epoch, logs={}):

        z_mean = predict_on_test_set(self.vae, 100)

        kmeans(epoch, self.vae, z_mean, 10, do_show=True)


if __name__ == "__main__":

    weights = False
    epochs = 100
    batch_size = 32

    vae = VAE(use_conv=True, latent_dim=10)
    vae.build_models()
    vae.build_losses()

    if weights:
        vae.vae.load_weights('vae_mlp_mnist.h5')
    else:
        # train the autoencoder
        vae.vae.fit(vae.x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(vae.x_test, None),
                    callbacks=[TestCallback(vae)],
                    verbose=0)
        vae.vae.save_weights('vae_mlp_mnist.h5')

    z_mean = predict_on_test_set(vae, batch_size)

    for digit in range(10):
        pass
        #cool_stats(z_mean, digit)

    kmeans(vae, z_mean, 10, do_show=True)

    #plot_results(vae, batch_size)