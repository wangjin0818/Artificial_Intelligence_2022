import os
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Reshape, Flatten

def create_generator(latent_dim, img_shape=(28, 28)):
    sequence = Input(shape=(latent_dim, ), dtype='float32')
    model = Dense(256)(sequence)
    model = LeakyReLU(alpha=0.2) (model)
    model = BatchNormalization(momentum=0.8)(model)
    model = Dense(512)(model)
    model = LeakyReLU(alpha=0.2) (model)
    model = BatchNormalization(momentum=0.8)(model)
    model = Dense(512)(model)
    model = LeakyReLU(alpha=0.2) (model)
    model = BatchNormalization(momentum=0.8)(model)
    model = Dense(np.prod(img_shape), activation='sigmoid')(model)
    output = Reshape(img_shape)(model)

    model = Model(inputs=sequence, outputs=output)
    model.summary()

    return model

def create_discriminator(img_shape=(28, 28)):
    sequence = Input(shape=img_shape)
    model = Flatten()(sequence)
    model = Dense(512)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dense(256)(model)
    model = LeakyReLU(alpha=0.2)(model)
    output = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=sequence, outputs=output)
    model.summary()

    return model

def sample_images(epoch, latent_dim, generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    print(gen_imgs.shape)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    plt.close()


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
print(x_train.shape)

batch_size = 32
epochs = 30000
latent_dim = 28*28
sample_interval = 2000

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

discriminator = create_discriminator()
discriminator.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

discriminator.trainable = False

generator = create_generator(latent_dim, )
z = Input(shape=(latent_dim, ))
img = generator(z)

validity = discriminator(img)
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer='adam')


for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs = x_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    gen_imgs = generator.predict(noise)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    g_loss = combined.train_on_batch(noise, valid)

    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    if epoch % sample_interval == 0:
        sample_images(epoch, latent_dim, generator)