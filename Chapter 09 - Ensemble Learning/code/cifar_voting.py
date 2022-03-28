import tensorflow as tf
import numpy as np
import scipy.stats

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
print(tf.__version__)

batch_size = 128
num_classes = 10
epochs = 10

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = tf.keras.utils.to_categorical(y_train)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=0, test_size=0.1)

print(y_train.shape)
print(y_test.shape)


def conv_bn(x, filters, kernel_size, activation):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal') (x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x


def create_cnn_bn():
    input_layer = tf.keras.Input(shape=(32, 32, 3))
    # reshape_layer = tf.keras.layers.Reshape((32, 32, 3))(input_layer)
    conv_layer_1 = conv_bn(input_layer, filters=64, kernel_size=(3, 3), activation='relu')
    conv_layer_2 = conv_bn(conv_layer_1, filters=64, kernel_size=(3, 3), activation='relu')
    pooling_layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_layer_2)

    conv_layer_3 = conv_bn(pooling_layer_1, filters=128, kernel_size=(3, 3), activation='relu')
    conv_layer_4 = conv_bn(conv_layer_3, filters=128, kernel_size=(3, 3), activation='relu')
    pooling_layer_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_layer_4)

    conv_layer_5 = conv_bn(pooling_layer_2, filters=256, kernel_size=(3, 3), activation='relu')
    conv_layer_6 = conv_bn(conv_layer_5, filters=256, kernel_size=(3, 3), activation='relu')
    pooling_layer_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_layer_6)

    flatten_layer = tf.keras.layers.Flatten()(pooling_layer_3)
    hidden_layer = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(flatten_layer)
    dropout_layer_2 = tf.keras.layers.Dropout(0.5)(hidden_layer)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout_layer_2)

    model = tf.keras.Model(input_layer, output_layer)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def create_vgg():
    input_layer = tf.keras.Input(shape=(32, 32, 3))
    vgg_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, pooling='max',
                                            input_tensor=input_layer)
    x = vgg_model.output
    flatten_layer = tf.keras.layers.Flatten()(x)
    hidden_layer = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(flatten_layer)
    dropout_layer_2 = tf.keras.layers.Dropout(0.25)(hidden_layer)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout_layer_2)

    model = tf.keras.Model(vgg_model.input, output_layer)

    # optimizer = tf.keras.optimizers.Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_resnet():
    input_layer = tf.keras.Input(shape=(32, 32, 3))
    vgg_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, pooling='max',
                                            input_tensor=input_layer)
    x = vgg_model.output
    flatten_layer = tf.keras.layers.Flatten()(x)
    hidden_layer = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(flatten_layer)
    dropout_layer_2 = tf.keras.layers.Dropout(0.25)(hidden_layer)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout_layer_2)

    model = tf.keras.Model(vgg_model.input, output_layer)

    # optimizer = tf.keras.optimizers.Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


print('Using real-time data augmentation.')
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

model = []
# model.append(create_cnn())
model.append(create_cnn_bn())
model.append(create_vgg())
model.append(create_resnet())

models = []
for i in range(len(model)):
    # model[i].fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model[i].fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_val, y_val))
    models.append(model[i])


# Predict labels with models
labels = []
for m in models:
    predicts = np.argmax(m.predict(x_test), axis=1)
    labels.append(predicts)

# Ensemble with voting
labels = np.array(labels)
labels = np.transpose(labels, (1, 0))
labels = scipy.stats.mode(labels, axis=1)[0]
y_pred = np.squeeze(labels)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))


