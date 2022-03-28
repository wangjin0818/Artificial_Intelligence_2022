import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# print(x_train[0], y_train[0])
y_train = tf.keras.utils.to_categorical(y_train)
y_test_val = tf.keras.utils.to_categorical(y_test)

batch_size = 32
num_classes = 10
epochs = 10


def conv_bn(x, filters, kernel_size, activation):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal') (x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation) (x)
    return x


input_layer = tf.keras.Input(shape=(32, 32, 3, ))
# reshape_layer = Reshape((32, 32, 3)) (input_layer)
conv_layer_1 = conv_bn(input_layer, filters=64, kernel_size=(3, 3), activation='relu')
conv_layer_2 = conv_bn(conv_layer_1, filters=64, kernel_size=(3, 3), activation='relu')
pooling_layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same') (conv_layer_2)

conv_layer_3 = conv_bn(pooling_layer_1, filters=128, kernel_size=(3, 3), activation='relu')
conv_layer_4 = conv_bn(conv_layer_3, filters=128, kernel_size=(3, 3), activation='relu')
pooling_layer_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (conv_layer_4)

conv_layer_5 = conv_bn(pooling_layer_2, filters=256, kernel_size=(3, 3), activation='relu')
conv_layer_6 = conv_bn(conv_layer_5, filters=256, kernel_size=(3, 3), activation='relu')
pooling_layer_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (conv_layer_6)

flatten_layer = tf.keras.layers.Flatten()(pooling_layer_3)
hidden_layer = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal') (flatten_layer)
dropout_layer_2 = tf.keras.layers.Dropout(0.5) (hidden_layer)
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal') (dropout_layer_2)

model = tf.keras.Model(input_layer, output_layer)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
y_pred = np.argmax(model.predict(x_test), axis=-1)
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))
