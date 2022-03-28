import tensorflow as tf
print(tf.__version__)

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

(x_train, y_train) , (x_test, y_test) = cifar10.load_data()
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

input_layer = tf.keras.Input(shape=(32, 32, 3, ))
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu') (input_layer)
pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (conv_layer)
dropout_layer_1 = tf.keras.layers.Dropout(0.25) (pooling_layer)
flatten_layer = tf.keras.layers.Flatten()(dropout_layer_1)
hidden_layer = tf.keras.layers.Dense(128, activation='relu') (flatten_layer)
dropout_layer_2 = tf.keras.layers.Dropout(0.5) (hidden_layer)
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax') (dropout_layer_2)

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