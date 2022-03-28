import tensorflow as tf
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

batch_size = 32
num_classes = 10
epochs = 10

# print(x_train[0], y_train[0])
y_train = tf.keras.utils.to_categorical(y_train)
y_test_val = tf.keras.utils.to_categorical(y_test)

from tensorflow.keras import applications
input_layer = tf.keras.Input(shape=(32, 32, 3))
# input_shape = (32, 32, 3)
vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, pooling='max', input_tensor=input_layer)

print(vgg_model.summary())

# Creating dictionary that maps layer names to the layers
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

print(layer_dict)

for layer in vgg_model.layers:
    layer.trainable = False

# Getting output tensor of the last VGG layer that we want to include
x = vgg_model.output

flatten_layer = tf.keras.layers.Flatten()(x)
hidden_layer = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal') (flatten_layer)
dropout_layer_2 = tf.keras.layers.Dropout(0.25)(hidden_layer)
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal') (dropout_layer_2)

model = tf.keras.Model(vgg_model.input, output_layer)

# optimizer = tf.keras.optimizers.Adam(lr=5e-4)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
y_pred = np.argmax(model.predict(x_test), axis=-1)

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))