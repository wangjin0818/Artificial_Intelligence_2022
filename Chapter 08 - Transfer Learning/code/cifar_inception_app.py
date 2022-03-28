import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

(x_train, y_train) , (x_test, y_test) = cifar10.load_data()

x_train_resized = []
for i, image in enumerate(x_train):
    resized_image = cv2.resize(image, (224, 224)) / 255.
    x_train_resized.append(resized_image)

x_test_resized = []
for i, image in enumerate(x_test):
    resized_image = cv2.resize(image, (224, 224)) / 255.
    x_test_resized.append(resized_image)

print(x_train_resized.shape)
print(x_test_resized.shape)
print(y_train.shape)
print(y_test.shape)

batch_size = 32
num_classes = 10
epochs = 10

# print(x_train[0], y_train[0])
y_train = tf.keras.utils.to_categorical(y_train)
y_test_val = tf.keras.utils.to_categorical(y_test)


from tensorflow.keras import applications
input_layer = tf.keras.Input(shape=(224, 224, 3))
# input_shape = (32, 32, 3)
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='max', input_tensor=input_layer)

x = base_model.output
flatten_layer = tf.keras.layers.Flatten()(x)
hidden_layer = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal') (flatten_layer)
dropout_layer_2 = tf.keras.layers.Dropout(0.25)(hidden_layer)
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal') (dropout_layer_2)

model = tf.keras.Model(base_model.input, output_layer)

print(model.summary())
# optimizer = tf.keras.optimizers.Adam(lr=5e-4)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
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

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        workers=4,
                        validation_data=(x_test, y_test_val))
y_pred = np.argmax(model.predict(x_test), axis=-1)

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))