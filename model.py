
import numpy as np
from keras.models import Sequential
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.utils.np_utils import to_categorical


def generate(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24, 24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)


training_batch_generator = generate('data/train', shuffle=True,
                                    batch_size=32, target_size=(24, 24))
validation_batch_generator = generate('data/valid', shuffle=True,
                                      batch_size=32, target_size=(24, 24))

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), input_shape=(
        24, 24, 1), activation='relu', ),
    MaxPooling2D(pool_size=(2, 2)),  # changed from 1,1 everywhere
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(training_batch_generator, validation_data=validation_batch_generator, epochs=10, steps_per_epoch=len(
    training_batch_generator.classes)//32, validation_steps=len(validation_batch_generator.classes)//32)
model.save('models/cnnCat2.h5', overwrite=True)
