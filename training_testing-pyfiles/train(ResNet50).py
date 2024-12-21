import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import pickle

train_data = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Train"
test_data = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Test"
val_data = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Val"
batch_size = 32
target_size = (125, 125)

# Data generators
train = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.40)

test = ImageDataGenerator(rescale=1/255.0)

train_generator = train.flow_from_directory(
    directory=train_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset='training')

valid_generator = train.flow_from_directory(
    directory=val_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True)

test_generator = test.flow_from_directory(
    directory=test_data,
    target_size=target_size,
    batch_size=1)

# Model definition
model = Sequential()
model.add(ResNet50(include_top=False, weights='imagenet', input_shape=(125, 125, 3)))
model.add(GlobalAveragePooling2D())
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training
hist = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    callbacks=[early_stopping])

# Saving model and history
model.save("model/ResNet50(200ep)/model.h5")

model_json = model.to_json()
with open("model/ResNet50(200ep)/model.json", "w") as json_file:
    json_file.write(model_json)

with open("model/ResNet50(200ep)/history.pckl", "wb") as f:
    pickle.dump(hist.history, f)



