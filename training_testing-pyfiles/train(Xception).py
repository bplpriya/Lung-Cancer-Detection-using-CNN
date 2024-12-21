import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Flatten, Dense # type: ignore
from tensorflow.keras.applications import Xception # type: ignore
import pickle
from tensorflow import keras


# Paths for lung cancer dataset
train_data = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Train"
val_data = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Val"  # Validation data path
test_data = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Test"  # Test data path

# Set batch size and target image size
batch_size = 32
target_size = (125, 125)

# ImageDataGenerator without validation split (since you have separate directories)
train = ImageDataGenerator(rescale=1/255.0)
validation = ImageDataGenerator(rescale=1/255.0)
test = ImageDataGenerator(rescale=1/255.0)

# Create train, validation, and test data generators
train_generator = train.flow_from_directory(
    directory=train_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True)

valid_generator = validation.flow_from_directory(
    directory=val_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True)

test_generator = test.flow_from_directory(
    directory=test_data,
    target_size=target_size,
    batch_size=1,  # Usually batch size = 1 for test data (one image at a time)
    class_mode="categorical",
    shuffle=False)  # Do not shuffle test data

# Define the model
model = Sequential()

# Add the Xception model as base, without its fully connected top layers
model.add(Xception(include_top=False,
                   weights=None,  # Use 'imagenet' if pre-trained weights are preferred
                   input_shape=(125, 125, 3)))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Adjust number of output units (3) based on your classes

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train the model
hist = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=4)  # Increase epochs as needed for better training

# Save model and training history
model_dir = "model/Xception_lung_cancer/"  # Directory for saving the model and history
os.makedirs(model_dir, exist_ok=True)

# Save the model architecture to JSON
model_json = model.to_json()
with open(f"{model_dir}/model.json", "w") as json_file:
    json_file.write(model_json)

# Save the training history to a pickle file
with open(f"{model_dir}/history.pckl", "wb") as f:
    pickle.dump(hist.history, f)

# Save the entire model (weights + architecture) for future use
model.save(f"{model_dir}/model.h5")

# If needed, evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
