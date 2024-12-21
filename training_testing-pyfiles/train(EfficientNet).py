import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.applications import EfficientNetB7 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore # Preprocessing function for EfficientNetB7


train_dir = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Train"
val_dir = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Val"
test_dir = r"C:\Project\Lung-Cancer-Detection-Using-CNN\training_testing-pyfiles\lung\Test"

BATCH_SIZE = 128
X = Y = 224  # Input size for EfficientNetB7

# Create separate data generators for train, validation, and test
data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Preprocessing for EfficientNet

# Train generator
train_generator = data_gen.flow_from_directory(
    directory=train_dir,
    target_size=(X, Y),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode="rgb",
    shuffle=True,
    seed=42
)

# Validation generator
val_generator = data_gen.flow_from_directory(
    directory=val_dir,
    target_size=(X, Y),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode="rgb",
    shuffle=False,
    seed=42
)

# Test generator (if you want to use it for evaluation later)
test_generator = data_gen.flow_from_directory(
    directory=test_dir,
    target_size=(X, Y),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode="rgb",
    shuffle=False,
    seed=42
)

# Load the EfficientNetB7 model without the top layers and freeze it
ptm = EfficientNetB7(
    input_shape=(X, Y, 3),
    weights='imagenet',  # Load pre-trained weights from ImageNet
    include_top=False    # Exclude the top fully connected layers
)

ptm.trainable = False

# Add custom layers on top of EfficientNetB7
x = GlobalAveragePooling2D()(ptm.output)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
y = Dense(len(folders), activation='softmax')(x) # type: ignore

model = Model(inputs=ptm.input, outputs=y)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
hist = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping]
)

# Save the model architecture and training history
model_json = model.to_json()
with open("model/EfficientNetB7/model.json", "w") as json_file:
    json_file.write(model_json)
with open('model/EfficientNetB7/history.pckl', 'wb') as f:
    pickle.dump(hist.history, f)

