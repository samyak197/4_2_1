import tensorflow as tf
import random
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
import os
import shutil
from pathlib import Path
import numpy as np
import base64


# Collecting data
def train_to_test(data_dir, label, no_of_images):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    label_train_dir = train_dir / label
    label_test_dir = test_dir / label
    no_of_images = int(no_of_images)

    # Ensure there are images available for selection
    image_files = [
        f
        for f in os.listdir(label_train_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    # Calculate the number of images for testing (25% of available images or specified number, whichever is smaller)
    images_for_test_dir = int(no_of_images * 0.25)

    # Select random images for testing
    selected_images = random.sample(image_files, images_for_test_dir)

    # Move selected images to the test directory
    for image in selected_images:
        source_path = os.path.join(label_train_dir, image)
        dest_path = os.path.join(label_test_dir, image)
        shutil.move(source_path, dest_path)

    print(f"{images_for_test_dir} random images moved to test_dir")


class_labels = []


def video_to_frame(data_dir, label, frames_data):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    class_labels.append(label)

    label_train_dir = train_dir / label
    label_test_dir = test_dir / label
    label_train_dir.mkdir()
    label_test_dir.mkdir()

    for index, frame_data in enumerate(frames_data):
        # Decode base64 image data
        _, frame_bytes = frame_data.split(",", 1)
        frame_nparr = np.frombuffer(base64.b64decode(frame_bytes), np.uint8)
        frame = cv2.imdecode(frame_nparr, cv2.IMREAD_COLOR)

        # Save the decoded frame
        frame_filename = os.path.join(label_train_dir, f"{label}_{index:04d}.jpg")
        cv2.imwrite(str(frame_filename), frame)
        print(f"Saved {frame_filename}")


def create_dirs(data_dir):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_dir.mkdir()
    print("train folder created")
    test_dir.mkdir()
    print("test folder created")
    return train_dir, test_dir


class_labels = []

print(class_labels)


# Model Training
def data_training(data_augmentation, data_dir, folder_name, no_of_epochs):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if data_augmentation:
        train_datagen_augmented = ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
        )
        print("Data augmented")
    else:
        train_datagen_augmented = ImageDataGenerator(rescale=1 / 255.0)
        print("Data not augmented")

    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    train_data_augmented = train_datagen_augmented.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=True,
    )
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=True,
    )

    model = Sequential(
        [
            Conv2D(
                filters=50, kernel_size=3, activation="relu", input_shape=(224, 224, 3)
            ),
            Conv2D(50, 3, activation="relu"),
            Conv2D(50, 3, activation="relu"),
            MaxPool2D(pool_size=2),
            Conv2D(30, 3, activation="relu"),
            Conv2D(30, 3, activation="relu"),
            MaxPool2D(),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    history = model.fit(
        train_data_augmented,
        epochs=int(no_of_epochs),
        steps_per_epoch=len(train_data_augmented),
        validation_data=test_data,
        validation_steps=len(test_data),
    )

    model.save(f"{folder_name}/{folder_name}_model.h5")

    # Print and return val_accuracy as percentage
    val_accuracy = history.history["val_accuracy"]
    val_accuracy_percentage = [round(acc * 100, 2) for acc in val_accuracy]
    print("Validation Accuracy (%):", val_accuracy_percentage)

    return val_accuracy_percentage


# test model


last_frame = None  # Define a global variable to store the last frame
class_labels = ["Class 0", "Class 1"]  # Define class labels globally


def get_label_and_prob(frame, model, object_1_label, object_2_label):
    class_labels = [object_1_label, object_2_label]
    img = tf.image.resize(frame, (224, 224))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    label = model.predict(img)

    # Calculate probabilities as percentages and round to two decimal places
    prob2 = round(label[0][0] * 100, 2)  # Probability for class 0
    prob1 = round(100 - prob2, 2)  # Probability for class 1

    return class_labels[np.argmax(label)], prob1, prob2


last_frame = None


def model_testing(folder_name, frame_data, object_1_label, object_2_label):

    global last_frame  # Reference the global variable

    frame_bytes = frame_data.read()
    frame_nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(frame_nparr, cv2.IMREAD_COLOR)
    model = load_model(f"{folder_name}/{folder_name}_model.h5")

    if last_frame is not None:
        same = np.array_equal(frame, last_frame)  # Check if the frames are equal
    else:
        same = False

    last_frame = frame.copy()  # Update the last frame
    label, prob1, prob2 = get_label_and_prob(
        frame, model, object_1_label, object_2_label
    )

    return label, prob1, prob2, same
