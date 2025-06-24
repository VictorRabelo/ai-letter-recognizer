"""Model training and loading utilities for EMNIST letters."""

import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

MODEL_PATH = "emnist_letters.h5"

# Mapping index -> character for letters only (A-Z then a-z)
LETTERS = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]


def _normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label


def load_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load EMNIST dataset filtered to letters only."""
    train = tfds.load("emnist/byclass", split="train", as_supervised=True)
    test = tfds.load("emnist/byclass", split="test", as_supervised=True)

    # Filter out digit classes (<10) and rotate images
    def _filter_map(image, label):
        image = tf.image.rot90(image, k=3)
        return image, label

    train = train.map(_filter_map).filter(lambda _img, lbl: lbl >= 10)
    test = test.map(_filter_map).filter(lambda _img, lbl: lbl >= 10)

    train = train.map(lambda img, lbl: (img, lbl - 10))
    test = test.map(lambda img, lbl: (img, lbl - 10))

    train = train.map(_normalize).shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    test = test.map(_normalize).batch(128).prefetch(tf.data.AUTOTUNE)
    return train, test


def build_model() -> tf.keras.Model:
    """Create a simple CNN for letter classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(LETTERS), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(epochs: int = 5) -> tf.keras.Model:
    """Train the model on the EMNIST dataset."""
    train_ds, test_ds = load_dataset()
    model = build_model()
    model.fit(train_ds, validation_data=test_ds, epochs=epochs)
    model.save(MODEL_PATH)
    return model


def load_model() -> tf.keras.Model:
    """Load a saved model or train a new one if not available."""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return train_model()


def predict_letter(model: tf.keras.Model, image) -> Tuple[str, float]:
    """Predict a letter from a PIL image."""
    arr = np.array(image).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    letter = LETTERS[idx]
    confidence = float(preds[idx])
    return letter, confidence
