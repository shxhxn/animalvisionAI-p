"""
model/model_architecture.py
----------------------------
Defines the CNN itself. Kept separate from train.py so the architecture can
be inspected, unit-tested, or swapped out without touching the training loop.

This is a CNN trained FROM SCRATCH (no transfer learning / pretrained
weights) — random-initialized weights that learn entirely from the
Animals-10 images we feed it. That's a deliberate portfolio choice: it
demonstrates understanding of conv/pool/dense mechanics rather than just
fine-tuning someone else's model.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

import config


def build_data_augmentation() -> tf.keras.Sequential:
    """
    Random, label-preserving transformations applied only during training.

    Why: with a few thousand images per class, the network can easily
    memorize the training set instead of learning general features. Randomly
    flipping/rotating/zooming images each epoch effectively gives the model
    more varied examples to learn from, which reduces overfitting.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def _conv_block(x, filters: int, block_name: str):
    """
    One reusable convolutional block: Conv -> BatchNorm -> ReLU -> Conv ->
    BatchNorm -> ReLU -> MaxPool -> Dropout.

    Two conv layers before pooling lets the network build slightly more
    complex features at each spatial resolution before downsampling.
    BatchNorm stabilizes/speeds up training; Dropout after pooling fights
    overfitting.
    """
    x = layers.Conv2D(filters, 3, padding="same", name=f"{block_name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", name=f"{block_name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu2")(x)

    x = layers.MaxPooling2D(pool_size=2, name=f"{block_name}_pool")(x)
    x = layers.Dropout(0.25, name=f"{block_name}_dropout")(x)
    return x


def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build and compile the CNN.

    Architecture summary:
        Input (128x128x3)
        -> Rescaling (pixels 0-255 -> 0-1)
        -> Data augmentation (train time only, no-op at inference)
        -> 4x conv blocks (32 -> 64 -> 128 -> 256 filters), each halving
           spatial resolution: 128 -> 64 -> 32 -> 16 -> 8
        -> GlobalAveragePooling (replaces Flatten + huge Dense layer,
           dramatically cutting parameter count and overfitting risk)
        -> Dense(256) + Dropout
        -> Dense(num_classes, softmax)
    """
    inputs = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS), name="input_image")

    x = layers.Rescaling(1.0 / 255, name="rescaling")(inputs)
    x = build_data_augmentation()(x)

    x = _conv_block(x, 32, "block1")
    x = _conv_block(x, 64, "block2")
    x = _conv_block(x, 128, "block3")
    x = _conv_block(x, 256, "block4")

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.5, name="dropout_final")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="animalvision_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check: build the model with a placeholder class count and
    # print its architecture. Useful for confirming shapes before a long
    # training run.
    model = build_model(num_classes=10)
    model.summary()
