"""
model/train.py
---------------
Orchestrates a full training run:
  1. Prepare the train/val/test split (skipped if it already exists).
  2. Build the CNN.
  3. Train it with callbacks for checkpointing, early stopping, and LR decay.
  4. Save the trained model, class names, and training history to disk.
  5. Plot accuracy/loss curves for the "Model Performance" page.

Run from the project root with:
    python -m model.train
"""

import json
import time

import matplotlib
matplotlib.use("Agg")  # headless backend — this script may run with no display
import matplotlib.pyplot as plt
import tensorflow as tf

import config
from model.data_loader import build_train_val_test_split, get_datasets
from model.model_architecture import build_model


def build_callbacks() -> list:
    """
    Callbacks are hooks Keras calls automatically during training. We use
    three, each solving a distinct problem:

    - ModelCheckpoint: saves the model every time validation accuracy
      improves, so we always keep the BEST version, not just the last one
      (the last epoch isn't necessarily the best — the model can overfit
      in later epochs).
    - EarlyStopping: stops training if validation loss hasn't improved in
      `patience` epochs, saving time and preventing overfitting.
    - ReduceLROnPlateau: shrinks the learning rate when progress stalls,
      which often squeezes out a bit more accuracy than a fixed rate.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def plot_training_history(history: dict) -> None:
    """
    Save two PNGs: accuracy (train vs val) and loss (train vs val) over
    epochs. These feed directly into the "Model Performance" page in the
    Flask app — generated once here rather than recomputed by the web app,
    since training history doesn't change between page loads.
    """
    epochs_range = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["accuracy"], label="Training Accuracy", marker="o")
    plt.plot(epochs_range, history["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.ACCURACY_PLOT_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["loss"], label="Training Loss", marker="o")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss", marker="o")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.LOSS_PLOT_PATH, dpi=150)
    plt.close()

    print(f"[train] Saved accuracy plot -> {config.ACCURACY_PLOT_PATH}")
    print(f"[train] Saved loss plot -> {config.LOSS_PLOT_PATH}")


def run_training(epochs: int = None) -> tf.keras.callbacks.History:
    """
    Main entry point for training. Returns the Keras History object in case
    a caller (e.g. a "retrain" route in the Flask app) wants to inspect it.
    """
    epochs = epochs or config.EPOCHS

    print("[train] Preparing dataset split...")
    build_train_val_test_split()

    print("[train] Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print(f"[train] Classes: {class_names}")

    print("[train] Building model...")
    model = build_model(num_classes=len(class_names))
    model.summary()

    print(f"[train] Starting training for up to {epochs} epochs...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=build_callbacks(),
    )
    elapsed = time.time() - start_time
    print(f"[train] Training finished in {elapsed / 60:.1f} minutes.")

    # The ModelCheckpoint callback already saved the BEST epoch's weights to
    # config.MODEL_PATH. We still call save() here as a safety net in case
    # training completes without validation accuracy ever improving (e.g. a
    # 1-epoch smoke test), so a model file always exists after this runs.
    model.save(config.MODEL_PATH)
    print(f"[train] Model saved -> {config.MODEL_PATH}")

    # Persist the raw history (accuracy/loss per epoch) as JSON so the
    # "Model Performance" page can display exact numbers, not just the plots.
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(config.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[train] History saved -> {config.TRAINING_HISTORY_PATH}")

    plot_training_history(history_dict)

    return history


if __name__ == "__main__":
    run_training()
