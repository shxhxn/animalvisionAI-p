"""
model/evaluate.py
------------------
Loads the trained model and measures how well it actually generalizes, using
the held-out TEST set (images the model never saw during training or
validation-based early stopping).

Produces:
  - static/results/confusion_matrix.png
  - static/results/metrics.json  (accuracy, precision, recall, F1 — overall
    and per-class)

Run from the project root with:
    python -m model.evaluate
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

import config
from model.data_loader import get_datasets


def load_trained_model() -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(config.MODEL_PATH)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. "
            "Run `python -m model.train` first."
        ) from exc


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple:
    """
    Run the model over an entire dataset and collect true vs predicted
    labels as flat arrays of class indices (what sklearn's metrics expect).
    """
    y_true, y_pred = [], []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"[evaluate] Saved confusion matrix -> {config.CONFUSION_MATRIX_PATH}")


def compute_and_save_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    """
    Computes overall accuracy plus precision/recall/F1 both as a single
    weighted-average number (for a quick headline stat) and broken down
    per-class (for the detail table on the Model Performance page).
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": report["accuracy"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "per_class": {
            class_name: {
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1_score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            }
            for class_name in class_names
        },
    }

    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Saved metrics -> {config.METRICS_JSON_PATH}")
    print(f"[evaluate] Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"[evaluate] Weighted F1:      {metrics['f1_weighted']:.4f}")

    return metrics


def run_evaluation() -> dict:
    model = load_trained_model()
    _, _, test_ds, class_names = get_datasets()

    print("[evaluate] Running inference on test set...")
    y_true, y_pred = collect_predictions(model, test_ds)

    plot_confusion_matrix(y_true, y_pred, class_names)
    metrics = compute_and_save_metrics(y_true, y_pred, class_names)
    return metrics


if __name__ == "__main__":
    run_evaluation()
