"""
model/data_loader.py
---------------------
Responsible for exactly one thing: turning the raw Animals-10 dataset on disk
into three ready-to-train tf.data.Dataset objects (train / validation / test).

Expected raw layout after downloading Animals-10 from Kaggle and extracting it
into data/raw/:

    data/raw/raw-img/cane/......jpeg
    data/raw/raw-img/cavallo/...jpeg
    data/raw/raw-img/elefante/..jpeg
    ... (folder names are in Italian)

This module:
  1. Renames/copies those Italian-named folders into English names
     (config.ANIMALS10_TRANSLATE) under data/processed/all/.
  2. Splits each class into train/val/test subfolders under data/processed/.
  3. Loads those subfolders as tf.data.Dataset objects, ready for model.fit().
"""

import os
import json
import random
import shutil

import tensorflow as tf

import config


def _find_raw_image_root() -> str:
    """
    Locate the folder that actually contains the per-class subfolders.

    Kaggle's Animals-10 zip extracts to data/raw/raw-img/<class>/*.jpeg, but
    people sometimes flatten it to data/raw/<class>/*.jpeg. Rather than force
    the user to match one exact layout, we search for it.
    """
    candidates = [config.RAW_DATA_DIR, os.path.join(config.RAW_DATA_DIR, "raw-img")]
    for candidate in candidates:
        if os.path.isdir(candidate):
            subfolders = [f for f in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, f))]
            # Heuristic: if the Italian class names are present, this is the root.
            if any(name in config.ANIMALS10_TRANSLATE for name in subfolders):
                return candidate

    raise FileNotFoundError(
        "Could not find the Animals-10 class folders under data/raw/. "
        "Download the dataset from Kaggle (alessiocorrado99/animals10), "
        "extract it, and place the 'raw-img' folder inside data/raw/."
    )


def _split_list(items: list, val_split: float, test_split: float) -> tuple:
    """Shuffle a list of filenames deterministically and split it 3 ways."""
    random.Random(config.RANDOM_SEED).shuffle(items)
    n = len(items)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    val_items = items[:n_val]
    test_items = items[n_val:n_val + n_test]
    train_items = items[n_val + n_test:]
    return train_items, val_items, test_items


def build_train_val_test_split(force: bool = False) -> None:
    """
    Create data/processed/{train,val,test}/<english_class_name>/ folders,
    populated with copies of the raw images.

    Idempotent: if data/processed/train already has content and force=False,
    this is skipped so re-running train.py doesn't redo the split every time.
    """
    train_dir = os.path.join(config.PROCESSED_DATA_DIR, "train")
    if os.path.isdir(train_dir) and os.listdir(train_dir) and not force:
        print("[data_loader] Processed split already exists — skipping split "
              "(pass force=True to redo it).")
        return

    raw_root = _find_raw_image_root()

    # Wipe any previous split so we don't mix old and new data.
    for split_name in ("train", "val", "test"):
        split_path = os.path.join(config.PROCESSED_DATA_DIR, split_name)
        if os.path.isdir(split_path):
            shutil.rmtree(split_path)

    for italian_name, english_name in config.ANIMALS10_TRANSLATE.items():
        class_dir = os.path.join(raw_root, italian_name)
        if not os.path.isdir(class_dir):
            print(f"[data_loader] WARNING: expected class folder '{italian_name}' "
                  f"not found, skipping.")
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        train_files, val_files, test_files = _split_list(
            image_files, config.VALIDATION_SPLIT, config.TEST_SPLIT
        )

        for split_name, files in (("train", train_files), ("val", val_files), ("test", test_files)):
            dest_dir = os.path.join(config.PROCESSED_DATA_DIR, split_name, english_name)
            os.makedirs(dest_dir, exist_ok=True)
            for filename in files:
                shutil.copy2(os.path.join(class_dir, filename), os.path.join(dest_dir, filename))

        print(f"[data_loader] {english_name:12s}: "
              f"{len(train_files)} train / {len(val_files)} val / {len(test_files)} test")


def get_datasets() -> tuple:
    """
    Load the train/val/test folders as tf.data.Dataset objects.

    Returns:
        (train_ds, val_ds, test_ds, class_names)
    """
    common_kwargs = dict(
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        seed=config.RANDOM_SEED,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.PROCESSED_DATA_DIR, "train"), shuffle=True, **common_kwargs
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.PROCESSED_DATA_DIR, "val"), shuffle=False, **common_kwargs
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(config.PROCESSED_DATA_DIR, "test"), shuffle=False, **common_kwargs
    )

    class_names = train_ds.class_names  # alphabetical order, e.g. ['butterfly', 'cat', ...]

    # Persist class names so the Flask app can map prediction indices -> labels
    # without needing to re-scan the dataset folders at inference time.
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    # Performance: cache in memory after first epoch and prefetch the next
    # batch while the GPU/CPU is busy with the current one.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    # Running this file directly lets you sanity-check the data pipeline
    # in isolation, without kicking off a full training run.
    build_train_val_test_split()
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print(f"[data_loader] Classes found: {class_names}")
    print(f"[data_loader] Train batches: {len(train_ds)}, "
          f"Val batches: {len(val_ds)}, Test batches: {len(test_ds)}")
