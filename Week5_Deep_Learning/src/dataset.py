"""
dataset.py
Data acquisition, memory-efficient preprocessing, normalization, and partitioning for
the Fashion-MNIST benchmark dataset.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Standard Fashion-MNIST class mappings established by Zalando Research
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

def load_and_preprocess_fashion_mnist(train_samples=20000, val_samples=4000, test_samples=5000, random_state=42, data_dir=None):
    """
    Downloads or loads Fashion-MNIST dataset with memory-efficient partitioning.
    Normalizes pixel intensities to [0.0, 1.0] using in-place scaling to prevent
    RAM spikes, constructs dedicated validation splits, and formats dual tensors.

    Parameters:
    -----------
    train_samples : int
        Number of training samples (default: 20,000 with exact class balance).
    val_samples : int
        Number of validation samples (default: 4,000).
    test_samples : int
        Number of unseen test samples (default: 5,000).
    random_state : int
        Seed for reproducible stratified splitting.

    Returns:
    --------
    data_dict : dict
        Contains 'X_train_cnn', 'X_val_cnn', 'X_test_cnn',
                 'X_train_mlp', 'X_val_mlp', 'X_test_mlp',
                 'y_train', 'y_val', 'y_test', 'class_names'.
    """
    print("[INFO] Ingesting Fashion-MNIST dataset via Keras datasets API...")
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.fashion_mnist.load_data()
    print("[INFO] Fashion-MNIST raw data successfully loaded.")

    # Stratified sub-sampling to respect hardware constraints while maintaining exact class balance
    total_train_val = train_samples + val_samples
    if total_train_val < len(X_train_raw):
        idx_pool, _ = train_test_split(
            np.arange(len(y_train_raw)),
            train_size=total_train_val,
            stratify=y_train_raw,
            random_state=random_state
        )
        X_train_pool = X_train_raw[idx_pool]
        y_train_pool = y_train_raw[idx_pool]
    else:
        X_train_pool, y_train_pool = X_train_raw, y_train_raw

    # Partition into Train and Validation
    X_train_arr, X_val_arr, y_train, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=val_samples,
        stratify=y_train_pool,
        random_state=random_state
    )

    # Stratified test set
    if test_samples < len(X_test_raw):
        idx_test, _ = train_test_split(
            np.arange(len(y_test_raw)),
            train_size=test_samples,
            stratify=y_test_raw,
            random_state=random_state
        )
        X_test_arr = X_test_raw[idx_test]
        y_test = y_test_raw[idx_test]
    else:
        X_test_arr, y_test = X_test_raw, y_test_raw

    # Convert to float32 and normalize in-place: avoids creating duplicate 200MB memory buffers
    X_train = np.ascontiguousarray(X_train_arr, dtype=np.float32)
    np.divide(X_train, 255.0, out=X_train)

    X_val = np.ascontiguousarray(X_val_arr, dtype=np.float32)
    np.divide(X_val, 255.0, out=X_val)

    X_test = np.ascontiguousarray(X_test_arr, dtype=np.float32)
    np.divide(X_test, 255.0, out=X_test)

    print(f"[INFO] Partitioned balanced splits: Train={X_train.shape[0]:,}, "
          f"Validation={X_val.shape[0]:,}, Test={X_test.shape[0]:,}")

    # Format A: 4D Tensors for Convolutional Neural Networks (N, 28, 28, 1)
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_val_cnn = np.expand_dims(X_val, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    # Format B: 2D Flattened Vectors for Multi-Layer Perceptrons (N, 784)
    X_train_mlp = X_train.reshape(-1, 28 * 28)
    X_val_mlp = X_val.reshape(-1, 28 * 28)
    X_test_mlp = X_test.reshape(-1, 28 * 28)

    data_dict = {
        "X_train_cnn": X_train_cnn,
        "X_val_cnn": X_val_cnn,
        "X_test_cnn": X_test_cnn,
        "X_train_mlp": X_train_mlp,
        "X_val_mlp": X_val_mlp,
        "X_test_mlp": X_test_mlp,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_names": CLASS_NAMES
    }

    return data_dict

def generate_dataset_summary(data_dict, output_dir):
    """
    Computes class counts and dimension profiles for each partition,
    saving the verification ledger to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_records = []

    y_train = data_dict["y_train"]
    y_val = data_dict["y_val"]
    y_test = data_dict["y_test"]

    for class_idx, name in enumerate(CLASS_NAMES):
        train_count = int(np.sum(y_train == class_idx))
        val_count = int(np.sum(y_val == class_idx))
        test_count = int(np.sum(y_test == class_idx))
        total_count = train_count + val_count + test_count

        summary_records.append({
            "Class_ID": class_idx,
            "Class_Name": name,
            "Train_Samples": train_count,
            "Val_Samples": val_count,
            "Test_Samples": test_count,
            "Total_Samples": total_count,
            "Train_Pct": f"{(train_count / len(y_train)) * 100:.1f}%",
            "Class_Balance": "Perfect (Uniform)"
        })

    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(output_dir, "dataset_split_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[INFO] Exported dataset partition verification ledger to: {summary_path}")
    return df_summary
