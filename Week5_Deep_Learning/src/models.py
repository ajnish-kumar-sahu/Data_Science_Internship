"""
models.py
Deep Learning Neural Network Architectures for Fashion Apparel Classification:
1. Baseline Multi-Layer Perceptron (MLP) - Unregularized Dense Network
2. Deep Regularized Convolutional Neural Network (CNN) - Multi-block ConvNet with
   Batch Normalization, Spatial Pooling, and Staged Dropout Regularization.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_baseline_mlp(input_shape=(784,), num_classes=10, learning_rate=0.001):
    """
    Constructs a baseline 3-layer fully connected Multi-Layer Perceptron (MLP).
    This model serves as our unregularized baseline to illustrate how dense
    connections struggle with spatial variance and suffer from overfitting.

    Architecture:
      Input (784) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Dense(10, Softmax)
    """
    model = keras.Sequential(name="Baseline_MLP_Classifier")

    # Layer 1: First hidden projection
    model.add(layers.Input(shape=input_shape, name="input_flattened_vector"))
    model.add(layers.Dense(256, activation="relu", name="dense_hidden_1"))

    # Layer 2: Second hidden representation
    model.add(layers.Dense(128, activation="relu", name="dense_hidden_2"))

    # Layer 3: Normalized probability distribution over 10 classes
    model.add(layers.Dense(num_classes, activation="softmax", name="output_probabilities"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_regularized_cnn(input_shape=(28, 28, 1), num_classes=10, learning_rate=0.001):
    """
    Constructs a modern deep Convolutional Neural Network (CNN) designed for
    spatial feature hierarchy extraction and strong generalization.

    Architecture Design Principles:
    1. Double Conv2D Blocks: Allows expanding receptive field while maintaining non-linearity.
    2. Batch Normalization: Re-centers internal covariate shifts, accelerating convergence
       and stabilizing weight gradients across layers.
    3. MaxPooling2D: Spatial dimension downsampling providing translation invariance.
    4. Staged Dropout: 0.25 after convolutional pooling blocks, and 0.50 prior to the final
       dense layer to aggressively counteract co-adaptation of features.
    """
    model = keras.Sequential(name="Deep_Regularized_CNN_Classifier")

    # ---- Conv Block 1: Low-level edge and texture detectors ----
    model.add(layers.Input(shape=input_shape, name="input_image_tensor"))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1_1"))
    model.add(layers.BatchNormalization(name="batchnorm_1_1"))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1_2"))
    model.add(layers.BatchNormalization(name="batchnorm_1_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"))
    model.add(layers.Dropout(0.25, name="dropout_spatial_1"))

    # ---- Conv Block 2: Mid-level shape and structural feature detectors ----
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2_1"))
    model.add(layers.BatchNormalization(name="batchnorm_2_1"))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2_2"))
    model.add(layers.BatchNormalization(name="batchnorm_2_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"))
    model.add(layers.Dropout(0.25, name="dropout_spatial_2"))

    # ---- Classification Head ----
    model.add(layers.Flatten(name="flatten_feature_maps"))
    model.add(layers.Dense(128, activation="relu", name="dense_latent_rep"))
    model.add(layers.BatchNormalization(name="batchnorm_dense"))
    model.add(layers.Dropout(0.50, name="dropout_dense_head"))
    model.add(layers.Dense(num_classes, activation="softmax", name="output_probabilities"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def export_model_architectures(mlp_model, cnn_model, output_dir):
    """
    Extracts tabular layer specifications and trainable parameter summaries
    for both models and exports them as CSV artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    def extract_summary(model):
        records = []
        for idx, layer in enumerate(model.layers):
            params = layer.count_params()
            try:
                if hasattr(layer, "output_shape"):
                    out_shape = str(layer.output_shape)
                elif hasattr(layer, "output"):
                    out_shape = str(layer.output.shape)
                else:
                    out_shape = "N/A"
            except Exception:
                out_shape = "N/A"
            layer_type = layer.__class__.__name__
            act = getattr(layer, "activation", None)
            act_name = act.__name__ if act else "None"

            records.append({
                "Layer_Idx": idx + 1,
                "Layer_Name": layer.name,
                "Layer_Type": layer_type,
                "Output_Shape": out_shape,
                "Parameters": params,
                "Activation": act_name
            })
        return pd.DataFrame(records)

    df_mlp = extract_summary(mlp_model)
    df_cnn = extract_summary(cnn_model)

    mlp_path = os.path.join(output_dir, "architecture_mlp_summary.csv")
    cnn_path = os.path.join(output_dir, "architecture_cnn_summary.csv")

    df_mlp.to_csv(mlp_path, index=False)
    df_cnn.to_csv(cnn_path, index=False)

    print(f"[INFO] Exported MLP Architecture Summary ({mlp_model.count_params():,} params) to: {mlp_path}")
    print(f"[INFO] Exported CNN Architecture Summary ({cnn_model.count_params():,} params) to: {cnn_path}")

    # Comparative summary
    trainable_mlp = int(sum([int(tf.size(w).numpy()) for w in mlp_model.trainable_weights])) if mlp_model.trainable_weights else mlp_model.count_params()
    trainable_cnn = int(sum([int(tf.size(w).numpy()) for w in cnn_model.trainable_weights])) if cnn_model.trainable_weights else cnn_model.count_params()

    comp_df = pd.DataFrame([
        {
            "Architecture": "Baseline MLP",
            "Total_Layers": len(mlp_model.layers),
            "Total_Parameters": mlp_model.count_params(),
            "Trainable_Parameters": trainable_mlp,
            "Regularization": "None (Null Regularization)",
            "Spatial_Inductive_Bias": "None (Flat input vector)"
        },
        {
            "Architecture": "Deep Regularized CNN",
            "Total_Layers": len(cnn_model.layers),
            "Total_Parameters": cnn_model.count_params(),
            "Trainable_Parameters": trainable_cnn,
            "Regularization": "Batch Normalization + Staged Dropout (0.25, 0.50)",
            "Spatial_Inductive_Bias": "Strong (Translation invariance via 2D Convolutions & MaxPool)"
        }
    ])
    comp_path = os.path.join(output_dir, "model_architectures_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    return comp_df
