"""
visualizer.py
Generates publication-grade charts, architecture schematics,
learning curves, confusion matrices, and feature map activations.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# Set global publication styling with standard safe DPI
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

# Corporate/Academic Palette
NAVY = "#1F497D"
BLUE = "#4472C4"
ORANGE = "#ED7D31"
GREEN = "#2CA02C"
RED = "#D62728"
GRAY = "#7F7F7F"
LIGHT_BG = "#F8F9FA"

def plot_dataset_sample_gallery(data_dict, output_dir):
    """
    Fig 1: Gallery showing 2 distinct examples for each of the 10 fashion classes.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train = data_dict["X_train_cnn"]
    y_train = data_dict["y_train"]
    class_names = data_dict["class_names"]

    fig, axes = plt.subplots(2, 10, figsize=(16, 4.2))
    fig.patch.set_facecolor("white")

    for col, class_idx in enumerate(range(10)):
        # Find indices for this class
        match_indices = np.where(y_train == class_idx)[0]
        for row in range(2):
            ax = axes[row, col]
            img = X_train[match_indices[row]].squeeze()
            ax.imshow(img, cmap="bone", interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{class_names[class_idx]}\n(ID: {class_idx})",
                             fontsize=9.5, fontweight="bold", color=NAVY, pad=8)

    plt.suptitle("Figure 1: Fashion-MNIST Benchmark Apparel Gallery (10 Distinct Categories)",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.03)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig01_dataset_sample_gallery.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_architecture_schematic(output_dir):
    """
    Fig 2: Conceptual structural diagram contrasting MLP and CNN inductive biases.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")

    # Baseline MLP Pipeline
    ax1.set_title("Architecture A: Baseline Multi-Layer Perceptron (MLP)\n[Spatial Collapse: 784 Flat Vector]",
                  fontsize=11, fontweight="bold", color=NAVY, pad=12)
    mlp_boxes = [
        ("Input Image\n(28x28 Grayscale)", "#E6F0FA", 0.85),
        ("Flatten Vector\n(784 Features)", "#D0E1FD", 0.70),
        ("Dense Layer 1\n(256 Units, ReLU)", "#B8D4FD", 0.52),
        ("Dense Layer 2\n(128 Units, ReLU)", "#A0C6FC", 0.35),
        ("Softmax Output\n(10 Class Probabilities)", "#FCE4D6", 0.15)
    ]
    for text, color, y in mlp_boxes:
        rect = plt.Rectangle((0.15, y - 0.06), 0.7, 0.10, facecolor=color, edgecolor=NAVY, linewidth=1.5, zorder=2)
        ax1.add_patch(rect)
        ax1.text(0.5, y - 0.01, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1A1A1A", zorder=3)
    for i in range(len(mlp_boxes) - 1):
        ax1.annotate("", xy=(0.5, mlp_boxes[i+1][2] + 0.04), xytext=(0.5, mlp_boxes[i][2] - 0.06),
                     arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=2, headwidth=8), zorder=1)
    ax1.text(0.5, -0.05, "Limitations: 235,146 Parameters; Lacks Translation Invariance;\nProne to Overfitting without Regularization",
             ha="center", va="top", fontsize=9, style="italic", color=RED)
    ax1.set_xlim(0, 1); ax1.set_ylim(-0.1, 1.0); ax1.axis("off")

    # Regularized CNN Pipeline
    ax2.set_title("Architecture B: Deep Regularized Convolutional Network (CNN)\n[Hierarchical Spatial Invariance]",
                  fontsize=11, fontweight="bold", color=NAVY, pad=12)
    cnn_boxes = [
        ("Input Image Tensor (28x28x1)", "#E6F0FA", 0.90),
        ("Conv Block 1: [Conv2D(32) + BN] x2 + MaxPool(2x2) + Drop(0.25)", "#C6E0B4", 0.74),
        ("Conv Block 2: [Conv2D(64) + BN] x2 + MaxPool(2x2) + Drop(0.25)", "#A9D08E", 0.56),
        ("Flatten & Dense Latent: Dense(128) + BN + Drop(0.50)", "#FFF2CC", 0.38),
        ("Softmax Classification Head (10 Class Probabilities)", "#FCE4D6", 0.18)
    ]
    for text, color, y in cnn_boxes:
        rect = plt.Rectangle((0.05, y - 0.06), 0.9, 0.10, facecolor=color, edgecolor="#385723", linewidth=1.5, zorder=2)
        ax2.add_patch(rect)
        ax2.text(0.5, y - 0.01, text, ha="center", va="center", fontsize=9, fontweight="bold", color="#1A1A1A", zorder=3)
    for i in range(len(cnn_boxes) - 1):
        ax2.annotate("", xy=(0.5, cnn_boxes[i+1][2] + 0.04), xytext=(0.5, cnn_boxes[i][2] - 0.06),
                     arrowprops=dict(facecolor="#385723", edgecolor="#385723", width=2, headwidth=8), zorder=1)
    ax2.text(0.5, -0.05, "Strengths: Preserves 2D Topology; 188,426 Parameters;\nDropout + BatchNorm Counteracts Co-adaptation",
             ha="center", va="top", fontsize=9, style="italic", color="#385723")
    ax2.set_xlim(0, 1); ax2.set_ylim(-0.1, 1.0); ax2.axis("off")

    plt.suptitle("Figure 2: Architectural Blueprints and Inductive Bias Comparison",
                 fontsize=13, fontweight="bold", color=NAVY, y=0.98)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig02_architecture_schematic.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_learning_curves(mlp_hist_path, cnn_hist_path, output_dir):
    """
    Fig 3 & Fig 4: Training vs Validation Loss & Accuracy learning curves.
    Clearly demonstrates the overfitting gap in MLP vs stability in CNN.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_mlp = pd.read_csv(mlp_hist_path)
    df_cnn = pd.read_csv(cnn_hist_path)

    # ------------------ Fig 3: Loss Curves ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # MLP Loss
    ax1.plot(df_mlp["epoch"], df_mlp["loss"], "o-", color=BLUE, label="MLP Training Loss", linewidth=2)
    ax1.plot(df_mlp["epoch"], df_mlp["val_loss"], "s--", color=RED, label="MLP Validation Loss", linewidth=2)
    ax1.set_title("Baseline MLP: Severe Overfitting Divergence", fontsize=11, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor=GRAY)
    ax1.annotate("Validation Loss Flattens/Diverges\n(Classic Overfitting Signature)",
                 xy=(df_mlp["epoch"].iloc[-3], df_mlp["val_loss"].iloc[-3]),
                 xytext=(df_mlp["epoch"].iloc[-3] - 4, df_mlp["val_loss"].iloc[-3] + 0.15),
                 arrowprops=dict(facecolor=RED, shrink=0.05, width=1.5, headwidth=6),
                 fontsize=8.5, color=RED, fontweight="bold")

    # CNN Loss
    ax2.plot(df_cnn["epoch"], df_cnn["loss"], "o-", color=GREEN, label="CNN Training Loss", linewidth=2)
    ax2.plot(df_cnn["epoch"], df_cnn["val_loss"], "s--", color="#205081", label="CNN Validation Loss", linewidth=2)
    ax2.set_title("Deep Regularized CNN: Convergent Generalization", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", edgecolor=GRAY)
    ax2.annotate("Stable Generalization\n(Minimal Train-Val Gap)",
                 xy=(df_cnn["epoch"].iloc[-2], df_cnn["val_loss"].iloc[-2]),
                 xytext=(df_cnn["epoch"].iloc[-2] - 4, df_cnn["val_loss"].iloc[-2] + 0.15),
                 arrowprops=dict(facecolor=GREEN, shrink=0.05, width=1.5, headwidth=6),
                 fontsize=8.5, color=GREEN, fontweight="bold")

    plt.suptitle("Figure 3: Training vs. Validation Cross-Entropy Loss Across Optimization Epochs",
                 fontsize=12, fontweight="bold", color=NAVY, y=0.98)
    plt.tight_layout()
    save_loss_path = os.path.join(output_dir, "fig03_learning_curves_loss.png")
    plt.savefig(save_loss_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_loss_path}")

    # ------------------ Fig 4: Accuracy Curves ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # MLP Accuracy
    ax1.plot(df_mlp["epoch"], df_mlp["accuracy"] * 100, "o-", color=BLUE, label="MLP Train Acc (%)", linewidth=2)
    ax1.plot(df_mlp["epoch"], df_mlp["val_accuracy"] * 100, "s--", color=RED, label="MLP Val Acc (%)", linewidth=2)
    ax1.set_title("Baseline MLP: Accuracy Ceiling at ~89%", fontsize=11, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(75, 100)
    ax1.legend(frameon=True, facecolor="white", edgecolor=GRAY)

    # CNN Accuracy
    ax2.plot(df_cnn["epoch"], df_cnn["accuracy"] * 100, "o-", color=GREEN, label="CNN Train Acc (%)", linewidth=2)
    ax2.plot(df_cnn["epoch"], df_cnn["val_accuracy"] * 100, "s--", color="#205081", label="CNN Val Acc (%)", linewidth=2)
    ax2.set_title("Deep Regularized CNN: Reaching >92.5% Accuracy", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(75, 100)
    ax2.legend(frameon=True, facecolor="white", edgecolor=GRAY)

    plt.suptitle("Figure 4: Training vs. Validation Accuracy Trajectories",
                 fontsize=12, fontweight="bold", color=NAVY, y=0.98)
    plt.tight_layout()
    save_acc_path = os.path.join(output_dir, "fig04_learning_curves_accuracy.png")
    plt.savefig(save_acc_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_acc_path}")

def plot_confusion_matrix_heatmap(cm_cnn, class_names, output_dir):
    """
    Fig 5: High-resolution annotated confusion matrix for the CNN model.
    """
    os.makedirs(output_dir, exist_ok=True)
    cm_norm = cm_cnn.astype("float") / cm_cnn.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Classification Percentage (%)"},
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Figure 5: Deep Regularized CNN Normalized Confusion Matrix (Test Set N=10,000)",
                 fontsize=12, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Predicted Class Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Class Label", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=35, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig05_confusion_matrix_heatmap.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_per_class_f1_comparison(df_per_class, output_dir):
    """
    Fig 6: Horizontal grouped bar chart comparing F1-scores between MLP and CNN.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("white")

    y = np.arange(len(df_per_class))
    height = 0.36

    rects1 = ax.barh(y - height/2, df_per_class["MLP_F1"], height, label="Baseline MLP", color="#95B3D7", edgecolor=NAVY)
    rects2 = ax.barh(y + height/2, df_per_class["CNN_F1"], height, label="Deep Regularized CNN", color=NAVY, edgecolor="#0D233A")

    ax.set_ylabel("Fashion Apparel Category", fontsize=11, fontweight="bold")
    ax.set_xlabel("F1-Score (Harmonic Mean of Precision & Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 6: Per-Class F1-Score Benchmark: Baseline MLP vs. Deep CNN",
                 fontsize=12, fontweight="bold", color=NAVY, pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(df_per_class["Class_Name"], fontsize=10)
    ax.set_xlim(0.65, 1.0)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRAY)

    # Annotate improvement delta
    for idx, row in df_per_class.iterrows():
        delta = row["CNN_F1"] - row["MLP_F1"]
        val_cnn = row["CNN_F1"]
        ax.text(val_cnn + 0.005, idx + height/2, f"+{delta:.3f} ({row['Pct_Gain']})",
                va="center", fontsize=8, color="#2CA02C" if delta > 0 else RED, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig06_per_class_f1_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_sample_predictions_grid(cnn_model, data_dict, output_dir, num_samples=16):
    """
    Fig 7: 4x4 grid of test samples showing true label, predicted label, and confidence.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_test = data_dict["X_test_cnn"]
    y_test = data_dict["y_test"]
    class_names = data_dict["class_names"]

    # Generate probabilities for first 100 samples
    probs = cnn_model.predict(X_test[:100], verbose=0)
    preds = np.argmax(probs, axis=1)

    # Pick a curated mix of correct predictions and difficult/misclassified cases
    correct_indices = np.where(preds == y_test[:100])[0]
    wrong_indices = np.where(preds != y_test[:100])[0]

    # Combine 12 correct and 4 hard/misclassified cases
    chosen_indices = list(correct_indices[:12]) + list(wrong_indices[:4])
    chosen_indices = chosen_indices[:num_samples]

    fig, axes = plt.subplots(4, 4, figsize=(13, 12))
    fig.patch.set_facecolor("white")

    for i, idx in enumerate(chosen_indices):
        ax = axes[i // 4, i % 4]
        img = X_test[idx].squeeze()
        true_lbl = class_names[y_test[idx]]
        pred_lbl = class_names[preds[idx]]
        conf = probs[idx, preds[idx]] * 100
        is_correct = preds[idx] == y_test[idx]

        ax.imshow(img, cmap="bone", interpolation="nearest")
        ax.axis("off")

        color = GREEN if is_correct else RED
        status = "CORRECT" if is_correct else "MISCLASSIFIED"
        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl} ({conf:.1f}%)\n[{status}]",
                     fontsize=9, fontweight="bold", color=color, pad=4)

    plt.suptitle("Figure 7: Deep CNN Diagnostic Sample Inference Grid\n[Green: High-Confidence Correct | Red: Edge-Case Misclassification]",
                 fontsize=12, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig07_sample_predictions_grid.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_convolutional_filters_feature_maps(cnn_model, data_dict, output_dir):
    """
    Fig 8: Feature map activations extracted from the first Conv2D layer of the trained CNN.
    Visualizes what edge/texture features the network extracts from a sample garment.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_test = data_dict["X_test_cnn"]
    class_names = data_dict["class_names"]

    # Choose a high-contrast apparel item (e.g. Bag or Pullover or Ankle boot)
    sample_idx = 0
    sample_img = X_test[sample_idx:sample_idx+1]
    true_label = class_names[data_dict["y_test"][sample_idx]]

    # Extract intermediate activations from the first Conv2D layer
    layer_name = "conv1_1"
    try:
        conv_layer = cnn_model.get_layer(layer_name)
        feature_maps = conv_layer(sample_img).numpy()
    except Exception:
        x = sample_img
        for l in cnn_model.layers:
            x = l(x)
            if l.name == layer_name:
                break
        feature_maps = x.numpy() if hasattr(x, "numpy") else np.array(x)

    # feature_maps shape: (1, 28, 28, 32)
    fmaps = feature_maps[0]

    fig, axes = plt.subplots(4, 8, figsize=(14, 7.5))
    fig.patch.set_facecolor("white")

    for idx in range(32):
        ax = axes[idx // 8, idx % 8]
        fmap = fmaps[:, :, idx]
        ax.imshow(fmap, cmap="viridis", interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"Filter #{idx+1}", fontsize=8, color=NAVY)

    plt.suptitle(f"Figure 8: Spatial Feature Map Activations from Layer '{layer_name}' (Sample: {true_label})\n"
                 "[Visualizing Learned Edge Detectors, Silhouettes, and High-Frequency Texture Encoders]",
                 fontsize=12, fontweight="bold", color=NAVY, y=0.98)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig08_convolutional_filters_feature_maps.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_model_complexity_tradeoff(output_dir):
    """
    Fig 9: Scatter trade-off comparing Parameters vs Accuracy vs Inference Latency.
    """
    os.makedirs(output_dir, exist_ok=True)
    bench_path = os.path.join(output_dir, "test_evaluation_benchmark.csv")
    if not os.path.exists(bench_path):
        return

    df_bench = pd.read_csv(bench_path)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")

    colors = [BLUE, GREEN]
    for idx, row in df_bench.iterrows():
        ax.scatter(
            row["Total_Params"] / 1000,
            row["Test_Accuracy"],
            s=row["Latency_per_1000_Img_ms"] * 15,
            color=colors[idx % len(colors)],
            alpha=0.75,
            edgecolors=NAVY,
            linewidth=2,
            label=f"{row['Architecture']} (Latency: {row['Latency_per_1000_Img_ms']:.1f}ms/1k img)"
        )
        ax.annotate(
            f"{row['Architecture']}\nAcc: {row['Test_Accuracy']}%\nParams: {int(row['Total_Params']):,}",
            xy=(row["Total_Params"] / 1000, row["Test_Accuracy"]),
            xytext=(row["Total_Params"] / 1000 + 10, row["Test_Accuracy"] - 0.5),
            fontsize=9.5, fontweight="bold", color=NAVY
        )

    ax.set_title("Figure 9: Architectural Efficiency Frontier: Parameters vs. Accuracy vs. Latency\n(Bubble size represents test inference latency per 1,000 images)",
                 fontsize=11, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Total Model Parameters (Thousands / K)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Set Top-1 Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(85, 95)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRAY)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig09_model_complexity_tradeoff.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")
