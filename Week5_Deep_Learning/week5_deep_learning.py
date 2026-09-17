"""
week5_deep_learning.py
Master pipeline for Week 5 Internship Task: Deep Learning Application in Data Science.
Comparative Fashion Apparel Classification & Overfitting Mitigation Using
Deep Convolutional Neural Networks (CNN) vs. Baseline Multi-Layer Perceptron (MLP).

Author: Ajnish Kumar | Roll No: 241809046713
Degree: BCA, Vinoba Bhave University, Hazaribag
Internship: Yuva Intern - Virtual Data Science with Python Trainee
Date: September 2026
"""

import os
import sys
import subprocess
import tensorflow as tf

# Add current folder and src to system path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.dataset import load_and_preprocess_fashion_mnist, generate_dataset_summary
from src.models import build_baseline_mlp, build_regularized_cnn, export_model_architectures
from src.train import train_models
from src.evaluate import evaluate_models
from src.visualizer import (
    plot_dataset_sample_gallery,
    plot_architecture_schematic,
    plot_learning_curves,
    plot_confusion_matrix_heatmap,
    plot_per_class_f1_comparison,
    plot_sample_predictions_grid,
    plot_convolutional_filters_feature_maps,
    plot_model_complexity_tradeoff
)

def main():
    print("=" * 80)
    print(" WEEK 5: DEEP LEARNING APPLICATION IN DATA SCIENCE")
    print(" Fashion-MNIST Computer Vision Classification & Overfitting Mitigation")
    print(" Author: Ajnish Kumar | Roll No: 241809046713 | Vinoba Bhave University")
    print("=" * 80)

    # Setup directories
    data_dir = os.path.join(CURRENT_DIR, "data")
    output_dir = os.path.join(CURRENT_DIR, "output")
    viz_dir = os.path.join(CURRENT_DIR, "visualizations")
    report_dir = os.path.join(CURRENT_DIR, "report")
    notebooks_dir = os.path.join(CURRENT_DIR, "notebooks")

    for d in [data_dir, output_dir, viz_dir, report_dir, notebooks_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Dataset Loading & Preprocessing
    print("\n--- PHASE 1: DATASET INGESTION & HYGIENE ---")
    data_dict = load_and_preprocess_fashion_mnist(train_samples=20000, val_samples=4000, test_samples=5000, random_state=42)
    generate_dataset_summary(data_dict, output_dir)
    plot_dataset_sample_gallery(data_dict, viz_dir)

    # 2. Architectural Blueprint & Model Construction
    print("\n--- PHASE 2: ARCHITECTURE DESIGN & INDUCTIVE BIASES ---")
    mlp_model = build_baseline_mlp(input_shape=(784,), num_classes=10)
    cnn_model = build_regularized_cnn(input_shape=(28, 28, 1), num_classes=10)
    export_model_architectures(mlp_model, cnn_model, output_dir)
    plot_architecture_schematic(viz_dir)

    # 3. Model Training & Optimization Dynamics
    print("\n--- PHASE 3: TRAINING DYNAMICS & CONVERGENCE MONITORING ---")
    # Training for 10 epochs captures the full divergence while keeping execution swift
    train_results = train_models(mlp_model, cnn_model, data_dict, output_dir, epochs=10, batch_size=128)

    mlp_hist_csv = os.path.join(output_dir, "mlp_training_history.csv")
    cnn_hist_csv = os.path.join(output_dir, "cnn_training_history.csv")
    plot_learning_curves(mlp_hist_csv, cnn_hist_csv, viz_dir)

    # 4. Statistical Evaluation & Diagnostics
    print("\n--- PHASE 4: TEST SET EVALUATION & ERROR MINING ---")
    eval_results = evaluate_models(mlp_model, cnn_model, data_dict, output_dir)

    plot_confusion_matrix_heatmap(eval_results["cm_cnn"], data_dict["class_names"], viz_dir)
    plot_per_class_f1_comparison(eval_results["df_per_class"], viz_dir)
    plot_sample_predictions_grid(cnn_model, data_dict, viz_dir)
    plot_convolutional_filters_feature_maps(cnn_model, data_dict, viz_dir)
    plot_model_complexity_tradeoff(output_dir)
    # Move fig09 to viz_dir if saved in output_dir
    fig9_old = os.path.join(output_dir, "fig09_model_complexity_tradeoff.png")
    fig9_new = os.path.join(viz_dir, "fig09_model_complexity_tradeoff.png")
    if os.path.exists(fig9_old) and not os.path.exists(fig9_new):
        os.rename(fig9_old, fig9_new)

    print("\n" + "=" * 80)
    print(" DEEP LEARNING PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
