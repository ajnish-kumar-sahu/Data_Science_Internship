# Week 5 Task: Deep Learning Application in Data Science
### Comparative Fashion Apparel Classification & Overfitting Mitigation Using Deep Convolutional Neural Networks (CNN) vs. Baseline Multi-Layer Perceptrons (MLP)

**Internship:** Virtual Data Science with Python Trainee | **Yuva Intern**  
**Author:** Ajnish Kumar | **Roll No:** 241809046713  
**Degree / Institution:** Bachelor of Computer Applications (BCA) | Vinoba Bhave University, Hazaribag  
**Date:** September 2026  

---

## 📌 Project Overview
This repository contains the complete implementation, production codebase, interactive notebooks, visualizations, and publication-grade technical report for the **Week 5 Internship Task: Deep Learning Application in Data Science**.

In this investigation, we design, train, regularize, and statistically evaluate deep neural networks applied to multi-class computer vision on the benchmark **Fashion-MNIST dataset** (70,000 $28 \times 28$ grayscale images across 10 fashion apparel categories). 

We empirically contrast two fundamentally different architectural paradigms:
1. **Baseline Multi-Layer Perceptron (MLP)**: An unregularized 3-layer fully connected network (235,146 parameters) that collapses spatial geometry into flat 1D vectors, demonstrating rapid overfitting and validation divergence.
2. **Deep Regularized Convolutional Neural Network (CNN)**: A multi-block ConvNet (188,426 parameters) featuring stacked $3 \times 3$ convolutions, Batch Normalization, spatial Max Pooling, and staged Dropout (0.25, 0.50), achieving state-of-the-art generalization and spatial invariance.

---

## 📂 Repository Structure

```
Week5_Deep_Learning/
├── README.md                           # Comprehensive documentation & execution guide
├── week5_deep_learning.py              # Master pipeline runner
├── generate_report.py                  # Generates formal Word document report (.docx)
├── write_notebook.py                   # Generates interactive Jupyter Notebook (.ipynb)
├── data/                               # Local dataset cache and archives
├── src/                                # Modular source code
│   ├── __init__.py
│   ├── dataset.py                      # Data ingestion, normalization, and partitioning
│   ├── models.py                       # MLP & CNN architectural definitions
│   ├── train.py                        # Model training with EarlyStopping & LR scheduling
│   ├── evaluate.py                     # Test set evaluation, metrics, and error mining
│   └── visualizer.py                   # High-res plotting suite (loss/acc curves, CM, filters)
├── output/                             # Exported CSV ledgers and serialized models
│   ├── dataset_split_summary.csv
│   ├── model_architectures_comparison.csv
│   ├── training_convergence_summary.csv
│   ├── test_evaluation_benchmark.csv
│   ├── classification_report_cnn.csv
│   ├── per_class_f1_comparison.csv
│   ├── top_misclassified_pairs_cnn.csv
│   └── models/                         # Serialized Keras models (.keras)
├── visualizations/                     # High-resolution figures (300 DPI)
│   ├── fig01_dataset_sample_gallery.png
│   ├── fig02_architecture_schematic.png
│   ├── fig03_learning_curves_loss.png
│   ├── fig04_learning_curves_accuracy.png
│   ├── fig05_confusion_matrix_heatmap.png
│   ├── fig06_per_class_f1_comparison.png
│   ├── fig07_sample_predictions_grid.png
│   ├── fig08_convolutional_filters_feature_maps.png
│   └── fig09_model_complexity_tradeoff.png
├── notebooks/
│   └── Week5_Deep_Learning.ipynb       # Interactive Jupyter Notebook
└── report/
    └── Week5_Deep_Learning_Report.docx # Formal publication-grade Word report
```

---

## 🚀 How to Run the Pipeline

Ensure your environment has Python 3.10+ and the required packages installed:
```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn python-docx
```

### 1. Execute the Master Pipeline
To run the complete data loading, model training, evaluation, and visualization suite:
```bash
python week5_deep_learning.py
```

### 2. Generate Interactive Jupyter Notebook
```bash
python write_notebook.py
```

### 3. Generate Formal Publication-Grade Word Report (.docx)
```bash
python generate_report.py
```

---

## 📊 Summary of Benchmark Results

| Architecture | Total Parameters | Test Loss | Test Accuracy | Top-2 Accuracy | Generalization Gap | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline MLP** | 235,146 | 0.3542 | 88.92% | 96.10% | +0.218 (Overfitting) | 0.22 ms / img |
| **Deep Regularized CNN** | **188,426** | **0.2184** | **92.54%** | **98.45%** | **+0.038 (Optimal)** | 0.38 ms / img |

### Key Insights:
- **Spatial Inductive Bias**: Convolutional filters and spatial pooling allow the CNN to achieve **+3.62% higher accuracy** while requiring **20% fewer parameters** than the MLP.
- **Overfitting Eradication**: Batch Normalization and Staged Dropout compressed the training-validation generalization gap from 0.218 down to 0.038.
- **Explainability**: Intermediate filter activations demonstrate that early convolutional layers autonomously learn edge detectors, contrast inverters, and garment silhouettes.
