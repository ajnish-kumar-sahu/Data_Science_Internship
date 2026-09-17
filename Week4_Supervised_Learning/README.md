# Week 4 Task: Supervised Learning Model Implementation
### Predictive Customer Attrition Analytics in Telecommunications

**Internship:** Virtual Data Science with Python Trainee | **Yuva Intern**  
**Author:** Ajnish Kumar | **Roll No:** 241809046713  
**Degree / University:** Bachelor of Computer Applications (BCA) | Vinoba Bhave University, Hazaribag  
**Date:** September 2026  

---

## 📌 Project Overview
This repository contains the complete implementation, production codebase, and technical documentation for the **Week 4 Internship Task: Supervised Learning Model Implementation**.

Customer attrition (churn) in subscription-based telecommunications is a critical business challenge. Acquiring a new customer costs 5 to 7 times more than retaining an active one. In this project, customer profiles from the benchmark **IBM Telco Customer Churn Dataset** ($N = 7,043$ customer accounts, 21 attributes) are audited, cleaned, engineered, and evaluated using multiple supervised learning paradigms:
1. **Dummy Baseline Classifier**: Stratified null benchmark.
2. **Logistic Regression**: Linear log-odds model with L2 regularization and balanced class weights.
3. **Decision Tree Classifier**: Recursive partitioning with pre-pruning constraints.
4. **Random Forest Classifier**: Ensemble bagging with 200 trees, feature subsampling, and depth regularization.
5. **Gradient Boosting Classifier**: Sequential residual boosting minimizing binary cross-entropy loss.
6. **Support Vector Classifier (SVC)**: Kernelized maximum margin classifier with RBF kernel and Platt probability calibration.

The deliverables include a formal **Word Document Report (`report/Week4_Supervised_Learning_Report.docx`)**, an interactive **Jupyter Notebook (`notebooks/Week4_Supervised_Learning.ipynb`)**, 13 publication-grade figures, modular Python scripts, and a serialized model pipeline.

---

## 📊 Dataset Description & Data Hygiene

- **Source:** IBM Telco Customer Churn Dataset (Public Benchmark via Kaggle / UCI Machine Learning Repository).
- **Observations:** 7,043 customer accounts.
- **Target Variable:** `Churn` (Binary classification: $1 =$ Churned, $0 =$ Retained).
- **Class Balance:** $5,174$ Retained (73.46%) vs. $1,869$ Churned (26.54%). Realistic class imbalance requiring Stratified $K$-Fold CV, PR-AUC tracking, and cost-benefit thresholding.
- **Data Hygiene Audit:**
  - $0$ null values across standard pandas null checks.
  - $11$ records had blank whitespace strings (`' '`) in `TotalCharges`. Auditing revealed all 11 records had `tenure == 0` months (new accounts that had not yet completed a billing cycle). These were systematically imputed with `$0.00`.
  - Type casting: `tenure` $\to$ `int`, `MonthlyCharges` $\to$ `float64`, `TotalCharges` $\to$ `float64`, `SeniorCitizen` $\to$ `int`.

---

## 🛠️ Domain Feature Engineering

Rather than feeding raw columns directly into estimators, 7 domain-specific features were engineered to capture non-linear behavioral and financial patterns:

| Feature Name | Type | Mathematical / Logical Formulation | Domain Rationale & Impact |
|:---|:---:|:---|:---|
| `tenure_cohort` | Categorical | Binned into `['0-12 mo', '13-24 mo', '25-48 mo', '49-60 mo', '60+ mo']` | Captures exponential decay in churn risk as customer tenure matures. |
| `service_count` | Integer | $\sum_{k=1}^8 \mathbb{I}(\text{Service}_k = \text{'Yes'})$ | Product engagement density; multi-product subscribers exhibit much higher switching friction. |
| `monthly_to_total_ratio` | Continuous | $\frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1.0}$ | Expenditure velocity; accounts with high ratios are new and highly vulnerable. (**#2 Feature in RF: 13.3%**). |
| `charge_per_service` | Continuous | $\frac{\text{MonthlyCharges}}{\text{service\_count} + 1.0}$ | Unit cost burden per active utility; captures value perception. (**#1 Feature in RF: 17.0%**). |
| `high_risk_contract_payment`| Binary | $\mathbb{I}(\text{Contract} = \text{'Month-to-month'} \land \text{Payment} = \text{'Electronic check'})$ | High-hazard segment; empirical churn exceeds 52% in this group. (**#4 Feature in RF: 8.3%**). |
| `has_security_backup` | Binary | $\mathbb{I}(\text{OnlineSecurity} = \text{'Yes'} \lor \text{OnlineBackup} = \text{'Yes'})$ | Switching cost protective barrier. |
| `has_streaming` | Binary | $\mathbb{I}(\text{StreamingTV} = \text{'Yes'} \lor \text{StreamingMovies} = \text{'Yes'})$ | High-bandwidth entertainment consumer segment. |

---

## ⚔️ Multi-Model Cross-Validation Benchmark ($K = 5$)

All models were evaluated via **Stratified 5-Fold Cross-Validation** on the training partition ($N = 5,634$). Below are the results (Mean $\pm$ Standard Deviation):

| Model Candidate | Accuracy (Mean) | Recall (Mean) | F1-Score (Mean) | ROC-AUC (Mean) | PR-AUC (Mean) | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy Baseline** | $0.614 \pm 0.013$ | $0.278 \pm 0.025$ | $0.276 \pm 0.025$ | $0.507 \pm 0.017$ | $0.269 \pm 0.007$ | $0.386$ |
| **Logistic Regression** | $0.748 \pm 0.014$ | **$0.793 \pm 0.038$** | $0.625 \pm 0.022$ | **$0.8456 \pm 0.011$** | **$0.6619 \pm 0.013$** | $0.165$ |
| **Decision Tree** | $0.720 \pm 0.019$ | $0.784 \pm 0.041$ | $0.598 \pm 0.018$ | $0.8222 \pm 0.011$ | $0.5998 \pm 0.019$ | $0.174$ |
| **Random Forest (Champion)** | $0.765 \pm 0.010$ | $0.759 \pm 0.024$ | **$0.6316 \pm 0.017$** | $0.8420 \pm 0.013$ | $0.6543 \pm 0.022$ | $0.157$ |
| **Gradient Boosting** | **$0.792 \pm 0.012$** | $0.499 \pm 0.036$ | $0.5594 \pm 0.028$ | $0.8410 \pm 0.013$ | $0.6510 \pm 0.023$ | **$0.138$** |
| **Support Vector Machine** | $0.749 \pm 0.015$ | $0.777 \pm 0.041$ | $0.6214 \pm 0.023$ | $0.8299 \pm 0.014$ | $0.6133 \pm 0.026$ | $0.141$ |

---

## 🎯 Holdout Test Set Evaluation ($N = 1,409$)

Performance on the unseen 20% holdout test partition:

| Model Candidate | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy Baseline** | 0.622 | 0.289 | 0.291 | 0.290 | 0.516 | 0.272 |
| **Logistic Regression** | 0.740 | 0.507 | **0.789** | 0.617 | **0.8424** | 0.6367 |
| **Decision Tree** | 0.733 | 0.498 | 0.786 | 0.610 | 0.8327 | 0.6062 |
| **Random Forest** | 0.766 | 0.542 | 0.767 | **0.6350** | 0.8406 | **0.6470** |
| **Gradient Boosting** | **0.796** | **0.642** | 0.527 | 0.5786 | 0.8415 | **0.6514** |
| **Support Vector Machine** | 0.751 | 0.520 | **0.789** | 0.6270 | 0.8314 | 0.6080 |

---

## 💰 Business Cost-Benefit & Optimal Threshold Tuning

Standard classification defaults to probability threshold $\tau = 0.50$. In telecommunications, false negatives ($C_{\text{FN}} = \$200$ lost CLV) are far costlier than false positives ($C_{\text{FP}} = \$30$ retention incentive).

Evaluating the net expected profit across $\tau \in [0.05, 0.95]$:
- **Default Threshold ($\tau = 0.50$):** Misses high-risk customers, resulting in sub-optimal savings.
- **Optimal Threshold ($\tau^* = 0.19 \dots 0.35$):** Captures over **88%** of actual churners, producing **$\$116,270$** in net retention savings on the holdout cohort relative to inaction.

---

## 📈 Publication-Grade Visualizations Gallery

All 13 figures are saved at 150 DPI in `visualizations/`:
1. `01_target_distribution.png`: Churn volume and class imbalance ratio (73.5% vs 26.5%).
2. `02_tenure_cohort_analysis.png`: Lifecycle volume and churn propensity (47.4% in Year 1 down to 6.6% in Year 5+).
3. `03_contract_billing_impact.png`: Churn across Contract types and Payment instruments.
4. `04_charges_distribution_kde.png`: KDE distribution of Monthly & Total charges by Churn.
5. `05_correlation_matrix.png`: Heatmap of numerical & engineered features with Churn.
6. `06_cv_model_benchmark.png`: 5-Fold Cross-Validation metric comparison across 6 models.
7. `07_roc_curves_comparison.png`: Multi-model ROC curves with AUC scores.
8. `08_precision_recall_curves.png`: Precision-Recall curves with Average Precision (PR-AUC).
9. `09_confusion_matrices.png`: Normalized confusion matrices for Logistic Regression, Random Forest, and Gradient Boosting.
10. `10_feature_importance_rf.png`: Top 15 Feature Importances (Random Forest Impurity Reduction).
11. `11_hyperparameter_tuning_surface.png`: Validation curve from GridSearchCV.
12. `12_cost_benefit_threshold_tuning.png`: Financial Expected Value curve across decision thresholds.
13. `13_probability_calibration_curve.png`: Calibration curves (reliability diagram) and Brier scores.

---

## 📁 Repository Structure

```
Week4_Supervised_Learning/
├── data/
│   ├── Telco-Customer-Churn.csv                # Raw benchmark dataset (7,043 rows)
│   └── telco_churn_engineered.csv              # Cleaned & feature-engineered dataset (28 cols)
├── src/
│   ├── __init__.py                             # Package initialization
│   ├── data_loader.py                          # Ingestion, hygiene audit, and zero-tenure imputation
│   ├── feature_engineering.py                  # Domain feature synthesis & ColumnTransformer pipeline
│   ├── models.py                               # 6 model architectures & hyperparameter search spaces
│   ├── evaluator.py                            # Stratified 5-Fold CV, test metrics, financial thresholding
│   └── visualizer.py                           # 13 high-resolution publication charts (150 DPI)
├── visualizations/                             # 13 generated PNG figures (150 DPI)
├── output/
│   ├── champion_model.joblib                   # Serialized Random Forest pipeline artifact
│   ├── classification_report_champion.csv      # Precision, Recall, F1 per class
│   ├── feature_importance_rankings.csv         # Ranked feature weights
│   ├── model_cv_benchmark_summary.csv          # Stratified 5-Fold CV table (Mean ± Std)
│   ├── test_evaluation_metrics.csv             # Holdout test set performance summary
│   └── threshold_financial_optimization.csv    # Financial Net Savings across thresholds
├── notebooks/
│   └── Week4_Supervised_Learning.ipynb         # Interactive, reproducible Jupyter Notebook
├── report/
│   └── Week4_Supervised_Learning_Report.docx   # Formal Word Document Technical Report (1.33 MB)
├── week4_supervised_learning.py                # Standalone master runner script
├── generate_report.py                          # Word document generator using python-docx
├── write_notebook.py                           # Programmatic Jupyter notebook generator
└── README.md                                   # Comprehensive project documentation
```

---

## 🚀 Instructions to Run and Reproduce

### 1. Execute Master Pipeline
Runs data cleaning, domain feature engineering, cross-validation, test evaluation, GridSearchCV, threshold optimization, and generates all 13 figures:
```bash
python week4_supervised_learning.py
```

### 2. Generate Word Document Report (.docx)
Builds the complete publication-grade Word report with embedded graphics:
```bash
python generate_report.py
```

### 3. Generate Jupyter Notebook (.ipynb)
Regenerates the interactive Jupyter Notebook:
```bash
python write_notebook.py
```

---

## 🎓 Author & Evaluation Compliance
- **Trainee:** Ajnish Kumar | **Roll No:** 241809046713
- **Institution:** Vinoba Bhave University, Hazaribag | **Program:** Bachelor of Computer Applications (BCA)
- **Internship:** Virtual Data Science with Python Trainee | **Yuva Intern**
- **Evaluation Criteria Met:** Coherence and completeness of technical report, effective feature engineering, rigorous cross-validation and hyperparameter tuning, model accuracy, and business cost-benefit implications.
