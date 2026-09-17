# Week 6: Integrative Capstone Project and Evaluation
## Enterprise Customer Behavioral Analytics, Predictive Attrition Modeling, and Lifetime Value Optimization

**Internship:** Virtual Data Science with Python Trainee | **Yuva Intern**  
**Author:** Ajnish Kumar | **Roll No:** 241809046713  
**Degree / Institution:** Bachelor of Computer Applications (BCA) | Vinoba Bhave University, Hazaribag  
**Date:** September 2026  

---

## Executive Overview
This Capstone Project represents the integrative culmination of the Yuva Intern Data Science with Python program. We establish an end-to-end data science architecture addressing enterprise subscription churn and customer value optimization:

1. **Data Acquisition & Hygiene Auditing**: Synthesized and audited 5,000 subscriber records across 22 operational, transactional, and financial features with missing value imputation and Tukey IQR outlier boundaries.
2. **Exploratory Diagnostics & Inferential Testing**: Conducted univariate profiling, bivariate cross-tabulations, and Welch's two-sample independent $t$-tests ($p < 10^{-15}$).
3. **Unsupervised Behavioral Segmentation**: Validated K-Means partitions ($k \in [2, 7]$) via Inertia and Silhouette optimization ($k=4$), projected latent clusters via 3-component PCA, and profiled 4 distinct enterprise personas.
4. **Supervised Churn Risk Modeling**: Benchmarked 5 classifiers (Dummy Baseline, Logistic Regression, Random Forest, Gradient Boosting, Deep MLP) using 5-Fold Stratified Cross-Validation and ROC-AUC / PR-AUC test evaluation. Optimized decision boundaries using empirical financial payoff matrices.
5. **Supervised CLV Continuous Forecasting**: Benchmarked 4 regression models (Dummy, Ridge, Random Forest, Gradient Boosting) achieving $R^2 = 0.942$ and RMSE of $124.30.
6. **Integrative Synthesis Engine**: Intersected unsupervised personas with supervised risk scores to formulate the Executive Value-at-Risk Matrix and simulated a 640% ROI targeted retention campaign.

---

## Repository Structure

```
Week6_Capstone_Project/
├── data/                               # Cached raw and cleaned customer datasets
│   └── raw_enterprise_customer_profiles.csv
├── src/                                # Modular enterprise pipeline components
│   ├── __init__.py
│   ├── data_loader.py                  # Ingestion, synthesis, imputation & scaling
│   ├── eda.py                          # Univariate profiling & Welch's t-tests
│   ├── clustering.py                   # K-Means, Elbow/Silhouette & PCA projection
│   ├── supervised_models.py            # Churn classification & CLV regression
│   ├── integrator.py                   # Value-at-Risk synthesis & ROI simulation
│   └── visualizer.py                   # Publication-grade 12-figure visualization suite
├── output/                             # Generated CSV ledgers & benchmark tables
│   ├── data_cleaning_audit.csv
│   ├── outlier_iqr_audit.csv
│   ├── eda_numerical_summary.csv
│   ├── eda_bivariate_hypothesis_tests.csv
│   ├── clustering_k_evaluation.csv
│   ├── customer_persona_profiles.csv
│   ├── pca_explained_variance.csv
│   ├── churn_cv_benchmark_summary.csv
│   ├── churn_test_evaluation_metrics.csv
│   ├── threshold_optimization_churn.csv
│   ├── clv_cv_benchmark_summary.csv
│   ├── clv_test_evaluation_metrics.csv
│   ├── capital_value_at_risk_summary.csv
│   └── retention_campaign_roi_simulation.csv
├── visualizations/                     # 12 high-resolution publication figures
│   ├── fig01_eda_feature_distributions.png
│   ├── fig02_bivariate_churn_interactions.png
│   ├── fig03_correlation_matrix_heatmap.png
│   ├── fig04_clustering_elbow_silhouette.png
│   ├── fig05_pca_persona_clusters_scatter.png
│   ├── fig06_persona_attribute_comparison.png
│   ├── fig07_supervised_churn_roc_curves.png
│   ├── fig08_churn_confusion_matrix_pr_curve.png
│   ├── fig09_clv_regression_actual_vs_predicted.png
│   ├── fig10_feature_importance_dual_benchmark.png
│   ├── fig11_executive_value_at_risk_matrix.png
│   └── fig12_retention_campaign_financial_curve.png
├── notebooks/                          # Interactive exploratory & modeling notebook
│   └── Week6_Capstone_Project.ipynb
├── report/                             # Formal Word document deliverable
│   └── Week6_Capstone_Project_Report.docx
├── week6_capstone_pipeline.py          # Master execution script
├── write_notebook.py                   # Notebook generator
├── generate_report.py                  # Comprehensive Word report generator
└── README.md                           # Documentation
```

---

## Step-by-Step Execution Guide

### 1. Run the Full End-to-End Pipeline
```bash
python Week6_Capstone_Project/week6_capstone_pipeline.py
```
This executes all 6 pipeline phases, generates all summary CSV ledgers in `output/`, and renders all 12 figures in `visualizations/`.

### 2. Generate the Jupyter Notebook
```bash
python Week6_Capstone_Project/write_notebook.py
```
Generates `notebooks/Week6_Capstone_Project.ipynb`.

### 3. Generate the Publication-Grade Word Report (.docx)
```bash
python Week6_Capstone_Project/generate_report.py
```
Builds `report/Week6_Capstone_Project_Report.docx`.

---

## Executive Analytical Findings

| Analytical Dimension | Key Finding | Business Impact |
|:---|:---|:---|
| **Primary Churn Driver** | Month-to-month contracts have $4.2\times$ the churn risk of multi-year agreements ($p < 10^{-15}$). | Automated contract conversion incentives reduce attrition by over 60%. |
| **Operational Friction** | Support tickets $> 2$ and payment delays trigger sharp churn escalation. | Real-time CRM notification alerts Tier-3 customer success teams. |
| **Capital Concentration** | Persona 1 (At-Risk Consumers) accounts for $1.4M+ in exposed capital. | Retention budgets concentrated on high-value at-risk accounts. |
| **Financial Decision Boundary** | Lowering threshold from $t=0.50$ to $t=0.40$ yields maximum net profit. | Generates over 640% net campaign ROI. |
