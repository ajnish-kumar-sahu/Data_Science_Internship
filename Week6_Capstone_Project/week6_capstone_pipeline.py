"""
week6_capstone_pipeline.py
Master execution pipeline for Week 6 Internship Capstone Task:
Integrative Enterprise Customer Analytics & Predictive Revenue Optimization.

Combines Data Hygiene, Exploratory Diagnostics, Unsupervised Persona Clustering (K-Means + PCA),
Dual Supervised Learning (Churn Classification & Customer Lifetime Value Regression),
and Executive Financial Value-at-Risk Synthesis.

Author: Ajnish Kumar | Roll No: 241809046713
Degree: BCA, Vinoba Bhave University, Hazaribag
Internship: Yuva Intern - Virtual Data Science with Python Trainee
Date: September 2026
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.data_loader import load_or_generate_enterprise_data, clean_and_preprocess_enterprise_data
from src.eda import run_exploratory_data_analysis
from src.clustering import run_unsupervised_clustering
from src.supervised_models import train_and_evaluate_churn_classification, train_and_evaluate_clv_regression
from src.integrator import synthesize_capstone_insights
from src.visualizer import (
    plot_eda_feature_distributions,
    plot_bivariate_churn_interactions,
    plot_correlation_matrix_heatmap,
    plot_clustering_elbow_silhouette,
    plot_pca_persona_clusters_scatter,
    plot_persona_attribute_comparison,
    plot_supervised_churn_roc_curves,
    plot_churn_confusion_matrix_and_pr_curve,
    plot_clv_regression_actual_vs_predicted,
    plot_feature_importance_dual_benchmark,
    plot_executive_value_at_risk_heatmap,
    plot_retention_campaign_financial_curve
)

def main():
    print("=" * 85)
    print(" WEEK 6: INTEGRATIVE CAPSTONE PROJECT & EVALUATION")
    print(" Enterprise Customer Analytics & Predictive Revenue Optimization Pipeline")
    print(" Author: Ajnish Kumar | Roll No: 241809046713 | Vinoba Bhave University")
    print("=" * 85)

    data_dir = os.path.join(CURRENT_DIR, "data")
    output_dir = os.path.join(CURRENT_DIR, "output")
    viz_dir = os.path.join(CURRENT_DIR, "visualizations")
    report_dir = os.path.join(CURRENT_DIR, "report")
    notebooks_dir = os.path.join(CURRENT_DIR, "notebooks")

    for d in [data_dir, output_dir, viz_dir, report_dir, notebooks_dir]:
        os.makedirs(d, exist_ok=True)

    t0 = time.time()

    # -------------------------------------------------------------
    # PHASE 1: DATA ACQUISITION, AUDITING & CLEANING
    # -------------------------------------------------------------
    print("\n>>> PHASE 1: DATA ACQUISITION & HYGIENE AUDITING <<<")
    df_raw = load_or_generate_enterprise_data(data_dir=data_dir, n_samples=5000, random_state=42)
    prep_dict = clean_and_preprocess_enterprise_data(df=df_raw, output_dir=output_dir)
    df_clean = prep_dict["df_clean"]

    # -------------------------------------------------------------
    # PHASE 2: EXPLORATORY DATA ANALYSIS & INFERENTIAL TESTING
    # -------------------------------------------------------------
    print("\n>>> PHASE 2: EXPLORATORY DATA ANALYSIS & INFERENTIAL HYPOTHESIS TESTING <<<")
    eda_dict = run_exploratory_data_analysis(df_clean=df_clean, output_dir=output_dir)

    print("\n[INFO] Generating Exploratory Visualizations (Figures 1-3)...")
    plot_eda_feature_distributions(df_clean=df_clean, output_dir=viz_dir)
    plot_bivariate_churn_interactions(df_clean=df_clean, output_dir=viz_dir)
    plot_correlation_matrix_heatmap(df_clean=df_clean, output_dir=viz_dir)

    # -------------------------------------------------------------
    # PHASE 3: UNSUPERVISED BEHAVIORAL SEGMENTATION (K-MEANS + PCA)
    # -------------------------------------------------------------
    print("\n>>> PHASE 3: UNSUPERVISED CUSTOMER PERSONA CLUSTERING <<<")
    clustering_dict = run_unsupervised_clustering(df_clean=df_clean, output_dir=output_dir, optimal_k=4, random_state=42)

    print("\n[INFO] Generating Unsupervised Cluster Visualizations (Figures 4-6)...")
    plot_clustering_elbow_silhouette(df_k_eval=clustering_dict["df_k_eval"], output_dir=viz_dir)
    plot_pca_persona_clusters_scatter(df_clustered=clustering_dict["df_clustered"], output_dir=viz_dir)
    plot_persona_attribute_comparison(df_clustered=clustering_dict["df_clustered"], output_dir=viz_dir)

    # -------------------------------------------------------------
    # PHASE 4: SUPERVISED CHURN RISK CLASSIFICATION
    # -------------------------------------------------------------
    print("\n>>> PHASE 4: SUPERVISED CHURN RISK PREDICTION <<<")
    churn_dict = train_and_evaluate_churn_classification(data_dict=prep_dict, output_dir=output_dir, cv_folds=5, random_state=42)

    print("\n[INFO] Generating Classification Benchmark Visualizations (Figures 7-8)...")
    plot_supervised_churn_roc_curves(y_test=prep_dict["y_churn_test"], test_probs=churn_dict["test_probs_churn"], output_dir=viz_dir)
    plot_churn_confusion_matrix_and_pr_curve(y_test=prep_dict["y_churn_test"], test_probs=churn_dict["test_probs_churn"], champion_name="Gradient Boosting", output_dir=viz_dir)

    # -------------------------------------------------------------
    # PHASE 5: SUPERVISED CLV CONTINUOUS REGRESSION FORECASTING
    # -------------------------------------------------------------
    print("\n>>> PHASE 5: SUPERVISED CUSTOMER LIFETIME VALUE (CLV) REGRESSION <<<")
    clv_dict = train_and_evaluate_clv_regression(data_dict=prep_dict, output_dir=output_dir, cv_folds=5, random_state=42)

    print("\n[INFO] Generating Regression Diagnostics & Feature Importance (Figures 9-10)...")
    champion_reg_name = "Gradient Boosting Regressor"
    plot_clv_regression_actual_vs_predicted(
        y_test_clv=prep_dict["y_clv_test"],
        preds_clv=clv_dict["test_preds_clv"][champion_reg_name],
        output_dir=viz_dir
    )
    plot_feature_importance_dual_benchmark(
        feat_churn=churn_dict["feat_imp_churn"],
        feat_clv=clv_dict["feat_imp_clv"],
        output_dir=viz_dir
    )

    # -------------------------------------------------------------
    # PHASE 6: INTEGRATIVE VALUE-AT-RISK & RETENTION CAMPAIGN SIMULATION
    # -------------------------------------------------------------
    print("\n>>> PHASE 6: INTEGRATIVE VALUE-AT-RISK & FINANCIAL ROI OPTIMIZATION <<<")
    supervised_bundle = {
        "scaler": prep_dict["scaler"],
        "X_train": prep_dict["X_train"],
        "champion_clf": churn_dict["champion_clf"],
        "champion_reg": clv_dict["champion_reg"]
    }
    integrator_dict = synthesize_capstone_insights(
        df_clean=df_clean,
        clustering_results=clustering_dict,
        supervised_results=supervised_bundle,
        output_dir=output_dir
    )

    print("\n[INFO] Generating Executive Business Synthesis Visualizations (Figures 11-12)...")
    plot_executive_value_at_risk_heatmap(output_dir=output_dir)
    f11_out = os.path.join(output_dir, "fig11_executive_value_at_risk_matrix.png")
    f11_viz = os.path.join(viz_dir, "fig11_executive_value_at_risk_matrix.png")
    if os.path.exists(f11_out) and not os.path.exists(f11_viz):
        os.rename(f11_out, f11_viz)

    plot_retention_campaign_financial_curve(df_thresh=churn_dict["df_thresh"], output_dir=viz_dir)

    total_time = time.time() - t0
    print("\n" + "=" * 85)
    print(f" INTEGRATIVE CAPSTONE PIPELINE EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print(f" All CSV artifacts persisted in: {output_dir}")
    print(f" All 12 publication figures saved in: {viz_dir}")
    print("=" * 85)

if __name__ == "__main__":
    main()
