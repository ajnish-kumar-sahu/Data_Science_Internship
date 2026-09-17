"""
week4_supervised_learning.py
Master execution pipeline for Week 4: Supervised Learning Model Implementation.
Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import sys
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix

# Add local src directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_raw_data, audit_and_clean_data
from feature_engineering import engineer_features, get_feature_lists, build_preprocessor
from models import get_model_pipelines, get_hyperparameter_grid
from evaluator import cross_validate_all_models, evaluate_on_test_set, optimize_threshold
import visualizer as viz

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Week4_Master")
warnings.filterwarnings("ignore")

# Define paths
DATA_DIR = os.path.join(BASE_DIR, "data")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
OUT_DIR = os.path.join(BASE_DIR, "output")
REPORT_DIR = os.path.join(BASE_DIR, "report")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

for p in [DATA_DIR, VIZ_DIR, OUT_DIR, REPORT_DIR, NOTEBOOKS_DIR]:
    os.makedirs(p, exist_ok=True)


def main():
    logger.info("================================================================")
    logger.info("  WEEK 4: SUPERVISED LEARNING MODEL IMPLEMENTATION (START)")
    logger.info("================================================================")

    # -------------------------------------------------------------------------
    # Step 1: Data Ingestion and Hygiene
    # -------------------------------------------------------------------------
    raw_csv = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")
    raw_df = load_raw_data(raw_csv)

    clean_df, audit = audit_and_clean_data(raw_df)
    logger.info(f"Audit Summary: {audit}")

    # -------------------------------------------------------------------------
    # Step 2: Feature Engineering
    # -------------------------------------------------------------------------
    logger.info("Applying domain feature engineering...")
    df_feat = engineer_features(clean_df)

    # Save engineered dataset
    engineered_csv = os.path.join(DATA_DIR, "telco_churn_engineered.csv")
    df_feat.to_csv(engineered_csv, index=False)
    logger.info(f"Saved engineered dataset ({df_feat.shape}) to {engineered_csv}")

    # -------------------------------------------------------------------------
    # Step 3: Exploratory Visualizations (Figures 1-5)
    # -------------------------------------------------------------------------
    logger.info("Generating exploratory data visualizations...")
    viz.plot_target_distribution(df_feat, os.path.join(VIZ_DIR, "01_target_distribution.png"))
    viz.plot_tenure_cohorts(df_feat, os.path.join(VIZ_DIR, "02_tenure_cohort_analysis.png"))
    viz.plot_contract_billing_impact(df_feat, os.path.join(VIZ_DIR, "03_contract_billing_impact.png"))
    viz.plot_charges_distribution(df_feat, os.path.join(VIZ_DIR, "04_charges_distribution_kde.png"))
    viz.plot_correlation_matrix(df_feat, os.path.join(VIZ_DIR, "05_correlation_matrix.png"))

    # -------------------------------------------------------------------------
    # Step 4: Train-Test Split (Holdout 80/20 Stratified)
    # -------------------------------------------------------------------------
    logger.info("Splitting dataset into stratified Train (80%) and Test (20%) partitions...")
    X = df_feat.drop(columns=["customerID", "Churn"])
    y = df_feat["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    # -------------------------------------------------------------------------
    # Step 5: Stratified 5-Fold Cross-Validation Across All Models
    # -------------------------------------------------------------------------
    logger.info("Instantiating model pipelines...")
    pipelines = get_model_pipelines()

    logger.info("Executing Stratified 5-Fold Cross-Validation across all 6 candidate models...")
    cv_summary_df = cross_validate_all_models(pipelines, X_train, y_train, cv=5)
    cv_csv_path = os.path.join(OUT_DIR, "model_cv_benchmark_summary.csv")
    cv_summary_df.to_csv(cv_csv_path, index=False)
    logger.info(f"Saved CV benchmark summary to {cv_csv_path}")
    print("\n--- Cross-Validation Benchmark Summary ---\n")
    print(cv_summary_df[["Model", "Accuracy_Mean", "Recall_Mean", "F1_Mean", "ROC_AUC_Mean", "PR_AUC_Mean"]].to_string(index=False))

    # Figure 6: CV Benchmark Bar Chart
    viz.plot_cv_benchmark(cv_summary_df, os.path.join(VIZ_DIR, "06_cv_model_benchmark.png"))

    # -------------------------------------------------------------------------
    # Step 6: Full Training & Holdout Test Set Evaluation
    # -------------------------------------------------------------------------
    logger.info("Fitting all models on full training partition and evaluating on holdout test set...")
    test_results = []
    trained_models = {}
    test_eval_dicts = {}

    for name, pipeline in pipelines.items():
        logger.info(f"Evaluating {name} on holdout test set...")
        eval_dict = evaluate_on_test_set(pipeline, X_train, y_train, X_test, y_test)
        trained_models[name] = pipeline
        test_eval_dicts[name] = eval_dict

        test_results.append({
            "Model": name,
            "Accuracy": eval_dict["accuracy"],
            "Precision": eval_dict["precision"],
            "Recall": eval_dict["recall"],
            "F1_Score": eval_dict["f1"],
            "ROC_AUC": eval_dict["roc_auc"],
            "PR_AUC": eval_dict["pr_auc"],
            "Brier_Score": eval_dict["brier_score"],
            "True_Negatives": eval_dict["tn"],
            "False_Positives": eval_dict["fp"],
            "False_Negatives": eval_dict["fn"],
            "True_Positives": eval_dict["tp"]
        })

    test_summary_df = pd.DataFrame(test_results)
    test_csv_path = os.path.join(OUT_DIR, "test_evaluation_metrics.csv")
    test_summary_df.to_csv(test_csv_path, index=False)
    logger.info(f"Saved test evaluation summary to {test_csv_path}")
    print("\n--- Holdout Test Set Evaluation ---\n")
    print(test_summary_df[["Model", "Accuracy", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # Step 7: Model Comparison Visualizations (Figures 7, 8, 9, 13)
    # -------------------------------------------------------------------------
    logger.info("Generating multi-model evaluation curves...")
    viz.plot_roc_curves(trained_models, X_test, y_test, os.path.join(VIZ_DIR, "07_roc_curves_comparison.png"))
    viz.plot_pr_curves(trained_models, X_test, y_test, os.path.join(VIZ_DIR, "08_precision_recall_curves.png"))
    viz.plot_calibration_curve(trained_models, X_test, y_test, os.path.join(VIZ_DIR, "13_probability_calibration_curve.png"))

    # Confusion matrices for selected top 3 models
    top_models_cm = {
        "Logistic Regression": test_eval_dicts["Logistic Regression"]["confusion_matrix"],
        "Random Forest": test_eval_dicts["Random Forest"]["confusion_matrix"],
        "Gradient Boosting": test_eval_dicts["Gradient Boosting"]["confusion_matrix"]
    }
    viz.plot_confusion_matrices(top_models_cm, os.path.join(VIZ_DIR, "09_confusion_matrices.png"))

    # -------------------------------------------------------------------------
    # Step 8: Hyperparameter Tuning (Random Forest Optimization)
    # -------------------------------------------------------------------------
    logger.info("Conducting GridSearchCV hyperparameter optimization on Random Forest...")
    rf_pipeline = pipelines["Random Forest"]
    rf_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [6, 10, 14],
        "classifier__min_samples_split": [2, 5]
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_grid,
        scoring="roc_auc",
        cv=skf,
        n_jobs=1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"Best GridSearchCV Hyperparameters: {grid_search.best_params_}")
    logger.info(f"Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    viz.plot_hyperparameter_tuning(cv_results_df, os.path.join(VIZ_DIR, "11_hyperparameter_tuning_surface.png"))

    # Set champion model
    champion_pipeline = grid_search.best_estimator_
    champion_eval = evaluate_on_test_set(champion_pipeline, X_train, y_train, X_test, y_test)

    # Save champion model artifact
    model_artifact_path = os.path.join(OUT_DIR, "champion_model.joblib")
    joblib.dump(champion_pipeline, model_artifact_path)
    logger.info(f"Serialized champion pipeline model to {model_artifact_path}")

    # Save champion classification report
    champ_report_path = os.path.join(OUT_DIR, "classification_report_champion.csv")
    champion_eval["classification_report_df"].to_csv(champ_report_path)
    logger.info(f"Saved champion classification report to {champ_report_path}")

    # -------------------------------------------------------------------------
    # Step 9: Feature Importance Analysis (Figure 10)
    # -------------------------------------------------------------------------
    logger.info("Extracting feature importances from Champion Random Forest...")
    rf_classifier = champion_pipeline.named_steps["classifier"]
    preprocessor = champion_pipeline.named_steps["preprocessor"]

    # Extract feature names from ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()
    # Clean prefixes
    clean_names = [name.split("__")[-1] for name in feature_names]

    importances = rf_classifier.feature_importances_
    importance_df = pd.DataFrame({
        "feature": clean_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    imp_csv_path = os.path.join(OUT_DIR, "feature_importance_rankings.csv")
    importance_df.to_csv(imp_csv_path, index=False)
    logger.info(f"Saved feature importance rankings to {imp_csv_path}")

    viz.plot_feature_importance(importance_df, os.path.join(VIZ_DIR, "10_feature_importance_rf.png"))

    # -------------------------------------------------------------------------
    # Step 10: Financial Decision Threshold Optimization (Figure 12)
    # -------------------------------------------------------------------------
    logger.info("Simulating business cost-benefit trade-offs across decision thresholds...")
    champ_probs = champion_eval["y_prob"]
    threshold_df = optimize_threshold(y_test.values, champ_probs, cost_fn=200.0, cost_fp=30.0, benefit_tp=170.0)

    thresh_csv_path = os.path.join(OUT_DIR, "threshold_financial_optimization.csv")
    threshold_df.to_csv(thresh_csv_path, index=False)
    logger.info(f"Saved threshold financial optimization table to {thresh_csv_path}")

    viz.plot_cost_benefit_curve(threshold_df, os.path.join(VIZ_DIR, "12_cost_benefit_threshold_tuning.png"))

    best_thresh_row = threshold_df.loc[threshold_df["net_savings_vs_baseline"].idxmax()]
    logger.info(f"Optimal Decision Threshold: {best_thresh_row['threshold']:.2f}")
    logger.info(f"Max Financial Net Savings: ${best_thresh_row['net_savings_vs_baseline']:,.2f} vs default $0.50 threshold")

    logger.info("================================================================")
    logger.info("  WEEK 4: PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("================================================================")


if __name__ == "__main__":
    main()
