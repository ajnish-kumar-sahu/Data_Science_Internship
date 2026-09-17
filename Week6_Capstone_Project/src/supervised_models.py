"""
supervised_models.py
Dual supervised machine learning pipeline:
1. Classification Sub-Pipeline: Customer Churn Probability Prediction & Cost-Benefit Thresholding
2. Regression Sub-Pipeline: Customer Lifetime Value (CLV in $) Continuous Forecasting

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix
)

def train_and_evaluate_churn_classification(data_dict, output_dir, cv_folds=5, random_state=42):
    """
    Trains and benchmarks 5 diverse classifiers for Churn Risk prediction.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train = data_dict["X_train_scaled"]
    y_train = data_dict["y_churn_train"]
    X_test = data_dict["X_test_scaled"]
    y_test = data_dict["y_churn_test"]

    classifiers = {
        "Dummy Baseline": DummyClassifier(strategy="stratified", random_state=random_state),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=random_state, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=random_state),
        "Deep MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=random_state, early_stopping=True)
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_records = []
    test_records = []
    fitted_models = {}
    test_probs = {}

    print(f"\n[INFO] Benchmarking {len(classifiers)} classification models with {cv_folds}-Fold Stratified CV...")
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

    for name, model in classifiers.items():
        # Cross-validation
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_records.append({
            "Model": name,
            "CV_Accuracy_Mean": round(cv_res["test_accuracy"].mean() * 100, 2),
            "CV_Precision_Mean": round(cv_res["test_precision"].mean() * 100, 2),
            "CV_Recall_Mean": round(cv_res["test_recall"].mean() * 100, 2),
            "CV_F1_Mean": round(cv_res["test_f1"].mean() * 100, 2),
            "CV_ROC_AUC_Mean": round(cv_res["test_roc_auc"].mean(), 4),
            "CV_PR_AUC_Mean": round(cv_res["test_average_precision"].mean(), 4)
        })

        # Test set evaluation
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = y_pred.astype(float)
        test_probs[name] = probs

        test_records.append({
            "Model": name,
            "Test_Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Test_Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Test_Recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Test_F1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Test_ROC_AUC": round(roc_auc_score(y_test, probs), 4),
            "Test_PR_AUC": round(average_precision_score(y_test, probs), 4),
            "Brier_Score": round(brier_score_loss(y_test, probs), 4)
        })

    df_cv_churn = pd.DataFrame(cv_records)
    df_test_churn = pd.DataFrame(test_records).sort_values(by="Test_ROC_AUC", ascending=False).reset_index(drop=True)

    df_cv_churn.to_csv(os.path.join(output_dir, "churn_cv_benchmark_summary.csv"), index=False)
    df_test_churn.to_csv(os.path.join(output_dir, "churn_test_evaluation_metrics.csv"), index=False)

    # 2. Champion Model Feature Importance (Gradient Boosting)
    champion_clf = fitted_models["Gradient Boosting"]
    feat_imp = pd.DataFrame({
        "Feature": X_train.columns,
        "Gini_Importance": champion_clf.feature_importances_
    }).sort_values(by="Gini_Importance", ascending=False).reset_index(drop=True)
    feat_imp.to_csv(os.path.join(output_dir, "feature_importance_churn.csv"), index=False)

    # 3. Cost-Benefit Threshold Optimization
    # Assume: Retaining a customer preserves $1,200 annual value; intervention cost is $60;
    # acceptance rate is 65%.
    gb_probs = test_probs["Gradient Boosting"]
    thresholds = np.linspace(0.10, 0.90, 41)
    thresh_records = []

    cost_per_contact = 60.0
    value_saved_if_churner = 1200.0 * 0.65  # $780 net preserved

    for t in thresholds:
        preds = (gb_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))
        tn = np.sum((preds == 0) & (y_test == 0))

        net_benefit = (tp * (value_saved_if_churner - cost_per_contact)) - (fp * cost_per_contact)
        roi_pct = (net_benefit / ((tp + fp) * cost_per_contact) * 100) if (tp + fp) > 0 else 0.0

        thresh_records.append({
            "Threshold": round(t, 2),
            "True_Positives": int(tp),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "Net_Financial_Value_Dollars": round(net_benefit, 2),
            "Intervention_ROI_Pct": round(roi_pct, 1),
            "F1_Score": round(f1_score(y_test, preds, zero_division=0), 4)
        })

    df_thresh = pd.DataFrame(thresh_records)
    df_thresh.to_csv(os.path.join(output_dir, "threshold_optimization_churn.csv"), index=False)

    print(f"[INFO] Classification benchmark completed. Champion model: Gradient Boosting (ROC-AUC: {df_test_churn.iloc[0]['Test_ROC_AUC']})")

    return {
        "df_cv_churn": df_cv_churn,
        "df_test_churn": df_test_churn,
        "feat_imp_churn": feat_imp,
        "df_thresh": df_thresh,
        "fitted_models_churn": fitted_models,
        "test_probs_churn": test_probs,
        "champion_clf": champion_clf
    }

def train_and_evaluate_clv_regression(data_dict, output_dir, cv_folds=5, random_state=42):
    """
    Trains and benchmarks continuous regression models for Customer Lifetime Value (CLV in $).
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train = data_dict["X_train_scaled"]
    y_train = data_dict["y_clv_train"]
    X_test = data_dict["X_test_scaled"]
    y_test = data_dict["y_clv_test"]

    regressors = {
        "Dummy Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=10.0, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=random_state)
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_records = []
    test_records = []
    fitted_regressors = {}
    test_preds = {}

    print(f"\n[INFO] Benchmarking {len(regressors)} regression models for Customer Lifetime Value (CLV)...")

    for name, model in regressors.items():
        # CV evaluation
        r2_scores = []
        rmse_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            preds_va = model.predict(X_va)
            r2_scores.append(r2_score(y_va, preds_va))
            rmse_scores.append(np.sqrt(mean_squared_error(y_va, preds_va)))

        cv_records.append({
            "Model": name,
            "CV_R2_Mean": round(np.mean(r2_scores), 4),
            "CV_R2_Std": round(np.std(r2_scores), 4),
            "CV_RMSE_Mean_Dollars": round(np.mean(rmse_scores), 2)
        })

        # Test set evaluation
        model.fit(X_train, y_train)
        fitted_regressors[name] = model

        preds_te = model.predict(X_test)
        test_preds[name] = preds_te

        rmse = np.sqrt(mean_squared_error(y_test, preds_te))
        mae = mean_absolute_error(y_test, preds_te)
        r2 = r2_score(y_test, preds_te)
        mape = np.mean(np.abs((y_test - preds_te) / y_test)) * 100

        test_records.append({
            "Model": name,
            "Test_R2": round(r2, 4),
            "Test_RMSE_Dollars": round(rmse, 2),
            "Test_MAE_Dollars": round(mae, 2),
            "Test_MAPE_Pct": round(mape, 2)
        })

    df_cv_clv = pd.DataFrame(cv_records)
    df_test_clv = pd.DataFrame(test_records).sort_values(by="Test_R2", ascending=False).reset_index(drop=True)

    df_cv_clv.to_csv(os.path.join(output_dir, "clv_cv_benchmark_summary.csv"), index=False)
    df_test_clv.to_csv(os.path.join(output_dir, "clv_test_evaluation_metrics.csv"), index=False)

    # Feature importances for CLV
    champion_reg = fitted_regressors["Gradient Boosting Regressor"]
    feat_imp_clv = pd.DataFrame({
        "Feature": X_train.columns,
        "Relative_Importance": champion_reg.feature_importances_
    }).sort_values(by="Relative_Importance", ascending=False).reset_index(drop=True)
    feat_imp_clv.to_csv(os.path.join(output_dir, "feature_importance_clv.csv"), index=False)

    print(f"[INFO] CLV Regression benchmark completed. Champion model: Gradient Boosting Regressor (R2: {df_test_clv.iloc[0]['Test_R2']})")

    return {
        "df_cv_clv": df_cv_clv,
        "df_test_clv": df_test_clv,
        "feat_imp_clv": feat_imp_clv,
        "fitted_regressors": fitted_regressors,
        "test_preds_clv": test_preds,
        "champion_reg": champion_reg
    }
