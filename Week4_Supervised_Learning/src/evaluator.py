"""
evaluator.py
Model evaluation, Stratified Cross-Validation, Hyperparameter Tuning,
and Financial Decision Threshold Optimization.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


def cross_validate_all_models(pipelines: dict, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> pd.DataFrame:
    """
    Performs Stratified K-Fold cross validation across all candidate models
    and computes comprehensive classification metrics.
    
    Parameters:
        pipelines (dict): Dictionary of name -> Pipeline objects.
        X (pd.DataFrame): Training feature matrix.
        y (pd.Series): Target binary vector.
        cv (int): Number of folds (default=5).
        
    Returns:
        pd.DataFrame: Summary benchmark table with mean and std for each metric.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    for name, pipeline in pipelines.items():
        logger.info(f"Cross-validating {name} across {cv} stratified folds...")
        fold_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
            "pr_auc": [],
            "brier": []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_val)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                y_prob = pipeline.decision_function(X_val)
            else:
                y_prob = y_pred.astype(float)
                
            fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
            fold_metrics["pr_auc"].append(average_precision_score(y_val, y_prob))
            fold_metrics["brier"].append(brier_score_loss(y_val, y_prob))
            
        row = {
            "Model": name,
            "Accuracy_Mean": np.mean(fold_metrics["accuracy"]),
            "Accuracy_Std": np.std(fold_metrics["accuracy"]),
            "Precision_Mean": np.mean(fold_metrics["precision"]),
            "Precision_Std": np.std(fold_metrics["precision"]),
            "Recall_Mean": np.mean(fold_metrics["recall"]),
            "Recall_Std": np.std(fold_metrics["recall"]),
            "F1_Mean": np.mean(fold_metrics["f1"]),
            "F1_Std": np.std(fold_metrics["f1"]),
            "ROC_AUC_Mean": np.mean(fold_metrics["roc_auc"]),
            "ROC_AUC_Std": np.std(fold_metrics["roc_auc"]),
            "PR_AUC_Mean": np.mean(fold_metrics["pr_auc"]),
            "PR_AUC_Std": np.std(fold_metrics["pr_auc"]),
            "Brier_Score_Mean": np.mean(fold_metrics["brier"]),
            "Brier_Score_Std": np.std(fold_metrics["brier"])
        }
        results.append(row)
        
    summary_df = pd.DataFrame(results)
    return summary_df


def evaluate_on_test_set(pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Fits the model pipeline on train data and generates thorough test set evaluations.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_prob = pipeline.decision_function(X_test)
    else:
        y_prob = y_pred.astype(float)
        
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "classification_report_df": report_df,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                       cost_fn: float = 200.0,
                       cost_fp: float = 30.0,
                       benefit_tp: float = 170.0) -> pd.DataFrame:
    """
    Simulates business financial cost-benefit outcomes across classification decision thresholds.
    
    Business assumptions:
    - Churn Loss (False Negative): Customer churns undetected, losing lifetime value ($200).
    - Retention Incentive (Cost on flagged customer): Proactive discount or retention package ($30).
    - True Positive Value: Intervened customer is retained, saving $200 CLV minus $30 promo = $170 net benefit.
    - True Negative Value: Loyal customer left untouched = $0 cost.
    - False Positive Cost: Loyal customer receives unneeded $30 discount = -$30.
    
    Parameters:
        y_true (np.ndarray): True binary labels.
        y_prob (np.ndarray): Predicted probability of churn.
        cost_fn (float): Cost of false negative.
        cost_fp (float): Cost of false positive.
        benefit_tp (float): Net benefit of true positive.
        
    Returns:
        pd.DataFrame: Table of metrics and net financial benefit across thresholds.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    results = []
    
    total_churners = int(np.sum(y_true == 1))
    baseline_loss = total_churners * cost_fn  # Doing nothing scenario
    
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = len(y_true) - int(np.sum(preds)), int(np.sum(preds)), 0, 0
            
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        # Financial Net Value:
        # We save $170 for each TP, spend $30 for each FP, and suffer $200 for each FN.
        net_financial_outcome = (tp * benefit_tp) - (fp * cost_fp) - (fn * cost_fn)
        # Net savings relative to doing nothing:
        net_savings_vs_baseline = baseline_loss + net_financial_outcome
        
        results.append({
            "threshold": round(thresh, 2),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "net_financial_outcome": round(net_financial_outcome, 2),
            "net_savings_vs_baseline": round(net_savings_vs_baseline, 2)
        })
        
    df_res = pd.DataFrame(results)
    return df_res
