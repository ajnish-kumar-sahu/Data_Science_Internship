"""
visualizer.py
High-resolution, publication-grade data visualizations for Supervised Learning.
All figures are styled using clean, professional palettes and saved at 150 DPI.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve

# Styling constants
DPI = 150
PALETTE_NAVY = "#1f497d"
PALETTE_BLUE = "#4472c4"
PALETTE_ORANGE = "#ed7d31"
PALETTE_GREEN = "#70ad47"
PALETTE_RED = "#c00000"
PALETTE_GRAY = "#7f7f7f"
MODEL_COLORS = {
    "Dummy Baseline": "#7f7f7f",
    "Logistic Regression": "#4472c4",
    "Decision Tree": "#ed7d31",
    "Random Forest": "#1f497d",
    "Gradient Boosting": "#70ad47",
    "Support Vector Machine": "#7030a0"
}


def setup_plot_style():
    """Sets consistent seaborn and matplotlib aesthetics."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.titlesize": 14,
        "figure.titleweight": "bold"
    })


def save_fig(fig, output_path: str):
    """Utility to save and close figures cleanly."""
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# 1. Target Class Distribution
def plot_target_distribution(df: pd.DataFrame, output_path: str):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    churn_counts = df["Churn"].value_counts().sort_index()
    labels = ["Retained (No)", "Churned (Yes)"]
    colors = [PALETTE_BLUE, PALETTE_ORANGE]
    
    # Bar Chart
    bars = ax1.bar(labels, churn_counts.values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    ax1.set_title("Customer Churn Volume (Absolute Count)", pad=12)
    ax1.set_ylabel("Number of Customers")
    ax1.set_ylim(0, max(churn_counts.values) * 1.15)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 80, f"{h:,}", ha="center", va="bottom", fontweight="bold")
        
    # Donut Chart
    wedges, texts, autotexts = ax2.pie(
        churn_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0, 0.08),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("Target Class Proportion (Imbalance Ratio)", pad=12)
    
    fig.suptitle("Figure 1: Target Variable (Churn) Distribution Analysis", y=1.03)
    save_fig(fig, output_path)


# 2. Tenure Cohort Analysis
def plot_tenure_cohorts(df: pd.DataFrame, output_path: str):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    cohort_stats = df.groupby("tenure_cohort", observed=True)["Churn"].agg(["count", "mean"]).reset_index()
    cohort_stats["churn_pct"] = cohort_stats["mean"] * 100
    
    # Ax1: Cohort Volume
    sns.barplot(data=cohort_stats, x="tenure_cohort", y="count", ax=ax1, palette="Blues_r", edgecolor="black")
    ax1.set_title("Customer Volume by Lifecycle Stage (Cohort)")
    ax1.set_xlabel("Tenure Cohort")
    ax1.set_ylabel("Total Customer Count")
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2, h + 30), ha="center", va="bottom")
        
    # Ax2: Churn Propensity
    bars = ax2.bar(cohort_stats["tenure_cohort"], cohort_stats["churn_pct"], color=PALETTE_ORANGE, edgecolor="black", width=0.55)
    ax2.set_title("Empirical Churn Rate (%) Across Tenure Cohorts")
    ax2.set_xlabel("Tenure Cohort")
    ax2.set_ylabel("Churn Rate (%)")
    ax2.set_ylim(0, 60)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax2.axhline(df["Churn"].mean() * 100, color=PALETTE_RED, linestyle="--", label=f"Average Churn ({df['Churn'].mean()*100:.1f}%)")
    ax2.legend()
    
    fig.suptitle("Figure 2: Impact of Customer Tenure on Churn Propensity", y=1.03)
    save_fig(fig, output_path)


# 3. Contract & Billing Impact
def plot_contract_billing_impact(df: pd.DataFrame, output_path: str):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Contract Churn Rate
    contract_df = df.groupby("Contract")["Churn"].agg(["count", "mean"]).reset_index()
    contract_df["churn_pct"] = contract_df["mean"] * 100
    sns.barplot(data=contract_df, x="Contract", y="churn_pct", ax=ax1, palette=["#c00000", "#4472c4", "#70ad47"], edgecolor="black")
    ax1.set_title("Churn Rate by Contract Commitment Type")
    ax1.set_ylabel("Churn Rate (%)")
    ax1.set_xlabel("Contract Agreement")
    ax1.set_ylim(0, 55)
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h + 1.2), ha="center", va="bottom", fontweight="bold")
        
    # Payment Method Churn Rate
    pay_df = df.groupby("PaymentMethod")["Churn"].agg(["count", "mean"]).reset_index()
    pay_df["churn_pct"] = pay_df["mean"] * 100
    sns.barplot(data=pay_df, y="PaymentMethod", x="churn_pct", ax=ax2, palette="Oranges_r", edgecolor="black")
    ax2.set_title("Churn Rate by Payment Instrument")
    ax2.set_xlabel("Churn Rate (%)")
    ax2.set_ylabel("")
    ax2.set_xlim(0, 55)
    for p in ax2.patches:
        w = p.get_width()
        ax2.annotate(f" {w:.1f}%", (w, p.get_y() + p.get_height() / 2), ha="left", va="center", fontweight="bold")
        
    fig.suptitle("Figure 3: Customer Churn Correlation with Contract and Payment Terms", y=1.03)
    save_fig(fig, output_path)


# 4. Charges Distribution (KDE)
def plot_charges_distribution(df: pd.DataFrame, output_path: str):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", common_norm=False, fill=True,
                palette=[PALETTE_BLUE, PALETTE_ORANGE], alpha=0.35, linewidth=2, ax=ax1)
    ax1.set_title("Monthly Charges Density by Churn Status")
    ax1.set_xlabel("Monthly Charges ($)")
    ax1.set_ylabel("Probability Density")
    
    sns.kdeplot(data=df, x="TotalCharges", hue="Churn", common_norm=False, fill=True,
                palette=[PALETTE_BLUE, PALETTE_ORANGE], alpha=0.35, linewidth=2, ax=ax2)
    ax2.set_title("Total Cumulative Charges Density by Churn Status")
    ax2.set_xlabel("Total Charges ($)")
    ax2.set_ylabel("Probability Density")
    
    fig.suptitle("Figure 4: Financial Expenditure Density (KDE) Split by Customer Churn", y=1.03)
    save_fig(fig, output_path)


# 5. Correlation Heatmap
def plot_correlation_matrix(df: pd.DataFrame, output_path: str):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    
    cols = [
        "tenure", "MonthlyCharges", "TotalCharges", "service_count",
        "monthly_to_total_ratio", "charge_per_service", "SeniorCitizen",
        "high_risk_contract_payment", "has_security_backup", "has_streaming", "Churn"
    ]
    sub_df = df[cols].copy()
    corr = sub_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.6, vmax=0.6,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Figure 5: Correlation Matrix of Numeric & Engineered Features with Target Churn", pad=14)
    save_fig(fig, output_path)


# 6. CV Model Benchmark Bar Chart
def plot_cv_benchmark(summary_df: pd.DataFrame, output_path: str):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    metrics = ["ROC_AUC_Mean", "PR_AUC_Mean", "F1_Mean", "Recall_Mean", "Accuracy_Mean"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1-Score", "Recall", "Accuracy"]
    
    df_plot = summary_df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")
    df_plot["Metric"] = df_plot["Metric"].map(dict(zip(metrics, metric_labels)))
    
    sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", ax=ax, palette="Blues_r", edgecolor="black")
    ax.set_title("Figure 6: Stratified 5-Fold Cross-Validation Performance Benchmark Across Models", pad=12)
    ax.set_ylabel("Metric Score (0.0 to 1.0)")
    ax.set_xlabel("Supervised Learning Model")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=15, ha="right")
    save_fig(fig, output_path)


# 7. Multi-Model ROC Curves
def plot_roc_curves(models_dict: dict, X_test, y_test, output_path: str):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    for name, pipeline in models_dict.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_test)
        else:
            y_prob = pipeline.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(name, "#333333")
        linestyle = "--" if "Dummy" in name else "-"
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color, linewidth=2.0, linestyle=linestyle)
        
    ax.plot([0, 1], [0, 1], "k:", label="Random Guess (AUC = 0.500)")
    ax.set_title("Figure 7: Receiver Operating Characteristic (ROC) Curves Comparison", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="lower right", frameon=True)
    save_fig(fig, output_path)


# 8. Precision-Recall Curves
def plot_pr_curves(models_dict: dict, X_test, y_test, output_path: str):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    baseline = np.mean(y_test == 1)
    
    for name, pipeline in models_dict.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_test)
        else:
            y_prob = pipeline.predict(X_test)
            
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        color = MODEL_COLORS.get(name, "#333333")
        linestyle = "--" if "Dummy" in name else "-"
        ax.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=2.0, linestyle=linestyle)
        
    ax.axhline(baseline, color="black", linestyle=":", label=f"Baseline Prevalence ({baseline:.2f})")
    ax.set_title("Figure 8: Precision-Recall Curves (Critical for Imbalanced Churn Evaluation)", pad=12)
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="upper right", frameon=True)
    save_fig(fig, output_path)


# 9. Confusion Matrices Comparison
def plot_confusion_matrices(cms: dict, output_path: str):
    setup_plot_style()
    n_models = len(cms)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.0))
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, cm) in zip(axes, cms.items()):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        # Display formatted strings with both count and percentage
        annot_text = np.array([
            [f"{cm[0,0]:,}\n({cm_norm[0,0]:.1%})", f"{cm[0,1]:,}\n({cm_norm[0,1]:.1%})"],
            [f"{cm[1,0]:,}\n({cm_norm[1,0]:.1%})", f"{cm[1,1]:,}\n({cm_norm[1,1]:.1%})"]
        ])
        
        sns.heatmap(cm_norm, annot=annot_text, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred Retain", "Pred Churn"], yticklabels=["True Retain", "True Churn"],
                    linewidths=1.0, linecolor="gray")
        ax.set_title(f"{name}\nConfusion Matrix")
        ax.set_ylabel("Actual Class")
        ax.set_xlabel("Predicted Class")
        
    fig.suptitle("Figure 9: Normalized Confusion Matrices on Holdout Test Set (N = 1,409)", y=1.05)
    save_fig(fig, output_path)


# 10. Feature Importance
def plot_feature_importance(importance_df: pd.DataFrame, output_path: str):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6.0))
    
    top_feats = importance_df.head(15).sort_values("importance", ascending=True)
    bars = ax.barh(top_feats["feature"], top_feats["importance"], color=PALETTE_BLUE, edgecolor="black", height=0.65)
    ax.set_title("Figure 10: Top 15 Feature Importances (Random Forest Impurity Reduction)", pad=12)
    ax.set_xlabel("Relative Importance Weight")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", ha="left", va="center", fontsize=9)
    save_fig(fig, output_path)


# 11. Hyperparameter Tuning Surface / Validation Curve
def plot_hyperparameter_tuning(cv_results: pd.DataFrame, output_path: str):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.0))
    
    # Check if max_depth and n_estimators exist in cv_results
    param_n_est = "param_classifier__n_estimators"
    param_depth = "param_classifier__max_depth"
    
    if param_n_est in cv_results.columns and param_depth in cv_results.columns:
        res = cv_results.copy()
        res[param_depth] = res[param_depth].fillna("None").astype(str)
        sns.lineplot(data=res, x=param_n_est, y="mean_test_score", hue=param_depth,
                     marker="o", palette="tab10", linewidth=2.0, ax=ax)
        ax.set_title("Figure 11: Hyperparameter Optimization Curve (GridSearchCV ROC-AUC)", pad=12)
        ax.set_xlabel("Number of Estimators (Trees)")
        ax.set_ylabel("Mean Cross-Validation ROC-AUC")
        ax.legend(title="Max Tree Depth", frameon=True)
    else:
        # Fallback generic rank plot
        ax.plot(cv_results["mean_test_score"].values, marker="o", color=PALETTE_NAVY)
        ax.set_title("Figure 11: Hyperparameter Candidates ROC-AUC Progression", pad=12)
        ax.set_xlabel("Hyperparameter Candidate Index")
        ax.set_ylabel("Mean CV ROC-AUC")
        
    save_fig(fig, output_path)


# 12. Cost-Benefit Financial Threshold Tuning
def plot_cost_benefit_curve(thresh_df: pd.DataFrame, output_path: str):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    
    best_idx = thresh_df["net_savings_vs_baseline"].idxmax()
    best_row = thresh_df.loc[best_idx]
    best_thresh = best_row["threshold"]
    best_savings = best_row["net_savings_vs_baseline"]
    
    # Upper Plot: Net Financial Savings
    ax1.plot(thresh_df["threshold"], thresh_df["net_savings_vs_baseline"], color=PALETTE_GREEN, linewidth=2.5, label="Net Financial Benefit ($)")
    ax1.axvline(0.50, color=PALETTE_GRAY, linestyle=":", label="Default Threshold (0.50)")
    ax1.axvline(best_thresh, color=PALETTE_RED, linestyle="--", label=f"Optimal Threshold ({best_thresh:.2f} -> ${best_savings:,.0f})")
    ax1.scatter([best_thresh], [best_savings], color=PALETTE_RED, s=90, zorder=5)
    ax1.set_title("Net Retention Savings vs. Inaction Baseline ($200 Churn Loss / $30 Incentive)")
    ax1.set_ylabel("Net Financial Savings ($)")
    ax1.legend(loc="lower center", frameon=True)
    
    # Lower Plot: Precision, Recall, F1
    ax2.plot(thresh_df["threshold"], thresh_df["precision"], label="Precision", color=PALETTE_BLUE, linewidth=2)
    ax2.plot(thresh_df["threshold"], thresh_df["recall"], label="Recall", color=PALETTE_ORANGE, linewidth=2)
    ax2.plot(thresh_df["threshold"], thresh_df["f1_score"], label="F1-Score", color=PALETTE_NAVY, linewidth=2)
    ax2.axvline(best_thresh, color=PALETTE_RED, linestyle="--")
    ax2.set_title("Classification Trade-offs (Precision vs Recall vs F1)")
    ax2.set_xlabel("Classification Decision Probability Threshold")
    ax2.set_ylabel("Score (0.0 to 1.0)")
    ax2.legend(loc="center right", frameon=True)
    
    fig.suptitle("Figure 12: Business Cost-Benefit Analysis and Optimal Threshold Selection", y=1.02)
    save_fig(fig, output_path)


# 13. Calibration Curve / Reliability Diagram
def plot_calibration_curve(models_dict: dict, X_test, y_test, output_path: str):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    for name, pipeline in models_dict.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
            color = MODEL_COLORS.get(name, "#333333")
            linestyle = "--" if "Dummy" in name else "-"
            ax.plot(prob_pred, prob_true, marker="s", label=f"{name}", color=color, linewidth=1.8, linestyle=linestyle)
            
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax.set_title("Figure 13: Model Probability Calibration (Reliability Diagram)", pad=12)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical True Fraction of Positives")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left", frameon=True)
    save_fig(fig, output_path)
