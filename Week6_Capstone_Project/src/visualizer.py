"""
visualizer.py
Comprehensive visualization suite for the Integrative Enterprise Customer Analytics Capstone.
Generates publication-grade figures for EDA, Unsupervised Clustering, Supervised Classification,
Regression Diagnostics, and Business Value-at-Risk Matrices.

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
from sklearn.metrics import roc_curve, precision_recall_curve

# Global publication styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

# Palette constants
NAVY = "#1F497D"
BLUE = "#4472C4"
ORANGE = "#ED7D31"
GREEN = "#2CA02C"
RED = "#D62728"
PURPLE = "#7030A0"
GRAY = "#7F7F7F"
LIGHT_BG = "#F8F9FA"

def plot_eda_feature_distributions(df_clean, output_dir):
    """
    Fig 1: 2x2 multi-panel distribution of key numerical variables.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("white")

    # 1. Tenure
    sns.histplot(df_clean["TenureMonths"], kde=True, color=NAVY, ax=axes[0, 0], bins=30)
    axes[0, 0].set_title("Customer Tenure Distribution (Months)", fontsize=11, fontweight="bold", color=NAVY)
    axes[0, 0].set_xlabel("Tenure (Months)", fontsize=10, fontweight="bold")
    axes[0, 0].axvline(df_clean["TenureMonths"].median(), color=RED, linestyle="--", label=f"Median: {df_clean['TenureMonths'].median():.0f}m")
    axes[0, 0].legend()

    # 2. Monthly Charges
    sns.histplot(df_clean["MonthlyCharges"], kde=True, color=BLUE, ax=axes[0, 1], bins=30)
    axes[0, 1].set_title("Monthly Billed Charges ($)", fontsize=11, fontweight="bold", color=NAVY)
    axes[0, 1].set_xlabel("Monthly Charges ($)", fontsize=10, fontweight="bold")
    axes[0, 1].axvline(df_clean["MonthlyCharges"].median(), color=RED, linestyle="--", label=f"Median: ${df_clean['MonthlyCharges'].median():.2f}")
    axes[0, 1].legend()

    # 3. Monthly Usage GB
    sns.histplot(df_clean["MonthlyUsageGB"], kde=True, color=ORANGE, ax=axes[1, 0], bins=30)
    axes[1, 0].set_title("Monthly Data Consumption (GB)", fontsize=11, fontweight="bold", color=NAVY)
    axes[1, 0].set_xlabel("Usage (GB)", fontsize=10, fontweight="bold")
    axes[1, 0].axvline(df_clean["MonthlyUsageGB"].median(), color=RED, linestyle="--", label=f"Median: {df_clean['MonthlyUsageGB'].median():.1f}GB")
    axes[1, 0].legend()

    # 4. Customer Lifetime Value
    sns.histplot(df_clean["CustomerLifetimeValue"], kde=True, color=GREEN, ax=axes[1, 1], bins=30)
    axes[1, 1].set_title("Customer Lifetime Value (CLV in $)", fontsize=11, fontweight="bold", color=NAVY)
    axes[1, 1].set_xlabel("Customer Lifetime Value ($)", fontsize=10, fontweight="bold")
    axes[1, 1].axvline(df_clean["CustomerLifetimeValue"].median(), color=RED, linestyle="--", label=f"Median: ${df_clean['CustomerLifetimeValue'].median():.2f}")
    axes[1, 1].legend()

    plt.suptitle("Figure 1: Exploratory Distribution Profiling of Primary Customer Attributes",
                 fontsize=13, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig01_eda_feature_distributions.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_bivariate_churn_interactions(df_clean, output_dir):
    """
    Fig 2: Churn rate broken down by contractual and operational drivers.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("white")

    # Contract Type
    contract_churn = df_clean.groupby("ContractType")["Churn"].mean() * 100
    axes[0, 0].bar(contract_churn.index, contract_churn.values, color=[RED, BLUE, GREEN], edgecolor=NAVY)
    axes[0, 0].set_title("Churn Rate by Contract Type", fontsize=11, fontweight="bold", color=NAVY)
    axes[0, 0].set_ylabel("Churn Rate (%)", fontsize=10, fontweight="bold")
    for i, v in enumerate(contract_churn.values):
        axes[0, 0].text(i, v + 1.0, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9.5)

    # Internet Service
    net_churn = df_clean.groupby("InternetService")["Churn"].mean() * 100
    axes[0, 1].bar(net_churn.index, net_churn.values, color=[BLUE, RED, GRAY], edgecolor=NAVY)
    axes[0, 1].set_title("Churn Rate by Internet Service Architecture", fontsize=11, fontweight="bold", color=NAVY)
    axes[0, 1].set_ylabel("Churn Rate (%)", fontsize=10, fontweight="bold")
    for i, v in enumerate(net_churn.values):
        axes[0, 1].text(i, v + 1.0, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9.5)

    # Support Tickets
    ticket_churn = df_clean.groupby("SupportTicketsLastYear")["Churn"].mean() * 100
    axes[1, 0].plot(ticket_churn.index, ticket_churn.values, "o-", color=RED, linewidth=2.5)
    axes[1, 0].set_title("Churn Rate vs. Support Tickets Opened", fontsize=11, fontweight="bold", color=NAVY)
    axes[1, 0].set_xlabel("Support Tickets (Last Year)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Churn Rate (%)", fontsize=10, fontweight="bold")

    # Satisfaction Score
    sat_churn = df_clean.groupby("CustomerSatisfactionScore")["Churn"].mean() * 100
    axes[1, 1].bar(sat_churn.index.astype(int), sat_churn.values, color=[RED, ORANGE, "#E5B800", "#70AD47", GREEN], edgecolor=NAVY)
    axes[1, 1].set_title("Churn Rate by Customer Satisfaction Score (1-5)", fontsize=11, fontweight="bold", color=NAVY)
    axes[1, 1].set_xlabel("Satisfaction Rating (1 = Poor, 5 = Excellent)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Churn Rate (%)", fontsize=10, fontweight="bold")
    for i, v in zip(sat_churn.index.astype(int), sat_churn.values):
        axes[1, 1].text(i, v + 1.0, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9.5)

    plt.suptitle("Figure 2: Empirical Bivariate Drivers of Customer Attrition",
                 fontsize=13, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig02_bivariate_churn_interactions.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_correlation_matrix_heatmap(df_clean, output_dir):
    """
    Fig 3: Correlation heatmap across primary numerical attributes, Churn, and CLV.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_cols = ["Age", "TenureMonths", "MonthlyUsageGB", "SupportTicketsLastYear",
                "PaymentDelaysCount", "CustomerSatisfactionScore", "MonthlyCharges",
                "TotalChargesHistorical", "Churn", "CustomerLifetimeValue"]

    corr = df_clean[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, linewidths=0.5, cbar_kws={"label": "Pearson Correlation"})
    ax.set_title("Figure 3: Inter-Feature Pearson Correlation Matrix", fontsize=12, fontweight="bold", color=NAVY, pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig03_correlation_matrix_heatmap.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_clustering_elbow_silhouette(df_k_eval, output_dir):
    """
    Fig 4: Dual-axis plot validating K-Means optimal k selection (Elbow vs Silhouette).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    color_inertia = NAVY
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", color=color_inertia, fontsize=10, fontweight="bold")
    line1 = ax1.plot(df_k_eval["k_clusters"], df_k_eval["Inertia"], "o-", color=color_inertia, linewidth=2.5, label="Inertia (Elbow)")
    ax1.tick_params(axis="y", labelcolor=color_inertia)

    # Optimal knee indicator
    ax1.axvline(4, color=RED, linestyle="--", linewidth=1.8, label="Optimal k = 4")

    ax2 = ax1.twinx()
    color_sil = ORANGE
    ax2.set_ylabel("Mean Silhouette Coefficient", color=color_sil, fontsize=10, fontweight="bold")
    line2 = ax2.plot(df_k_eval["k_clusters"], df_k_eval["Silhouette_Score"], "s-", color=color_sil, linewidth=2.5, label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor=color_sil)
    ax2.grid(False)

    lines = line1 + line2 + [plt.Line2D([0], [0], color=RED, linestyle="--", label="Selected Partition (k=4)")]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", edgecolor=GRAY)

    plt.title("Figure 4: Unsupervised Hyperparameter Validation: Inertia vs. Silhouette Coefficient",
              fontsize=11, fontweight="bold", color=NAVY, pad=12)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig04_clustering_elbow_silhouette.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_pca_persona_clusters_scatter(df_clustered, output_dir):
    """
    Fig 5: 2D PCA scatter plot showing the 4 distinct customer personas.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    persona_map = {
        0: ("High-Value Enterprise Loyalists", NAVY),
        1: ("At-Risk Month-to-Month Consumers", RED),
        2: ("Tech-Savvy Heavy Streamers", BLUE),
        3: ("Budget-Conscious Minimalists", GREEN)
    }

    for c_id, (p_name, color) in persona_map.items():
        sub = df_clustered[df_clustered["Cluster_ID"] == c_id]
        ax.scatter(sub["PCA_1"], sub["PCA_2"], c=color, label=f"{p_name} (N={len(sub):,})",
                   alpha=0.60, edgecolors="none", s=28)

        # Centroid
        cx, cy = sub["PCA_1"].mean(), sub["PCA_2"].mean()
        ax.scatter(cx, cy, c="white", edgecolors="black", s=140, marker="P", linewidth=1.5, zorder=5)
        ax.text(cx, cy + 0.15, p_name, fontsize=8.5, fontweight="bold", color=color, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9), zorder=6)

    ax.set_title("Figure 5: Principal Component Analysis (PCA) 2D Latent Manifold Projection\n[Unsupervised Customer Behavioral Segmentation]",
                 fontsize=11, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("First Principal Component (PC1 - Scale & Value Driver)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Second Principal Component (PC2 - Usage & Risk Driver)", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRAY)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig05_pca_persona_clusters_scatter.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_persona_attribute_comparison(df_clustered, output_dir):
    """
    Fig 6: Comparative multi-attribute profile of the 4 discovered customer personas.
    """
    os.makedirs(output_dir, exist_ok=True)
    features = ["TenureMonths", "MonthlyCharges", "MonthlyUsageGB", "SupportTicketsLastYear",
                "CustomerSatisfactionScore", "Churn", "CustomerLifetimeValue"]
    display_names = ["Tenure (m)", "Monthly ($)", "Usage (GB)", "Tickets",
                     "Satisfaction (1-5)", "Churn Rate (%)", "CLV ($)"]

    persona_map = {
        0: "High-Value Enterprise Loyalists",
        1: "At-Risk Month-to-Month Consumers",
        2: "Tech-Savvy Heavy Streamers",
        3: "Budget-Conscious Minimalists"
    }

    df_copy = df_clustered.copy()
    df_copy["Persona"] = df_copy["Cluster_ID"].map(persona_map)

    means = df_copy.groupby("Persona")[features].mean()
    # Normalize each column 0-1 for radar / spider comparison
    norm_means = (means - means.min()) / (means.max() - means.min() + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")

    # Panel A: Grouped Bar Chart of Raw Means
    colors = [NAVY, RED, BLUE, GREEN]
    means.T.plot(kind="bar", ax=axes[0], color=colors, edgecolor="black", width=0.8)
    axes[0].set_title("Panel A: Raw Attribute Means by Behavioral Persona", fontsize=11, fontweight="bold", color=NAVY)
    axes[0].set_xticklabels(display_names, rotation=35, ha="right", fontsize=9.5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Log Scale Mean Value", fontsize=10, fontweight="bold")
    axes[0].legend(title="Customer Persona", fontsize=8.5, loc="upper right")

    # Panel B: Relative Attribute Index (Normalized Scale 0.0 - 1.0)
    norm_means.T.plot(kind="bar", ax=axes[1], color=colors, edgecolor="black", width=0.8)
    axes[1].set_title("Panel B: Relative Feature Index Profile (Normalized Min-Max)", fontsize=11, fontweight="bold", color=NAVY)
    axes[1].set_xticklabels(display_names, rotation=35, ha="right", fontsize=9.5)
    axes[1].set_ylabel("Relative Intensity (0 = Min, 1 = Max)", fontsize=10, fontweight="bold")
    axes[1].legend(title="Customer Persona", fontsize=8.5, loc="upper right")

    plt.suptitle("Figure 6: Multi-Dimensional Behavioral Persona Profiles & Attribute Signatures",
                 fontsize=13, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig06_persona_attribute_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_supervised_churn_roc_curves(y_test, test_probs, output_dir):
    """
    Fig 7: Multi-model ROC Curve Benchmark.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor("white")

    colors = {"Dummy Baseline": GRAY, "Logistic Regression": BLUE, "Random Forest": ORANGE,
              "Gradient Boosting": GREEN, "Deep MLP Classifier": PURPLE}

    for name, probs in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        from sklearn.metrics import roc_auc_score
        auc_val = roc_auc_score(y_test, probs)
        lw = 2.5 if name == "Gradient Boosting" else 1.8
        ls = "--" if name == "Dummy Baseline" else "-"
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc_val:.4f})", color=colors.get(name, BLUE), linewidth=lw, linestyle=ls)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance Line (AUC = 0.5000)")
    ax.set_title("Figure 7: Supervised Churn Risk Multi-Model ROC Benchmark (Test Set N=1,000)",
                 fontsize=11, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRAY)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig07_supervised_churn_roc_curves.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_churn_confusion_matrix_and_pr_curve(y_test, test_probs, champion_name, output_dir):
    """
    Fig 8: Confusion Matrix Heatmap for Champion Model & Precision-Recall Curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Panel A: Champion Confusion Matrix
    from sklearn.metrics import confusion_matrix
    probs = test_probs[champion_name]
    preds = (probs >= 0.50).astype(int)
    cm = confusion_matrix(y_test, preds)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"],
                annot_kws={"size": 14, "weight": "bold"})
    ax1.set_title(f"Panel A: Champion Model Confusion Matrix\n[{champion_name} @ Threshold=0.50]",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Ground Truth Class", fontsize=10, fontweight="bold")

    # Panel B: Precision-Recall Curves
    colors = {"Dummy Baseline": GRAY, "Logistic Regression": BLUE, "Random Forest": ORANGE,
              "Gradient Boosting": GREEN, "Deep MLP Classifier": PURPLE}
    for name, pr_probs in test_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, pr_probs)
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_test, pr_probs)
        lw = 2.5 if name == champion_name else 1.8
        ls = "--" if name == "Dummy Baseline" else "-"
        ax2.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=colors.get(name, BLUE), linewidth=lw, linestyle=ls)

    ax2.set_title("Panel B: Precision-Recall (PR) Curves Benchmark",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=10, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRAY)

    plt.suptitle("Figure 8: Supervised Churn Diagnostic: Confusion Matrix & Precision-Recall Trade-off",
                 fontsize=12, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig08_churn_confusion_matrix_pr_curve.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_clv_regression_actual_vs_predicted(y_test_clv, preds_clv, output_dir):
    """
    Fig 9: Actual vs. Predicted Customer Lifetime Value with residual distribution.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    # Scatter
    ax1.scatter(y_test_clv, preds_clv, alpha=0.55, color=BLUE, edgecolors="none", s=25)
    min_val = min(y_test_clv.min(), preds_clv.min())
    max_val = max(y_test_clv.max(), preds_clv.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Forecast Line (y = x)")
    ax1.set_title("Actual vs. Predicted Customer Lifetime Value ($)", fontsize=11, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Actual CLV ($)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Predicted CLV ($)", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left")

    # Residuals
    residuals = y_test_clv - preds_clv
    sns.histplot(residuals, kde=True, color=NAVY, ax=ax2, bins=30)
    ax2.set_title("Regression Residual Distribution (Actual - Predicted)", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_xlabel("Residual Error ($)", fontsize=10, fontweight="bold")
    ax2.axvline(0, color=RED, linestyle="--", linewidth=1.5)

    plt.suptitle("Figure 9: Customer Lifetime Value (CLV) Regression Diagnostics",
                 fontsize=12, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig09_clv_regression_actual_vs_predicted.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_feature_importance_dual_benchmark(feat_churn, feat_clv, output_dir):
    """
    Fig 10: Side-by-side bar chart of top drivers for Churn vs top drivers for CLV.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # Top 8 Churn Features
    top_churn = feat_churn.head(8)
    ax1.barh(top_churn["Feature"][::-1], top_churn["Gini_Importance"][::-1], color=RED, edgecolor=NAVY)
    ax1.set_title("Top 8 Predictors of Customer Churn Risk", fontsize=11, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Gini Importance / Gain", fontsize=10, fontweight="bold")

    # Top 8 CLV Features
    top_clv = feat_clv.head(8)
    ax2.barh(top_clv["Feature"][::-1], top_clv["Relative_Importance"][::-1], color=GREEN, edgecolor=NAVY)
    ax2.set_title("Top 8 Predictors of Customer Lifetime Value (CLV)", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_xlabel("Relative Importance / Gain", fontsize=10, fontweight="bold")

    plt.suptitle("Figure 10: Dual Supervised Feature Importance Benchmark",
                 fontsize=12, fontweight="bold", color=NAVY, y=0.98)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig10_feature_importance_dual_benchmark.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_executive_value_at_risk_heatmap(output_dir):
    """
    Fig 11: Heatmap visualizing total capital exposure by Customer Persona and Churn Risk Tier.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "capital_value_at_risk_summary.csv")
    if not os.path.exists(summary_path):
        return

    df_var = pd.read_csv(summary_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    y = np.arange(len(df_var))
    bars = ax.barh(y, df_var["Total_Capital_at_Risk_Dollars"] / 1000.0, color=[RED, ORANGE, BLUE, GREEN], edgecolor=NAVY, height=0.55)

    ax.set_yticks(y)
    ax.set_yticklabels(df_var["Customer_Persona"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Capital Exposure at Critical Risk (Thousands of USD / $K)", fontsize=10, fontweight="bold")
    ax.set_title("Figure 11: Executive Value-at-Risk by Customer Behavioral Persona\n[Total Projected CLV Exposed to Critical Churn Risk]",
                 fontsize=11, fontweight="bold", color=NAVY, pad=12)

    for idx, row in df_var.iterrows():
        val = row["Total_Capital_at_Risk_Dollars"] / 1000.0
        ax.text(val + 5.0, idx, f"${row['Total_Capital_at_Risk_Dollars']:,.0f} ({row['Cohort_Risk_Rate_Pct']} at risk)",
                va="center", fontsize=9, fontweight="bold", color=NAVY)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig11_executive_value_at_risk_matrix.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")

def plot_retention_campaign_financial_curve(df_thresh, output_dir):
    """
    Fig 12: Financial Net Value ($) and ROI (%) curve across decision thresholds.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    color_val = GREEN
    ax1.set_xlabel("Classification Decision Threshold", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Net Financial Payoff ($)", color=color_val, fontsize=10, fontweight="bold")
    line1 = ax1.plot(df_thresh["Threshold"], df_thresh["Net_Financial_Value_Dollars"], "o-", color=color_val, linewidth=2.5, label="Net Saved Value ($)")
    ax1.tick_params(axis="y", labelcolor=color_val)

    # Max profit point
    max_idx = df_thresh["Net_Financial_Value_Dollars"].idxmax()
    best_t = df_thresh.loc[max_idx, "Threshold"]
    best_val = df_thresh.loc[max_idx, "Net_Financial_Value_Dollars"]
    ax1.axvline(best_t, color=RED, linestyle="--", linewidth=1.8, label=f"Optimal Threshold (t = {best_t:.2f})")

    ax2 = ax1.twinx()
    color_roi = BLUE
    ax2.set_ylabel("Retention Intervention ROI (%)", color=color_roi, fontsize=10, fontweight="bold")
    line2 = ax2.plot(df_thresh["Threshold"], df_thresh["Intervention_ROI_Pct"], "s--", color=color_roi, linewidth=2, label="Campaign ROI (%)")
    ax2.tick_params(axis="y", labelcolor=color_roi)
    ax2.grid(False)

    lines = line1 + line2 + [plt.Line2D([0], [0], color=RED, linestyle="--", label=f"Optimal Profit: ${best_val:,.0f}")]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", edgecolor=GRAY)

    plt.title("Figure 12: Cost-Benefit Decision Boundary Optimization for Retention Marketing",
              fontsize=11, fontweight="bold", color=NAVY, pad=12)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "fig12_retention_campaign_financial_curve.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {save_path}")
