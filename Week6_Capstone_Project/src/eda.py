"""
eda.py
Comprehensive Exploratory Data Analysis, distributional profiling,
and bivariate hypothesis testing for the Enterprise Customer Analytics Pipeline.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

def run_exploratory_data_analysis(df_clean, output_dir):
    """
    Executes automated exploratory statistical analysis, computing univariate profiles,
    correlation matrices, and inferential hypothesis test statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Numerical Feature Profiling
    num_cols = ["Age", "TenureMonths", "MonthlyUsageGB", "SupportTicketsLastYear",
                "PaymentDelaysCount", "CustomerSatisfactionScore", "MonthlyCharges",
                "TotalChargesHistorical", "CustomerLifetimeValue"]

    records_num = []
    for col in num_cols:
        series = df_clean[col].dropna()
        records_num.append({
            "Feature": col,
            "Mean": round(series.mean(), 2),
            "Std_Dev": round(series.std(), 2),
            "Min": round(series.min(), 2),
            "Q25": round(series.quantile(0.25), 2),
            "Median": round(series.median(), 2),
            "Q75": round(series.quantile(0.75), 2),
            "Max": round(series.max(), 2),
            "Skewness": round(series.skew(), 3),
            "Kurtosis": round(series.kurtosis(), 3)
        })
    df_num_profile = pd.DataFrame(records_num)
    df_num_profile.to_csv(os.path.join(output_dir, "eda_numerical_summary.csv"), index=False)

    # 2. Correlation Matrix
    corr_matrix = df_clean[num_cols + ["Churn"]].corr().round(4)
    corr_matrix.to_csv(os.path.join(output_dir, "eda_correlation_matrix.csv"))

    # 3. Inferential Bivariate Hypothesis Tests (Churn vs Non-Churn)
    churn_0 = df_clean[df_clean["Churn"] == 0]
    churn_1 = df_clean[df_clean["Churn"] == 1]

    hypothesis_tests = []
    for col in num_cols:
        # Two-sample independent t-test (Welch's t-test unequal variances)
        t_stat, p_val = stats.ttest_ind(churn_1[col].dropna(), churn_0[col].dropna(), equal_var=False)
        mean_churn = churn_1[col].mean()
        mean_retain = churn_0[col].mean()
        diff = mean_churn - mean_retain

        sig = "Statistically Significant (p < 0.001)" if p_val < 0.001 else ("Significant (p < 0.05)" if p_val < 0.05 else "Not Significant")

        hypothesis_tests.append({
            "Feature": col,
            "Mean_Retained": round(mean_retain, 2),
            "Mean_Churned": round(mean_churn, 2),
            "Mean_Difference": round(diff, 2),
            "t_Statistic": round(t_stat, 3),
            "p_Value": f"{p_val:.2e}" if p_val < 0.0001 else f"{p_val:.4f}",
            "Significance": sig
        })

    df_hypo = pd.DataFrame(hypothesis_tests)
    df_hypo.to_csv(os.path.join(output_dir, "eda_bivariate_hypothesis_tests.csv"), index=False)

    print("[INFO] Exploratory Data Analysis completed. Summaries exported.")
    return {
        "df_num_profile": df_num_profile,
        "corr_matrix": corr_matrix,
        "df_hypo": df_hypo
    }
