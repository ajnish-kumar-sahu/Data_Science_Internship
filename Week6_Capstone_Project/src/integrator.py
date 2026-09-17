"""
integrator.py
Integrative Synthesis Engine: Cross-referencing Unsupervised Behavioral Personas
with Supervised Churn Risk & Customer Lifetime Value (CLV) forecasts.
Constructs the Executive Value-at-Risk Matrix and executes Retention Campaign Financial ROI Simulations.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import numpy as np
import pandas as pd

def synthesize_capstone_insights(df_clean, clustering_results, supervised_results, output_dir):
    """
    Synthesizes unsupervised personas with supervised predictions across the full dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_master = df_clean.copy()

    # 1. Attach Cluster IDs and Persona Names
    df_master["Cluster_ID"] = clustering_results["df_clustered"]["Cluster_ID"]
    persona_map = {
        0: "High-Value Enterprise Loyalists",
        1: "At-Risk Month-to-Month Consumers",
        2: "Tech-Savvy Heavy Streamers",
        3: "Budget-Conscious Minimalists"
    }
    df_master["Customer_Persona"] = df_master["Cluster_ID"].map(persona_map)

    # 2. Attach Full-Dataset Supervised Predictions
    scaler = supervised_results["scaler"]
    drop_cols = ["CustomerID", "Churn", "CustomerLifetimeValue", "Cluster_ID", "Customer_Persona"]
    feature_cols = [c for c in df_master.columns if c not in drop_cols]

    # One-hot encode exactly as trained
    categorical_cols = df_master[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    X_full_enc = pd.get_dummies(df_master[feature_cols], columns=categorical_cols, drop_first=True, dtype=float)

    # Align columns
    train_cols = supervised_results["X_train"].columns
    for c in train_cols:
        if c not in X_full_enc.columns:
            X_full_enc[c] = 0.0
    X_full_enc = X_full_enc[train_cols]

    X_full_scaled = scaler.transform(X_full_enc)

    # Churn probability via champion model (Gradient Boosting)
    champion_clf = supervised_results["champion_clf"]
    df_master["Predicted_Churn_Prob"] = champion_clf.predict_proba(X_full_scaled)[:, 1].round(4)
    df_master["Predicted_Churn_Binary"] = (df_master["Predicted_Churn_Prob"] >= 0.45).astype(int)

    # Predicted CLV via champion regressor (Gradient Boosting Regressor)
    champion_reg = supervised_results["champion_reg"]
    df_master["Predicted_CLV_Dollars"] = champion_reg.predict(X_full_scaled).round(2)

    # 3. Stratify Risk and Value Tiers
    # Risk Tiers
    risk_bins = [0.0, 0.30, 0.60, 1.01]
    risk_labels = ["Low Risk (<30%)", "Medium Risk (30-60%)", "Critical Risk (>60%)"]
    df_master["Churn_Risk_Tier"] = pd.cut(df_master["Predicted_Churn_Prob"], bins=risk_bins, labels=risk_labels, right=False)

    # CLV Tiers
    clv_bins = [0.0, 1200.0, 2500.0, 10000.0]
    clv_labels = ["Budget Value (<$1.2k)", "Mid Value ($1.2k-$2.5k)", "High Value (>$2.5k)"]
    df_master["CLV_Value_Tier"] = pd.cut(df_master["Predicted_CLV_Dollars"], bins=clv_bins, labels=clv_labels, right=False)

    # 4. Construct Executive Value-at-Risk Matrix
    risk_matrix = pd.crosstab(
        index=df_master["Customer_Persona"],
        columns=[df_master["CLV_Value_Tier"], df_master["Churn_Risk_Tier"]],
        margins=True,
        margins_name="Total_Accounts"
    )
    risk_matrix.to_csv(os.path.join(output_dir, "executive_value_at_risk_matrix.csv"))

    # Financial Exposure at Risk (Total CLV in Critical Risk Tiers)
    var_summary = []
    for persona in df_master["Customer_Persona"].unique():
        sub = df_master[df_master["Customer_Persona"] == persona]
        total_accounts = len(sub)
        critical_accounts = sub[sub["Churn_Risk_Tier"] == "Critical Risk (>60%)"]
        crit_count = len(critical_accounts)
        clv_at_risk = critical_accounts["Predicted_CLV_Dollars"].sum()
        mean_clv_risk = critical_accounts["Predicted_CLV_Dollars"].mean() if crit_count > 0 else 0.0

        var_summary.append({
            "Customer_Persona": persona,
            "Total_Cohort_Size": total_accounts,
            "Critical_Risk_Accounts": crit_count,
            "Cohort_Risk_Rate_Pct": f"{(crit_count / total_accounts) * 100:.1f}%",
            "Mean_CLV_at_Risk_Dollars": round(mean_clv_risk, 2),
            "Total_Capital_at_Risk_Dollars": round(clv_at_risk, 2)
        })

    df_var = pd.DataFrame(var_summary).sort_values(by="Total_Capital_at_Risk_Dollars", ascending=False).reset_index(drop=True)
    df_var.to_csv(os.path.join(output_dir, "capital_value_at_risk_summary.csv"), index=False)

    # 5. Financial ROI Simulation of Targeted Retention Campaign
    # Strategy: Intervene only on accounts with Churn Risk >= 0.40 AND Predicted CLV >= $1,500
    eligible_mask = (df_master["Predicted_Churn_Prob"] >= 0.40) & (df_master["Predicted_CLV_Dollars"] >= 1500.0)
    targeted_df = df_master[eligible_mask]

    n_targeted = len(targeted_df)
    cost_per_package = 65.0  # $65 loyalty credit / premium router upgrade
    total_campaign_cost = n_targeted * cost_per_package

    # Assume 65% retention success rate among contacted customers who would have churned
    retention_success_rate = 0.65
    gross_clv_saved = targeted_df["Predicted_CLV_Dollars"].sum() * retention_success_rate
    net_campaign_benefit = gross_clv_saved - total_campaign_cost
    roi_percent = (net_campaign_benefit / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0.0

    simulation_records = [
        {"Metric": "Targeted Customer Accounts", "Value": f"{n_targeted:,} accounts"},
        {"Metric": "Targeting Criteria", "Value": "Churn Prob >= 40% & Predicted CLV >= $1,500"},
        {"Metric": "Per-Account Intervention Package", "Value": f"${cost_per_package:.2f}"},
        {"Metric": "Total Retention Campaign Budget", "Value": f"${total_campaign_cost:,.2f}"},
        {"Metric": "Simulated Retention Uplift", "Value": f"{retention_success_rate * 100:.0f}% of contacted churners preserved"},
        {"Metric": "Gross Customer Lifetime Value Preserved", "Value": f"${gross_clv_saved:,.2f}"},
        {"Metric": "Net Enterprise Financial Gain", "Value": f"${net_campaign_benefit:,.2f}"},
        {"Metric": "Projected Return on Investment (ROI)", "Value": f"{roi_percent:,.1f}%"}
    ]
    df_sim = pd.DataFrame(simulation_records)
    df_sim.to_csv(os.path.join(output_dir, "retention_campaign_roi_simulation.csv"), index=False)

    # Save integrated customer master ledger
    master_path = os.path.join(output_dir, "integrated_customer_master_ledger.csv")
    df_master.to_csv(master_path, index=False)
    print(f"[INFO] Integrative Synthesis Engine completed. Master ledger exported to: {master_path}")

    return {
        "df_master": df_master,
        "risk_matrix": risk_matrix,
        "df_var": df_var,
        "df_sim": df_sim
    }
