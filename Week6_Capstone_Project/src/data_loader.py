"""
data_loader.py
Data acquisition, synthesis, auditing, cleaning, and preprocessing for the
Integrative Enterprise Customer Analytics Capstone Pipeline.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_or_generate_enterprise_data(data_dir, n_samples=5000, random_state=42):
    """
    Loads or generates an enterprise customer analytics dataset with realistic
    transactional, operational, usage, and financial distributions.
    """
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "raw_enterprise_customer_profiles.csv")

    if os.path.exists(raw_path):
        print(f"[INFO] Ingesting existing enterprise dataset from: {raw_path}")
        df = pd.read_csv(raw_path)
        return df

    print(f"[INFO] Generating enterprise customer profile dataset (N = {n_samples:,})...")
    np.random.seed(random_state)

    # 1. Customer IDs and Demographics
    cust_ids = [f"CUST_{i+10001:05d}" for i in range(n_samples)]
    age = np.random.normal(loc=42.5, scale=13.0, size=n_samples).clip(18, 80).astype(int)
    gender = np.random.choice(["Female", "Male"], size=n_samples, p=[0.50, 0.50])
    senior_citizen = (age >= 65).astype(int)
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes", np.random.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45]), "No")

    # 2. Account & Tenure (Beta distribution skewed toward newer accounts with long-tail veterans)
    tenure_months = (np.random.beta(a=1.2, b=1.5, size=n_samples) * 72).astype(int)

    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract_type = []
    for t in tenure_months:
        if t < 12:
            p = [0.75, 0.18, 0.07]
        elif t < 36:
            p = [0.45, 0.35, 0.20]
        else:
            p = [0.20, 0.40, 0.40]
        contract_type.append(np.random.choice(contract_choices, p=p))
    contract_type = np.array(contract_type)

    payment_methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    payment_method = np.random.choice(payment_methods, size=n_samples, p=[0.34, 0.22, 0.22, 0.22])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])

    # 3. Services and Data Usage
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.44, 0.44, 0.12])
    tech_support = np.where(internet_service == "No", "No internet service",
                            np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65]))
    online_security = np.where(internet_service == "No", "No internet service",
                               np.random.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62]))
    streaming_services = np.where(internet_service == "No", "No internet service",
                                  np.random.choice(["Yes", "No"], size=n_samples, p=[0.49, 0.51]))

    # Data usage in GB per month (log-normal with outliers)
    base_usage = np.where(internet_service == "Fiber optic", 350.0,
                          np.where(internet_service == "DSL", 120.0, 15.0))
    monthly_usage_gb = np.random.normal(loc=base_usage, scale=base_usage * 0.35).clip(5.0, 1200.0)

    # 4. Operational Interactions & Sentiment
    support_tickets = np.random.poisson(lam=np.where(contract_type == "Month-to-month", 2.2, 0.8), size=n_samples).clip(0, 12)
    payment_delays = np.random.poisson(lam=np.where(payment_method == "Electronic check", 1.4, 0.3), size=n_samples).clip(0, 8)
    satisfaction_score = (5 - 0.4 * support_tickets - 0.3 * payment_delays + np.random.normal(0, 0.5, size=n_samples)).clip(1, 5).round().astype(float)

    # 5. Financials: MonthlyCharges and TotalCharges
    base_charge = np.where(internet_service == "Fiber optic", 78.0,
                           np.where(internet_service == "DSL", 48.0, 20.0))
    add_ons = (tech_support == "Yes") * 15.0 + (online_security == "Yes") * 12.0 + (streaming_services == "Yes") * 22.0
    monthly_charges = (base_charge + add_ons + np.random.normal(0, 4.0, size=n_samples)).clip(18.5, 128.0).round(2)
    total_charges = (monthly_charges * tenure_months + np.random.normal(0, 25.0, size=n_samples)).clip(0.0).round(2)

    # 6. Target 1: Customer Churn (Log-odds function mimicking true human behavior)
    log_odds = (
        -1.8
        + 1.35 * (np.array(contract_type) == "Month-to-month")
        - 1.20 * (np.array(contract_type) == "Two year")
        + 0.55 * (payment_method == "Electronic check")
        + 0.35 * (internet_service == "Fiber optic")
        + 0.28 * support_tickets
        + 0.32 * payment_delays
        - 0.045 * tenure_months
        - 0.40 * (satisfaction_score - 3.0)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-log_odds))
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    # 7. Target 2: Customer Lifetime Value (CLV in $)
    # Formulated as expected discounted forward revenue based on margin, tenure stability, and usage
    clv = (
        monthly_charges * 24.0 * (1.0 - 0.65 * churn)
        + 12.5 * tenure_months
        + 0.85 * monthly_usage_gb
        + np.random.normal(0, 150.0, size=n_samples)
    ).clip(150.0, 5500.0).round(2)

    df = pd.DataFrame({
        "CustomerID": cust_ids,
        "Age": age,
        "Gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "TenureMonths": tenure_months,
        "ContractType": contract_type,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "StreamingServices": streaming_services,
        "MonthlyUsageGB": monthly_usage_gb.round(1),
        "SupportTicketsLastYear": support_tickets,
        "PaymentDelaysCount": payment_delays,
        "CustomerSatisfactionScore": satisfaction_score,
        "MonthlyCharges": monthly_charges,
        "TotalChargesHistorical": total_charges,
        "Churn": churn,
        "CustomerLifetimeValue": clv
    })

    # Introduce realistic data flaws for data hygiene auditing:
    # A few missing values in TotalChargesHistorical and Satisfaction
    missing_tc_idx = np.random.choice(df.index[df["TenureMonths"] == 0], size=min(12, len(df.index[df["TenureMonths"] == 0])), replace=False)
    df.loc[missing_tc_idx, "TotalChargesHistorical"] = np.nan

    missing_sat_idx = np.random.choice(df.index, size=18, replace=False)
    df.loc[missing_sat_idx, "CustomerSatisfactionScore"] = np.nan

    df.to_csv(raw_path, index=False)
    print(f"[INFO] Raw enterprise dataset successfully cached to: {raw_path}")
    return df

def clean_and_preprocess_enterprise_data(df, output_dir):
    """
    Audits missing values, handles data cleaning, encodes categorical features,
    and partitions the data into training and test sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_clean = df.copy()

    # 1. Missing Value Audit & Imputation
    null_counts = df_clean.isnull().sum()
    cleaning_records = []
    for col, count in null_counts.items():
        if count > 0:
            strategy = "Impute with 0.0 (tenure=0 accounts)" if col == "TotalChargesHistorical" else "Impute with median score"
            cleaning_records.append({
                "Feature": col,
                "Null_Count": int(count),
                "Null_Percentage": f"{(count / len(df_clean)) * 100:.2f}%",
                "Remediation_Strategy": strategy
            })

    # Impute
    df_clean["TotalChargesHistorical"] = df_clean["TotalChargesHistorical"].fillna(0.0)
    df_clean["CustomerSatisfactionScore"] = df_clean["CustomerSatisfactionScore"].fillna(df_clean["CustomerSatisfactionScore"].median())

    df_cleaning = pd.DataFrame(cleaning_records)
    df_cleaning.to_csv(os.path.join(output_dir, "data_cleaning_audit.csv"), index=False)

    # 2. Outlier Auditing via Tukey's IQR
    outlier_records = []
    numeric_cols = ["Age", "TenureMonths", "MonthlyUsageGB", "MonthlyCharges", "TotalChargesHistorical", "CustomerLifetimeValue"]
    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
        outlier_records.append({
            "Numeric_Feature": col,
            "Q1": round(q1, 2),
            "Median": round(df_clean[col].median(), 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower_Bound": round(lower, 2),
            "Upper_Bound": round(upper, 2),
            "Outlier_Count": int(outliers),
            "Outlier_Pct": f"{(outliers / len(df_clean)) * 100:.2f}%"
        })
    df_outliers = pd.DataFrame(outlier_records)
    df_outliers.to_csv(os.path.join(output_dir, "outlier_iqr_audit.csv"), index=False)

    # 3. Categorical Encoding (One-Hot Encoding)
    drop_cols = ["CustomerID", "Churn", "CustomerLifetimeValue"]
    X_raw = df_clean.drop(columns=drop_cols)
    y_churn = df_clean["Churn"]
    y_clv = df_clean["CustomerLifetimeValue"]

    categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=float)

    # 4. Train / Test Splitting (80% Train, 20% Test, Stratified by Churn)
    X_train, X_test, y_churn_train, y_churn_test, y_clv_train, y_clv_test = train_test_split(
        X_encoded, y_churn, y_clv,
        test_size=0.20,
        random_state=42,
        stratify=y_churn
    )

    # 5. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print(f"[INFO] Data cleaning complete. Train features: {X_train_scaled.shape}, Test features: {X_test_scaled.shape}")
    print(f"[INFO] Churn Balance: Train={y_churn_train.mean()*100:.1f}%, Test={y_churn_test.mean()*100:.1f}%")

    # Export cleaned dataset
    clean_path = os.path.join(output_dir, "cleaned_enterprise_customers.csv")
    df_clean.to_csv(clean_path, index=False)

    return {
        "df_clean": df_clean,
        "df_cleaning": df_cleaning,
        "df_outliers": df_outliers,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_churn_train": y_churn_train,
        "y_churn_test": y_churn_test,
        "y_clv_train": y_clv_train,
        "y_clv_test": y_clv_test,
        "scaler": scaler
    }
