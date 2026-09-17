"""
feature_engineering.py
Domain feature engineering and leak-free preprocessing pipeline constructor.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs domain-specific features for customer churn prediction:
    1. tenure_cohort: Categorical lifecycle bins.
    2. service_count: Aggregate count of active products/services.
    3. monthly_to_total_ratio: Financial expenditure velocity.
    4. charge_per_service: Unit cost per active service subscribed.
    5. high_risk_contract_payment: Interaction term for Month-to-month + Electronic check.
    6. has_security_backup: Indicator for switching-cost protective services.
    7. has_streaming: Indicator for high-bandwidth entertainment services.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataframe.
        
    Returns:
        pd.DataFrame: Feature-engineered dataframe.
    """
    df_feat = df.copy()
    
    # 1. Tenure Cohorts
    bins = [-1, 12, 24, 48, 60, 100]
    labels = ["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "60+ mo"]
    df_feat["tenure_cohort"] = pd.cut(df_feat["tenure"], bins=bins, labels=labels)
    
    # 2. Service Count
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Count occurrences where value is 'Yes'
    df_feat["service_count"] = 0
    for col in service_cols:
        if col in df_feat.columns:
            df_feat["service_count"] += (df_feat[col] == "Yes").astype(int)
            
    # 3. Monthly to Total Charges Ratio
    # Adding small epsilon to avoid division by zero
    df_feat["monthly_to_total_ratio"] = df_feat["MonthlyCharges"] / (df_feat["TotalCharges"] + 1.0)
    
    # 4. Charge Per Service
    df_feat["charge_per_service"] = df_feat["MonthlyCharges"] / (df_feat["service_count"] + 1.0)
    
    # 5. High Risk Interaction Flag: Month-to-month + Electronic check
    contract_mask = df_feat["Contract"] == "Month-to-month"
    payment_mask = df_feat["PaymentMethod"] == "Electronic check"
    df_feat["high_risk_contract_payment"] = (contract_mask & payment_mask).astype(int)
    
    # 6. Protective Security / Backup Engagement
    sec_mask = df_feat["OnlineSecurity"] == "Yes"
    bak_mask = df_feat["OnlineBackup"] == "Yes"
    df_feat["has_security_backup"] = (sec_mask | bak_mask).astype(int)
    
    # 7. Entertainment Streaming Engagement
    tv_mask = df_feat["StreamingTV"] == "Yes"
    mov_mask = df_feat["StreamingMovies"] == "Yes"
    df_feat["has_streaming"] = (tv_mask | mov_mask).astype(int)
    
    return df_feat


def get_feature_lists():
    """
    Returns the organized feature lists for preprocessing pipelines.
    """
    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "service_count",
        "monthly_to_total_ratio",
        "charge_per_service"
    ]
    
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure_cohort"
    ]
    
    binary_passthrough = [
        "SeniorCitizen",
        "high_risk_contract_payment",
        "has_security_backup",
        "has_streaming"
    ]
    
    return numeric_features, categorical_features, binary_passthrough


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer ensuring no data leakage:
    - StandardScaler applied to continuous numerical variables.
    - OneHotEncoder(drop='first', sparse_output=False) applied to nominal categoricals.
    - Passthrough for binary zero/one indicator features.
    
    Returns:
        ColumnTransformer: Preprocessing transformer.
    """
    numeric_features, categorical_features, binary_passthrough = get_feature_lists()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
            ("bin", "passthrough", binary_passthrough)
        ],
        remainder="drop"
    )
    return preprocessor


if __name__ == "__main__":
    from data_loader import load_raw_data, audit_and_clean_data
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = load_raw_data(os.path.join(base_dir, "data", "Telco-Customer-Churn.csv"))
    clean, _ = audit_and_clean_data(raw)
    feat = engineer_features(clean)
    prep = build_preprocessor()
    X = feat.drop(columns=["customerID", "Churn"])
    X_trans = prep.fit_transform(X)
    print("Engineered Shape:", feat.shape)
    print("Transformed Matrix Shape:", X_trans.shape)
