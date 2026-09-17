"""
data_loader.py
Data loading, cleaning, inspection, outlier detection, and scaling pipeline
for Mall Customer Segmentation dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_and_preprocess_data(csv_path=None):
    """
    Loads Mall_Customers.csv, performs data auditing, outlier analysis,
    feature encoding, and scaling.
    
    Returns:
        df (pd.DataFrame): Raw dataframe with cleaned column names
        df_processed (pd.DataFrame): Dataframe with encoded & scaled features
        X_scaled (np.ndarray): Scaled features for clustering [Income, Spending]
        X_scaled_all (np.ndarray): Scaled features across [Age, Income, Spending, Gender]
        scaler (StandardScaler): Fitted StandardScaler instance
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "Mall_Customers.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    col_mapping = {
        "CustomerID": "CustomerID",
        "Genre": "Gender",
        "Gender": "Gender",
        "Age": "Age",
        "Annual Income (k$)": "Annual_Income_k",
        "Spending Score (1-100)": "Spending_Score"
    }
    df.rename(columns=col_mapping, inplace=True)
    
    # Missing value verification
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"[Warning] Missing values detected:\n{null_counts[null_counts > 0]}")
        df.dropna(inplace=True)
    else:
        print("[Info] Data Hygiene: Zero missing values detected (100% complete).")
    
    # Outlier Detection via IQR
    outlier_info = {}
    for col in ["Age", "Annual_Income_k", "Spending_Score"]:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            "Q25": q25,
            "Q75": q75,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outlier_Count": len(outliers),
            "Outlier_Indices": list(outliers.index)
        }
    
    df_processed = df.copy()
    
    # Categorical encoding: Female=0, Male=1
    df_processed["Gender_Code"] = (df_processed["Gender"].str.strip().str.capitalize() == "Male").astype(int)
    
    # Standard Scaling (Euclidean space distance preservation)
    scaler_2d = StandardScaler()
    feature_cols_2d = ["Annual_Income_k", "Spending_Score"]
    X_scaled_2d = scaler_2d.fit_transform(df_processed[feature_cols_2d])
    
    scaler_all = StandardScaler()
    feature_cols_all = ["Age", "Annual_Income_k", "Spending_Score", "Gender_Code"]
    X_scaled_all = scaler_all.fit_transform(df_processed[feature_cols_all])
    
    # MinMax Scaler for alternative comparison
    minmax_scaler = MinMaxScaler()
    X_minmax_2d = minmax_scaler.fit_transform(df_processed[feature_cols_2d])
    
    # Add scaled columns to df_processed
    df_processed["Annual_Income_Scaled"] = X_scaled_2d[:, 0]
    df_processed["Spending_Score_Scaled"] = X_scaled_2d[:, 1]
    
    # Save preprocessed dataset
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    preprocessed_path = os.path.join(output_dir, "Mall_Customers_preprocessed.csv")
    df_processed.to_csv(preprocessed_path, index=False)
    print(f"[Info] Preprocessed dataset saved at: {preprocessed_path}")
    
    return {
        "df_raw": df,
        "df_processed": df_processed,
        "X_scaled_2d": X_scaled_2d,
        "X_scaled_all": X_scaled_all,
        "X_minmax_2d": X_minmax_2d,
        "scaler_2d": scaler_2d,
        "scaler_all": scaler_all,
        "outlier_info": outlier_info,
        "feature_cols_2d": feature_cols_2d,
        "feature_cols_all": feature_cols_all
    }

if __name__ == "__main__":
    data_dict = load_and_preprocess_data()
    print("Shape:", data_dict["df_raw"].shape)
    print("Columns:", list(data_dict["df_raw"].columns))
    print("Outlier summary:", data_dict["outlier_info"])
