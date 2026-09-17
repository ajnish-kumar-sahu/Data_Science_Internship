"""
data_loader.py
Data ingestion, cleaning, and validation module for Telco Customer Churn dataset.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV dataset from disk.
    
    Parameters:
        filepath (str): Absolute or relative path to the CSV file.
        
    Returns:
        pd.DataFrame: Raw dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw dataset from {filepath} with shape: {df.shape}")
    return df


def audit_and_clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Performs data cleaning, type casting, and sanity validation:
    1. Trims whitespace from string column values.
    2. Identifies and handles whitespace strings in 'TotalCharges' (present when tenure == 0).
    3. Converts 'TotalCharges' to float64.
    4. Encodes target 'Churn' as binary integer {0, 1}.
    5. Preserves customerID for tracking while ensuring it is excluded from model training.
    
    Parameters:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        tuple[pd.DataFrame, dict]: Cleaned dataframe and metadata audit dictionary.
    """
    df_clean = df.copy()
    
    # Strip whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # Strip whitespace across string objects
    str_cols = df_clean.select_dtypes(include=['object']).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        
    # Audit TotalCharges
    # 11 records have tenure = 0 and TotalCharges = '' or ' '
    blank_total_mask = (df_clean['TotalCharges'] == '') | (df_clean['TotalCharges'] == ' ')
    num_blanks = int(blank_total_mask.sum())
    logger.info(f"Detected {num_blanks} blank strings in TotalCharges corresponding to tenure=0.")
    
    # Replace blank strings with 0.0 (customers with 0 tenure have incurred $0 total charges)
    df_clean.loc[blank_total_mask, 'TotalCharges'] = '0.0'
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Ensure numeric columns are strictly float/int
    df_clean['tenure'] = df_clean['tenure'].astype(int)
    df_clean['MonthlyCharges'] = df_clean['MonthlyCharges'].astype(float)
    df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(int)
    
    # Encode target variable
    if 'Churn' in df_clean.columns:
        churn_map = {'Yes': 1, 'No': 0, '1': 1, '0': 0, 1: 1, 0: 0}
        df_clean['Churn'] = df_clean['Churn'].map(churn_map)
        churn_counts = df_clean['Churn'].value_counts().to_dict()
        churn_rate = float(df_clean['Churn'].mean())
        logger.info(f"Target variable Churn distribution: {churn_counts} (Churn Rate: {churn_rate:.2%})")
    else:
        churn_counts = {}
        churn_rate = 0.0

    audit_summary = {
        "initial_rows": len(df),
        "initial_cols": df.shape[1],
        "blank_total_charges_imputed": num_blanks,
        "remaining_nulls": int(df_clean.isnull().sum().sum()),
        "churn_rate": churn_rate,
        "churn_counts": churn_counts
    }
    
    return df_clean, audit_summary


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "Telco-Customer-Churn.csv")
    raw = load_raw_data(data_path)
    clean, audit = audit_and_clean_data(raw)
    print("Audit summary:", audit)
    print("Clean head:\n", clean.head(2))
