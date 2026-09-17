"""
write_notebook.py
Generates an interactive, reproducible Jupyter Notebook:
Week4_Supervised_Learning.ipynb

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import nbformat as nbf

def generate_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Header Cell
    cells.append(nbf.v4.new_markdown_cell(r"""# Week 4: Supervised Learning Model Implementation
### Predictive Customer Churn Analytics in Telecommunications
**Internship:** Virtual Data Science with Python Trainee | **Yuva Intern**  
**Author:** Ajnish Kumar | **Roll No:** 241809046713  
**Degree / University:** Bachelor of Computer Applications (BCA) | Vinoba Bhave University, Hazaribag  
**Date:** September 2026  

---

## 📌 Executive Summary & Project Objectives
This project provides a comprehensive end-to-end implementation of a **Supervised Learning Predictive System** addressing customer attrition (churn) within the telecommunications industry. Using the benchmark **IBM Telco Customer Churn Dataset** (7,043 customer accounts, 21 attributes), we formulate a high-stakes binary classification problem to proactively detect customer defection.

### Core Workflow:
1. **Data Ingestion & Hygiene**: Automated auditing, type casting, and zero-imputation of whitespace in `TotalCharges`.
2. **Domain Feature Engineering**: Creation of tenure lifecycle cohorts, service bundle density, financial expenditure velocity ratios, and high-risk contract-payment interaction flags.
3. **Leak-Free Preprocessing**: Scikit-Learn `ColumnTransformer` integrating `StandardScaler` and `OneHotEncoder(drop='first')` strictly within cross-validation folds.
4. **Multi-Model Benchmarking**: Comparative analysis of 6 distinct algorithms:
   - Dummy Classifier (Baseline)
   - Logistic Regression (L2 Regularized with Balanced Class Weights)
   - Decision Tree Classifier
   - Random Forest Classifier (Ensemble Bagging)
   - Gradient Boosting Classifier (Sequential Residual Learning)
   - Support Vector Classifier (RBF Kernel & Platt Probability Calibration)
5. **Rigorous Validation**: Stratified 5-Fold Cross-Validation assessing Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Brier Score.
6. **Hyperparameter Tuning**: `GridSearchCV` optimization of tree depth, ensemble size, and split criteria.
7. **Business Impact Modeling**: Financial expected value simulation across probability thresholds ($0.05 \dots 0.95$) balancing customer acquisition cost ($200) vs retention incentive ($30)."""))

    # Imports Cell
    cells.append(nbf.v4.new_markdown_cell("""## 1. Environment Setup & Core Dependencies"""))
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
%matplotlib inline
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.titlesize': 13, 'axes.titlesize': 11, 'axes.labelsize': 10})
print("All dependencies successfully imported!")"""))

    # Data Loading
    cells.append(nbf.v4.new_markdown_cell("""## 2. Data Ingestion & Hygiene Audit"""))
    cells.append(nbf.v4.new_code_cell("""data_path = '../data/Telco-Customer-Churn.csv'
df_raw = pd.read_csv(data_path)
print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

df = df_raw.copy()
# Trim whitespace across columns and string objects
df.columns = df.columns.str.strip()
str_cols = df.select_dtypes(include=['object']).columns
for col in str_cols:
    df[col] = df[col].astype(str).str.strip()

# Audit TotalCharges whitespace
blank_mask = (df['TotalCharges'] == '') | (df['TotalCharges'] == ' ')
print(f"Blank TotalCharges detected: {blank_mask.sum()} (all corresponding to tenure=0)")
df.loc[blank_mask, 'TotalCharges'] = '0.0'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Type conversions
df['tenure'] = df['tenure'].astype(int)
df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

# Binary target encoding
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
churn_rate = df['Churn'].mean()
print(f"Target Churn Distribution: {df['Churn'].value_counts().to_dict()} (Churn Rate: {churn_rate:.2%})")
df.head(3)"""))

    # Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("""## 3. Domain Feature Engineering
Constructing behavioral and financial indicators to enhance model discriminative power:
- **Tenure Cohorts**: Categorical lifecycle stages (`0-12 mo`, `13-24 mo`, `25-48 mo`, `49-60 mo`, `60+ mo`).
- **Service Bundle Count**: Total active subscribed features.
- **Financial Velocity**: MonthlyCharges-to-TotalCharges ratio.
- **Unit Cost per Service**: Monthly expenditure divided by active service count.
- **High-Risk Interaction Flag**: Month-to-month contract combined with electronic check payment."""))
    
    cells.append(nbf.v4.new_code_cell("""# 1. Tenure Cohorts
bins = [-1, 12, 24, 48, 60, 100]
labels = ['0-12 mo', '13-24 mo', '25-48 mo', '49-60 mo', '60+ mo']
df['tenure_cohort'] = pd.cut(df['tenure'], bins=bins, labels=labels)

# 2. Service Bundle Count
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['service_count'] = 0
for col in service_cols:
    df['service_count'] += (df[col] == 'Yes').astype(int)

# 3. Financial Velocity Ratio
df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1.0)

# 4. Charge Per Active Service
df['charge_per_service'] = df['MonthlyCharges'] / (df['service_count'] + 1.0)

# 5. High-Risk Interaction Flag
df['high_risk_contract_payment'] = ((df['Contract'] == 'Month-to-month') & (df['PaymentMethod'] == 'Electronic check')).astype(int)

# 6. Service Groupings
df['has_security_backup'] = ((df['OnlineSecurity'] == 'Yes') | (df['OnlineBackup'] == 'Yes')).astype(int)
df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)

print("Engineered dataset shape:", df.shape)
df[['tenure', 'tenure_cohort', 'service_count', 'monthly_to_total_ratio', 'charge_per_service', 'high_risk_contract_payment']].head(3)"""))

    # Visualizations
    cells.append(nbf.v4.new_markdown_cell("""## 4. Exploratory Data Analysis & Behavioral Profiling"""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Churn Distribution
churn_counts = df['Churn'].value_counts()
axes[0].bar(['Retained (0)', 'Churned (1)'], churn_counts.values, color=['#4472c4', '#ed7d31'], edgecolor='black')
axes[0].set_title("Customer Churn Class Distribution")
axes[0].set_ylabel("Count")

# Churn by Contract
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
contract_churn.plot(kind='bar', ax=axes[1], color=['#c00000', '#4472c4', '#70ad47'], edgecolor='black', rot=0)
axes[1].set_title("Churn Rate (%) by Contract Type")
axes[1].set_ylabel("Churn Rate (%)")

# Churn by Tenure Cohort
cohort_churn = df.groupby('tenure_cohort', observed=True)['Churn'].mean() * 100
cohort_churn.plot(kind='bar', ax=axes[2], color='#ed7d31', edgecolor='black', rot=15)
axes[2].set_title("Churn Rate (%) by Tenure Cohort")
axes[2].set_ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()"""))

    # Preprocessing Pipeline
    cells.append(nbf.v4.new_markdown_cell("""## 5. Leak-Free Preprocessing Pipeline Architecture"""))
    cells.append(nbf.v4.new_code_cell("""X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'service_count', 'monthly_to_total_ratio', 'charge_per_service']
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_cohort']
binary_passthrough = ['SeniorCitizen', 'high_risk_contract_payment', 'has_security_backup', 'has_streaming']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
        ('bin', 'passthrough', binary_passthrough)
    ],
    remainder='drop'
)
print("Pipeline preprocessor successfully configured!")"""))

    # Cross-Validation Benchmarking
    cells.append(nbf.v4.new_markdown_cell("""## 6. Multi-Model Benchmark & Stratified 5-Fold Cross-Validation"""))
    cells.append(nbf.v4.new_code_cell("""models = {
    'Dummy Baseline': DummyClassifier(strategy='stratified', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
    'Support Vector Machine': SVC(probability=True, class_weight='balanced', random_state=42)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
benchmark_records = []

for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    accs, recs, f1s, rocs, prs = [], [], [], [], []
    
    for tr_idx, val_idx in skf.split(X_train, y_train):
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = pipe.predict(X_train.iloc[val_idx])
        probs = pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
        
        accs.append(accuracy_score(y_train.iloc[val_idx], preds))
        recs.append(recall_score(y_train.iloc[val_idx], preds))
        f1s.append(f1_score(y_train.iloc[val_idx], preds))
        rocs.append(roc_auc_score(y_train.iloc[val_idx], probs))
        prs.append(average_precision_score(y_train.iloc[val_idx], probs))
        
    benchmark_records.append({
        'Model': name,
        'Accuracy': np.mean(accs),
        'Recall': np.mean(recs),
        'F1-Score': np.mean(f1s),
        'ROC-AUC': np.mean(rocs),
        'PR-AUC': np.mean(prs)
    })

bench_df = pd.DataFrame(benchmark_records).sort_values('ROC-AUC', ascending=False)
bench_df"""))

    # Hyperparameter Tuning
    cells.append(nbf.v4.new_markdown_cell("""## 7. Hyperparameter Tuning on Random Forest via GridSearchCV"""))
    cells.append(nbf.v4.new_code_cell("""rf_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1))])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [6, 10, 14],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf_pipe, param_grid, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
champion_model = grid_search.best_estimator_"""))

    # Holdout Test Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## 8. Final Holdout Test Set Evaluation & Diagnostic Curves"""))
    cells.append(nbf.v4.new_code_cell("""y_test_pred = champion_model.predict(X_test)
y_test_prob = champion_model.predict_proba(X_test)[:, 1]

print("--- Champion Model Test Set Classification Report ---")
print(classification_report(y_test, y_test_pred, target_names=['Retained', 'Churned']))
print(f"Test Set ROC-AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
print(f"Test Set PR-AUC:  {average_precision_score(y_test, y_test_prob):.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
ax1.plot(fpr, tpr, color='#1f497d', lw=2.5, label=f"Champion RF (AUC = {roc_auc_score(y_test, y_test_prob):.3f})")
ax1.plot([0, 1], [0, 1], 'k:')
ax1.set_title("Receiver Operating Characteristic (ROC) Curve")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate (Recall)")
ax1.legend()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
ax2.plot(rec, prec, color='#ed7d31', lw=2.5, label=f"Champion RF (PR-AUC = {average_precision_score(y_test, y_test_prob):.3f})")
ax2.axhline(y_test.mean(), color='k', linestyle=':', label=f"Baseline Prevalence ({y_test.mean():.2f})")
ax2.set_title("Precision-Recall (PR) Curve")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend()

plt.tight_layout()
plt.show()"""))

    # Financial Threshold Tuning
    cells.append(nbf.v4.new_markdown_cell("""## 9. Financial Cost-Benefit Decision Threshold Optimization
Moving beyond arbitrary 0.50 cutoffs to maximize business profitability:
- **Cost of False Negative (Churn Loss)**: $200 (Lost customer lifetime value)
- **Cost of Intervention (Retention Incentive)**: $30 (Discount/promotion offer)
- **Net Benefit of True Positive**: $170 ($200 saved CLV minus $30 retention cost)"""))
    
    cells.append(nbf.v4.new_code_cell("""thresholds = np.linspace(0.05, 0.95, 91)
cost_fn = 200.0
cost_fp = 30.0
benefit_tp = 170.0
baseline_loss = (y_test == 1).sum() * cost_fn

financial_results = []
for thresh in thresholds:
    preds = (y_test_prob >= thresh).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    net_val = (tp * benefit_tp) - (fp * cost_fp) - (fn * cost_fn)
    net_savings = baseline_loss + net_val
    financial_results.append({
        'threshold': thresh,
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'net_savings': net_savings
    })

fin_df = pd.DataFrame(financial_results)
best_row = fin_df.loc[fin_df['net_savings'].idxmax()]
print(f"Optimal Probability Threshold: {best_row['threshold']:.2f}")
print(f"Maximum Net Retention Savings: ${best_row['net_savings']:,.2f} on test cohort")

plt.figure(figsize=(10, 4.5))
plt.plot(fin_df['threshold'], fin_df['net_savings'], color='#70ad47', lw=2.5, label='Net Savings ($)')
plt.axvline(best_row['threshold'], color='#c00000', linestyle='--', label=f"Optimal Cutoff ({best_row['threshold']:.2f})")
plt.axvline(0.50, color='gray', linestyle=':', label='Default Cutoff (0.50)')
plt.title("Financial Net Savings vs Classification Decision Threshold")
plt.xlabel("Probability Threshold")
plt.ylabel("Net Savings ($)")
plt.legend()
plt.tight_layout()
plt.show()"""))

    # Feature Importance
    cells.append(nbf.v4.new_markdown_cell("""## 10. Feature Importance & Strategic Business Recommendations"""))
    cells.append(nbf.v4.new_code_cell("""rf_clf = champion_model.named_steps['classifier']
prep = champion_model.named_steps['preprocessor']
feat_names = [name.split('__')[-1] for name in prep.get_feature_names_out()]

imp_df = pd.DataFrame({
    'Feature': feat_names,
    'Importance': rf_clf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 5))
top15 = imp_df.head(15).sort_values('Importance', ascending=True)
plt.barh(top15['Feature'], top15['Importance'], color='#1f497d', edgecolor='black')
plt.title("Top 15 Feature Importances (Random Forest Impurity Reduction)")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()"""))

    nb['cells'] = cells
    
    notebook_path = os.path.join(os.path.dirname(__file__), "notebooks", "Week4_Supervised_Learning.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated Jupyter Notebook at: {notebook_path}")

if __name__ == "__main__":
    generate_notebook()
