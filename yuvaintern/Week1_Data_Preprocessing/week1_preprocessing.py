# ============================================================
# WEEK 1 PROJECT
# Data Acquisition, Cleaning and Preprocessing
# Student Performance Dataset
# ============================================================

# ------------------------------------------------------------
# 1. Import Libraries
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------

# The UCI Student Performance dataset uses ';' as separator
df = pd.read_csv("data/student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ------------------------------------------------------------
# 3. Initial Data Exploration
# ------------------------------------------------------------

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFO ==========")
print(df.info())


# ------------------------------------------------------------
# 4. Missing Value Analysis
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

print(missing[missing > 0])

if df.isnull().sum().sum() == 0:
    print("No missing values found.")


# ------------------------------------------------------------
# 5. Check G3 = 0 Cases
# ------------------------------------------------------------

print("\n========== G3 = 0 ANALYSIS ==========")

zero_g3 = df[df["G3"] == 0]

print("Students with G3 = 0:", len(zero_g3))

print("\nSample of G3 = 0 students:")
print(
    zero_g3[
        ["G1", "G2", "G3", "failures", "absences"]
    ].head(10)
)


# ------------------------------------------------------------
# 6. Descriptive Statistics
# ------------------------------------------------------------

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(df.describe())


# ------------------------------------------------------------
# 7. Duplicate Check
# ------------------------------------------------------------

print("\n========== DUPLICATE CHECK ==========")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)


# ------------------------------------------------------------
# 8. Check Absences
# ------------------------------------------------------------

print("\n========== ABSENCES ANALYSIS ==========")

print(df["absences"].describe())

Q1 = df["absences"].quantile(0.25)
Q3 = df["absences"].quantile(0.75)

IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

outliers = df[df["absences"] > upper_limit]

print("Students above upper limit:", len(outliers))


# ------------------------------------------------------------
# 9. Visualize Absence Outliers
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(x=df["absences"])

plt.title("Boxplot of Student Absences")
plt.xlabel("Number of Absences")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. Cap Absences at 95th Percentile
# ------------------------------------------------------------

cap_value = df["absences"].quantile(0.95)

print("\n95th percentile cap value:", cap_value)

df["absences"] = df["absences"].clip(
    upper=cap_value
)

print("New maximum absences:", df["absences"].max())


# ------------------------------------------------------------
# 11. Check Age
# ------------------------------------------------------------

print("\n========== AGE ANALYSIS ==========")

print(
    df["age"]
    .value_counts()
    .sort_index()
)

print(
    "\nMinimum age:",
    df["age"].min()
)

print(
    "Maximum age:",
    df["age"].max()
)


# ------------------------------------------------------------
# 12. Check Categorical Columns
# ------------------------------------------------------------

print("\n========== CATEGORICAL VALUES ==========")

categorical_columns = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian"
]

for col in categorical_columns:

    print("\n", col)

    print(df[col].unique())


# ------------------------------------------------------------
# 13. Encode Categorical Variables
# ------------------------------------------------------------

print("\n========== CATEGORICAL ENCODING ==========")

df_encoded = df.copy()

label_encoder = LabelEncoder()

categorical_columns_all = (
    df_encoded
    .select_dtypes(include="object")
    .columns
    .tolist()
)

print(
    "Categorical columns:",
    categorical_columns_all
)

for col in categorical_columns_all:

    df_encoded[col] = (
        label_encoder
        .fit_transform(df_encoded[col])
    )

print("\nEncoding completed.")

print(
    df_encoded.dtypes.unique()
)


# ------------------------------------------------------------
# 14. Create Total Grade Feature
# ------------------------------------------------------------

print("\n========== FEATURE ENGINEERING ==========")

df_encoded["total_grade"] = (
    df_encoded["G1"]
    + df_encoded["G2"]
    + df_encoded["G3"]
)

print(
    "total_grade column created."
)


# ------------------------------------------------------------
# 15. Create Pass/Fail Feature
# ------------------------------------------------------------

df_encoded["pass_fail"] = (
    df_encoded["G3"] >= 10
).astype(int)

print(
    "pass_fail column created."
)

print(
    "\nPass rate:",
    df_encoded["pass_fail"].mean().round(2)
)


# ------------------------------------------------------------
# 16. Correlation Analysis
# ------------------------------------------------------------

print("\n========== CORRELATION ANALYSIS ==========")

correlation_columns = [
    "G1",
    "G2",
    "G3",
    "studytime",
    "failures",
    "absences"
]

grade_corr = (
    df_encoded[correlation_columns]
    .corr()
)

print(
    grade_corr.round(2)
)


# ------------------------------------------------------------
# 17. Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    grade_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Between Student Performance Variables")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 18. Feature Scaling
# ------------------------------------------------------------

print("\n========== FEATURE SCALING ==========")

numeric_columns = [
    "age",
    "absences",
    "studytime",
    "failures",
    "G1",
    "G2",
    "G3",
    "total_grade"
]

scaler = StandardScaler()

df_scaled = df_encoded.copy()

df_scaled[numeric_columns] = (
    scaler.fit_transform(
        df_encoded[numeric_columns]
    )
)

print(
    df_scaled[numeric_columns]
    .describe()
    .round(2)
)


# ------------------------------------------------------------
# 19. Final Validation
# ------------------------------------------------------------

print("\n========== FINAL VALIDATION ==========")

print(
    "Final shape:",
    df_scaled.shape
)

print(
    "Total missing values:",
    df_scaled.isnull().sum().sum()
)

print(
    "Data types:",
    df_scaled.dtypes.unique()
)


# ------------------------------------------------------------
# 20. Save Processed Dataset
# ------------------------------------------------------------

output_file = (
    "output/student_performance_cleaned.csv"
)

df_scaled.to_csv(
    output_file,
    index=False
)

print(
    "\nCleaned dataset saved successfully!"
)

print(
    "File:",
    output_file
)


# ------------------------------------------------------------
# END OF WEEK 1 PROJECT
# ------------------------------------------------------------

print("\n========================================")
print("WEEK 1 DATA PREPROCESSING COMPLETED")
print("========================================")