import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR  = os.path.join(BASE_DIR, "visualizations")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
COLORS = ["#C44E52", "#55A868"]
DPI = 150

def save(fig, name):
    fig.savefig(os.path.join(VIZ_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name}")

# ===========================================================================
# Step 1: Load dataset
# ===========================================================================
print("=== Step 1: Loading Titanic Dataset ===")
df_raw = sns.load_dataset("titanic")
print(f"Shape: {df_raw.shape}")
df_raw.to_csv(os.path.join(DATA_DIR, "titanic_raw.csv"), index=False)

# ===========================================================================
# Step 2: Initial Exploration
# ===========================================================================
print("\n=== Step 2: Initial Exploration ===")
print(df_raw.dtypes)
print(df_raw.describe().round(2))
missing = df_raw.isnull().sum()
miss_pct = (missing / len(df_raw) * 100).round(2)
miss_df = pd.DataFrame({"Count": missing, "Pct%": miss_pct})
print(miss_df[missing > 0])

# Fig 1: Missing Values Heatmap
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(df_raw.isnull(), cbar=True, cmap="YlOrRd", yticklabels=False, ax=ax)
ax.set_title("Fig 1 - Missing Values Heatmap (Yellow = Missing)", pad=18, fontsize=14, fontweight="bold")
ax.set_xlabel("Columns")
fig.tight_layout()
save(fig, "fig01_missing_heatmap.png")

# ===========================================================================
# Step 3: Data Cleaning and Feature Engineering
# ===========================================================================
print("\n=== Step 3: Data Cleaning ===")
df = df_raw.copy()
df.drop(columns=["deck", "embark_town", "alive", "class"], inplace=True)
df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda x: x.fillna(x.median()))
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
df["family_size"] = df["sibsp"] + df["parch"]
df["is_alone"] = (df["family_size"] == 0).astype(int)
age_labels = ["Child(0-12)", "Teen(13-18)", "Young Adult(19-35)", "Middle-Aged(36-60)", "Senior(61+)"]
df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100], labels=age_labels)
fare_labels = ["Low", "Medium", "High", "Very High"]
df["fare_band"] = pd.qcut(df["fare"], q=4, labels=fare_labels)
df.to_csv(os.path.join(DATA_DIR, "titanic_cleaned.csv"), index=False)
print(f"Cleaned shape: {df.shape}, nulls remaining: {df.isnull().sum().sum()}")

# ===========================================================================
# Fig 2: Overall Survival Distribution
# ===========================================================================
print("\n=== Fig 2: Survival Distribution ===")
surv = df["survived"].value_counts()
labs = ["Did Not Survive", "Survived"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].pie(surv, labels=labs, colors=COLORS, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[0].set_title("Survival Breakdown (%)", fontsize=13, fontweight="bold")
sns.barplot(x=labs, y=surv.values, palette=COLORS, ax=axes[1], edgecolor="black")
for bar, val in zip(axes[1].patches, surv.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha="center", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 620)
axes[1].set_title("Survival Count", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Survival Status")
axes[1].set_ylabel("Count")
fig.suptitle("Fig 2 - Overall Survival Distribution (n=891)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig02_survival_distribution.png")

# ===========================================================================
# Fig 3: Survival by Sex
# ===========================================================================
print("\n=== Fig 3: Survival by Sex ===")
ss = df.groupby(["sex", "survived"]).size().unstack(fill_value=0)
ss.columns = ["Did Not Survive", "Survived"]
ssp = ss.div(ss.sum(axis=1), axis=0) * 100
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ss.plot(kind="bar", ax=axes[0], color=COLORS, edgecolor="black", rot=0)
axes[0].set_title("Survival Count by Sex", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Sex")
axes[0].set_ylabel("Count")
axes[0].legend(fontsize=10)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 int(bar.get_height()), ha="center", fontsize=10, fontweight="bold")
ssp.plot(kind="bar", ax=axes[1], color=COLORS, edgecolor="black", rot=0)
axes[1].set_title("Survival Rate by Sex (%)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Sex")
axes[1].set_ylabel("Percentage (%)")
axes[1].legend(fontsize=10)
axes[1].set_ylim(0, 105)
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
fig.suptitle("Fig 3 - Survival by Sex", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig03_survival_by_sex.png")
print(ssp.round(1))

# ===========================================================================
# Fig 4: Survival by Passenger Class
# ===========================================================================
print("\n=== Fig 4: Survival by Class ===")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.countplot(data=df, x="pclass", hue="survived",
              palette={0: COLORS[0], 1: COLORS[1]}, edgecolor="black", ax=axes[0])
axes[0].set_title("Survival Count by Passenger Class", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Passenger Class")
axes[0].set_ylabel("Count")
axes[0].legend(["Did Not Survive", "Survived"], fontsize=10)
cr = df.groupby("pclass")["survived"].mean() * 100
brs = axes[1].bar(cr.index, cr.values, color=["#4C72B0", "#55A868", "#DD8452"], edgecolor="black")
for bar, val in zip(brs, cr.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
axes[1].set_title("Survival Rate by Class (%)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Passenger Class")
axes[1].set_ylabel("Survival Rate (%)")
axes[1].set_ylim(0, 80)
axes[1].set_xticks([1, 2, 3])
fig.suptitle("Fig 4 - Survival by Passenger Class", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig04_survival_by_class.png")

# ===========================================================================
# Fig 5: Age Distribution
# ===========================================================================
print("\n=== Fig 5: Age Distribution ===")
age_mean = df["age"].mean()
age_med  = df["age"].median()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["age"].dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(age_mean, color="red", linestyle="--", linewidth=2, label=f"Mean={age_mean:.1f}")
axes[0].axvline(age_med,  color="orange", linestyle="--", linewidth=2, label=f"Median={age_med:.1f}")
axes[0].set_title("Age Distribution (All Passengers)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=10)
for surv_val, col, lbl in zip([0, 1], COLORS, ["Did Not Survive", "Survived"]):
    axes[1].hist(df[df["survived"] == surv_val]["age"].dropna(),
                 bins=25, alpha=0.55, color=col, edgecolor="white", label=lbl)
axes[1].set_title("Age Distribution by Survival", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Frequency")
axes[1].legend(fontsize=10)
fig.suptitle("Fig 5 - Age Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig05_age_distribution.png")

# ===========================================================================
# Fig 6: Fare Distribution
# ===========================================================================
print("\n=== Fig 6: Fare Distribution ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
dfp = df[["pclass", "fare"]].copy()
dfp["pclass"] = dfp["pclass"].astype(str)
sns.boxplot(data=dfp, x="pclass", y="fare",
            palette=["#4C72B0", "#55A868", "#DD8452"],
            order=["1", "2", "3"], ax=axes[0], linewidth=1.5)
axes[0].set_title("Fare Distribution by Class", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Passenger Class")
axes[0].set_ylabel("Fare (GBP)")
axes[0].set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
dfp2 = df[["survived", "fare"]].copy()
dfp2["survived"] = dfp2["survived"].map({0: "Did Not Survive", 1: "Survived"})
dfp2 = dfp2[dfp2["fare"] < 300]
sns.violinplot(data=dfp2, x="survived", y="fare",
               palette={"Did Not Survive": COLORS[0], "Survived": COLORS[1]},
               inner="quartile", ax=axes[1])
axes[1].set_title("Fare Distribution by Survival", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Survival Status")
axes[1].set_ylabel("Fare (GBP, capped at 300)")
fig.suptitle("Fig 6 - Fare Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig06_fare_distribution.png")

# ===========================================================================
# Fig 7: Survival by Age Group
# ===========================================================================
print("\n=== Fig 7: Survival by Age Group ===")
ags = df.groupby(["age_group", "survived"], observed=True).size().unstack(fill_value=0)
ags.columns = ["Did Not Survive", "Survived"]
agp = ags.div(ags.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 5))
xv = np.arange(len(agp))
w = 0.38
b1 = ax.bar(xv - w / 2, agp["Did Not Survive"], w, label="Did Not Survive", color=COLORS[0], edgecolor="black")
b2 = ax.bar(xv + w / 2, agp["Survived"],        w, label="Survived",        color=COLORS[1], edgecolor="black")
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{bar.get_height():.0f}%", ha="center", fontsize=9.5, fontweight="bold")
ax.set_title("Fig 7 - Survival Rate by Age Group (%)", fontsize=14, fontweight="bold", pad=18)
ax.set_xlabel("Age Group")
ax.set_ylabel("Percentage (%)")
ax.set_xticks(xv)
ax.set_xticklabels(agp.index, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 100)
fig.tight_layout()
save(fig, "fig07_survival_by_age_group.png")
print(agp.round(1))

# ===========================================================================
# Fig 8: Correlation Heatmap
# ===========================================================================
print("\n=== Fig 8: Correlation Heatmap ===")
nc = ["survived", "pclass", "age", "sibsp", "parch", "fare", "family_size", "is_alone"]
corr = df[nc].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
            center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
            annot_kws={"size": 11}, ax=ax)
ax.set_title("Fig 8 - Correlation Heatmap (Numeric Features)", fontsize=14, fontweight="bold", pad=18)
fig.tight_layout()
save(fig, "fig08_correlation_heatmap.png")
print(corr["survived"].drop("survived").sort_values(ascending=False).round(3))

# ===========================================================================
# Fig 9: Pairplot
# ===========================================================================
print("\n=== Fig 9: Pairplot ===")
dpp = df[["survived", "age", "fare", "pclass", "family_size"]].copy()
dpp["survived"] = dpp["survived"].map({0: "No", 1: "Yes"})
g = sns.pairplot(dpp, hue="survived", palette={"No": COLORS[0], "Yes": COLORS[1]},
                 diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20})
g.figure.suptitle("Fig 9 - Pairplot of Key Features (by Survival)", fontsize=13, fontweight="bold", y=1.01)
g.figure.savefig(os.path.join(VIZ_DIR, "fig09_pairplot.png"), dpi=DPI, bbox_inches="tight")
plt.close(g.figure)
print("  [saved] fig09_pairplot.png")

# ===========================================================================
# Fig 10: Embarkation Port and Family Size
# ===========================================================================
print("\n=== Fig 10: Embarkation and Family Size ===")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
er = df.groupby("embarked")["survived"].mean() * 100
pm = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
el = [pm.get(p, p) for p in er.index]
brs = axes[0].bar(el, er.values, color=["#4C72B0", "#55A868", "#DD8452"], edgecolor="black")
for bar, val in zip(brs, er.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
axes[0].set_title("Survival Rate by Embarkation Port", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Port of Embarkation")
axes[0].set_ylabel("Survival Rate (%)")
axes[0].set_ylim(0, 75)
frv = df.groupby("family_size")["survived"].mean() * 100
axes[1].plot(frv.index, frv.values, marker="o", linewidth=2.5, markersize=8,
             color="#4C72B0", markerfacecolor="#DD8452")
for xv2, yv in zip(frv.index, frv.values):
    axes[1].annotate(f"{yv:.0f}%", xy=(xv2, yv), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
axes[1].set_title("Survival Rate by Family Size", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Family Size (sibsp + parch)")
axes[1].set_ylabel("Survival Rate (%)")
axes[1].set_xticks(frv.index)
axes[1].set_ylim(0, 90)
fig.suptitle("Fig 10 - Embarkation Port and Family Size", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig10_embarkation_family.png")

# ===========================================================================
# Fig 11: Class x Sex Heatmap
# ===========================================================================
print("\n=== Fig 11: Class x Sex Heatmap ===")
piv = df.pivot_table(values="survived", index="sex", columns="pclass", aggfunc="mean") * 100
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(piv, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=0.5,
            linecolor="white", annot_kws={"size": 13, "weight": "bold"}, vmin=0, vmax=100, ax=ax)
ax.set_title("Fig 11 - Survival Rate (%) by Class and Sex", fontsize=14, fontweight="bold", pad=18)
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Sex")
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"], fontsize=11)
fig.tight_layout()
save(fig, "fig11_class_sex_heatmap.png")
print(piv.round(1))

# ===========================================================================
# Fig 12: Fare Band Survival
# ===========================================================================
print("\n=== Fig 12: Fare Band Survival ===")
fbs = df.groupby("fare_band", observed=True)["survived"].mean() * 100
fig, ax = plt.subplots(figsize=(9, 5))
brs2 = ax.bar(fbs.index.astype(str), fbs.values,
              color=["#C44E52", "#DD8452", "#4C72B0", "#55A868"], edgecolor="black")
for bar, val in zip(brs2, fbs.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_title("Fig 12 - Survival Rate by Fare Band", fontsize=14, fontweight="bold", pad=18)
ax.set_xlabel("Fare Band (quartile-based)")
ax.set_ylabel("Survival Rate (%)")
ax.set_ylim(0, 75)
fig.tight_layout()
save(fig, "fig12_fare_band_survival.png")

# ===========================================================================
# Statistical Tests
# ===========================================================================
print("\n=== Statistical Tests (Chi-Square) ===")
def chi2t(col):
    ct = pd.crosstab(df[col], df["survived"])
    chi2_val, p, dof, _ = stats.chi2_contingency(ct)
    sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"
    print(f"  {col:15s}: chi2={chi2_val:.2f}, p={p:.4f}  -> {sig}")

for col in ["sex", "pclass", "embarked", "is_alone", "fare_band", "age_group"]:
    chi2t(col)

surv_total = df["survived"].sum()
surv_rate  = df["survived"].mean() * 100
alone_total = df["is_alone"].sum()
print(f"\nTotal passengers : {len(df)}")
print(f"Survivors        : {surv_total} ({surv_rate:.1f}%)")
print(f"Mean age         : {df['age'].mean():.1f}")
print(f"Median fare      : {df['fare'].median():.2f}")
print(f"Travelling alone : {alone_total} ({df['is_alone'].mean()*100:.1f}%)")
print(f"\nAll visualizations saved to: {VIZ_DIR}")
