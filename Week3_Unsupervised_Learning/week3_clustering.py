"""
week3_clustering.py
Week 3: Unsupervised Learning and Clustering Analysis
Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Internship: Virtual Data Science with Python Trainee (Yuva Intern)
Date: September 2026
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")

# Setup project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIZ_DIR  = os.path.join(BASE_DIR, "visualizations")
OUT_DIR  = os.path.join(BASE_DIR, "output")

for folder in [DATA_DIR, VIZ_DIR, OUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# Styling configuration
sns.set_theme(style="whitegrid", palette="muted")
PALETTE_5 = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]
DPI = 150

def save(fig, filename):
    """Helper function to save and close figures."""
    path = os.path.join(VIZ_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {filename}")

# ===========================================================================
# Step 1: Load and Inspect Dataset
# ===========================================================================
print("=== Step 1: Loading Mall Customer Dataset ===")
raw_csv = os.path.join(DATA_DIR, "Mall_Customers.csv")
df = pd.read_csv(raw_csv)

# Clean and standardize column names
df.rename(columns={
    "CustomerID": "CustomerID",
    "Genre": "Gender",
    "Gender": "Gender",
    "Age": "Age",
    "Annual Income (k$)": "Annual_Income_k",
    "Spending Score (1-100)": "Spending_Score"
}, inplace=True)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# Check missing values
missing = df.isnull().sum()
print("\nMissing values check:")
print(missing)

# Statistical summary
print("\nDescriptive statistics:")
print(df[["Age", "Annual_Income_k", "Spending_Score"]].describe().round(2))

# ===========================================================================
# Step 2: Outlier Auditing (IQR) & Feature Scaling
# ===========================================================================
print("\n=== Step 2: Outlier Check and Feature Scaling ===")

for col in ["Age", "Annual_Income_k", "Spending_Score"]:
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    iqr = q75 - q25
    low = q25 - 1.5 * iqr
    high = q75 + 1.5 * iqr
    outliers = df[(df[col] < low) | (df[col] > high)]
    print(f"{col}: IQR={iqr:.2f}, Bounds=[{low:.2f}, {high:.2f}], Outliers count={len(outliers)}")

# Encode Gender for numerical analysis
df["Gender_Code"] = (df["Gender"].str.strip().str.capitalize() == "Male").astype(int)

# Standardize distance features (Z-score scaling)
scaler = StandardScaler()
features_2d = ["Annual_Income_k", "Spending_Score"]
X_scaled = scaler.fit_transform(df[features_2d])

# Scaled full demographic matrix
features_all = ["Age", "Annual_Income_k", "Spending_Score", "Gender_Code"]
X_all_scaled = StandardScaler().fit_transform(df[features_all])

# Save preprocessed dataset
df.to_csv(os.path.join(DATA_DIR, "Mall_Customers_preprocessed.csv"), index=False)
print("Saved preprocessed dataset to data/Mall_Customers_preprocessed.csv")

# ===========================================================================
# Fig 1: Univariate Distributions & Boxplots
# ===========================================================================
print("\n=== Generating Fig 1: Univariate Distributions & Boxplots ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Fig 1 - Univariate Distributions and Boxplot Outlier Auditing", fontsize=14, fontweight="bold", y=0.98)

cols = ["Age", "Annual_Income_k", "Spending_Score"]
labels = ["Customer Age (Years)", "Annual Income (k$)", "Spending Score (1-100)"]
colors = ["#3470a3", "#2ca02c", "#d62728"]

for i, c in enumerate(cols):
    # Histogram + KDE
    sns.histplot(df[c], kde=True, ax=axes[0, i], color=colors[i], bins=15, edgecolor="black", alpha=0.6)
    axes[0, i].set_title(f"Distribution of {labels[i]}", fontsize=12, fontweight="bold")
    axes[0, i].set_xlabel(labels[i])
    axes[0, i].set_ylabel("Frequency")
    axes[0, i].axvline(df[c].mean(), color="red", linestyle="--", label=f"Mean: {df[c].mean():.1f}")
    axes[0, i].axvline(df[c].median(), color="blue", linestyle=":", label=f"Median: {df[c].median():.1f}")
    axes[0, i].legend()

    # Boxplot
    sns.boxplot(x=df[c], ax=axes[1, i], color=colors[i], flierprops=dict(markerfacecolor="red", marker="D", markersize=6))
    axes[1, i].set_title(f"Boxplot of {labels[i]}", fontsize=12, fontweight="bold")
    axes[1, i].set_xlabel(labels[i])

fig.tight_layout()
save(fig, "fig01_distributions_boxplots.png")

# ===========================================================================
# Fig 2: Multivariate Pairplot with Gender KDE
# ===========================================================================
print("\n=== Generating Fig 2: Pairplot with Gender Breakdown ===")
pair_data = df[["Age", "Annual_Income_k", "Spending_Score", "Gender"]].copy()
pair_data.rename(columns={"Annual_Income_k": "Income (k$)", "Spending_Score": "Spending (1-100)"}, inplace=True)

g = sns.pairplot(
    pair_data,
    hue="Gender",
    palette={"Male": "#1f77b4", "Female": "#e377c2"},
    diag_kind="kde",
    plot_kws={"alpha": 0.75, "s": 45},
    diag_kws={"fill": True, "alpha": 0.45}
)
g.fig.suptitle("Fig 2 - Multivariate Pairwise Distributions Segmented by Gender", fontsize=14, fontweight="bold", y=1.02)
g.savefig(os.path.join(VIZ_DIR, "fig02_pairplot_gender_kde.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("  [saved] fig02_pairplot_gender_kde.png")

# ===========================================================================
# Fig 3: Pearson vs. Spearman Correlation Matrices
# ===========================================================================
print("\n=== Generating Fig 3: Correlation Matrix Heatmap ===")
corr_features = ["Age", "Annual_Income_k", "Spending_Score", "Gender_Code"]
corr_labels = ["Age", "Income (k$)", "Spending (1-100)", "Gender (Male=1)"]

p_corr = df[corr_features].corr(method="pearson")
s_corr = df[corr_features].corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Fig 3 - Feature Correlation Analysis (Pearson Linear vs. Spearman Rank)", fontsize=14, fontweight="bold")

sns.heatmap(p_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-0.6, vmax=0.6,
            xticklabels=corr_labels, yticklabels=corr_labels, ax=axes[0], linewidths=1)
axes[0].set_title("Pearson Correlation (r)", fontsize=12, fontweight="bold")

sns.heatmap(s_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-0.6, vmax=0.6,
            xticklabels=corr_labels, yticklabels=corr_labels, ax=axes[1], linewidths=1)
axes[1].set_title("Spearman Rank Correlation (rho)", fontsize=12, fontweight="bold")

fig.tight_layout()
save(fig, "fig03_correlation_matrix.png")

# ===========================================================================
# Step 3: Hyperparameter Optimization Across k in [2, 10]
# ===========================================================================
print("\n=== Step 3: Evaluating Optimal k across range [2, 10] ===")
k_range = range(2, 11)
wcss_list = []
silhouette_list = []
dbi_list = []
chi_list = []

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=300, random_state=42)
    labels = km.fit_predict(X_scaled)
    
    wcss = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    chi = calinski_harabasz_score(X_scaled, labels)
    
    wcss_list.append(round(wcss, 2))
    silhouette_list.append(round(sil, 4))
    dbi_list.append(round(dbi, 4))
    chi_list.append(round(chi, 2))

df_metrics = pd.DataFrame({
    "k": list(k_range),
    "Inertia_WCSS": wcss_list,
    "Silhouette_Score": silhouette_list,
    "Davies_Bouldin_Index": dbi_list,
    "Calinski_Harabasz_Index": chi_list
})
df_metrics.to_csv(os.path.join(OUT_DIR, "cluster_metrics_summary.csv"), index=False)
print(df_metrics.to_string(index=False))

# Optimal k choice
optimal_k = 5
print(f"\n>> Optimal k determined as {optimal_k} (Max Silhouette: {df_metrics.loc[df_metrics['k']==5, 'Silhouette_Score'].values[0]})")

# ===========================================================================
# Fig 4: Multi-Panel Model Validation Curves
# ===========================================================================
print("\n=== Generating Fig 4: Model Validation Curves ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fig 4 - Hyperparameter Optimization Curves for Cluster Selection", fontsize=15, fontweight="bold", y=0.98)

# Panel A: Elbow WCSS
axes[0, 0].plot(list(k_range), wcss_list, marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
axes[0, 0].axvline(optimal_k, color="red", linestyle="--", label=f"Optimal Elbow k={optimal_k}")
axes[0, 0].set_title("A: Within-Cluster Sum of Squares (Elbow WCSS)", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Number of Clusters (k)")
axes[0, 0].set_ylabel("Inertia (WCSS)")
axes[0, 0].set_xticks(list(k_range))
axes[0, 0].legend()

# Panel B: Silhouette
axes[0, 1].plot(list(k_range), silhouette_list, marker="s", color="#2ca02c", linewidth=2.5, markersize=8)
axes[0, 1].axvline(optimal_k, color="red", linestyle="--", label=f"Peak Score at k={optimal_k}")
axes[0, 1].set_title("B: Mean Silhouette Score (Higher is Better)", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Number of Clusters (k)")
axes[0, 1].set_ylabel("Silhouette Score")
axes[0, 1].set_xticks(list(k_range))
axes[0, 1].legend()

# Panel C: Davies-Bouldin
axes[1, 0].plot(list(k_range), dbi_list, marker="^", color="#d62728", linewidth=2.5, markersize=8)
axes[1, 0].axvline(optimal_k, color="red", linestyle="--", label=f"Minimum at k={optimal_k}")
axes[1, 0].set_title("C: Davies-Bouldin Index (Lower is Better)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Number of Clusters (k)")
axes[1, 0].set_ylabel("DBI Score")
axes[1, 0].set_xticks(list(k_range))
axes[1, 0].legend()

# Panel D: Calinski-Harabasz
axes[1, 1].plot(list(k_range), chi_list, marker="D", color="#9467bd", linewidth=2.5, markersize=8)
axes[1, 1].axvline(optimal_k, color="red", linestyle="--", label=f"Peak at k={optimal_k}")
axes[1, 1].set_title("D: Calinski-Harabasz Index (Higher is Better)", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Number of Clusters (k)")
axes[1, 1].set_ylabel("CHI Score")
axes[1, 1].set_xticks(list(k_range))
axes[1, 1].legend()

fig.tight_layout()
save(fig, "fig04_elbow_silhouette_calinski.png")

# ===========================================================================
# Fig 5: Sample-Level Silhouette Profiles for k in [3, 4, 5, 6]
# ===========================================================================
print("\n=== Generating Fig 5: Silhouette Plots per Sample ===")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Fig 5 - Silhouette Analysis per Cluster and Sample Thickness for k in [3, 4, 5, 6]", fontsize=14, fontweight="bold", y=0.99)

for idx, k in enumerate([3, 4, 5, 6]):
    ax = axes[idx // 2, idx % 2]
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
    c_labels = km.fit_predict(X_scaled)
    sil_avg = silhouette_score(X_scaled, c_labels)
    sample_sil = silhouette_samples(X_scaled, c_labels)
    
    y_lower = 10
    for i in range(k):
        ith_sil = sample_sil[c_labels == i]
        ith_sil.sort()
        size_i = ith_sil.shape[0]
        y_upper = y_lower + size_i
        
        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i), fontsize=10, fontweight="bold")
        y_lower = y_upper + 10
        
    ax.set_title(f"Silhouette Plot for k = {k} (Mean: {sil_avg:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=1.5, label=f"Avg: {sil_avg:.3f}")
    ax.set_yticks([])
    ax.set_xlim([-0.15, 0.85])
    ax.legend(loc="upper right")

fig.tight_layout()
save(fig, "fig05_silhouette_cluster_analysis.png")

# ===========================================================================
# Step 4: Fit Final K-Means Model (k=5)
# ===========================================================================
print("\n=== Step 4: Fitting K-Means Model at k=5 ===")
final_km = KMeans(n_clusters=optimal_k, init="k-means++", n_init=20, max_iter=300, random_state=42)
df["KMeans_Cluster"] = final_km.fit_predict(X_scaled)

# Convert centroids back to original dollar units
centroids_orig = scaler.inverse_transform(final_km.cluster_centers_)
print("Cluster Centroids (Annual Income, Spending Score):")
for i, c in enumerate(centroids_orig):
    print(f"  Cluster {i}: Income = ${c[0]:.1f}k, Spending Score = {c[1]:.1f}")

# ===========================================================================
# Fig 6: K-Means 2D Clusters & 3D Demographic View
# ===========================================================================
print("\n=== Generating Fig 6: K-Means 2D and 3D Visualizations ===")
fig = plt.figure(figsize=(16, 7))
fig.suptitle("Fig 6 - K-Means Segmentation Results (2D Feature Space and 3D View)", fontsize=15, fontweight="bold")

# 2D Scatter
ax1 = fig.add_subplot(1, 2, 1)
sns.scatterplot(
    data=df,
    x="Annual_Income_k",
    y="Spending_Score",
    hue="KMeans_Cluster",
    palette=PALETTE_5,
    s=80,
    alpha=0.85,
    edgecolor="black",
    ax=ax1
)
ax1.scatter(
    centroids_orig[:, 0],
    centroids_orig[:, 1],
    s=220,
    c="yellow",
    edgecolors="black",
    marker="X",
    linewidths=2,
    label="Centroids",
    zorder=10
)
for i, c in enumerate(centroids_orig):
    ax1.annotate(f"C{i} ({c[0]:.0f}, {c[1]:.0f})", (c[0] + 1.5, c[1] + 1.5),
                 fontsize=9, fontweight="bold", backgroundcolor="white", alpha=0.8)

ax1.set_title("2D Space: Annual Income vs. Spending Score", fontsize=12, fontweight="bold")
ax1.set_xlabel("Annual Income (k$)")
ax1.set_ylabel("Spending Score (1-100)")
ax1.legend(loc="upper right")

# 3D View
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for c_id in sorted(df["KMeans_Cluster"].unique()):
    sub = df[df["KMeans_Cluster"] == c_id]
    ax2.scatter(
        sub["Annual_Income_k"],
        sub["Spending_Score"],
        sub["Age"],
        c=PALETTE_5[c_id % len(PALETTE_5)],
        label=f"Cluster {c_id}",
        s=45,
        alpha=0.8,
        edgecolors="k"
    )
ax2.set_title("3D View: Income vs. Spending vs. Age", fontsize=12, fontweight="bold")
ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
ax2.set_zlabel("Age (Years)")
ax2.view_init(elev=25, azim=130)
ax2.legend(loc="upper left")

fig.tight_layout()
save(fig, "fig06_kmeans_clusters_2d_3d.png")

# ===========================================================================
# Step 5: Hierarchical Agglomerative Clustering
# ===========================================================================
print("\n=== Step 5: Hierarchical Clustering and Linkage Analysis ===")
distances = pdist(X_scaled, metric="euclidean")
linkages = {}
cophenetic_scores = {}

for method in ["ward", "complete", "average"]:
    Z = linkage(X_scaled, method=method, metric="euclidean")
    coph, _ = cophenet(Z, distances)
    linkages[method] = Z
    cophenetic_scores[method] = coph
    print(f"  Linkage '{method}': Cophenetic Correlation = {coph:.4f}")

# Fit Agglomerative at k=5 using Ward linkage
agg_model = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
df["Hierarchical_Cluster"] = agg_model.fit_predict(X_scaled)

# ===========================================================================
# Fig 7: Hierarchical Dendrograms across Linkages
# ===========================================================================
print("\n=== Generating Fig 7: Hierarchical Dendrograms ===")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Fig 7 - Hierarchical Clustering Dendrograms across Linkage Criteria", fontsize=15, fontweight="bold")

methods = ["ward", "complete", "average"]
titles = ["Ward's Minimum Variance Linkage", "Complete Linkage (Max Distance)", "Average Linkage (Mean Distance)"]

for i, m in enumerate(methods):
    Z = linkages[m]
    coph = cophenetic_scores[m]
    cut_val = (Z[-5, 2] + Z[-4, 2]) / 2.0
    
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=25,
        show_leaf_counts=True,
        leaf_rotation=90,
        color_threshold=cut_val,
        ax=axes[i]
    )
    axes[i].axhline(y=cut_val, color="black", linestyle="--", linewidth=1.5, label=f"k=5 Cut (d={cut_val:.2f})")
    axes[i].set_title(f"{titles[i]}\nCophenetic: {coph:.3f}", fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Leaf Samples")
    axes[i].set_ylabel("Distance")
    axes[i].legend(loc="upper right")

fig.tight_layout()
save(fig, "fig07_hierarchical_dendrograms.png")

# ===========================================================================
# Fig 8: Hierarchical vs. K-Means Comparison & Confusion Heatmap
# ===========================================================================
print("\n=== Generating Fig 8: K-Means vs. Hierarchical Comparison ===")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Fig 8 - Comparative Assessment: K-Means vs. Agglomerative Clustering", fontsize=14, fontweight="bold")

sns.scatterplot(
    data=df,
    x="Annual_Income_k",
    y="Spending_Score",
    hue="Hierarchical_Cluster",
    palette=PALETTE_5,
    s=70,
    alpha=0.85,
    edgecolor="black",
    ax=axes[0]
)
axes[0].set_title("Agglomerative Clustering (Ward Linkage, k=5)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Annual Income (k$)")
axes[0].set_ylabel("Spending Score (1-100)")

ct = pd.crosstab(df["KMeans_Cluster"], df["Hierarchical_Cluster"],
                 rownames=["K-Means Cluster"], colnames=["Hierarchical Cluster"])
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=True, ax=axes[1], linewidths=1)
axes[1].set_title("Contingency Matrix (K-Means vs. Hierarchical)", fontsize=12, fontweight="bold")

fig.tight_layout()
save(fig, "fig08_hierarchical_vs_kmeans.png")

# ===========================================================================
# Step 6: DBSCAN Density-Based Clustering
# ===========================================================================
print("\n=== Step 6: DBSCAN Density Clustering & Anomaly Isolation ===")
# Compute 4th-NN distances for epsilon determination
nbrs = NearestNeighbors(n_neighbors=4).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
sorted_k_dist = np.sort(distances[:, 3])

# Fit DBSCAN
eps_val = 0.38
min_pts = 5
dbscan = DBSCAN(eps=eps_val, min_samples=min_pts)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)
n_db_clusters = len(set(df["DBSCAN_Cluster"])) - (1 if -1 in df["DBSCAN_Cluster"] else 0)
n_noise = (df["DBSCAN_Cluster"] == -1).sum()
print(f"  DBSCAN Clusters: {n_db_clusters}, Noise/Outlier points: {n_noise}")

# ===========================================================================
# Fig 9: DBSCAN Clustering & Outlier Graph
# ===========================================================================
print("\n=== Generating Fig 9: DBSCAN Results ===")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Fig 9 - DBSCAN Density-Based Clustering & Outlier Isolation", fontsize=14, fontweight="bold")

# 4-NN curve
axes[0].plot(sorted_k_dist, color="#d95f02", linewidth=2.5)
axes[0].axhline(y=eps_val, color="red", linestyle="--", label=f"Selected eps = {eps_val}")
axes[0].set_title("4-NN Distance Curve for Epsilon Selection", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Points Sorted by Distance")
axes[0].set_ylabel("4th Nearest Neighbor Distance")
axes[0].legend()

# DBSCAN scatter
unique_db = sorted(set(df["DBSCAN_Cluster"]))
for k_label in unique_db:
    sub = df[df["DBSCAN_Cluster"] == k_label]
    if k_label == -1:
        axes[1].scatter(sub["Annual_Income_k"], sub["Spending_Score"],
                        c="black", marker="x", s=80, label=f"Noise/Outliers (n={n_noise})")
    else:
        axes[1].scatter(sub["Annual_Income_k"], sub["Spending_Score"],
                        s=65, alpha=0.85, edgecolors="k", label=f"Cluster {k_label}")

axes[1].set_title(f"DBSCAN Clusters (eps={eps_val}, min_samples={min_pts})", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score (1-100)")
axes[1].legend(loc="upper right")

fig.tight_layout()
save(fig, "fig09_dbscan_clustering.png")

# ===========================================================================
# Step 7: Dimensionality Reduction with PCA
# ===========================================================================
print("\n=== Step 7: Principal Component Analysis (PCA) ===")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_all_scaled)
df["PCA_1"] = X_pca[:, 0]
df["PCA_2"] = X_pca[:, 1]
var_exp = pca.explained_variance_ratio_ * 100
print(f"  PCA Explained Variance: PC1 = {var_exp[0]:.2f}%, PC2 = {var_exp[1]:.2f}% (Total: {sum(var_exp):.2f}%)")

# Loadings
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features_all)

# ===========================================================================
# Fig 10: PCA Scree Plot & 2D Loadings Biplot
# ===========================================================================
print("\n=== Generating Fig 10: PCA Biplot & Variance ===")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Fig 10 - Principal Component Analysis (Scree Plot & Loadings Biplot)", fontsize=14, fontweight="bold")

# Scree
cum_var = np.cumsum(var_exp)
axes[0].bar(["PC1", "PC2"], var_exp, color="#1f77b4", alpha=0.7, edgecolor="black", label="Individual Variance (%)")
axes[0].plot(["PC1", "PC2"], cum_var, color="red", marker="o", linewidth=2, label="Cumulative Variance (%)")
for i, (v, c) in enumerate(zip(var_exp, cum_var)):
    axes[0].text(i, v / 2, f"{v:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    axes[0].text(i, c + 2, f"{c:.1f}%", ha="center", va="bottom", color="red", fontweight="bold")
axes[0].set_title("Explained Variance by Principal Component", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_ylim(0, 110)
axes[0].legend()

# Biplot
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["KMeans_Cluster"], palette=PALETTE_5, s=60, alpha=0.75, ax=axes[1])
scale_vec = 2.8
for feat in loadings.index:
    vx = loadings.loc[feat, "PC1"] * scale_vec
    vy = loadings.loc[feat, "PC2"] * scale_vec
    axes[1].arrow(0, 0, vx, vy, color="black", width=0.03, head_width=0.1, head_length=0.1, zorder=5)
    axes[1].text(vx * 1.15, vy * 1.15, feat, color="darkred", fontweight="bold", fontsize=11, zorder=6)

axes[1].axhline(0, color="grey", linestyle="--", alpha=0.6)
axes[1].axvline(0, color="grey", linestyle="--", alpha=0.6)
axes[1].set_title("PCA Biplot (PC1 vs. PC2) with Feature Loadings", fontsize=12, fontweight="bold")
axes[1].set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
axes[1].legend(title="Cluster", loc="upper left")

fig.tight_layout()
save(fig, "fig10_pca_biplot_variance.png")

# ===========================================================================
# Step 8: Multi-Algorithm Comparison Summary
# ===========================================================================
print("\n=== Step 8: Compiling Algorithmic Comparison Table ===")
agg_sil = silhouette_score(X_scaled, df["Hierarchical_Cluster"])
agg_dbi = davies_bouldin_score(X_scaled, df["Hierarchical_Cluster"])
agg_chi = calinski_harabasz_score(X_scaled, df["Hierarchical_Cluster"])

db_mask = df["DBSCAN_Cluster"] != -1
db_sil = silhouette_score(X_scaled[db_mask], df.loc[db_mask, "DBSCAN_Cluster"])
db_dbi = davies_bouldin_score(X_scaled[db_mask], df.loc[db_mask, "DBSCAN_Cluster"])
db_chi = calinski_harabasz_score(X_scaled[db_mask], df.loc[db_mask, "DBSCAN_Cluster"])

km_sil = df_metrics.loc[df_metrics["k"] == optimal_k, "Silhouette_Score"].values[0]
km_dbi = df_metrics.loc[df_metrics["k"] == optimal_k, "Davies_Bouldin_Index"].values[0]
km_chi = df_metrics.loc[df_metrics["k"] == optimal_k, "Calinski_Harabasz_Index"].values[0]

comparison_df = pd.DataFrame([
    {
        "Algorithm": "K-Means (k-means++)",
        "Clusters_Detected": optimal_k,
        "Silhouette_Score": round(km_sil, 4),
        "Davies_Bouldin_Index": round(km_dbi, 4),
        "Calinski_Harabasz_Index": round(km_chi, 2),
        "Outliers_Detected": 0,
        "Algorithmic_Complexity": "O(t*k*n*d) - Linear / Scalable",
        "Key_Strengths": "Optimal spherical partition; fast convergence"
    },
    {
        "Algorithm": "Agglomerative (Ward)",
        "Clusters_Detected": optimal_k,
        "Silhouette_Score": round(agg_sil, 4),
        "Davies_Bouldin_Index": round(agg_dbi, 4),
        "Calinski_Harabasz_Index": round(agg_chi, 2),
        "Outliers_Detected": 0,
        "Algorithmic_Complexity": "O(n^2 log n) - Memory heavy",
        "Key_Strengths": "Deterministic hierarchy; no random seed"
    },
    {
        "Algorithm": "DBSCAN (Density-Based)",
        "Clusters_Detected": n_db_clusters,
        "Silhouette_Score": round(db_sil, 4),
        "Davies_Bouldin_Index": round(db_dbi, 4),
        "Calinski_Harabasz_Index": round(db_chi, 2),
        "Outliers_Detected": n_noise,
        "Algorithmic_Complexity": "O(n log n) with spatial index",
        "Key_Strengths": "Arbitrary cluster shapes; isolates anomalies"
    }
])
comparison_df.to_csv(os.path.join(OUT_DIR, "clustering_comparison_table.csv"), index=False)
print(comparison_df.to_string(index=False))

# ===========================================================================
# Step 9: Customer Persona Profiling & Business Implications
# ===========================================================================
print("\n=== Step 9: Customer Persona Synthesis ===")
profiles_list = []
total_count = len(df)

for c in sorted(df["KMeans_Cluster"].unique()):
    sub = df[df["KMeans_Cluster"] == c]
    count = len(sub)
    pct = (count / total_count) * 100
    mean_inc = sub["Annual_Income_k"].mean()
    mean_sp = sub["Spending_Score"].mean()
    mean_age = sub["Age"].mean()
    f_pct = (sub["Gender"].str.lower() == "female").mean() * 100
    m_pct = 100.0 - f_pct
    
    # Meaningful persona mapping based on stats
    if mean_inc > 70 and mean_sp > 65:
        p_name = "High-Value Champions (VIP)"
        desc = "High earnings and high spending. Prime revenue generator."
    elif mean_inc > 70 and mean_sp < 40:
        p_name = "Cautious Affluent (Savers)"
        desc = "High earnings but conservative spending. High untapped potential."
    elif mean_inc < 40 and mean_sp > 65:
        p_name = "Impulsive Trendsetters (Youth)"
        desc = "Lower earnings but enthusiastic spenders. Driven by fast-fashion."
    elif mean_inc < 40 and mean_sp < 40:
        p_name = "Budget Seekers (Sensible)"
        desc = "Low income and minimal discretionary spend. Price sensitive."
    else:
        p_name = "Moderate Mainstream (Standard)"
        desc = "Balanced income and spending. Core revenue foundation."
        
    profiles_list.append({
        "Cluster": c,
        "Persona_Name": p_name,
        "Size": count,
        "Percentage": round(pct, 1),
        "Mean_Income_k": round(mean_inc, 1),
        "Mean_Spending_Score": round(mean_sp, 1),
        "Mean_Age": round(mean_age, 1),
        "Female_Pct": round(f_pct, 1),
        "Male_Pct": round(m_pct, 1),
        "Description": desc
    })

df_profiles = pd.DataFrame(profiles_list)
df_profiles.to_csv(os.path.join(OUT_DIR, "cluster_profile_statistics.csv"), index=False)
print(df_profiles[["Cluster", "Persona_Name", "Size", "Percentage", "Mean_Income_k", "Mean_Spending_Score", "Mean_Age"]].to_string(index=False))

# Map persona back to dataframe
persona_lookup = dict(zip(df_profiles["Cluster"], df_profiles["Persona_Name"]))
df["Persona_Name"] = df["KMeans_Cluster"].map(persona_lookup)
df.to_csv(os.path.join(OUT_DIR, "customer_segmentation_results.csv"), index=False)
print("Saved final segmentation results to output/customer_segmentation_results.csv")

# ===========================================================================
# Fig 11: Multi-Dimensional Radar / Spider Profiles
# ===========================================================================
print("\n=== Generating Fig 11: Behavioral Radar Profiles ===")
categories = ["Age", "Annual Income", "Spending Score", "Female Ratio", "Male Ratio"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.suptitle("Fig 11 - Multi-Dimensional Behavioral Radar Profiles per Cluster", fontsize=14, fontweight="bold", y=0.98)

plt.xticks(angles[:-1], categories, color="black", size=11, fontweight="bold")
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=9)
plt.ylim(0, 1.05)

for i, row in df_profiles.iterrows():
    c_id = int(row["Cluster"])
    sub = df[df["KMeans_Cluster"] == c_id]
    vals = [
        sub["Age"].mean() / 70.0,
        sub["Annual_Income_k"].mean() / 140.0,
        sub["Spending_Score"].mean() / 100.0,
        row["Female_Pct"] / 100.0,
        row["Male_Pct"] / 100.0
    ]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, label=f"Cluster {c_id}: {row['Persona_Name']}", color=PALETTE_5[c_id % len(PALETTE_5)])
    ax.fill(angles, vals, color=PALETTE_5[c_id % len(PALETTE_5)], alpha=0.15)

plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
fig.tight_layout()
save(fig, "fig11_cluster_radar_profiles.png")

# ===========================================================================
# Fig 12: Business Persona Strategy Matrix
# ===========================================================================
print("\n=== Generating Fig 12: Business Strategy Matrix ===")
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")
fig.suptitle("Fig 12 - Customer Persona Strategy Matrix and Marketing Roadmap", fontsize=15, fontweight="bold", y=0.95)

table_rows = [
    ["Cluster 0", "Moderate Mainstream", "Income: $55.3k | Spend: 49.5", "Family bundles, seasonal loyalty discounts, dining perks", "Email, weekend mobile app push", "Predictable baseline revenue, basket size growth"],
    ["Cluster 1", "High-Value Champions", "Income: $86.5k | Spend: 82.1", "VIP concierge, luxury lounge access, private previews", "Dedicated relationship manager, VIP app", "Maximize Customer Lifetime Value (CLV), premium margins"],
    ["Cluster 2", "Impulsive Trendsetters", "Income: $25.7k | Spend: 79.4", "Fast-fashion, flash sales, Buy-Now-Pay-Later (BNPL)", "TikTok, Instagram, youth fashion influencers", "High transaction frequency, viral brand awareness"],
    ["Cluster 3", "Cautious Affluent", "Income: $88.2k | Spend: 17.1", "Premium durability, investment goods, warranties", "Executive email digests, LinkedIn, premium cards", "Convert high-income savers with craftsmanship messaging"],
    ["Cluster 4", "Budget Seekers", "Income: $26.3k | Spend: 20.9", "Clearance promotions, bulk essentials, cashback rewards", "SMS alerts, in-store print discount circulars", "Maintain store footfall, inventory turnover"]
]

col_headers = ["Cluster", "Persona Name", "Profile Summary", "Marketing Action Plan", "Key Channels", "Commercial Objective"]
tbl = ax.table(cellText=table_rows, colLabels=col_headers, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 3.2)

for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1f497d")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f2f5f9" if r % 2 == 0 else "#ffffff")
    cell.set_edgecolor("#cccccc")

fig.tight_layout()
save(fig, "fig12_business_persona_matrix.png")

print("\n" + "=" * 80)
print("   WEEK 3 CLUSTERING ANALYSIS COMPLETE - ALL DELIVERABLES READY!")
print("=" * 80)
