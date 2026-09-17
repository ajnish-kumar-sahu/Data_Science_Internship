"""
write_notebook.py
Generates the interactive Jupyter Notebook Week3_Clustering_Analysis.ipynb
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb['cells'] = [
    nbf.v4.new_markdown_cell('# Week 3: Unsupervised Learning & Clustering Analysis\n**Author:** Ajnish Kumar | **Roll No:** 241809046713\n**Degree:** BCA, Vinoba Bhave University, Hazaribag\n**Internship:** Virtual Data Science with Python Trainee (Yuva Intern)\n\n---\n\n## Objective\nThis notebook provides a complete implementation of unsupervised customer segmentation on the Mall Customer dataset using:\n1. **K-Means Clustering** with exhaustive hyperparameter tuning (Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz).\n2. **Agglomerative Hierarchical Clustering** with Ward, Complete, and Average linkages.\n3. **DBSCAN Density-Based Clustering** with k-NN distance elbow analysis.\n4. **Principal Component Analysis (PCA)** for multidimensional projection.\n5. **Business Persona Synthesis** and strategic marketing recommendations.'),

    nbf.v4.new_code_cell("""import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist

warnings.filterwarnings('ignore')
%matplotlib inline
sns.set_theme(style='whitegrid')
"""),

    nbf.v4.new_markdown_cell('## 1. Load and Preprocess the Dataset\nLoading the Mall Customer dataset and verifying data hygiene.'),

    nbf.v4.new_code_cell("""# Load data
df = pd.read_csv('../data/Mall_Customers.csv')
df.rename(columns={
    'Annual Income (k$)': 'Annual_Income_k',
    'Spending Score (1-100)': 'Spending_Score'
}, inplace=True)

print(f"Dataset Shape: {df.shape}")
print(f"Missing Values: {df.isnull().sum().sum()}")
df.head()
"""),

    nbf.v4.new_markdown_cell('## 2. Feature Standardization (Z-score Scaling)\nDistance-based algorithms (K-Means, Hierarchical, DBSCAN) mandate scaling to ensure equal geometric weighting.'),

    nbf.v4.new_code_cell("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Annual_Income_k', 'Spending_Score']])
print("X_scaled shape:", X_scaled.shape)
print("Mean:", np.round(X_scaled.mean(axis=0), 4))
print("Std Dev:", np.round(X_scaled.std(axis=0), 4))
"""),

    nbf.v4.new_markdown_cell('## 3. Hyperparameter Optimization: Evaluating k ∈ [2, 10]\nEvaluating Within-Cluster Sum of Squares (Inertia/WCSS), Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.'),

    nbf.v4.new_code_cell("""metrics_list = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=300, random_state=42)
    labels = km.fit_predict(X_scaled)
    metrics_list.append({
        'k': k,
        'Inertia (WCSS)': km.inertia_,
        'Silhouette': silhouette_score(X_scaled, labels),
        'Davies-Bouldin': davies_bouldin_score(X_scaled, labels),
        'Calinski-Harabasz': calinski_harabasz_score(X_scaled, labels)
    })

metrics_df = pd.DataFrame(metrics_list)
metrics_df
"""),

    nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# WCSS Elbow
axes[0, 0].plot(metrics_df['k'], metrics_df['Inertia (WCSS)'], marker='o', color='#1f77b4', linewidth=2)
axes[0, 0].axvline(5, color='red', linestyle='--', label='Elbow at k=5')
axes[0, 0].set_title('Elbow Curve (WCSS vs. k)')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].legend()

# Silhouette
axes[0, 1].plot(metrics_df['k'], metrics_df['Silhouette'], marker='s', color='#2ca02c', linewidth=2)
axes[0, 1].axvline(5, color='red', linestyle='--', label='Max Silhouette at k=5')
axes[0, 1].set_title('Mean Silhouette Score vs. k')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].legend()

# Davies-Bouldin
axes[1, 0].plot(metrics_df['k'], metrics_df['Davies-Bouldin'], marker='^', color='#d62728', linewidth=2)
axes[1, 0].axvline(5, color='red', linestyle='--', label='Minimum DBI at k=5')
axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('DBI')
axes[1, 0].legend()

# Calinski-Harabasz
axes[1, 1].plot(metrics_df['k'], metrics_df['Calinski-Harabasz'], marker='D', color='#9467bd', linewidth=2)
axes[1, 1].axvline(5, color='red', linestyle='--', label='Peak CHI at k=5')
axes[1, 1].set_title('Calinski-Harabasz Index (Higher is Better)')
axes[1, 1].set_xlabel('k')
axes[1, 1].set_ylabel('CHI')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
"""),

    nbf.v4.new_markdown_cell('## 4. Fitting the Optimal K-Means Model (k=5)\nExtracting centroids in both scaled and original dollar units.'),

    nbf.v4.new_code_cell("""optimal_k = 5
km_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, max_iter=300, random_state=42)
df['KMeans_Cluster'] = km_final.fit_predict(X_scaled)
centroids_orig = scaler.inverse_transform(km_final.cluster_centers_)

plt.figure(figsize=(11, 6))
sns.scatterplot(
    data=df,
    x='Annual_Income_k',
    y='Spending_Score',
    hue='KMeans_Cluster',
    palette='tab10',
    s=85,
    alpha=0.85
)
plt.scatter(
    centroids_orig[:, 0],
    centroids_orig[:, 1],
    s=250,
    c='yellow',
    marker='X',
    edgecolor='black',
    linewidth=2,
    label='Cluster Centroids'
)

for i, c in enumerate(centroids_orig):
    plt.annotate(f"Centroid {i}: ({c[0]:.0f}k, {c[1]:.0f})", (c[0] + 1.5, c[1] + 1.5),
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

plt.title('K-Means Customer Segmentation (k=5)', fontsize=14, fontweight='bold')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='upper right')
plt.show()
"""),

    nbf.v4.new_markdown_cell('## 5. Agglomerative Hierarchical Clustering\nGenerating linkage dendrograms and validating with Cophenetic Correlation.'),

    nbf.v4.new_code_cell("""# Linkage matrices
dists = pdist(X_scaled, metric='euclidean')
Z_ward = linkage(X_scaled, method='ward')
coph_corr, _ = cophenet(Z_ward, dists)
print(f"Ward Linkage Cophenetic Correlation: {coph_corr:.4f}")

plt.figure(figsize=(13, 5))
dendrogram(Z_ward, truncate_mode='lastp', p=25, leaf_rotation=90)
plt.axhline(y=3.82, color='red', linestyle='--', label='Cut for k=5 (d=3.82)')
plt.title('Agglomerative Hierarchical Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold')
plt.xlabel('Cluster / Leaf Index')
plt.ylabel('Euclidean Distance')
plt.legend()
plt.show()

# Fit Agglomerative model
agg = AgglomerativeClustering(n_clusters=5, linkage='ward')
df['Hierarchical_Cluster'] = agg.fit_predict(X_scaled)

# Cross-tabulation contingency table
print("Contingency Table (K-Means vs. Hierarchical):")
pd.crosstab(df['KMeans_Cluster'], df['Hierarchical_Cluster'], rownames=['K-Means'], colnames=['Hierarchical'])
"""),

    nbf.v4.new_markdown_cell('## 6. DBSCAN Density-Based Clustering & Outlier Isolation\nUsing 4-NN distance elbow curve to determine epsilon (ε) and isolate noise.'),

    nbf.v4.new_code_cell("""# 4-NN distance graph
nbrs = NearestNeighbors(n_neighbors=4).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
sorted_distances = np.sort(distances[:, 3])

plt.figure(figsize=(9, 4))
plt.plot(sorted_distances, color='#d95f02', linewidth=2.5)
plt.axhline(y=0.38, color='red', linestyle='--', label='Selected ε = 0.38')
plt.title('4-NN Distance Curve for Epsilon Selection')
plt.xlabel('Points Sorted by Distance')
plt.ylabel('4th Nearest Neighbor Distance')
plt.legend()
plt.show()

# Fit DBSCAN
dbscan = DBSCAN(eps=0.38, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)
n_noise = (df['DBSCAN_Cluster'] == -1).sum()
print(f"DBSCAN Detected {len(set(df['DBSCAN_Cluster'])) - 1} Dense Clusters and {n_noise} Noise/Outlier points.")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Annual_Income_k',
    y='Spending_Score',
    hue='DBSCAN_Cluster',
    palette='Spectral',
    s=75,
    style=(df['DBSCAN_Cluster'] == -1)
)
plt.title('DBSCAN Density Clustering (Crosses = Anomalies/Noise)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
"""),

    nbf.v4.new_markdown_cell('## 7. Dimensionality Reduction with PCA\nProjecting multidimensional customer demographic profiles into 2D principal space.'),

    nbf.v4.new_code_cell("""all_features = ['Age', 'Annual_Income_k', 'Spending_Score']
X_all_scaled = StandardScaler().fit_transform(df[all_features])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_all_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Cumulative Variance:", np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['KMeans_Cluster'], palette='tab10', s=75)
plt.title('PCA 2D Projection of Customer Clusters', fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
plt.show()
"""),

    nbf.v4.new_markdown_cell('## 8. Customer Persona Profiling & Business Implications\nSynthesizing behavioral characteristics for the 5 discovered commercial personas.'),

    nbf.v4.new_code_cell("""persona_names = {
    0: "Moderate Mainstream (Standard)",
    1: "High-Value Champions (VIP)",
    2: "Impulsive Trendsetters (Youth)",
    3: "Cautious Affluent (Savers)",
    4: "Budget Seekers (Sensible)"
}
df['Persona_Name'] = df['KMeans_Cluster'].map(persona_names)

profile_table = df.groupby('Persona_Name').agg(
    Customer_Count=('CustomerID', 'count'),
    Mean_Income_k=('Annual_Income_k', 'mean'),
    Mean_Spending_Score=('Spending_Score', 'mean'),
    Mean_Age=('Age', 'mean')
).round(1)

profile_table['Population_Share_%'] = (profile_table['Customer_Count'] / len(df) * 100).round(1)
profile_table.sort_values(by='Mean_Income_k', ascending=False)
""")
]

nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks', 'Week3_Clustering_Analysis.ipynb')
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully written to: {nb_path}")
print(f"Notebook size: {os.path.getsize(nb_path)} bytes")
