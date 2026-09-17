"""
visualizer.py
Publication-grade visualizer generating 12 figures for Unsupervised
Learning and Clustering Analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples

# Configure styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 150
})

PALETTE_5 = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]
DPI = 150

def save_fig(fig, output_path):
    """Saves figure and closes current matplotlib plot."""
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {os.path.basename(output_path)}")

# 1. Feature Distributions & Boxplots
def plot_distributions_and_boxplots(df, out_path):
    features = ["Age", "Annual_Income_k", "Spending_Score"]
    titles = ["Customer Age (Years)", "Annual Income (k$)", "Spending Score (1-100)"]
    colors = ["#3470a3", "#2ca02c", "#d62728"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Figure 1: Univariate Distributions and Boxplot Outlier Auditing", fontsize=15, fontweight="bold", y=0.98)
    
    for i, col in enumerate(features):
        # Histogram with KDE
        sns.histplot(df[col], kde=True, ax=axes[0, i], color=colors[i], bins=15, edgecolor="black", alpha=0.6)
        axes[0, i].set_title(f"Distribution of {titles[i]}", fontweight="semibold")
        axes[0, i].set_xlabel(titles[i])
        axes[0, i].set_ylabel("Frequency")
        
        # Mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[0, i].axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        axes[0, i].axvline(median_val, color="blue", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")
        axes[0, i].legend(loc="upper right", frameon=True)
        
        # Boxplot
        sns.boxplot(x=df[col], ax=axes[1, i], color=colors[i], flierprops=dict(markerfacecolor="red", marker="D", markersize=6))
        axes[1, i].set_title(f"Boxplot: {titles[i]}", fontweight="semibold")
        axes[1, i].set_xlabel(titles[i])
        
    plt.tight_layout()
    save_fig(fig, out_path)

# 2. Pairplot with Gender KDE
def plot_pairplot_gender(df, out_path):
    pair_df = df[["Age", "Annual_Income_k", "Spending_Score", "Gender"]].copy()
    pair_df.rename(columns={
        "Annual_Income_k": "Income (k$)",
        "Spending_Score": "Spending (1-100)"
    }, inplace=True)
    
    g = sns.pairplot(
        pair_df,
        hue="Gender",
        palette={"Male": "#1f77b4", "Female": "#e377c2"},
        diag_kind="kde",
        plot_kws={"alpha": 0.75, "s": 45, "edgecolor": "none"},
        diag_kws={"fill": True, "alpha": 0.45},
        corner=False
    )
    g.fig.suptitle("Figure 2: Multivariate Pairwise Distributions Segmented by Gender", fontsize=14, fontweight="bold", y=1.02)
    g.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {os.path.basename(out_path)}")

# 3. Correlation Matrix Heatmaps
def plot_correlation_matrices(df, out_path):
    corr_cols = ["Age", "Annual_Income_k", "Spending_Score", "Gender_Code"]
    labels = ["Age", "Income (k$)", "Spending (1-100)", "Gender (Male=1)"]
    
    p_corr = df[corr_cols].corr(method="pearson")
    s_corr = df[corr_cols].corr(method="spearman")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 3: Feature Correlation Analysis (Pearson Linear vs. Spearman Monotonic)", fontsize=15, fontweight="bold")
    
    mask = np.triu(np.ones_like(p_corr, dtype=bool))
    
    sns.heatmap(p_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-0.6, vmax=0.6,
                xticklabels=labels, yticklabels=labels, cbar=True, ax=axes[0], linewidths=1)
    axes[0].set_title("Pearson Correlation Coefficient (r)", fontweight="semibold")
    
    sns.heatmap(s_corr, annot=True, fmt=".2f", cmap="vlag", vmin=-0.6, vmax=0.6,
                xticklabels=labels, yticklabels=labels, cbar=True, ax=axes[1], linewidths=1)
    axes[1].set_title("Spearman Rank Correlation (ρ)", fontweight="semibold")
    
    plt.tight_layout()
    save_fig(fig, out_path)

# 4. Multi-Panel Model Validation (Elbow, Silhouette, DBI, CHI)
def plot_cluster_validation_curves(df_metrics, optimal_k, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Figure 4: Hyperparameter Optimization and Cluster Validation Curves", fontsize=15, fontweight="bold", y=0.98)
    
    k_vals = df_metrics["k"]
    
    # 1. Inertia / Elbow WCSS
    axes[0, 0].plot(k_vals, df_metrics["Inertia_WCSS"], marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
    axes[0, 0].axvline(optimal_k, color="red", linestyle="--", alpha=0.8, label=f"Optimal Elbow k={optimal_k}")
    axes[0, 0].set_title("A: Within-Cluster Sum of Squares (WCSS / Inertia)", fontweight="semibold")
    axes[0, 0].set_xlabel("Number of Clusters (k)")
    axes[0, 0].set_ylabel("Inertia (WCSS)")
    axes[0, 0].set_xticks(k_vals)
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    
    # 2. Silhouette Score
    axes[0, 1].plot(k_vals, df_metrics["Silhouette_Score"], marker="s", color="#2ca02c", linewidth=2.5, markersize=8)
    axes[0, 1].axvline(optimal_k, color="red", linestyle="--", alpha=0.8, label=f"Max Score at k={optimal_k}")
    axes[0, 1].set_title("B: Mean Silhouette Coefficient (Higher is Better)", fontweight="semibold")
    axes[0, 1].set_xlabel("Number of Clusters (k)")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].set_xticks(k_vals)
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    
    # 3. Davies-Bouldin Index (lower is better)
    axes[1, 0].plot(k_vals, df_metrics["Davies_Bouldin_Index"], marker="^", color="#d62728", linewidth=2.5, markersize=8)
    axes[1, 0].axvline(optimal_k, color="red", linestyle="--", alpha=0.8, label=f"Minimum at k={optimal_k}")
    axes[1, 0].set_title("C: Davies-Bouldin Index (Lower is Better)", fontweight="semibold")
    axes[1, 0].set_xlabel("Number of Clusters (k)")
    axes[1, 0].set_ylabel("Davies-Bouldin Index")
    axes[1, 0].set_xticks(k_vals)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    
    # 4. Calinski-Harabasz Index (higher is better)
    axes[1, 1].plot(k_vals, df_metrics["Calinski_Harabasz_Index"], marker="D", color="#9467bd", linewidth=2.5, markersize=8)
    axes[1, 1].axvline(optimal_k, color="red", linestyle="--", alpha=0.8, label=f"Peak at k={optimal_k}")
    axes[1, 1].set_title("D: Calinski-Harabasz Index (Variance Ratio - Higher is Better)", fontweight="semibold")
    axes[1, 1].set_xlabel("Number of Clusters (k)")
    axes[1, 1].set_ylabel("Calinski-Harabasz Score")
    axes[1, 1].set_xticks(k_vals)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    save_fig(fig, out_path)

# 5. Silhouette Analysis per Sample (k=3, 4, 5, 6)
def plot_silhouette_per_sample(X, out_path, k_list=[3, 4, 5, 6], random_state=42):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Figure 5: Silhouette Analysis per Cluster and Sample Thickness for k ∈ [3, 4, 5, 6]", fontsize=15, fontweight="bold", y=0.99)
    
    for idx, k in enumerate(k_list):
        ax = axes[idx // 2, idx % 2]
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=random_state)
        cluster_labels = km.fit_predict(X)
        
        sil_avg = silhouette_score(X, cluster_labels)
        sample_sil_values = silhouette_samples(X, cluster_labels)
        
        y_lower = 10
        for i in range(k):
            ith_cluster_values = sample_sil_values[cluster_labels == i]
            ith_cluster_values.sort()
            
            size_cluster_i = ith_cluster_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=10, fontweight="bold")
            y_lower = y_upper + 10
            
        ax.set_title(f"Silhouette Plot for k = {k} (Mean Score: {sil_avg:.3f})", fontweight="semibold")
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=1.5, label=f"Avg: {sil_avg:.3f}")
        ax.set_yticks([])
        ax.set_xlim([-0.15, 0.85])
        ax.legend(loc="upper right")
        
    plt.tight_layout()
    save_fig(fig, out_path)

# 6. K-Means Clusters 2D & 3D Projections
def plot_kmeans_clusters_2d_3d(df, centroids_original, out_path):
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("Figure 6: K-Means Segmentation Results (2D Feature Space and 3D Demographic Projection)", fontsize=15, fontweight="bold")
    
    # Left: 2D Scatter with Centroids
    ax1 = fig.add_subplot(1, 2, 1)
    
    cluster_names = {
        0: "Cluster 0: Sensible / Budget",
        1: "Cluster 1: Careless / Impulsive",
        2: "Cluster 2: Standard / Moderate",
        3: "Cluster 3: Target / High-Spenders",
        4: "Cluster 4: Cautious / Savers"
    }
    
    scatter = sns.scatterplot(
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
    
    # Plot Centroids
    ax1.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        s=220,
        c="yellow",
        edgecolors="black",
        marker="X",
        linewidths=2,
        label="Cluster Centroids (k-means++)",
        zorder=10
    )
    
    # Annotate Centroids
    for i, c in enumerate(centroids_original):
        ax1.annotate(f"C{i} ({c[0]:.0f}, {c[1]:.0f})", (c[0] + 1.5, c[1] + 1.5),
                     fontsize=9, fontweight="bold", backgroundcolor="white", alpha=0.8)
        
    ax1.set_title("2D Customer Segmentation: Annual Income vs. Spending Score", fontweight="semibold")
    ax1.set_xlabel("Annual Income (k$)")
    ax1.set_ylabel("Spending Score (1-100)")
    ax1.legend(loc="upper right", frameon=True)
    
    # Right: 3D Projection
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
        
    ax2.set_title("3D View: Income vs. Spending vs. Age", fontweight="semibold")
    ax2.set_xlabel("Annual Income (k$)")
    ax2.set_ylabel("Spending Score (1-100)")
    ax2.set_zlabel("Age (Years)")
    ax2.view_init(elev=25, azim=130)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.95))
    
    plt.tight_layout()
    save_fig(fig, out_path)

# 7. Hierarchical Dendrograms
def plot_hierarchical_dendrograms(linkage_dict, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Figure 7: Hierarchical Clustering Dendrograms across Linkage Criteria (Cut at k=5)", fontsize=15, fontweight="bold")
    
    methods = ["ward", "complete", "average"]
    titles = ["Ward's Minimum Variance Linkage", "Complete Linkage (Maximum Distance)", "Average Linkage (Mean Distance)"]
    
    for i, method in enumerate(methods):
        Z = linkage_dict[method]["linkage_matrix"]
        coph = linkage_dict[method]["cophenetic_corr"]
        
        # Determine cut threshold distance for 5 clusters
        # 5 clusters means 4 joins remaining from top
        dist_cut = (Z[-5, 2] + Z[-4, 2]) / 2.0
        
        dendrogram(
            Z,
            truncate_mode="lastp",
            p=25,
            show_leaf_counts=True,
            leaf_rotation=90,
            leaf_font_size=9,
            color_threshold=dist_cut,
            ax=axes[i]
        )
        axes[i].axhline(y=dist_cut, color="black", linestyle="--", linewidth=1.5,
                        label=f"k=5 Cut (d={dist_cut:.2f})")
        axes[i].set_title(f"{titles[i]}\nCophenetic Corr: {coph:.3f}", fontweight="semibold")
        axes[i].set_xlabel("Cluster Sample / Leaf Index")
        axes[i].set_ylabel("Euclidean Distance (Merge Cost)")
        axes[i].legend(loc="upper right")
        
    plt.tight_layout()
    save_fig(fig, out_path)

# 8. Hierarchical vs. K-Means Comparison & Confusion Heatmap
def plot_hierarchical_vs_kmeans(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Figure 8: Comparative Assessment: K-Means vs. Agglomerative Hierarchical Clustering", fontsize=15, fontweight="bold")
    
    # Scatter side by side
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
    axes[0].set_title("Agglomerative Clustering (Ward Linkage, k=5)", fontweight="semibold")
    axes[0].set_xlabel("Annual Income (k$)")
    axes[0].set_ylabel("Spending Score (1-100)")
    
    # Confusion / Contingency Cross-tabulation
    ct = pd.crosstab(df["KMeans_Cluster"], df["Hierarchical_Cluster"],
                     rownames=["K-Means Cluster"], colnames=["Hierarchical Cluster"])
    
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=True, ax=axes[1], linewidths=1)
    axes[1].set_title("Contingency Matrix: K-Means vs. Hierarchical Assignments", fontweight="semibold")
    
    plt.tight_layout()
    save_fig(fig, out_path)

# 9. DBSCAN Clustering & Outlier Detection
def plot_dbscan_clustering(X_scaled, sorted_k_distances, dbscan_res, df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Figure 9: DBSCAN Density-Based Clustering and Anomaly/Noise Identification", fontsize=15, fontweight="bold")
    
    # Left: k-distance elbow graph
    axes[0].plot(sorted_k_distances, color="#d95f02", linewidth=2.5)
    axes[0].axhline(y=dbscan_res["eps"], color="red", linestyle="--", linewidth=1.5,
                    label=f"Selected ε = {dbscan_res['eps']}")
    axes[0].set_title("4-NN Distance Graph for Optimal Epsilon (ε) Selection", fontweight="semibold")
    axes[0].set_xlabel("Points Sorted by 4th-NN Distance")
    axes[0].set_ylabel("Distance to 4th Nearest Neighbor")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Right: DBSCAN scatter
    db_labels = dbscan_res["labels"]
    unique_labels = sorted(set(db_labels))
    
    colors = [cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k_label, col in zip(unique_labels, colors):
        if k_label == -1:
            # Outliers / Noise in black
            col = [0, 0, 0, 1]
            label_name = f"Noise / Outliers (n={dbscan_res['n_noise']})"
            marker = "x"
            size = 80
        else:
            label_name = f"Dense Cluster {k_label}"
            marker = "o"
            size = 65
            
        class_member_mask = (db_labels == k_label)
        xy = df[class_member_mask]
        
        axes[1].scatter(
            xy["Annual_Income_k"],
            xy["Spending_Score"],
            c=[col],
            label=label_name,
            marker=marker,
            s=size,
            alpha=0.85,
            edgecolors="k" if marker == "o" else "r"
        )
        
    axes[1].set_title(f"DBSCAN Clusters (ε={dbscan_res['eps']}, min_samples={dbscan_res['min_samples']})", fontweight="semibold")
    axes[1].set_xlabel("Annual Income (k$)")
    axes[1].set_ylabel("Spending Score (1-100)")
    axes[1].legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    save_fig(fig, out_path)

# 10. PCA Biplot and Explained Variance
def plot_pca_biplot_variance(pca_res, df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Figure 10: Principal Component Analysis (Scree Plot and 2D Loadings Biplot)", fontsize=15, fontweight="bold")
    
    # Left: Scree Plot
    var_exp = pca_res["pca_2d"].explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_exp)
    
    axes[0].bar(["PC1", "PC2"], var_exp, color="#1f77b4", alpha=0.7, edgecolor="black", label="Individual Variance (%)")
    axes[0].plot(["PC1", "PC2"], cum_var, color="red", marker="o", linewidth=2, label="Cumulative Variance (%)")
    
    for i, (v, c) in enumerate(zip(var_exp, cum_var)):
        axes[0].text(i, v / 2, f"{v:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        axes[0].text(i, c + 2, f"{c:.1f}%", ha="center", va="bottom", color="red", fontweight="bold")
        
    axes[0].set_title("Scree Plot: Explained Variance by Principal Component", fontweight="semibold")
    axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_ylim(0, 110)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Right: PCA Biplot
    X_pca = pca_res["X_pca_2d"]
    loadings = pca_res["loadings"]
    
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df["KMeans_Cluster"],
        palette=PALETTE_5,
        s=60,
        alpha=0.75,
        ax=axes[1]
    )
    
    # Add loading vectors
    scale_factor = 2.8
    feature_labels = {
        "Age": "Age",
        "Annual_Income_k": "Income",
        "Spending_Score": "Spending",
        "Gender_Code": "Gender"
    }
    
    for feat in loadings.index:
        x_vec = loadings.loc[feat, "PC1"] * scale_factor
        y_vec = loadings.loc[feat, "PC2"] * scale_factor
        axes[1].arrow(0, 0, x_vec, y_vec, color="black", width=0.03, head_width=0.1, head_length=0.1, zorder=5)
        disp_name = feature_labels.get(feat, feat)
        axes[1].text(x_vec * 1.15, y_vec * 1.15, disp_name, color="darkred", fontweight="bold", fontsize=11, zorder=6)
        
    axes[1].axhline(0, color="grey", linestyle="--", alpha=0.6)
    axes[1].axvline(0, color="grey", linestyle="--", alpha=0.6)
    axes[1].set_title("PCA Biplot (PC1 vs PC2) with Feature Loadings", fontweight="semibold")
    axes[1].set_xlabel(f"PC1 ({var_exp[0]:.1f}% Variance)")
    axes[1].set_ylabel(f"PC2 ({var_exp[1]:.1f}% Variance)")
    axes[1].legend(title="Cluster", loc="upper left")
    
    plt.tight_layout()
    save_fig(fig, out_path)

# 11. Cluster Radar / Spider Profiles
def plot_cluster_radar_profiles(df, out_path):
    categories = ["Age", "Annual Income", "Spending Score", "Female Ratio", "Male Ratio"]
    N = len(categories)
    
    cluster_means = []
    for c in sorted(df["KMeans_Cluster"].unique()):
        sub = df[df["KMeans_Cluster"] == c]
        norm_age = sub["Age"].mean() / 70.0
        norm_inc = sub["Annual_Income_k"].mean() / 140.0
        norm_sp = sub["Spending_Score"].mean() / 100.0
        f_ratio = (sub["Gender"].str.lower() == "female").mean()
        m_ratio = 1.0 - f_ratio
        vals = [norm_age, norm_inc, norm_sp, f_ratio, m_ratio]
        vals += vals[:1]  # Close polygon
        cluster_means.append(vals)
        
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle("Figure 11: Multi-Dimensional Behavioral Radar Profiles per Cluster", fontsize=15, fontweight="bold", y=0.98)
    
    plt.xticks(angles[:-1], categories, color="black", size=11, fontweight="semibold")
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=9)
    plt.ylim(0, 1.05)
    
    persona_labels = [
        "Cluster 0: Budget Seekers",
        "Cluster 1: Impulsive Trendsetters",
        "Cluster 2: Moderate Mainstream",
        "Cluster 3: High-Value Champions",
        "Cluster 4: Cautious Affluent"
    ]
    
    for i, vals in enumerate(cluster_means):
        ax.plot(angles, vals, linewidth=2, linestyle="solid", label=persona_labels[i], color=PALETTE_5[i % len(PALETTE_5)])
        ax.fill(angles, vals, color=PALETTE_5[i % len(PALETTE_5)], alpha=0.15)
        
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
    plt.tight_layout()
    save_fig(fig, out_path)

# 12. Strategic Business Persona Matrix
def plot_business_persona_matrix(df_profiles, out_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    fig.suptitle("Figure 12: Customer Persona Segmentation Matrix & Actionable Strategic Roadmap", fontsize=16, fontweight="bold", y=0.95)
    
    personas_data = [
        ["Cluster 0", "Budget Seekers", "Low Income\nLow Spend", "Essential discounts, clearance items, value coupons", "SMS, in-store flyers, budget push notifications", "Price sensitivity protection, baseline footfall retention"],
        ["Cluster 1", "Impulsive Trendsetters", "Low Income\nHigh Spend", "Trendy fashion drops, installment plans (BNPL), FOMO events", "Instagram, TikTok, youth influencer partnerships", "High margin fast-fashion monetization, brand buzz creation"],
        ["Cluster 2", "Moderate Mainstream", "Moderate Income\nModerate Spend", "Seasonal promotions, loyalty rewards, family bundles", "Email newsletters, loyalty app perks, weekend sales", "Reliable recurring revenue, basket size expansion"],
        ["Cluster 3", "High-Value Champions", "High Income\nHigh Spend", "VIP concierge, luxury lounges, exclusive previews, personalized sales", "Direct VIP relationship manager, concierge app, private invitations", "Maximum Customer Lifetime Value (CLV), premium profitability"],
        ["Cluster 4", "Cautious Affluent", "High Income\nLow Spend", "High-utility premium goods, investment value, quality guarantee", "Targeted LinkedIn/email, quality brand editorial, premium perks", "Converting high-income savers into luxury consumers via value messaging"]
    ]
    
    columns = ["Cluster", "Persona Name", "Income / Spend Profile", "Strategic Marketing Offer", "Primary Channels", "Commercial Objective"]
    
    table = ax.table(
        cellText=personas_data,
        colLabels=columns,
        cellLoc="center",
        loc="center"
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 3.2)
    
    # Styling table
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f497d")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cluster_idx = row - 1
            if cluster_idx % 2 == 0:
                cell.set_facecolor("#f2f5f9")
            else:
                cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#cccccc")
        
    plt.tight_layout()
    save_fig(fig, out_path)
