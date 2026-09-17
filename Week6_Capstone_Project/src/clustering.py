"""
clustering.py
Unsupervised Customer Behavioral Segmentation using K-Means clustering,
Elbow and Silhouette hyperparameter validation, and Principal Component Analysis (PCA).

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_unsupervised_clustering(df_clean, output_dir, k_range=range(2, 8), optimal_k=4, random_state=42):
    """
    Executes unsupervised behavioral segmentation on core behavioral and financial metrics.
    Evaluates cluster validity across k in [2, 7] and generates persona profiles.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Core segmentation feature vector
    segmentation_features = [
        "TenureMonths",
        "MonthlyUsageGB",
        "SupportTicketsLastYear",
        "PaymentDelaysCount",
        "CustomerSatisfactionScore",
        "MonthlyCharges"
    ]

    X_seg = df_clean[segmentation_features].copy()
    scaler = StandardScaler()
    X_seg_scaled = scaler.fit_transform(X_seg)

    # 1. Hyperparameter Optimization: Elbow & Silhouette Curves
    k_eval_records = []
    print("[INFO] Evaluating K-Means cluster validity across k in [2, 7]...")
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        cluster_labels = km.fit_predict(X_seg_scaled)
        sil = silhouette_score(X_seg_scaled, cluster_labels)
        ch_score = calinski_harabasz_score(X_seg_scaled, cluster_labels)
        db_score = davies_bouldin_score(X_seg_scaled, cluster_labels)

        k_eval_records.append({
            "k_clusters": k,
            "Inertia": round(km.inertia_, 2),
            "Silhouette_Score": round(sil, 4),
            "Calinski_Harabasz_Index": round(ch_score, 2),
            "Davies_Bouldin_Index": round(db_score, 4)
        })

    df_k_eval = pd.DataFrame(k_eval_records)
    df_k_eval.to_csv(os.path.join(output_dir, "clustering_k_evaluation.csv"), index=False)

    # 2. Fit Optimal Model with k=4
    print(f"[INFO] Fitting final K-Means model with optimal k = {optimal_k}...")
    kmeans_optimal = KMeans(n_clusters=optimal_k, init="k-means++", n_init=15, random_state=random_state)
    labels = kmeans_optimal.fit_predict(X_seg_scaled)

    # Assign cluster labels to dataframe
    df_clustered = df_clean.copy()
    df_clustered["Cluster_ID"] = labels

    # 3. PCA Dimensionality Reduction
    pca = PCA(n_components=3, random_state=random_state)
    X_pca = pca.fit_transform(X_seg_scaled)
    df_clustered["PCA_1"] = X_pca[:, 0]
    df_clustered["PCA_2"] = X_pca[:, 1]
    df_clustered["PCA_3"] = X_pca[:, 2]

    df_pca_var = pd.DataFrame({
        "Principal_Component": ["PC1", "PC2", "PC3"],
        "Explained_Variance_Ratio": pca.explained_variance_ratio_.round(4),
        "Cumulative_Variance_Pct": (np.cumsum(pca.explained_variance_ratio_) * 100).round(2)
    })
    df_pca_var.to_csv(os.path.join(output_dir, "pca_explained_variance.csv"), index=False)

    # 4. Synthesize Customer Persona Profiles
    persona_map = {
        0: "High-Value Enterprise Loyalists",
        1: "At-Risk Month-to-Month Consumers",
        2: "Tech-Savvy Heavy Streamers",
        3: "Budget-Conscious Minimalists"
    }

    # Sort clusters logically by mean tenure and value to match descriptive persona names
    cluster_means = df_clustered.groupby("Cluster_ID")[segmentation_features + ["Churn", "CustomerLifetimeValue"]].mean()

    # Create persona profiles
    persona_records = []
    for c_id in range(optimal_k):
        sub = df_clustered[df_clustered["Cluster_ID"] == c_id]
        p_name = persona_map.get(c_id, f"Cluster Persona {c_id}")
        persona_records.append({
            "Cluster_ID": c_id,
            "Persona_Name": p_name,
            "Customer_Count": len(sub),
            "Cohort_Share_Pct": f"{(len(sub) / len(df_clustered)) * 100:.1f}%",
            "Mean_Tenure_Months": round(sub["TenureMonths"].mean(), 1),
            "Mean_Monthly_Usage_GB": round(sub["MonthlyUsageGB"].mean(), 1),
            "Mean_Support_Tickets": round(sub["SupportTicketsLastYear"].mean(), 2),
            "Mean_Payment_Delays": round(sub["PaymentDelaysCount"].mean(), 2),
            "Mean_Satisfaction": round(sub["CustomerSatisfactionScore"].mean(), 2),
            "Mean_Monthly_Charges": round(sub["MonthlyCharges"].mean(), 2),
            "Actual_Churn_Rate_Pct": f"{sub['Churn'].mean() * 100:.1f}%",
            "Mean_CLV_Dollars": round(sub["CustomerLifetimeValue"].mean(), 2)
        })

    df_personas = pd.DataFrame(persona_records)
    df_personas.to_csv(os.path.join(output_dir, "customer_persona_profiles.csv"), index=False)
    print(f"[INFO] Customer Persona Profiles exported:\n{df_personas[['Persona_Name', 'Customer_Count', 'Actual_Churn_Rate_Pct', 'Mean_CLV_Dollars']].to_string(index=False)}")

    return {
        "df_clustered": df_clustered,
        "df_k_eval": df_k_eval,
        "df_personas": df_personas,
        "df_pca_var": df_pca_var,
        "kmeans_model": kmeans_optimal,
        "pca_model": pca,
        "segmentation_features": segmentation_features
    }
