"""
evaluation_metrics.py
Quantitative evaluation metrics for unsupervised clustering:
Inertia (WCSS), Silhouette Score & Samples, Davies-Bouldin Index,
Calinski-Harabasz Index, and Cluster Profiling summaries.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)

def evaluate_cluster_range(X, k_range=range(2, 11), random_state=42):
    """
    Evaluates clustering quality across a range of k values using:
    - Inertia (WCSS)
    - Silhouette Score
    - Davies-Bouldin Index
    - Calinski-Harabasz Index
    """
    records = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=300, random_state=random_state)
        labels = km.fit_predict(X)
        
        inertia = km.inertia_
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        chi = calinski_harabasz_score(X, labels)
        
        records.append({
            "k": k,
            "Inertia_WCSS": round(inertia, 2),
            "Silhouette_Score": round(sil, 4),
            "Davies_Bouldin_Index": round(dbi, 4),
            "Calinski_Harabasz_Index": round(chi, 2)
        })
        
    df_metrics = pd.DataFrame(records)
    return df_metrics

def compute_silhouette_details(X, labels):
    """
    Computes sample-level silhouette scores and cluster-level averages.
    """
    sample_scores = silhouette_samples(X, labels)
    overall_score = silhouette_score(X, labels)
    
    cluster_silhouettes = {}
    for c in np.unique(labels):
        if c == -1:
            continue
        cluster_scores = sample_scores[labels == c]
        cluster_silhouettes[c] = {
            "mean": np.mean(cluster_scores),
            "min": np.min(cluster_scores),
            "max": np.max(cluster_scores),
            "negative_count": np.sum(cluster_scores < 0)
        }
        
    return {
        "overall_score": overall_score,
        "sample_scores": sample_scores,
        "cluster_silhouettes": cluster_silhouettes
    }

def generate_cluster_profiles(df, cluster_col="KMeans_Cluster"):
    """
    Generates detailed behavioral and demographic summaries for each cluster.
    """
    profile_rows = []
    total_customers = len(df)
    
    for c in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == c]
        count = len(sub)
        pct = (count / total_customers) * 100
        
        mean_age = sub["Age"].mean()
        mean_income = sub["Annual_Income_k"].mean()
        mean_spending = sub["Spending_Score"].mean()
        
        pct_female = (sub["Gender"].str.lower() == "female").mean() * 100
        pct_male = (sub["Gender"].str.lower() == "male").mean() * 100
        
        profile_rows.append({
            "Cluster": c,
            "Size": count,
            "Percentage": round(pct, 1),
            "Mean_Age": round(mean_age, 1),
            "Mean_Income_k": round(mean_income, 1),
            "Mean_Spending_Score": round(mean_spending, 1),
            "Female_Pct": round(pct_female, 1),
            "Male_Pct": round(pct_male, 1)
        })
        
    df_profiles = pd.DataFrame(profile_rows)
    return df_profiles
