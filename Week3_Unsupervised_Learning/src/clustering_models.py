"""
clustering_models.py
Implementation of K-Means, Hierarchical (Agglomerative) Clustering,
DBSCAN, and PCA Dimensionality Reduction.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

class ClusteringPipeline:
    """
    Unified pipeline executing K-Means, Agglomerative Clustering,
    DBSCAN, and PCA with validation metrics.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.kmeans_model = None
        self.agg_model = None
        self.dbscan_model = None
        self.pca_2d = None
        self.pca_3d = None

    def fit_kmeans(self, X, n_clusters=5):
        """Fits K-Means algorithm on feature matrix X."""
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=self.random_state
        )
        labels = self.kmeans_model.fit_predict(X)
        centroids = self.kmeans_model.cluster_centers_
        inertia = self.kmeans_model.inertia_
        return {
            "model": self.kmeans_model,
            "labels": labels,
            "centroids": centroids,
            "inertia": inertia,
            "n_clusters": n_clusters
        }

    def compute_hierarchical_linkages(self, X):
        """
        Computes linkage matrices for Ward, Complete, and Average methods
        and evaluates Cophenetic Correlation Coefficients.
        """
        distances = pdist(X, metric="euclidean")
        linkage_methods = ["ward", "complete", "average"]
        results = {}
        
        for method in linkage_methods:
            Z = linkage(X, method=method, metric="euclidean")
            coph_corr, _ = cophenet(Z, distances)
            results[method] = {
                "linkage_matrix": Z,
                "cophenetic_corr": coph_corr
            }
        return results

    def fit_agglomerative(self, X, n_clusters=5, linkage_method="ward"):
        """Fits Agglomerative Hierarchical Clustering."""
        self.agg_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        labels = self.agg_model.fit_predict(X)
        return {
            "model": self.agg_model,
            "labels": labels,
            "n_clusters": n_clusters,
            "linkage_method": linkage_method
        }

    def compute_k_distances(self, X, k=4):
        """
        Calculates k-nearest neighbor distances for determining DBSCAN epsilon.
        """
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        sorted_distances = np.sort(distances[:, k - 1])
        return sorted_distances

    def fit_dbscan(self, X, eps=0.35, min_samples=5):
        """Fits DBSCAN density-based clustering algorithm."""
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan_model.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        core_sample_indices = self.dbscan_model.core_sample_indices_
        return {
            "model": self.dbscan_model,
            "labels": labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "core_sample_indices": core_sample_indices,
            "eps": eps,
            "min_samples": min_samples
        }

    def fit_pca(self, X, feature_names=None):
        """
        Fits PCA for 2D and 3D projections and computes feature loadings.
        """
        pca_2 = PCA(n_components=2, random_state=self.random_state)
        X_pca_2d = pca_2.fit_transform(X)
        var_ratio_2d = pca_2.explained_variance_ratio_
        
        pca_3 = PCA(n_components=3, random_state=self.random_state)
        X_pca_3d = pca_3.fit_transform(X)
        var_ratio_3d = pca_3.explained_variance_ratio_

        loadings = None
        if feature_names is not None:
            loadings = pd.DataFrame(
                pca_2.components_.T,
                columns=["PC1", "PC2"],
                index=feature_names
            )

        return {
            "pca_2d": pca_2,
            "pca_3d": pca_3,
            "X_pca_2d": X_pca_2d,
            "X_pca_3d": X_pca_3d,
            "var_ratio_2d": var_ratio_2d,
            "var_ratio_3d": var_ratio_3d,
            "loadings": loadings
        }
