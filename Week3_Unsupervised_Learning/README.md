# Week 3 Task: Unsupervised Learning and Clustering Analysis

**Internship:** Virtual Data Science with Python Trainee | **Yuva Intern**  
**Author:** Ajnish Kumar | **Roll No:** 241809046713  
**Degree / University:** Bachelor of Computer Applications (BCA) | Vinoba Bhave University, Hazaribag  
**Date:** September 2026  

---

## 📌 Project Overview
This repository contains the complete implementation and technical documentation for the **Week 3 Internship Task: Unsupervised Learning and Clustering Analysis**. 

Unlike supervised paradigms that depend on predefined target labels, unsupervised clustering discovers intrinsic geometric manifolds and hidden patterns within unstructured multidimensional data. In this project, consumer behavioral profiles from the benchmark **Mall Customer Segmentation Dataset** are audited, standardized, and partitioned using multiple clustering algorithms:
1. **K-Means Clustering** with $k$-means++ seeding and multi-metric validation across $k \in [2, 10]$.
2. **Agglomerative Hierarchical Clustering** evaluating Ward's, Complete, and Average linkage criteria with Cophenetic Correlation validation.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** with $4$-NN distance curve tuning for anomaly and noise isolation.
4. **Principal Component Analysis (PCA)** for 2D/3D dimensionality reduction and feature vector biplot loadings.

The deliverables include a formal **Word Document Report (`report/Week3_Clustering_Report.docx`)**, an interactive **Jupyter Notebook (`notebooks/Week3_Clustering_Analysis.ipynb`)**, 12 publication-grade figures, modular Python scripts, and customer persona datasets.

---

## 📊 Dataset Description & Data Hygiene

- **Source:** Mall Customer Segmentation Dataset (Public Benchmark from Kaggle / UCI Machine Learning Repository).
- **Observations:** 200 customer membership profiles.
- **Attributes:**
  - `CustomerID`: Unique customer identifier (dropped from distance metrics).
  - `Gender`: Biological gender (Female: 56.0%, Male: 44.0%).
  - `Age`: Customer age in years (Range: 18 - 70; Mean: 38.85 ± 13.97).
  - `Annual Income (k$)`: Annual household earnings in thousands of USD (Range: $15k - $137k; Mean: $60.56k).
  - `Spending Score (1-100)`: Behavioral index computed by mall analytics based on purchase frequency, volume, and recency (Range: 1 - 99; Mean: 50.20 ± 25.82).
- **Data Completeness:** 100% complete (0 null or missing values).
- **Outlier Auditing:** Two entries (IDs 199 and 200 with income of $137k) marginally exceeded Tukey's IQR upper bound ($132.75k). Retained as authentic affluent market representation.
- **Feature Scaling:** Transformed via `StandardScaler` ($z = (x - \mu)/\sigma$) to ensure distance-metric invariance in Euclidean space.

---

## 🔍 Hyperparameter Optimization ($k \in [2, 10]$)

An exhaustive search across $k \in [2, 10]$ established **$k = 5$** as the global structural optimum:

| Cluster Count ($k$) | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Evaluation Verdict |
|:---:|:---:|:---:|:---:|:---:|:---|
| 2 | 269.69 | 0.3213 | 1.2670 | 95.67 | Under-partitioned |
| 3 | 157.70 | 0.4666 | 0.7165 | 151.34 | Over-aggregated |
| 4 | 108.92 | 0.4939 | 0.7096 | 174.60 | Intermediate |
| **5** | **65.57** | **0.5547** | **0.5722** | **248.65** | ★ **Global Optimum (Elbow, Max Sil, Min DBI)** |
| 6 | 55.06 | 0.5399 | 0.6546 | 243.09 | Sub-optimal (Cluster fragmentation) |
| 7 | 44.86 | 0.5281 | 0.7148 | 254.62 | Diminishing returns |
| 8 | 37.15 | 0.4567 | 0.7579 | 267.91 | Over-segmented |
| 9 | 32.39 | 0.4571 | 0.7632 | 270.95 | Over-segmented |
| 10 | 29.69 | 0.4362 | 0.7645 | 263.35 | Over-segmented |

---

## ⚔️ Multi-Algorithm Benchmark Comparison

| Algorithm | Clusters Detected | Silhouette Score | Davies-Bouldin Index | Computational Complexity | Primary Strengths & Limitations |
|:---|:---:|:---:|:---:|:---|:---|
| **K-Means ($k$-means++)** | 5 | **0.5547** | **0.5722** | $\mathcal{O}(t \cdot k \cdot n \cdot d)$ (Linear / Scalable) | **Strength:** Optimal spherical partition, fast convergence.<br>**Limitation:** Assumes spherical clusters, requires pre-specifying $k$. |
| **Agglomerative (Ward)** | 5 | 0.5547 | 0.5722 | $\mathcal{O}(n^2 \log n)$ (Memory-heavy) | **Strength:** Deterministic hierarchy, no random seed, visual tree.<br>**Limitation:** Irreversible merge decisions, computationally expensive. |
| **DBSCAN ($\varepsilon=0.38$)** | 4 Dense + Noise | 0.3541 | 1.1210 | $\mathcal{O}(n \log n)$ to $\mathcal{O}(n^2)$ | **Strength:** Arbitrary cluster shapes, isolates anomalies/outliers.<br>**Limitation:** Sensitive to $\varepsilon$ and `min_samples`, struggles with variable density. |

---

## 👥 Customer Personas & Strategic Business Roadmap

| Cluster | Persona Name | Share (%) | Mean Income | Mean Spending | Mean Age | Strategic Value & Omnichannel Actions |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **Cluster 1** | **High-Value Champions (VIP)** | 19.5% | $86.5k | 82.1 | 32.7 yrs | **Commercial Engine:** Deploy VIP concierge styling, private preview galas, luxury brand partnerships, dedicated clienteling app. |
| **Cluster 3** | **Cautious Affluent (Savers)** | 17.5% | $88.2k | 17.1 | 41.1 yrs | **Untapped Liquidity:** Avoid aggressive discounts. Market investment-grade goods, premium electronics, craftsmanship, and warranty guarantees. |
| **Cluster 2** | **Impulsive Trendsetters (Youth)** | 11.0% | $25.7k | 79.4 | 25.3 yrs | **High Velocity:** Fast fashion, TikTok/Instagram creator partnerships, pop-ups, Buy-Now-Pay-Later (BNPL Klarna/Afterpay) integrations. |
| **Cluster 4** | **Budget Seekers (Sensible)** | 11.5% | $26.3k | 20.9 | 45.2 yrs | **Footfall Anchors:** Curate clearance racks, multi-buy utility essentials, digital coupon SMS alerts, everyday cashback loyalty credits. |
| **Cluster 0** | **Moderate Mainstream (Core)** | 40.5% | $55.3k | 49.5 | 42.7 yrs | **Revenue Baseline:** Family tiered loyalty programs, weekend promotions, casual dining vouchers, seasonal holiday bundles. |

---

## 📈 Visualizations Catalog (12 Publication-Grade Figures)

The analysis produces 12 high-resolution figures saved in `visualizations/`:
1. `fig01_distributions_boxplots.png`: Univariate histograms with KDE and Tukey boxplot outlier fences.
2. `fig02_pairplot_gender_kde.png`: Multivariate pairplot showing bivariate KDE density curves segmented by Gender.
3. `fig03_correlation_matrix.png`: Pearson linear correlation vs. Spearman rank monotonic correlation heatmaps.
4. `fig04_elbow_silhouette_calinski.png`: Multi-panel hyperparameter validation curves across $k \in [2, 10]$.
5. `fig05_silhouette_cluster_analysis.png`: Sample-level silhouette profiles for $k = 3, 4, 5, 6$ testing cluster thickness and negative scores.
6. `fig06_kmeans_clusters_2d_3d.png`: 2D scatter of Annual Income vs. Spending Score with annotated centroids and 3D demographic projection.
7. `fig07_hierarchical_dendrograms.png`: Hierarchical dendrograms across Ward, Complete, and Average linkages with $k=5$ cut line.
8. `fig08_hierarchical_vs_kmeans.png`: Scatter comparison and confusion contingency cross-tabulation heatmap.
9. `fig09_dbscan_clustering.png`: 4-NN distance elbow curve and DBSCAN density clusters with anomaly isolation.
10. `fig10_pca_biplot_variance.png`: Scree plot of explained variance and 2D feature loadings biplot vectors.
11. `fig11_cluster_radar_profiles.png`: Multi-dimensional behavioral spider/radar profiles per cluster.
12. `fig12_business_persona_matrix.png`: Strategic executive matrix detailing offers, channels, and commercial objectives.

---

## 📂 Repository File Structure

```
Week3_Unsupervised_Learning/
├── data/
│   ├── Mall_Customers.csv                # Raw benchmark dataset (200 records)
│   └── Mall_Customers_preprocessed.csv   # Encoded, scaled dataset
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── data_loader.py                    # Hygiene audit, IQR outlier detection, StandardScaler
│   ├── clustering_models.py              # KMeans, Agglomerative, DBSCAN, and PCA pipelines
│   ├── evaluation_metrics.py             # WCSS, Silhouette, DBI, CHI, and profile calculations
│   └── visualizer.py                     # Plotting engine for all 12 publication figures
├── notebooks/
│   └── Week3_Clustering_Analysis.ipynb   # Interactive Jupyter notebook
├── output/
│   ├── cluster_metrics_summary.csv       # Multi-metric evaluation table for k=2..10
│   ├── cluster_profile_statistics.csv    # Statistical demographic breakdown per cluster
│   ├── clustering_comparison_table.csv   # Algorithmic benchmark summary
│   └── customer_segmentation_results.csv # Final customer records with assigned cluster labels
├── visualizations/
│   ├── fig01_distributions_boxplots.png
│   ├── ...
│   └── fig12_business_persona_matrix.png
├── report/
│   └── Week3_Clustering_Report.docx      # 2.69 MB comprehensive Word report deliverable
├── week3_clustering.py                   # Master end-to-end execution pipeline
├── generate_report.py                    # Automated python-docx report compilation script
├── write_notebook.py                     # Automated Jupyter notebook generation script
└── README.md                             # Project documentation
```

---

## 🚀 How to Run

### 1. Prerequisites
Ensure Python 3.10+ is installed with scientific libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy python-docx nbformat
```

### 2. Execute Master Pipeline
Run the full data loading, model fitting, and visualization pipeline:
```bash
cd Week3_Unsupervised_Learning
python week3_clustering.py
```

### 3. Generate Word Document (.docx) Report
To rebuild the Word report deliverable:
```bash
python generate_report.py
```
The resulting document is saved at `report/Week3_Clustering_Report.docx`.

### 4. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/Week3_Clustering_Analysis.ipynb
```

---

## 🎯 Conclusion & Deliverables Summary
All requirements outlined in the Week 3 brief have been completely fulfilled:
- Selected and audited an authentic public dataset (`Mall_Customers.csv`).
- Applied distance-preserving standardization and outlier inspections.
- Implemented and tuned **K-Means**, **Hierarchical Clustering**, **DBSCAN**, and **PCA**.
- Evaluated optimal cluster selection using Elbow, Silhouette, DBI, and CHI.
- Synthesized 12 publication-grade figures and 5 actionable customer personas.
- Generated the required **DOCX report deliverable (`report/Week3_Clustering_Report.docx`)**, complete with theoretical math, code snippets, embedded visuals, and strategic marketing playbooks.
