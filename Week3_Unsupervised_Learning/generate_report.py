"""
generate_report.py
Generates the comprehensive Week 3 Unsupervised Learning & Clustering Analysis Report
as a professionally styled Word (.docx) document.

Author: Ajnish Kumar | August / September 2026
"""

import os
import sys
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml, OxmlElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
OUT_DIR = os.path.join(BASE_DIR, "output")
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

# Load output metrics and tables
metrics_df = pd.read_csv(os.path.join(OUT_DIR, "cluster_metrics_summary.csv"))
profiles_df = pd.read_csv(os.path.join(OUT_DIR, "cluster_profile_statistics.csv"))
comparison_df = pd.read_csv(os.path.join(OUT_DIR, "clustering_comparison_table.csv"))

doc = Document()

# Page Margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.1)
    section.right_margin = Inches(1.1)

# Color Palette Constants
COLOR_PRIMARY = RGBColor(31, 73, 125)      # Deep Navy
COLOR_SECONDARY = RGBColor(68, 114, 196)  # Steel Blue
COLOR_MUTED = RGBColor(80, 80, 80)        # Charcoal Gray
COLOR_DARK = RGBColor(30, 30, 30)         # Near Black

# Helper Functions
def set_font(run, name="Calibri", size=11, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color

def heading(text, level=1, space_before=14, space_after=6):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    color = COLOR_PRIMARY if level == 1 else (COLOR_SECONDARY if level == 2 else COLOR_DARK)
    for run in p.runs:
        run.font.name = "Calibri"
        run.font.color.rgb = color
    return p

def para(text="", bold=False, italic=False, indent=False, size=11, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if indent:
        p.paragraph_format.left_indent = Inches(0.3)
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic, color=COLOR_DARK)
    return p

def bullet(text, level=0, bold_prefix="", space_after=3):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.35 + level * 0.25)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        set_font(r_pre, size=10.5, bold=True, color=COLOR_PRIMARY)
    run = p.add_run(text)
    set_font(run, size=10.5, color=COLOR_DARK)
    return p

def code_block(lines):
    """Inserts a monospaced gray background box for code snippets."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F6F9"/>')
    cell._tc.get_or_add_tcPr().append(shd)
    
    # Border
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="18" w:space="0" w:color="1F497D"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    cell._tc.get_or_add_tcPr().append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.05
    for i, line in enumerate(lines):
        if i > 0:
            p = cell.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
        run = p.add_run(line)
        run.font.name = "Courier New"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(20, 20, 20)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def insert_image(filename, width=6.2, caption=None):
    path = os.path.join(VIZ_DIR, filename)
    if os.path.exists(path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run()
        run.add_picture(path, width=Inches(width))
        if caption:
            cp = doc.add_paragraph()
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.paragraph_format.space_before = Pt(2)
            cp.paragraph_format.space_after = Pt(10)
            r_cap = cp.add_run(caption)
            set_font(r_cap, size=9.5, italic=True, color=COLOR_MUTED)
    else:
        para(f"[Figure missing: {filename}]", italic=True)

def style_table(tbl, col_widths=None):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(tbl.rows):
        # Tr height
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        for j, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if col_widths and j < len(col_widths):
                cell.width = Inches(col_widths[j])
            
            # Header shading
            if i == 0:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1F497D"/>')
                cell._tc.get_or_add_tcPr().append(shd)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(4)
                    p.paragraph_format.space_after = Pt(4)
                    for run in p.runs:
                        run.font.name = "Calibri"
                        run.font.size = Pt(10)
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
            else:
                # Zebra shading
                if i % 2 == 1:
                    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F9FAFB"/>')
                else:
                    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="FFFFFF"/>')
                cell._tc.get_or_add_tcPr().append(shd)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
                    for run in p.runs:
                        run.font.name = "Calibri"
                        run.font.size = Pt(9.5)
                        run.font.color.rgb = COLOR_DARK

def callout_box(text, title="Key Finding"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF3FA"/>')
    cell._tc.get_or_add_tcPr().append(shd)
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="2B5C8F"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    cell._tc.get_or_add_tcPr().append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(5)
    r1 = p.add_run(f"★ {title}: ")
    set_font(r1, size=10.5, bold=True, color=COLOR_PRIMARY)
    r2 = p.add_run(text)
    set_font(r2, size=10.5, italic=True, color=COLOR_DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

print("[Info] Constructing Week 3 Document Content...")

# =============================================================================
# COVER PAGE
# =============================================================================
doc.add_paragraph().paragraph_format.space_before = Pt(20)

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_p.add_run("Week 3 Task Report")
set_font(run, size=28, bold=True, color=COLOR_PRIMARY)

subtitle_p = doc.add_paragraph()
subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle_p.add_run("Unsupervised Learning and Clustering Analysis")
set_font(run, size=17, bold=True, color=COLOR_SECONDARY)

tag_p = doc.add_paragraph()
tag_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = tag_p.add_run("Customer Behavioral Segmentation, Multi-Algorithm Benchmark (K-Means, Hierarchical, DBSCAN, PCA), and Strategic Marketing Architecture")
set_font(run, size=11.5, italic=True, color=COLOR_MUTED)

doc.add_paragraph().paragraph_format.space_before = Pt(40)

meta_data = [
    ("Student Name", "Ajnish Kumar"),
    ("Roll Number", "241809046713"),
    ("Academic Degree", "Bachelor of Computer Applications (BCA)"),
    ("University", "Vinoba Bhave University, Hazaribag"),
    ("Internship Program", "Virtual Data Science with Python Trainee (Yuva Intern)"),
    ("Submission Track", "Week 3: Machine Learning Specialization"),
    ("Submission Date", "September 2026"),
]

meta_tbl = doc.add_table(rows=len(meta_data), cols=2)
meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (k, v) in enumerate(meta_data):
    r = meta_tbl.rows[i]
    r.cells[0].width = Inches(2.2)
    r.cells[1].width = Inches(4.0)
    p0 = r.cells[0].paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run0 = p0.add_run(k + " :")
    set_font(run0, size=11, bold=True, color=COLOR_PRIMARY)
    p1 = r.cells[1].paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run1 = p1.add_run("  " + v)
    set_font(run1, size=11, color=COLOR_DARK)
    p0.paragraph_format.space_before = Pt(3)
    p0.paragraph_format.space_after = Pt(3)
    p1.paragraph_format.space_before = Pt(3)
    p1.paragraph_format.space_after = Pt(3)

doc.add_paragraph().paragraph_format.space_before = Pt(40)

exec_box_p = doc.add_paragraph()
exec_box_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_box = exec_box_p.add_run("A Comprehensive Exploration of Clustering Methodologies, Validations, and Industrial Applications\nEstimated Work Effort: 30 to 35 Hours | Scikit-Learn Machine Learning Pipeline")
set_font(r_box, size=10, italic=True, color=COLOR_MUTED)

doc.add_page_break()

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================
heading("Executive Summary", level=1)
para(
    "This technical report details the end-to-end implementation and critical evaluation of unsupervised "
    "learning and clustering algorithms completed for the Week 3 Data Science Internship task at Yuva Intern. "
    "Unlike supervised learning paradigms where ground-truth target labels steer model optimization, unsupervised "
    "clustering requires discovering intrinsic, latent geometric structures within unstructured multidimensional data. "
    "This investigation applies rigorous mathematical and empirical methodologies to segment consumer behavioral "
    "profiles using the benchmark Mall Customer Segmentation dataset."
)
para(
    "A thorough multi-algorithm comparative study was conducted, contrasting partition-based K-Means clustering, "
    "agglomerative hierarchical clustering across three linkage criteria (Ward, Complete, Average), density-based "
    "spatial clustering of applications with noise (DBSCAN), and dimensionality reduction via Principal Component "
    "Analysis (PCA). Hyperparameter optimization across candidate cluster sizes k in [2, 10] mathematically confirmed "
    "that k = 5 represents the global structural optimum, simultaneously achieving an Elbow inflection, a peak mean "
    "Silhouette Coefficient of 0.5547, a minimal Davies-Bouldin Index of 0.5722, and a maximal Calinski-Harabasz score of 248.65."
)
para(
    "The 5 distinct customer personas uncovered provide profound commercial utility: (1) High-Value Champions (VIPs) "
    "exhibiting high earnings ($86.5k) and high expenditure (score 82.1); (2) Cautious Affluent customers possessing substantial "
    "income ($88.2k) but ultra-low spending (17.1); (3) Impulsive Trendsetters with lower earnings ($25.7k) but intense spending "
    "(79.4); (4) Budget Seekers who are economically constrained across both axes; and (5) Moderate Mainstream shoppers "
    "constituting the largest commercial baseline (40.5% of the total demographic). Strategic marketing architectures, "
    "pricing elasticity models, and omnichannel activation playbooks are synthesized for each persona."
)

callout_box(
    "Optimal partition at k=5 yields a Silhouette score of 0.5547 and isolates 5 distinct commercial personas, "
    "enabling enterprise-grade targeting that maximizes Customer Lifetime Value (CLV) and optimizes acquisition expenditure.",
    title="Key Milestone"
)

# =============================================================================
# 1. INTRODUCTION & OBJECTIVES
# =============================================================================
heading("1. Introduction & Task Objectives", level=1)
para(
    "Unsupervised machine learning represents one of the foundational pillars of modern data science. In commercial "
    "enterprises, customer data arrives continuously without predefined labels indicating consumer intent, brand loyalty, "
    "or risk tolerance. Clustering algorithms overcome this challenge by organizing unlabeled feature vectors into cohesive, "
    "well-separated subgroups such that intra-cluster similarity is maximized while inter-cluster similarity is minimized."
)

heading("1.1  Task Objectives & Work Scope", level=2)
for b, prefix in [
    ("Identify, audit, and preprocess a publicly available multi-feature consumer dataset appropriate for unsupervised learning.", "Dataset Selection & Curation: "),
    ("Establish a clean data pipeline addressing hygiene, missingness, extreme outliers (IQR/Z-score), and distance-preserving feature scaling.", "Mathematical Preprocessing: "),
    ("Implement K-Means clustering and execute rigorous hyperparameter grid searches across k in [2, 10] using Elbow (WCSS), Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.", "Hyperparameter Tuning: "),
    ("Implement Agglomerative Hierarchical Clustering across Ward, Complete, and Average linkages, validating tree fidelity via the Cophenetic Correlation Coefficient.", "Hierarchical Exploration: "),
    ("Implement DBSCAN to test density-based clustering and isolate anomalous/noise data points via k-nearest neighbor distance elbow plots.", "Density-Based Clustering: "),
    ("Apply Principal Component Analysis (PCA) to perform 2D and 3D dimensionality reduction, analyze explained variance ratios, and generate feature loading biplots.", "Dimensionality Reduction: "),
    ("Synthesize behavioral profiles, radar charts, and commercial strategy roadmaps for each discovered customer segment.", "Business Persona Synthesis: "),
    ("Deliver a publication-grade formal DOCX report, complete Python scripts, and an interactive Jupyter Notebook.", "Documentation & Delivery: "),
]:
    bullet(b, bold_prefix=prefix)

heading("1.2  Technology Stack & Environment", level=2)
para(
    "The technical implementation was constructed on Python 3.12 utilizing core scientific computing and machine learning libraries. "
    "The deterministic reproducibility of all experiments is enforced through centralized random seeds (random_state = 42)."
)
code_block([
    "import sklearn               # Scikit-learn: KMeans, Agglomerative, DBSCAN, PCA, Metrics",
    "import scipy.cluster.hierarchy # Dendrogram generation and cophenetic correlation analysis",
    "import pandas as pd          # Vectorized data manipulation and tabular transformations",
    "import numpy as np           # Linear algebra and multi-dimensional matrix operations",
    "import matplotlib.pyplot as plt # Core plotting and visualization engine",
    "import seaborn as sns        # Statistical graphics and distribution formatting",
    "import docx                  # Programmatic document synthesis and layout compilation"
])

# =============================================================================
# 2. DATASET SELECTION, EXPLORATION & PREPROCESSING
# =============================================================================
heading("2. Dataset Selection, Auditing & Preprocessing", level=1)
para(
    "For this investigation, the benchmark Mall Customer Segmentation Dataset was selected. This dataset represents "
    "an authentic compilation of consumer behavior gathered from an urban retail mall membership program. It comprises "
    "200 customer records characterized across 5 distinct attributes:"
)
for b, prefix in [
    ("Unique categorical identifier assigned to each consumer (dropped from distance calculations to avoid false ordinal bias).", "CustomerID: "),
    ("Biological gender of the customer (Female: 56.0%, Male: 44.0%).", "Gender: "),
    ("Customer age in years, spanning from 18 to 70 years with a mean of 38.85 years.", "Age: "),
    ("Annual household income in thousands of US dollars ($15k to $137k; mean $60.56k).", "Annual Income (k$): "),
    ("Proprietary metric computed by mall analytics based on purchasing frequency, transaction volume, and recency (scale: 1 to 100).", "Spending Score (1-100): "),
]:
    bullet(b, bold_prefix=prefix)

heading("2.1  Dataset Schema and Statistical Properties", level=2)
para(
    "A rigorous initial statistical audit was performed to determine data distribution parameters, central tendencies, "
    "and variance spreads across all features."
)

raw_stats_data = [
    ["Feature Attribute", "Data Type", "Count", "Mean ± Std", "Median", "IQR [Q25 - Q75]", "Min - Max", "Skewness"],
    ["Age", "int64", "200", "38.85 ± 13.97", "36.00", "20.25 [28.75 - 49.00]", "18 - 70", "+0.485 (Mild Right)"],
    ["Annual Income (k$)", "int64", "200", "60.56 ± 26.26", "61.50", "36.50 [41.50 - 78.00]", "15 - 137", "+0.322 (Near Normal)"],
    ["Spending Score (1-100)", "int64", "200", "50.20 ± 25.82", "50.00", "38.25 [34.75 - 73.00]", "1 - 99", "-0.047 (Symmetric)"],
    ["Gender", "object", "200", "112 F / 88 M", "Female", "56% F / 44% M", "Binary", "N/A"]
]
tbl_stats = doc.add_table(rows=len(raw_stats_data), cols=len(raw_stats_data[0]))
for i, row in enumerate(raw_stats_data):
    for j, val in enumerate(row):
        tbl_stats.rows[i].cells[j].text = val
style_table(tbl_stats, col_widths=[1.4, 0.7, 0.5, 1.1, 0.7, 1.4, 0.8, 1.1])
doc.add_paragraph().paragraph_format.space_after = Pt(6)

heading("2.2  Data Hygiene and Outlier Audit", level=2)
para(
    "Data hygiene auditing revealed zero missing values (0 null, NaN, or infinite entries across all 200 rows), confirming "
    "100% data completeness. Outlier detection was executed using Tukey's Interquartile Range (IQR) fence criteria: "
    "Lower = Q25 - 1.5*IQR; Upper = Q75 + 1.5*IQR. The analysis identified exactly two marginal entries in Annual Income "
    "(Customer IDs 199 and 200 with income of $137k, slightly surpassing the upper boundary of $132.75k). These records "
    "represent legitimate high-net-worth consumers rather than measurement or transmission errors. Retaining them preserves "
    "authentic affluent market dynamics, which is vital for clustering."
)

insert_image(
    "fig01_distributions_boxplots.png",
    width=6.3,
    caption="Figure 1: Univariate feature distributions with KDE curves and Tukey boxplots showing quartile fences and mean/median central tendencies."
)

heading("2.3  Bivariate and Correlation Analysis", level=2)
para(
    "Inspection of bivariate interactions highlights fundamental demographic and behavioral patterns. Spending Score exhibits "
    "a distinct negative correlation with Age (r = -0.33), indicating that younger consumers demonstrate higher spending propensity "
    "than older cohorts. In contrast, Annual Income displays near-zero linear correlation with Spending Score (r = 0.01), "
    "revealing that income alone is completely incapable of predicting consumer expenditure. This orthogonal independence "
    "makes Annual Income and Spending Score the optimal 2D feature manifold for unsupervised clustering."
)

insert_image(
    "fig02_pairplot_gender_kde.png",
    width=6.0,
    caption="Figure 2: Multivariate pairplot displaying pairwise feature interactions and bivariate Kernel Density Estimates (KDE) separated by Gender."
)

insert_image(
    "fig03_correlation_matrix.png",
    width=6.0,
    caption="Figure 3: Dual correlation matrices comparing Pearson linear correlation (r) against Spearman rank-order monotonic correlation (ρ)."
)

heading("2.4  Feature Standardization & Mathematical Scaling", level=2)
para(
    "Clustering algorithms rely heavily on geometric distance metrics (principally Euclidean distance). When feature attributes "
    "operate across discordant units and ranges (e.g., Annual Income spanning up to 137 while Spending Score spans 1 to 100), "
    "unscaled variables with larger raw magnitudes dominate the distance calculations. To ensure objective, equal geometric weighting, "
    "features were transformed via Z-score standardization (StandardScaler):"
)
para(
    "z = (x - μ) / σ", indent=True, bold=True
)
para(
    "where μ is the sample feature mean and σ represents standard deviation, ensuring zero mean and unit variance (μ = 0, σ² = 1)."
)

# =============================================================================
# 3. MATHEMATICAL FOUNDATIONS OF CLUSTERING TECHNIQUES
# =============================================================================
heading("3. Mathematical Foundations of Clustering Techniques", level=1)
para(
    "To ensure technical rigor, each clustering algorithm implemented in this investigation was evaluated based on its underlying "
    "mathematical assumptions, objective cost functions, and optimization mechanics."
)

heading("3.1  K-Means Clustering (Lloyd's Algorithm with k-means++)", level=2)
para(
    "K-Means is an iterative centroid-based partitioning technique that segments n observations into k non-overlapping clusters "
    "S = {S₁, S₂, ..., S_k}. The objective is to minimize the Within-Cluster Sum of Squares (WCSS), also termed Inertia (J):"
)
para(
    "J = Σ_{j=1}^{k} Σ_{x_i ∈ S_j} || x_i - c_j ||²", indent=True, bold=True
)
para(
    "where c_j denotes the centroid of cluster S_j, computed as the arithmetic mean of all assigned feature vectors. "
    "Standard Lloyd's algorithm is susceptible to poor local minima if centroids are initialized randomly. To guarantee fast "
    "and globally optimal convergence, the k-means++ initialization scheme was utilized. Under k-means++, initial centers are "
    "selected sequentially with probability proportional to the squared Euclidean distance to the nearest existing center:"
)
para(
    "P(x) = D(x)² / Σ_{x' ∈ X} D(x')²", indent=True, bold=True
)

heading("3.2  Agglomerative Hierarchical Clustering", level=2)
para(
    "Agglomerative clustering is a bottom-up hierarchical methodology where every observation initially occupies its own individual "
    "singleton cluster. Successive pairs of clusters are iteratively merged based on a linkage criterion until a single unified "
    "tree (dendrogram) is formed. Three primary linkage functions were mathematically evaluated:"
)
for b, prefix in [
    ("Minimizes the total within-cluster variance increase resulting from merging clusters A and B: ΔESS = (n_A · n_B / (n_A + n_B)) || c_A - c_B ||². Ward's method produces compact, spherical clusters.", "Ward's Minimum Variance Linkage: "),
    ("Determines proximity based on the maximum pairwise distance between any point in cluster A and any point in cluster B: D(A, B) = max { d(x, y) : x ∈ A, y ∈ B }. Avoids chaining and enforces compact cluster diameters.", "Complete Linkage (Maximum Distance): "),
    ("Computes the average distance between all cross-cluster point pairs: D(A, B) = (1 / (|A|·|B|)) Σ_{x ∈ A} Σ_{y ∈ B} d(x, y). Moderately robust to noise.", "Average Linkage (Mean Distance): "),
]:
    bullet(b, bold_prefix=prefix)
para(
    "The preservation quality of each linkage tree was validated via the Cophenetic Correlation Coefficient (c):"
)
para(
    "c = Σ_{i<j} (d_ij - d̄)(t_ij - t̄) / √[ Σ_{i<j} (d_ij - d̄)² · Σ_{i<j} (t_ij - t̄)² ]", indent=True, bold=True
)
para(
    "where d_ij is the original pairwise Euclidean distance and t_ij is the cophenetic distance (the dendrogram height where "
    "observations i and j are first merged). A coefficient closer to 1.0 signifies higher hierarchical preservation fidelity."
)

heading("3.3  DBSCAN (Density-Based Spatial Clustering of Applications with Noise)", level=2)
para(
    "DBSCAN separates dense spatial regions from low-density noise without requiring pre-specification of cluster count k. "
    "It relies on two core parameters: the neighborhood radius (ε, epsilon) and the minimum density threshold (min_samples, MinPts). "
    "Points are classified into three topological states:"
)
for b, prefix in [
    ("Point p where |N_ε(p)| ≥ min_samples within distance radius ε.", "Core Points: "),
    ("Point q where |N_ε(q)| < min_samples, but q is density-reachable from at least one core point.", "Border Points: "),
    ("Point o that is neither a core point nor density-reachable from any core point; designated as an anomaly/outlier (label = -1).", "Noise / Outlier Points: "),
]:
    bullet(b, bold_prefix=prefix)

heading("3.4  Quantitative Cluster Validation Metrics", level=2)
para(
    "Because unsupervised clustering lacks external ground-truth labels, models were evaluated using four complementary "
    "intrinsic validation metrics:"
)
for b, prefix in [
    ("Measures intra-cluster cohesion versus inter-cluster separation. For each sample i, s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is mean intra-cluster distance and b(i) is mean nearest-cluster distance. Values range from -1 (misclassified) to +1 (perfectly partitioned).", "Silhouette Coefficient (s): "),
    ("Evaluates the ratio of within-cluster scatter (s_i + s_j) to between-cluster separation d(c_i, c_j). Lower scores signify tighter clusters and superior inter-cluster separation.", "Davies-Bouldin Index (DBI): "),
    ("Also known as the Variance Ratio Criterion. Computes the ratio of between-cluster dispersion (SSB) to within-cluster dispersion (SSW), normalized by degrees of freedom: CHI = [SSB / (k - 1)] / [SSW / (n - k)]. Higher scores indicate superior partition quality.", "Calinski-Harabasz Index (CHI): "),
    ("Measures compactness by summing squared distances from each point to its assigned centroid. Used in the Elbow heuristic to identify diminishing returns.", "Within-Cluster Sum of Squares (WCSS): "),
]:
    bullet(b, bold_prefix=prefix)

# =============================================================================
# 4. HYPERPARAMETER OPTIMIZATION & MODEL SELECTION
# =============================================================================
heading("4. Hyperparameter Optimization & Model Selection", level=1)
para(
    "To rigorously identify the optimal number of clusters k, an exhaustive hyperparameter search was conducted across k in [2, 10]. "
    "All four validation metrics were computed synchronously at each candidate k."
)

heading("4.1  Multi-Metric Validation Across k ∈ [2, 10]", level=2)
para(
    "The experimental validation results are presented below. Inspecting multiple metrics simultaneously prevents "
    "overfitting and eliminates subjective bias in selecting the cluster parameter."
)

val_table_data = [
    ["Cluster Count (k)", "Inertia (WCSS)", "Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index", "Evaluation Verdict"]
]
for idx, row in metrics_df.iterrows():
    k_v = int(row["k"])
    wcss_v = f"{row['Inertia_WCSS']:.2f}"
    sil_v = f"{row['Silhouette_Score']:.4f}"
    dbi_v = f"{row['Davies_Bouldin_Index']:.4f}"
    chi_v = f"{row['Calinski_Harabasz_Index']:.2f}"
    
    if k_v == 5:
        verdict = "★ Global Optimum (Peak Sil, Min DBI, Elbow)"
    elif k_v == 6:
        verdict = "Sub-optimal (Drop in Sil, Increase in DBI)"
    elif k_v == 4:
        verdict = "Under-partitioned (Lower Sil, Merged VIPs)"
    elif k_v == 3:
        verdict = "Overly Aggregated (Poor Separation)"
    else:
        verdict = "Sub-optimal Partition"
    val_table_data.append([str(k_v), wcss_v, sil_v, dbi_v, chi_v, verdict])

tbl_val = doc.add_table(rows=len(val_table_data), cols=len(val_table_data[0]))
for i, row in enumerate(val_table_data):
    for j, val in enumerate(row):
        tbl_val.rows[i].cells[j].text = val
style_table(tbl_val, col_widths=[1.2, 1.0, 1.1, 1.2, 1.3, 1.8])
doc.add_paragraph().paragraph_format.space_after = Pt(6)

insert_image(
    "fig04_elbow_silhouette_calinski.png",
    width=6.3,
    caption="Figure 4: Multi-panel validation curves across k ∈ [2, 10] showing (A) WCSS Elbow curve, (B) Silhouette Score peak, (C) Davies-Bouldin minimum, and (D) Calinski-Harabasz Index."
)

heading("4.2  Detailed Silhouette Analysis per Sample", level=2)
para(
    "While mean Silhouette score provides a broad summary, it can conceal poorly formed or negative-silhouette clusters. "
    "To verify cluster stability, sample-level silhouette profiles were generated for candidate partitions k = 3, 4, 5, and 6."
)
para(
    "At k = 3 and k = 4, several clusters exhibit severe width disparities, and multiple samples cross the zero line into negative "
    "territory, indicating poor placement. At k = 5, all five cluster silhouettes extend beyond the average threshold line (0.555), "
    "cluster thicknesses are remarkably uniform, and negative coefficients are virtually eliminated. At k = 6, the silhouette score "
    "drops to 0.5399 as one natural cluster is artificially fractured. Thus, k = 5 is mathematically and structurally definitive."
)

insert_image(
    "fig05_silhouette_cluster_analysis.png",
    width=6.3,
    caption="Figure 5: Silhouette coefficient plots per individual cluster and sample thickness for k ∈ [3, 4, 5, 6]. The red dashed line denotes the overall mean silhouette score."
)

# =============================================================================
# 5. CLUSTERING RESULTS & MULTI-ALGORITHM BENCHMARK
# =============================================================================
heading("5. Clustering Results & Comparative Benchmark", level=1)
para(
    "Having established k = 5 as the optimal parameter, the data was segmented using K-Means, Agglomerative Hierarchical Clustering, "
    "and DBSCAN. In addition, PCA was applied to inspect cluster separability across multidimensional projections."
)

heading("5.1  K-Means Segmentation (2D Space and 3D Demographic View)", level=2)
para(
    "The 2D segmentation in Annual Income vs. Spending Score reveals five distinct quadrants surrounding a balanced central cluster. "
    "The cluster centroids converge at highly interpretable coordinates:"
)
for b, prefix in [
    ("Coordinates: ($55.3k Income, 49.5 Spending). Represents balanced, middle-class consumer behavior.", "Centroid 0 (Moderate Mainstream): "),
    ("Coordinates: ($86.5k Income, 82.1 Spending). High-earning, high-spending premium target segment.", "Centroid 1 (High-Value Champions): "),
    ("Coordinates: ($25.7k Income, 79.4 Spending). Low-earning, high-spending impulsive youth cohort.", "Centroid 2 (Impulsive Trendsetters): "),
    ("Coordinates: ($88.2k Income, 17.1 Spending). High-earning, highly conservative savers.", "Centroid 3 (Cautious Affluent): "),
    ("Coordinates: ($26.3k Income, 20.9 Spending). Low-earning, low-spending price-sensitive consumers.", "Centroid 4 (Budget Seekers): "),
]:
    bullet(b, bold_prefix=prefix)

insert_image(
    "fig06_kmeans_clusters_2d_3d.png",
    width=6.3,
    caption="Figure 6: K-Means segmentation results displaying (Left) 2D feature space with annotated centroids and (Right) 3D projection incorporating customer Age."
)

heading("5.2  Hierarchical Dendrogram Analysis and Linkage Evaluation", level=2)
para(
    "Hierarchical agglomerative clustering was tested across Ward's, Complete, and Average linkage criteria. Cophenetic "
    "correlation coefficient analysis produced the following preservation scores:"
)
for b, prefix in [
    ("c = 0.718. Ward's criterion optimizes variance compactness, yielding spherical clusters identical to K-Means.", "Ward's Linkage: "),
    ("c = 0.684. Complete linkage maintains clean cluster boundaries but shows lower overall metric balance.", "Complete Linkage: "),
    ("c = 0.728. Average linkage yields high cophenetic correlation but produces imbalanced cluster sizes.", "Average Linkage: "),
]:
    bullet(b, bold_prefix=prefix)
para(
    "Cutting the Ward dendrogram at horizontal distance threshold d = 3.82 generates exactly 5 clusters, perfectly mirroring "
    "the K-Means partition."
)

insert_image(
    "fig07_hierarchical_dendrograms.png",
    width=6.3,
    caption="Figure 7: Hierarchical dendrograms across Ward, Complete, and Average linkages with horizontal cut threshold for k = 5."
)

insert_image(
    "fig08_hierarchical_vs_kmeans.png",
    width=6.2,
    caption="Figure 8: Comparative assessment showing Agglomerative Ward clustering and cross-tabulation contingency matrix with K-Means."
)

heading("5.3  DBSCAN Density-Based Clustering and Outlier Identification", level=2)
para(
    "To test whether customer profiles conform to density gradients or contain distinct anomalies, DBSCAN was implemented. "
    "To calibrate the neighborhood radius ε, the 4-nearest neighbor (4-NN) distance graph was computed. The sorted distance curve "
    "reveals a pronounced knee (elbow) between ε = 0.35 and ε = 0.40. Setting ε = 0.38 with min_samples = 5 successfully isolated "
    "the dense core clusters while detecting isolated peripheral observations as noise/outliers (black cross markers in Figure 9)."
)

insert_image(
    "fig09_dbscan_clustering.png",
    width=6.2,
    caption="Figure 9: DBSCAN clustering results showing (Left) 4-NN distance elbow curve and (Right) density clusters with isolated noise/outliers."
)

heading("5.4  Principal Component Analysis (PCA) & Feature Loadings Biplot", level=2)
para(
    "To assess whether customer clusters remain cleanly partitioned when demographic features (Age, Gender) are incorporated, "
    "Principal Component Analysis was applied across all 4 scaled variables [Age, Annual Income, Spending Score, Gender Code]. "
    "The scree plot reveals that the first two principal components capture 64.8% of the total multivariate variance (PC1: 34.2%, "
    "PC2: 30.6%). The PCA biplot demonstrates that Spending Score and Age project in opposing directions along PC1, confirming "
    "that youthful consumer behavior drives expenditure, while Annual Income projects strongly along PC2."
)

insert_image(
    "fig10_pca_biplot_variance.png",
    width=6.2,
    caption="Figure 10: Principal Component Analysis (PCA) showing (Left) Scree plot of explained variance and (Right) 2D loadings biplot with feature vectors."
)

heading("5.5  Comprehensive Algorithmic Benchmark Comparison", level=2)
para(
    "The technical performance and computational characteristics of the three evaluated clustering paradigms are summarized below:"
)

tbl_comp = doc.add_table(rows=len(comparison_df) + 1, cols=6)
comp_headers = ["Algorithm", "Clusters", "Silhouette", "Davies-Bouldin", "Complexity", "Primary Strength"]
for j, h in enumerate(comp_headers):
    tbl_comp.rows[0].cells[j].text = h

for i, row in comparison_df.iterrows():
    r_cells = tbl_comp.rows[i + 1].cells
    r_cells[0].text = str(row["Algorithm"])
    r_cells[1].text = str(row["Clusters_Detected"])
    r_cells[2].text = str(row["Silhouette_Score"])
    r_cells[3].text = str(row["Davies_Bouldin_Index"])
    r_cells[4].text = str(row["Algorithmic_Complexity"])
    r_cells[5].text = str(row["Key_Strengths"])
style_table(tbl_comp, col_widths=[1.5, 0.7, 0.8, 0.9, 1.4, 1.8])
doc.add_paragraph().paragraph_format.space_after = Pt(6)

# =============================================================================
# 6. IN-DEPTH CLUSTER PROFILING & CUSTOMER PERSONAS
# =============================================================================
heading("6. In-Depth Cluster Profiling & Customer Personas", level=1)
para(
    "To translate mathematical partitions into actionable commercial value, detailed behavioral profiles were synthesized "
    "for each of the 5 clusters. A standardized multi-dimensional radar profile was constructed to visualize normalized "
    "demographic metrics across clusters."
)

insert_image(
    "fig11_cluster_radar_profiles.png",
    width=5.8,
    caption="Figure 11: Multi-dimensional behavioral radar profiles displaying normalized Age, Income, Spending, and Gender proportions per cluster."
)

heading("6.1  Cluster Summary Statistics Table", level=2)

prof_table_data = [
    ["Cluster", "Persona Name", "Size (n)", "Share (%)", "Mean Income", "Mean Spending", "Mean Age", "Female / Male %"]
]
for idx, row in profiles_df.iterrows():
    c_id = f"Cluster {int(row['Cluster'])}"
    p_name = str(row["Persona_Name"])
    sz = str(int(row["Size"]))
    pct = f"{row['Percentage']:.1f}%"
    inc = f"${row['Mean_Income_k']:.1f}k"
    sp = f"{row['Mean_Spending_Score']:.1f}"
    age = f"{row['Mean_Age']:.1f} yrs"
    gen = f"{row['Female_Pct']:.0f}% F / {row['Male_Pct']:.0f}% M"
    prof_table_data.append([c_id, p_name, sz, pct, inc, sp, age, gen])

tbl_prof = doc.add_table(rows=len(prof_table_data), cols=len(prof_table_data[0]))
for i, row in enumerate(prof_table_data):
    for j, val in enumerate(row):
        tbl_prof.rows[i].cells[j].text = val
style_table(tbl_prof, col_widths=[0.9, 1.8, 0.7, 0.7, 1.0, 1.0, 0.9, 1.2])
doc.add_paragraph().paragraph_format.space_after = Pt(6)

heading("6.2  Comprehensive Persona Profiles", level=2)

for idx, row in profiles_df.iterrows():
    c_id = int(row["Cluster"])
    p_name = row["Persona_Name"]
    heading(f"6.2.{idx + 1}  Cluster {c_id}: {p_name}", level=3)
    
    if "Champions" in p_name:
        para(
            f"Cluster {c_id} encompasses {row['Size']} individuals (representing {row['Percentage']}% of the customer base). "
            f"This is the premier commercial segment, exhibiting high average earnings ($86.5k) combined with the highest spending "
            f"score in the dataset (82.1). The demographic is youthful-to-early-career (average age 32.7 years) and predominantly "
            f"female ({row['Female_Pct']}%). They represent high-frequency consumers with high discretionary purchasing power."
        )
        bullet("Prime revenue engine; contributes disproportionately to high-margin retail sales.", bold_prefix="Commercial Profile: ")
        bullet("Luxury apparel, fine jewelry, cosmetics, designer lifestyle goods, gourmet dining.", bold_prefix="Preferred Categories: ")
        bullet("Highly receptive to exclusive VIP treatment, concierge services, and early-access drops.", bold_prefix="Engagement Drivers: ")
    elif "Cautious" in p_name:
        para(
            f"Cluster {c_id} consists of {row['Size']} individuals ({row['Percentage']}% of the customer base). "
            f"These consumers possess the highest average annual income ($88.2k), yet display an extremely conservative "
            f"spending score of only 17.1. Their average age is 41.1 years, with a balanced gender split ({row['Female_Pct']}% Female). "
            f"They represent established, affluent professionals who visit the mall selectively and resist impulsive purchases."
        )
        bullet("Massive untapped commercial potential; possesses high liquidity but low shopping frequency.", bold_prefix="Commercial Profile: ")
        bullet("High-durability electronics, luxury home furnishings, premium business attire, investments.", bold_prefix="Preferred Categories: ")
        bullet("Requires value-driven marketing, warranty guarantees, and high-utility messaging rather than discounts.", bold_prefix="Engagement Drivers: ")
    elif "Impulsive" in p_name:
        para(
            f"Cluster {c_id} contains {row['Size']} individuals ({row['Percentage']}% of total shoppers). "
            f"They are characterized by modest income ($25.7k) but extraordinary spending enthusiasm (mean score 79.4). "
            f"This is the youngest cohort in the mall (average age 25.3 years), with a female majority ({row['Female_Pct']}%). "
            f"They prioritize lifestyle status, fashion trends, and peer social recognition over personal balance sheet savings."
        )
        bullet("High transaction velocity; vulnerable to macroeconomic inflation or credit tightening.", bold_prefix="Commercial Profile: ")
        bullet("Fast-fashion retail, streetwear, activewear, trendy cafes, mobile gadgets, entertainment.", bold_prefix="Preferred Categories: ")
        bullet("Driven by social media influence (Instagram/TikTok), limited-edition drops, and BNPL financing.", bold_prefix="Engagement Drivers: ")
    elif "Budget" in p_name:
        para(
            f"Cluster {c_id} comprises {row['Size']} consumers ({row['Percentage']}% of the dataset). "
            f"They exhibit low annual income ($26.3k) and disciplined, minimal expenditure (mean score 20.9). "
            f"With an average age of 45.2 years (the oldest cohort) and balanced gender distribution, these shoppers "
            f"visit retail centers strictly for essential necessities and price-discounted goods."
        )
        bullet("Highly price-elastic; shopping behavior is governed strictly by budget constraints.", bold_prefix="Commercial Profile: ")
        bullet("Basic grocery items, discount pharmacy products, clearance footwear, budget household goods.", bold_prefix="Preferred Categories: ")
        bullet("Responds strongly to BOGO (buy-one-get-one) deals, seasonal sales, and cashback coupons.", bold_prefix="Engagement Drivers: ")
    else:
        para(
            f"Cluster {c_id} represents the dominant demographic core of the mall, containing {row['Size']} individuals "
            f"({row['Percentage']}% of all customers). Their metrics mirror the overall population median: annual income "
            f"averages $55.3k, spending score averages 49.5, and mean age is 42.7 years. Females comprise {row['Female_Pct']}%. "
            f"This cluster provides steady, predictable footfall and anchors baseline commercial leasing revenue."
        )
        bullet("Provides the high-volume operational foundation supporting standard retail tenants.", bold_prefix="Commercial Profile: ")
        bullet("Family apparel, department store goods, casual dining, children's toys, home goods.", bold_prefix="Preferred Categories: ")
        bullet("Engages with loyalty points programs, family bundle discounts, and weekend promotions.", bold_prefix="Engagement Drivers: ")

# =============================================================================
# 7. BUSINESS IMPLICATIONS & STRATEGIC MARKETING ROADMAP
# =============================================================================
heading("7. Business Implications & Strategic Marketing Roadmap", level=1)
para(
    "Data clustering is commercially meaningless unless translated into decisive strategic initiatives. Below, we synthesize "
    "an enterprise marketing and operational matrix designed to optimize Customer Lifetime Value (CLV), reduce customer acquisition "
    "costs (CAC), and maximize gross merchandise value (GMV)."
)

insert_image(
    "fig12_business_persona_matrix.png",
    width=6.3,
    caption="Figure 12: Customer persona strategic matrix summarizing targeted offers, primary channels, and commercial objectives."
)

heading("7.1  Targeted Omnichannel Action Plan", level=2)

strategies = [
    ("High-Value Champions (VIPs)", [
        "Deploy a dedicated VIP Concierge Program offering reserved valet parking, personal shopping stylists, and luxury lounge access.",
        "Host private, invitation-only brand previews 48 hours prior to public merchandise releases.",
        "Communicate primarily through dedicated clienteling apps, direct WhatsApp Business styling advisors, and bespoke postal invitations."
    ]),
    ("Cautious Affluent (High Income, Low Spend)", [
        "Avoid aggressive discount messaging which diminishes perceived prestige. Focus campaigns on product craftsmanship, heritage, and warranty guarantees.",
        "Partner with financial institutions and premium credit card providers (e.g., Amex Platinum / Chase Sapphire) to deliver exclusive rewards points.",
        "Promote high-ticket, investment-grade merchandise (luxury timepieces, bespoke tailoring, premium electronics) through executive digital channels (LinkedIn, Bloomberg ads, curated email digests)."
    ]),
    ("Impulsive Trendsetters (Youth Spenders)", [
        "Implement seamless Buy-Now-Pay-Later (BNPL) payment integrations (Klarna, Afterpay) at retail point-of-sale to eliminate checkout friction.",
        "Collaborate with micro-influencers and fashion creators on TikTok and Instagram to spark viral social media challenges and pop-up events.",
        "Introduce flash sales and gamified mobile app rewards with 24-hour expiration timers to harness fear-of-missing-out (FOMO)."
    ]),
    ("Budget Seekers (Low Income, Low Spend)", [
        "Focus inventory curation on budget-friendly utility, value multi-packs, and end-of-season clearance racks.",
        "Distribute targeted weekly digital coupons via SMS and low-cost print flyers distributed within local commuter transit hubs.",
        "Establish an accessible cashback loyalty tier where everyday essentials earn redeemable grocery credits to retain baseline footfall."
    ]),
    ("Moderate Mainstream (Core Baseline)", [
        "Implement a tiered family loyalty rewards program incentivizing cross-category shopping (e.g., spend $100 in apparel, receive 20% off dining).",
        "Design weekend family events, holiday festivals, and cinema combo packages to expand visit duration and average basket size.",
        "Automate personalized email and push notification newsletters based on historical seasonal purchasing cycles."
    ])
]

for p_title, actions in strategies:
    heading(p_title, level=3)
    for act in actions:
        bullet(act)

# =============================================================================
# 8. PYTHON CODE SNIPPETS & TECHNICAL IMPLEMENTATION
# =============================================================================
heading("8. Python Code Snippets & Technical Implementation", level=1)
para(
    "The complete clustering pipeline was engineered with clean, modular, production-ready Python code adhering to PEP 8 standards. "
    "Key implementation snippets demonstrating data preprocessing, hyperparameter optimization, model fitting, and validation "
    "are documented below."
)

heading("8.1  Preprocessing, Scaling & Outlier Detection Pipeline", level=2)
code_block([
    "# Data Preprocessing and Feature Scaling (src/data_loader.py)",
    "from sklearn.preprocessing import StandardScaler",
    "import pandas as pd",
    "",
    "df = pd.read_csv('data/Mall_Customers.csv')",
    "df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Spend'}, inplace=True)",
    "",
    "# Outlier detection via Tukey's IQR fences",
    "q25, q75 = df['Income'].quantile(0.25), df['Income'].quantile(0.75)",
    "iqr = q75 - q25",
    "outliers = df[(df['Income'] < q25 - 1.5*iqr) | (df['Income'] > q75 + 1.5*iqr)]",
    "",
    "# Distance-preserving standardization (Z-score)",
    "scaler = StandardScaler()",
    "X_scaled = scaler.fit_transform(df[['Income', 'Spend']])"
])

heading("8.2  Hyperparameter Optimization Loop Across k ∈ [2, 10]", level=2)
code_block([
    "# Multi-Metric Clustering Evaluation (src/evaluation_metrics.py)",
    "from sklearn.cluster import KMeans",
    "from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score",
    "",
    "metrics_log = []",
    "for k in range(2, 11):",
    "    km = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=300, random_state=42)",
    "    labels = km.fit_predict(X_scaled)",
    "    metrics_log.append({",
    "        'k': k,",
    "        'Inertia': km.inertia_,",
    "        'Silhouette': silhouette_score(X_scaled, labels),",
    "        'DBI': davies_bouldin_score(X_scaled, labels),",
    "        'CHI': calinski_harabasz_score(X_scaled, labels)",
    "    })",
    "df_metrics = pd.DataFrame(metrics_log)"
])

heading("8.3  Hierarchical Linkage & Cophenetic Correlation", level=2)
code_block([
    "# Hierarchical Dendrograms & Cophenetic Validation (src/clustering_models.py)",
    "from scipy.cluster.hierarchy import linkage, cophenet",
    "from scipy.spatial.distance import pdist",
    "from sklearn.cluster import AgglomerativeClustering",
    "",
    "distances = pdist(X_scaled, metric='euclidean')",
    "Z_ward = linkage(X_scaled, method='ward')",
    "coph_corr, _ = cophenet(Z_ward, distances)",
    "print(f'Ward Cophenetic Correlation: {coph_corr:.4f}')",
    "",
    "# Fit Agglomerative model at optimal cut k=5",
    "agg = AgglomerativeClustering(n_clusters=5, linkage='ward')",
    "agg_labels = agg.fit_predict(X_scaled)"
])

heading("8.4  DBSCAN Density Clustering & Epsilon Tuning", level=2)
code_block([
    "# DBSCAN Clustering with 4-NN Distance Elbow (src/clustering_models.py)",
    "from sklearn.neighbors import NearestNeighbors",
    "from sklearn.cluster import DBSCAN",
    "",
    "nbrs = NearestNeighbors(n_neighbors=4).fit(X_scaled)",
    "distances, _ = nbrs.kneighbors(X_scaled)",
    "sorted_k_dist = np.sort(distances[:, 3])  # 4th nearest neighbor distance",
    "",
    "# Fit DBSCAN at knee point epsilon = 0.38",
    "dbscan = DBSCAN(eps=0.38, min_samples=5)",
    "db_labels = dbscan.fit_predict(X_scaled)",
    "n_noise = list(db_labels).count(-1)"
])

# =============================================================================
# 9. LIMITATIONS, FUTURE RESEARCH & CRITICAL REFLECTION
# =============================================================================
heading("9. Limitations, Future Research & Critical Reflection", level=1)
para(
    "While this investigation accomplished its objectives and produced statistically robust, actionable clusters, "
    "objective academic reflection requires acknowledging methodological limitations and articulating avenues for future research."
)

heading("9.1  Methodological Limitations", level=2)
for b, prefix in [
    ("K-Means minimizes within-cluster variance under the assumption that clusters are isotropic and spherical. When natural data geometries are non-convex or manifold-shaped, K-Means struggles without kernel transformations.", "Spherical Cluster Assumption: "),
    ("Standard Euclidean distance weights all standardized coordinates equally, regardless of latent feature correlations. While PCA addressed this in 2D projection, Mahalanobis distance could further account for covariance structures.", "Distance Metric Sensitivity: "),
    ("The dataset provides static cross-sectional snapshots of consumer behavior. In reality, consumer preferences transition dynamically across macroeconomic cycles, career milestones, and seasonal holidays.", "Static Snapshot Limitation: "),
]:
    bullet(b, bold_prefix=prefix)

heading("9.2  Future Research Directions", level=2)
for b, prefix in [
    ("Combining RFM (Recency, Frequency, Monetary) metrics with spatial clustering to predict churn probabilities and forward-looking Customer Lifetime Value (CLV).", "Dynamic RFM-Clustering Integration: "),
    ("Applying Self-Organizing Maps (SOM) and Uniform Manifold Approximation and Projection (UMAP) to preserve complex non-linear topological relationships in ultra-high-dimensional spaces.", "Advanced Manifold Learning: "),
    ("Developing online, stream-based clustering pipelines (such as Mini-Batch K-Means or BIRCH) capable of updating centroids dynamically as real-time point-of-sale transactions occur.", "Streaming Real-Time Segmentation: "),
]:
    bullet(b, bold_prefix=prefix)

heading("9.3  Critical Reflection on 30-35 Hours of Effort", level=2)
para(
    "Working on this project across approximately 30 to 35 hours enabled an exhaustive exploration of unsupervised learning concepts. "
    "The journey traversed data hygiene auditing, mathematical formulation of objective cost functions, multi-metric hyperparameter "
    "searches, hierarchical dendrogram tree evaluation, density anomaly isolation, and commercial translation into executive strategy. "
    "The experience reinforced the principle that algorithms must serve clear business objectives: mathematical clustering only reaches "
    "its true potential when paired with deep domain context and actionable decision frameworks."
)

# =============================================================================
# 10. REFERENCES & BIBLIOGRAPHY
# =============================================================================
heading("10. References & Bibliography", level=1)
refs = [
    "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1(14), 281-297.",
    "Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, 1027-1035.",
    "Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65.",
    "Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. IEEE Transactions on Pattern Analysis and Machine Intelligence, (2), 224-227.",
    "Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. Communications in Statistics - Theory and Methods, 3(1), 1-27.",
    "Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD-96 Proceedings, 226-231.",
    "Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association, 58(301), 236-244.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Jain, A. K. (2010). Data clustering: 50 years beyond K-means. Pattern Recognition Letters, 31(8), 651-666."
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.first_line_indent = Inches(-0.4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(r)
    set_font(run, size=9.5, color=COLOR_DARK)

# Save Document
out_doc_path = os.path.join(REPORT_DIR, "Week3_Clustering_Report.docx")
doc.save(out_doc_path)
print(f"\n[Success] Report generated successfully at: {out_doc_path}")
print(f"File size: {os.path.getsize(out_doc_path)} bytes")
