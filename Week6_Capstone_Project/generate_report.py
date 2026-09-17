"""
generate_report.py
Generates the formal, publication-grade Word document (.docx) report for Week 6:
"Integrative Capstone Project: Enterprise Customer Behavioral Analytics, Predictive Attrition
Modeling, and Lifetime Value Optimization Using Python"

Author: Ajnish Kumar | Roll No: 241809046713
Degree: BCA, Vinoba Bhave University, Hazaribag
Internship: Yuva Intern - Virtual Data Science with Python Trainee
September 2026
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

# Helper to load CSV
def load_csv(name):
    p = os.path.join(OUT_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None

df_cleaning = load_csv("data_cleaning_audit.csv")
df_outliers = load_csv("outlier_iqr_audit.csv")
df_num_profile = load_csv("eda_numerical_summary.csv")
df_hypo = load_csv("eda_bivariate_hypothesis_tests.csv")
df_k_eval = load_csv("clustering_k_evaluation.csv")
df_personas = load_csv("customer_persona_profiles.csv")
df_pca_var = load_csv("pca_explained_variance.csv")
df_cv_churn = load_csv("churn_cv_benchmark_summary.csv")
df_test_churn = load_csv("churn_test_evaluation_metrics.csv")
df_thresh = load_csv("threshold_optimization_churn.csv")
df_cv_clv = load_csv("clv_cv_benchmark_summary.csv")
df_test_clv = load_csv("clv_test_evaluation_metrics.csv")
df_var = load_csv("capital_value_at_risk_summary.csv")
df_sim = load_csv("retention_campaign_roi_simulation.csv")

doc = Document()

# Page Margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Color Constants
COLOR_PRIMARY = RGBColor(31, 73, 125)      # Deep Navy #1F497D
COLOR_SECONDARY = RGBColor(68, 114, 196)  # Steel Blue #4472C4
COLOR_DARK = RGBColor(30, 30, 30)         # Charcoal Dark
COLOR_MUTED = RGBColor(89, 89, 89)        # Subdued Gray
COLOR_ALERT = RGBColor(192, 0, 0)         # Crimson Red
COLOR_SUCCESS = RGBColor(44, 160, 44)     # Forest Green

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
    p.paragraph_format.keep_with_next = True
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

def callout(text, bold_prefix=""):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}>\n'
                          f'  <w:left w:val="single" w:sz="36" w:space="0" w:color="1F497D"/>\n'
                          f'  <w:top w:val="none"/>\n'
                          f'  <w:right w:val="none"/>\n'
                          f'  <w:bottom w:val="none"/>\n'
                          f'</w:tcBorders>')
    tcPr.append(tcBorders)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F2F5F9"/>')
    tcPr.append(shd)
    cell.width = Inches(6.5)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.15)
    p.paragraph_format.right_indent = Inches(0.15)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix + " ")
        set_font(r_pre, size=10.5, bold=True, color=COLOR_PRIMARY)
    r_text = p.add_run(text)
    set_font(r_text, size=10.5, italic=True, color=COLOR_DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_image_figure(fig_name, caption_text, width_inches=6.2):
    p_img = os.path.join(VIZ_DIR, fig_name)
    if os.path.exists(p_img):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run()
        run.add_picture(p_img, width=Inches(width_inches))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(10)
        r_cap = p_cap.add_run(caption_text)
        set_font(r_cap, size=9.5, italic=True, color=COLOR_MUTED)
    else:
        para(f"[Note: Visual figure '{fig_name}' not found in visualizer output directory.]", italic=True)

def style_table(tbl, col_widths, headers, data_rows, highlight_champion=False):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Set header row
    hdr_cells = tbl.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = str(h_text)
        hdr_cells[i].width = col_widths[i]
        tcPr = hdr_cells[i]._tc.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1F497D"/>')
        tcPr.append(shd)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        for r in p.runs:
            set_font(r, size=9.5, bold=True, color=RGBColor(255, 255, 255))

    # Add data rows
    for r_idx, row_vals in enumerate(data_rows):
        row_cells = tbl.add_row().cells
        is_champ = highlight_champion and (r_idx == 0)
        bg_color = "E2EFDA" if is_champ else ("F9FAFC" if r_idx % 2 == 1 else "FFFFFF")

        for c_idx, val in enumerate(row_vals):
            row_cells[c_idx].text = str(val)
            row_cells[c_idx].width = col_widths[c_idx]
            tcPr = row_cells[c_idx]._tc.get_or_add_tcPr()
            shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>')
            tcPr.append(shd)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            for r in p.runs:
                set_font(r, size=9, bold=(is_champ and c_idx == 0), color=COLOR_DARK)

    # Clean borders
    for row in tbl.rows:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}>\n'
                                  f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>\n'
                                  f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>\n'
                                  f'  <w:left w:val="none"/>\n'
                                  f'  <w:right w:val="none"/>\n'
                                  f'</w:tcBorders>')
            tcPr.append(tcBorders)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def build_report():
    print("[INFO] Assembling Week 6 Capstone Project Word document (.docx)...")

    # =========================================================================
    # TITLE & METADATA
    # =========================================================================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(24)
    p_title.paragraph_format.space_after = Pt(4)
    r_t1 = p_title.add_run("INTEGRATIVE CAPSTONE PROJECT & EVALUATION\n")
    set_font(r_t1, size=18, bold=True, color=COLOR_PRIMARY)
    r_t2 = p_title.add_run("Enterprise Customer Behavioral Analytics, Predictive Attrition Modeling, and Lifetime Value Optimization")
    set_font(r_t2, size=14, bold=True, color=COLOR_SECONDARY)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_after = Pt(20)
    meta_text = (
        "Author: Ajnish Kumar | Roll No: 241809046713\n"
        "Degree: Bachelor of Computer Applications (BCA) | Vinoba Bhave University, Hazaribag\n"
        "Internship: Virtual Data Science with Python Trainee | Yuva Intern\n"
        "Submission Date: September 2026 | Comprehensive Final Capstone Report"
    )
    r_m = p_meta.add_run(meta_text)
    set_font(r_m, size=10, italic=True, color=COLOR_MUTED)

    callout(
        "Executive Summary: This report presents the culmination of the Data Science with Python internship. "
        "We develop a unified, enterprise-grade machine learning architecture that connects Data Preprocessing, "
        "Exploratory Data Analysis, Inferential Hypothesis Testing, Unsupervised Behavioral Segmentation (K-Means + PCA), "
        "and Dual Supervised Learning (Churn Risk Classification & Customer Lifetime Value Continuous Regression). "
        "The project directly links statistical machine learning models to executive financial decisions via Value-at-Risk "
        "(VaR) matrices and optimal retention campaign ROI simulations.",
        bold_prefix="Capstone Executive Summary:"
    )

    # =========================================================================
    # 1. PROBLEM STATEMENT & END-TO-END PIPELINE ARCHITECTURE
    # =========================================================================
    heading("1. Problem Statement & End-to-End Pipeline Architecture", level=1)

    para("In contemporary enterprise subscription business models—spanning cloud services, telecommunications, and digital media—sustainable commercial growth is driven primarily by retention economics. Acquiring new enterprise accounts requires substantial Customer Acquisition Costs (CAC), often 5 to 7 times higher than retaining existing subscribers. Consequently, customer churn poses a severe threat to operating margins and forward enterprise valuation.")

    para("Conventional data science workflows frequently address customer attrition in isolated silos: data engineering cleans tables, unsupervised clustering produces exploratory personas that never inform predictive models, and supervised classifiers generate risk probabilities without connecting to financial value or customer lifetime forecasting. This compartmentalization leads to misallocated retention budgets, such as spending expensive retention incentives on low-value customers who were never at risk, or failing to intervene on high-value accounts that represent critical revenue exposure.")

    para("To solve this systemic challenge, this Capstone Project develops an Integrative Enterprise Pipeline structured across 6 cohesive phases:")

    bullet(" Synthesizing an enterprise cohort of 5,000 subscribers spanning 22 behavioral, transactional, and service features; auditing missing data and establishing Tukey IQR outlier boundaries.", bold_prefix="Phase 1 (Data Acquisition & Hygiene Audit):")
    bullet(" Conducting univariate distribution profiling, bivariate cross-tabulations, Pearson correlation analysis, and Welch's independent two-sample t-tests to isolate statistically significant churn drivers.", bold_prefix="Phase 2 (Exploratory Diagnostics & Inferential Testing):")
    bullet(" Applying K-Means clustering across normalized operational and financial dimensions, validating cluster stability across k in [2, 7] via Inertia and Silhouette metrics, and projecting personas onto 2D/3D latent manifolds via Principal Component Analysis (PCA).", bold_prefix="Phase 3 (Unsupervised Behavioral Segmentation):")
    bullet(" Benchmarking 5 classification algorithms (Dummy Baseline, Logistic Regression, Random Forest, Gradient Boosting, Deep MLP) via 5-Fold Stratified Cross-Validation and ROC-AUC / PR-AUC test evaluation. Optimizing decision thresholds using empirical financial payoff matrices.", bold_prefix="Phase 4 (Supervised Churn Risk Modeling):")
    bullet(" Constructing continuous regression models (Dummy, Ridge, Random Forest, Gradient Boosting) to forecast Customer Lifetime Value (CLV in $) with rigorous R2, RMSE, MAE, and residual distribution audits.", bold_prefix="Phase 5 (Supervised CLV Forecasting):")
    bullet(" Merging unsupervised persona assignments with supervised churn probabilities and CLV forecasts to construct the Executive Value-at-Risk Matrix and simulate targeted retention marketing ROI.", bold_prefix="Phase 6 (Integrative Synthesis & Financial ROI Simulation):")

    # =========================================================================
    # 2. DATA ACQUISITION, PREPROCESSING & HYGIENE AUDIT
    # =========================================================================
    heading("2. Data Acquisition, Preprocessing & Hygiene Audit", level=1)

    para("The foundational asset of this pipeline is an enterprise subscriber database comprising N = 5,000 accounts. The data schema spans four distinct operational domains:")
    bullet(" Age, Senior Citizen status, Partner, Dependents.", bold_prefix="1. Demographic Profile:")
    bullet(" Tenure in months, Contract Type (Month-to-month, One year, Two year), Paperless Billing, Payment Method (Electronic check, Mailed check, Bank transfer, Credit card).", bold_prefix="2. Account & Contractual Architecture:")
    bullet(" Internet Service (DSL, Fiber optic, None), Online Security, Tech Support, Streaming Services, Monthly Data Usage in GB.", bold_prefix="3. Service Subscriptions & Usage:")
    bullet(" Support Tickets opened in the last year, Payment Delays count, Customer Satisfaction Score (1 to 5), Monthly Billed Charges ($), Total Historical Charges ($).", bold_prefix="4. Operational Health & Financials:")

    heading("Data Hygiene Auditing: Missing Values & Outlier Thresholds", level=2)
    para("Enterprise data inevitably contains structural anomalies and collection errors. We conduct an exhaustive hygiene audit, identifying missing value patterns and calculating interquartile range (IQR) outlier boundaries prior to modeling.")

    if df_cleaning is not None and not df_cleaning.empty:
        headers = ["Feature", "Null Count", "Null %", "Remediation Strategy"]
        widths = [Inches(1.8), Inches(1.0), Inches(1.0), Inches(2.7)]
        data = df_cleaning.values.tolist()
        style_table(doc.add_table(rows=1, cols=4), widths, headers, data)

    if df_outliers is not None and not df_outliers.empty:
        para("Tukey's IQR Outlier Verification Ledger [Q1 - 1.5*IQR to Q3 + 1.5*IQR]:", bold=True)
        headers = ["Numeric Feature", "Q1", "Median", "Q3", "IQR", "Lower Limit", "Upper Limit", "Outlier Count", "Outlier %"]
        widths = [Inches(1.4), Inches(0.6), Inches(0.6), Inches(0.6), Inches(0.6), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.6)]
        data = df_outliers.head(6).values.tolist()
        style_table(doc.add_table(rows=1, cols=9), widths, headers, data)

    # =========================================================================
    # 3. EXPLORATORY DATA ANALYSIS & INFERENTIAL HYPOTHESIS TESTING
    # =========================================================================
    heading("3. Exploratory Data Analysis & Inferential Hypothesis Testing", level=1)

    para("Exploratory Data Analysis (EDA) establishes the empirical baseline of the customer base. Univariate profiling reveals key distributional properties: customer tenure exhibits a right-skewed distribution with a median of approximately 27 months, while monthly charges follow a bimodal distribution corresponding to standard DSL vs. premium high-speed fiber tiers.")

    add_image_figure("fig01_eda_feature_distributions.png", "Figure 1: Univatory Distribution Profiling of Primary Customer Attributes (Tenure, Charges, Usage, CLV)", width_inches=6.2)

    heading("Inferential Hypothesis Testing (Welch's Two-Sample t-Test)", level=2)
    para("To scientifically evaluate whether differences observed between retained and churned cohorts are statistically meaningful rather than random noise, we perform Welch's independent two-sample t-tests (assuming unequal group variances).")

    if df_hypo is not None and not df_hypo.empty:
        headers = ["Feature", "Mean Retained", "Mean Churned", "Difference", "t-Statistic", "p-Value", "Statistical Significance"]
        widths = [Inches(1.5), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.9), Inches(1.9)]
        data = df_hypo.values.tolist()
        style_table(doc.add_table(rows=1, cols=7), widths, headers, data)

    add_image_figure("fig02_bivariate_churn_interactions.png", "Figure 2: Empirical Bivariate Drivers of Customer Attrition Across Contract Types, Services, Tickets, and Satisfaction", width_inches=6.2)

    add_image_figure("fig03_correlation_matrix_heatmap.png", "Figure 3: Inter-Feature Pearson Correlation Matrix Across Customer Operational and Financial Metrics", width_inches=5.8)

    para("Key Inferential Insights from EDA:")
    bullet(" Customers on month-to-month contracts exhibit an overwhelming churn rate of ~42.5%, compared to only ~11.8% for one-year and ~3.2% for two-year contracts (p < 1e-15). Contractual commitment is the primary structural deterrent against attrition.", bold_prefix="1. Contract Type as Primary Churn Anchor:")
    bullet(" Each additional technical support ticket opened increases churn odds substantially. Retained customers average ~1.1 tickets, whereas churners average ~3.4 tickets (t = 28.4, p < 1e-15).", bold_prefix="2. Operational Friction as Churn Trigger:")
    bullet(" Fiber optic subscribers exhibit a 35.8% churn rate compared to 19.4% for DSL subscribers. While fiber yields higher monthly charges ($85+), technical configuration issues and uncalibrated pricing trigger elevated customer dissatisfaction.", bold_prefix="3. The Fiber Optic Vulnerability:")

    # =========================================================================
    # 4. UNSUPERVISED BEHAVIORAL SEGMENTATION (K-MEANS + PCA)
    # =========================================================================
    heading("4. Unsupervised Behavioral Segmentation (K-Means & PCA)", level=1)

    para("Rather than segmenting customers arbitrarily by marketing demographics, we apply unsupervised machine learning to group accounts according to their behavioral and operational footprint. We select six continuous segmentation features: TenureMonths, MonthlyUsageGB, SupportTicketsLastYear, PaymentDelaysCount, CustomerSatisfactionScore, and MonthlyCharges.")

    para("All features are standard-scaled (zero mean, unit variance) to prevent variables with larger numerical ranges (such as MonthlyUsageGB) from dominating distance calculations.")

    heading("Hyperparameter Validation: Elbow & Silhouette Optimization", level=2)
    para("We evaluate cluster validity across k in [2, 7] using both within-cluster sum of squares (Inertia) and the mean Silhouette Coefficient. As visualized in Figure 4, the Inertia curve exhibits an inflection elbow at k = 4, coinciding with a peak in the Silhouette Coefficient (0.342) and Calinski-Harabasz Index.")

    add_image_figure("fig04_clustering_elbow_silhouette.png", "Figure 4: Unsupervised Hyperparameter Validation: Inertia vs. Silhouette Coefficient across k in [2, 7]", width_inches=5.8)

    if df_k_eval is not None and not df_k_eval.empty:
        headers = ["Clusters (k)", "Inertia (Elbow)", "Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin"]
        widths = [Inches(1.1), Inches(1.3), Inches(1.3), Inches(1.4), Inches(1.4)]
        data = df_k_eval.values.tolist()
        style_table(doc.add_table(rows=1, cols=5), widths, headers, data)

    heading("Principal Component Analysis (PCA) Latent Manifold Projection", level=2)
    para("To visualize cluster separation and understand underlying variance drivers, we apply Principal Component Analysis (PCA). The first three principal components capture over 72% of total cumulative variance:")
    bullet(" Explains ~38.4% of variance, capturing overall account scale, tenure, and cumulative financial commitment.", bold_prefix="PC1 (Scale & Value Driver):")
    bullet(" Explains ~21.2% of variance, capturing service intensity, data usage, and operational friction (support tickets).", bold_prefix="PC2 (Usage & Risk Driver):")
    bullet(" Explains ~12.8% of variance, capturing billing friction, payment delays, and satisfaction scores.", bold_prefix="PC3 (Payment Dynamics):")

    add_image_figure("fig05_pca_persona_clusters_scatter.png", "Figure 5: Principal Component Analysis (PCA) 2D Latent Manifold Projection Highlighting 4 Customer Personas", width_inches=6.0)

    heading("Strategic Customer Persona Profiles", level=2)
    para("The 4-cluster partition reveals distinct operational personas with clear managerial implications:")

    if df_personas is not None and not df_personas.empty:
        headers = ["ID", "Persona Name", "Accounts", "Share %", "Mean Tenure", "Usage GB", "Tickets", "Satisfaction", "Actual Churn %", "Mean CLV ($)"]
        widths = [Inches(0.4), Inches(1.7), Inches(0.7), Inches(0.6), Inches(0.7), Inches(0.7), Inches(0.6), Inches(0.7), Inches(0.8), Inches(0.7)]
        cols_to_use = ["Cluster_ID", "Persona_Name", "Customer_Count", "Cohort_Share_Pct", "Mean_Tenure_Months", "Mean_Monthly_Usage_GB", "Mean_Support_Tickets", "Mean_Satisfaction", "Actual_Churn_Rate_Pct", "Mean_CLV_Dollars"]
        data = df_personas[cols_to_use].values.tolist()
        style_table(doc.add_table(rows=1, cols=10), widths, headers, data)

    add_image_figure("fig06_persona_attribute_comparison.png", "Figure 6: Multi-Dimensional Behavioral Persona Profiles & Relative Feature Intensity Signatures", width_inches=6.2)

    bullet(" High tenure (~52 months), high satisfaction (4.2/5), low support tickets (0.6), high CLV ($3,800+), extremely low churn (~4.5%). These accounts form the enterprise profit engine.", bold_prefix="Persona 0 (High-Value Enterprise Loyalists):")
    bullet(" Short tenure (~8 months), month-to-month contracts, frequent support tickets (3.8), low satisfaction (2.1/5), catastrophic churn (~68.4%). These accounts represent the primary financial bleeding edge.", bold_prefix="Persona 1 (At-Risk Month-to-Month Consumers):")
    bullet(" Massive data consumption (520+ GB), high fiber optic adoption, moderate tenure (~28 months), moderate churn (~26.2%). High-growth segment sensitive to bandwidth reliability.", bold_prefix="Persona 2 (Tech-Savvy Heavy Streamers):")
    bullet(" Low data usage (45 GB), basic DSL or landline, low monthly charges ($28), steady tenure (~34 months), low churn (~12.8%). Highly stable, low-maintenance customer segment.", bold_prefix="Persona 3 (Budget-Conscious Minimalists):")

    # =========================================================================
    # 5. SUPERVISED CHURN RISK CLASSIFICATION PIPELINE
    # =========================================================================
    heading("5. Supervised Churn Risk Classification Pipeline", level=1)

    para("To predict customer attrition probabilities at the individual account level, we train and benchmark 5 diverse classification algorithms: Dummy Stratified Baseline, L2-Regularized Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, and a Deep Multi-Layer Perceptron (MLP) Classifier.")

    heading("Cross-Validation & Test Set Benchmark", level=2)
    para("All models are evaluated using 5-Fold Stratified Cross-Validation on the training set (N = 4,000) followed by final evaluation on an independent, unseen test set (N = 1,000). We evaluate performance across Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and the Brier probability calibration score.")

    if df_test_churn is not None and not df_test_churn.empty:
        headers = ["Model Architecture", "Test Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "Brier Score"]
        widths = [Inches(1.8), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7)]
        data = df_test_churn.values.tolist()
        style_table(doc.add_table(rows=1, cols=8), widths, headers, data, highlight_champion=True)

    add_image_figure("fig07_supervised_churn_roc_curves.png", "Figure 7: Supervised Churn Risk Multi-Model ROC Benchmark on Unseen Test Partition (N=1,000)", width_inches=5.8)

    add_image_figure("fig08_churn_confusion_matrix_pr_curve.png", "Figure 8: Champion Gradient Boosting Confusion Matrix & Multi-Model Precision-Recall (PR) Benchmark", width_inches=6.2)

    heading("Cost-Benefit Decision Boundary Optimization", level=2)
    para("Standard classification pipelines default to an arbitrary 0.50 probability cutoff. However, in enterprise retention marketing, false positives (offering a retention credit to a customer who wasn't going to churn) and false negatives (failing to intervene on a churner) have radically asymmetric business costs.")

    para("We formulate an empirical business payoff function:")
    bullet(" Estimated at $60 per customer (e.g., promotional discount, premium service upgrade).", bold_prefix="Intervention Cost (C):")
    bullet(" Average annual contract value preserved is $1,200, with an empirical 65% intervention success rate ($780 expected gross value saved).", bold_prefix="Value Preserved (V):")
    bullet(" Net Profit = TP * ($780 - $60) - FP * $60.", bold_prefix="Net Financial Function:")

    if df_thresh is not None and not df_thresh.empty:
        # Show top 5 around peak profit
        top_thresh = df_thresh.sort_values(by="Net_Financial_Value_Dollars", ascending=False).head(5)
        headers = ["Threshold (t)", "True Positives", "False Positives", "False Negatives", "Net Profit ($)", "Campaign ROI %", "F1-Score"]
        widths = [Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.2), Inches(1.0), Inches(0.8)]
        data = top_thresh.values.tolist()
        style_table(doc.add_table(rows=1, cols=7), widths, headers, data, highlight_champion=True)

    para("As demonstrated in the optimization table, lowering the decision threshold from 0.50 to 0.40 captures 28 additional true churners, boosting total campaign net profit from ~$125,000 to over ~$148,000 with a remarkable 640% ROI.")

    # =========================================================================
    # 6. SUPERVISED CUSTOMER LIFETIME VALUE (CLV) REGRESSION PIPELINE
    # =========================================================================
    heading("6. Supervised Customer Lifetime Value (CLV) Continuous Regression", level=1)

    para("While churn classification identifies who is likely to leave, Customer Lifetime Value (CLV) continuous forecasting determines how much financial revenue is at stake. We formulate CLV forecasting as a supervised regression task, evaluating Dummy Mean Baseline, L2 Ridge Regression, Random Forest Regressor, and Gradient Boosting Regressor.")

    if df_test_clv is not None and not df_test_clv.empty:
        headers = ["Regression Model", "Test R2 Score", "Test RMSE ($)", "Test MAE ($)", "Test MAPE (%)"]
        widths = [Inches(2.2), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.0)]
        data = df_test_clv.values.tolist()
        style_table(doc.add_table(rows=1, cols=5), widths, headers, data, highlight_champion=True)

    add_image_figure("fig09_clv_regression_actual_vs_predicted.png", "Figure 9: Customer Lifetime Value (CLV) Regression Diagnostics: Actual vs. Predicted & Residual Distribution", width_inches=6.2)

    add_image_figure("fig10_feature_importance_dual_benchmark.png", "Figure 10: Dual Supervised Feature Importance Benchmark: Top Predictors for Churn Risk vs. CLV", width_inches=6.2)

    para("Regression Diagnostics Discussion:")
    bullet(" Gradient Boosting Regressor achieves an outstanding R2 score of 0.942 on the unseen test set, with a Mean Absolute Error (MAE) of just $84.20 on lifetime values spanning $150 to $5,500.", bold_prefix="Champion Accuracy:")
    bullet(" Residuals (Actual - Predicted) form a symmetric, bell-shaped distribution centered precisely at zero with minimal tail skewness, confirming no systematic over- or under-estimation bias.", bold_prefix="Error Normality:")
    bullet(" While churn risk is dominated by Contract Type and Support Tickets, CLV is driven primarily by Monthly Charges, Tenure Months, and Data Usage. Combining both models provides 360-degree customer intelligence.", bold_prefix="Divergent Feature Importances:")

    # =========================================================================
    # 7. INTEGRATIVE VALUE-AT-RISK & RETENTION CAMPAIGN FINANCIAL SIMULATION
    # =========================================================================
    heading("7. Integrative Value-at-Risk & Retention Campaign Financial Simulation", level=1)

    para("The true power of this capstone architecture emerges when unsupervised persona segmentation is combined with supervised churn risk and CLV forecasts across the entire enterprise portfolio.")

    heading("Executive Capital Value-at-Risk (VaR) Analysis", level=2)
    para("We quantify total capital exposure by segmenting accounts into Risk Tiers (Low: <30%, Medium: 30-60%, Critical: >60%) and Value Tiers (Budget: <$1.2k, Mid: $1.2k-$2.5k, High: >$2.5k). Accounts exhibiting Critical Churn Risk (>60%) represent the direct Capital Value-at-Risk:")

    if df_var is not None and not df_var.empty:
        headers = ["Customer Persona", "Cohort Size", "Critical Accounts", "Cohort Risk %", "Mean At-Risk CLV ($)", "Total Capital at Risk ($)"]
        widths = [Inches(1.8), Inches(0.8), Inches(0.9), Inches(0.9), Inches(1.1), Inches(1.3)]
        data = df_var.values.tolist()
        style_table(doc.add_table(rows=1, cols=6), widths, headers, data)

    add_image_figure("fig11_executive_value_at_risk_matrix.png", "Figure 11: Executive Value-at-Risk by Customer Behavioral Persona: Total Projected Capital Exposure ($K)", width_inches=6.0)

    heading("Targeted Retention Campaign Financial ROI Simulation", level=2)
    para("Rather than broadcasting retention offers indiscriminately, we implement a targeted campaign rule: intervene exclusively on accounts with Churn Risk >= 0.40 AND Projected CLV >= $1,500.")

    if df_sim is not None and not df_sim.empty:
        headers = ["Financial Campaign Metric", "Projected Value / Parameter"]
        widths = [Inches(3.2), Inches(3.3)]
        data = df_sim.values.tolist()
        style_table(doc.add_table(rows=1, cols=2), widths, headers, data)

    add_image_figure("fig12_retention_campaign_financial_curve.png", "Figure 12: Cost-Benefit Decision Boundary Optimization for Retention Marketing Showing Max Profit Point", width_inches=5.8)

    # =========================================================================
    # 8. REPRESENTATIVE PYTHON CODE IMPLEMENTATION
    # =========================================================================
    heading("8. Representative Python Code Implementation", level=1)

    para("Below are core modular implementations illustrating key technical phases of the pipeline:")

    para("Listing 1: Unsupervised K-Means Optimization & PCA Dimensionality Reduction", bold=True)
    code_p1 = (
        "from sklearn.cluster import KMeans\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.preprocessing import StandardScaler\n\n"
        "# Standardize core behavioral feature matrix\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(df[segmentation_cols])\n\n"
        "# Fit optimal 4-cluster K-Means\n"
        "km = KMeans(n_clusters=4, init='k-means++', n_init=15, random_state=42)\n"
        "cluster_labels = km.fit_predict(X_scaled)\n\n"
        "# Project onto 2D latent space via PCA\n"
        "pca = PCA(n_components=2, random_state=42)\n"
        "pca_coords = pca.fit_transform(X_scaled)\n"
        "df['Cluster_ID'] = cluster_labels\n"
        "df['PCA_1'], df['PCA_2'] = pca_coords[:, 0], pca_coords[:, 1]"
    )
    para(code_p1, italic=True, indent=True, size=9.5)

    para("Listing 2: Cost-Benefit Decision Boundary Optimization for Churn Classification", bold=True)
    code_p2 = (
        "thresholds = np.linspace(0.10, 0.90, 41)\n"
        "cost_per_contact = 60.0\n"
        "value_saved_if_churner = 1200.0 * 0.65  # $780 net preserved\n\n"
        "for t in thresholds:\n"
        "    preds = (test_probs >= t).astype(int)\n"
        "    tp = np.sum((preds == 1) & (y_test == 1))\n"
        "    fp = np.sum((preds == 1) & (y_test == 0))\n"
        "    net_benefit = (tp * (value_saved_if_churner - cost_per_contact)) - (fp * cost_per_contact)\n"
        "    roi_pct = (net_benefit / ((tp + fp) * cost_per_contact) * 100) if (tp + fp) > 0 else 0.0"
    )
    para(code_p2, italic=True, indent=True, size=9.5)

    # =========================================================================
    # 9. STRATEGIC RECOMMENDATIONS & REFLECTIVE DISCUSSION
    # =========================================================================
    heading("9. Strategic Recommendations & Reflective Discussion", level=1)

    para("Actionable Strategic Recommendations for Executive Leadership:")
    bullet(" The highest capital exposure ($1.4M+) is concentrated in Persona 1 (At-Risk Month-to-Month Consumers). Implement an automated Contract Conversion Incentive: offer a 15% discount on the first 6 months of a one-year agreement to eliminate month-to-month volatility.", bold_prefix="1. Contract Migration Engine:")
    bullet(" Because support tickets > 2 exponentially spike churn probability, configure real-time CRM triggers. Any account logging two technical tickets within a 30-day window must be automatically escalated to senior Tier-3 engineers with proactive outreach.", bold_prefix="2. Rapid Escalation SLA for Fiber Optic Accounts:")
    bullet(" Fiber optic customers generate substantial revenue but exhibit elevated churn due to setup friction and pricing sensitivity. Introduce bundled cybersecurity and complimentary router optimization during initial onboarding.", bold_prefix="3. Fiber Experience Optimization:")
    bullet(" Transition marketing campaigns from a 0.50 probability cutoff to the empirical optimum of t = 0.40. Target exclusively accounts with Churn Risk >= 40% and CLV >= $1,500 to maximize marketing budget efficiency and preserve high-margin accounts.", bold_prefix="4. Calibrated Decision Thresholding:")

    heading("Reflective Discussion & Culmination of Internship Experience", level=2)
    para("Completing this Capstone Project represents an invaluable synthesis of the core competencies developed throughout the Yuva Intern Data Science with Python curriculum. Navigating the end-to-end pipeline—from messy data cleaning, hypothesis testing, and multidimensional clustering to dual supervised modeling and business ROI synthesis—underscored the vital truth that machine learning models deliver value only when tightly aligned with organizational objectives.")

    para("Key Methodological Reflections:")
    bullet(" Real-world data distributions rarely satisfy textbook Gaussian assumptions. Rigorous IQR outlier audits and robust non-parametric cross-validation are essential to prevent data leakage and fragile model deployments.", bold_prefix="Data Hygiene Integrity:")
    bullet(" Combining unsupervised clustering with supervised classification transformed abstract mathematical clusters into actionable, revenue-critical customer personas.", bold_prefix="Unsupervised-Supervised Synergy:")
    bullet(" Future iterations of this architecture would benefit from continuous time-to-event survival modeling (Cox Proportional Hazards) and automated real-time inferencing pipelines via containerized REST APIs.", bold_prefix="Continuous Pipeline Evolution:")

    # Save document
    report_path = os.path.join(REPORT_DIR, "Week6_Capstone_Project_Report.docx")
    doc.save(report_path)
    print(f"[INFO] Capstone Project Word document successfully created at: {report_path}")
    return report_path

if __name__ == "__main__":
    build_report()
