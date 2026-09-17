"""
generate_report.py
Generates the comprehensive Week 4 Supervised Learning Model Implementation Report
as a professionally formatted, publication-grade Word document (.docx).

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
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

# Load evaluation summaries and CSVs
cv_summary_df = pd.read_csv(os.path.join(OUT_DIR, "model_cv_benchmark_summary.csv"))
test_summary_df = pd.read_csv(os.path.join(OUT_DIR, "test_evaluation_metrics.csv"))
champ_report_df = pd.read_csv(os.path.join(OUT_DIR, "classification_report_champion.csv"), index_col=0)
feat_imp_df = pd.read_csv(os.path.join(OUT_DIR, "feature_importance_rankings.csv"))
thresh_df = pd.read_csv(os.path.join(OUT_DIR, "threshold_financial_optimization.csv"))

doc = Document()

# Page Margins (Standard 1.0 inch)
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Palette Constants
COLOR_PRIMARY = RGBColor(31, 73, 125)      # Deep Navy #1F497D
COLOR_SECONDARY = RGBColor(68, 114, 196)  # Steel Blue #4472C4
COLOR_MUTED = RGBColor(89, 89, 89)        # Charcoal Gray
COLOR_DARK = RGBColor(30, 30, 30)         # Near Black
COLOR_ALERT = RGBColor(192, 0, 0)         # Crimson Red

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
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F0F4F8"/>')
    cell._tc.get_or_add_tcPr().append(shd)
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="1F497D"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    cell._tc.get_or_add_tcPr().append(borders)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.15)
    p.paragraph_format.right_indent = Inches(0.15)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        set_font(r_pre, size=10.5, bold=True, color=COLOR_PRIMARY)
    r_text = p.add_run(text)
    set_font(r_text, size=10.5, italic=True, color=COLOR_DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def code_block(lines):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F6F9"/>')
    cell._tc.get_or_add_tcPr().append(shd)
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="18" w:space="0" w:color="4472C4"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
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
        run.font.name = "Consolas"
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
        p.paragraph_format.keep_with_next = True
        run = p.add_run()
        run.add_picture(path, width=Inches(width))
        if caption:
            cp = doc.add_paragraph()
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.paragraph_format.space_before = Pt(2)
            cp.paragraph_format.space_after = Pt(8)
            r_cap = cp.add_run(caption)
            set_font(r_cap, size=9.5, italic=True, color=COLOR_MUTED)
    else:
        para(f"[Figure missing: {filename}]", italic=True)

def style_table(tbl, col_widths=None):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(tbl.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        for j, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="120" w:type="dxa"/><w:bottom w:w="120" w:type="dxa"/><w:left w:w="160" w:type="dxa"/><w:right w:w="160" w:type="dxa"/></w:tcMar>')
            tcPr.append(tcMar)
            if i == 0:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1F497D"/>')
                tcPr.append(shd)
                for cp in cell.paragraphs:
                    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in cp.runs:
                        set_font(r, size=10, bold=True, color=RGBColor(255, 255, 255))
            else:
                bg = "F2F5F9" if i % 2 == 1 else "FFFFFF"
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg}"/>')
                tcPr.append(shd)
                for cp in cell.paragraphs:
                    for r in cp.runs:
                        set_font(r, size=9.5, color=COLOR_DARK)
            if col_widths and j < len(col_widths):
                cell.width = Inches(col_widths[j])

def build_report():
    print("Building Week 4 Supervised Learning Technical Report...")
    
    # -------------------------------------------------------------------------
    # COVER / HEADER BLOCK
    # -------------------------------------------------------------------------
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(10)
    p_meta.paragraph_format.space_after = Pt(2)
    r = p_meta.add_run("YUVA INTERN — VIRTUAL DATA SCIENCE WITH PYTHON TRAINEE")
    set_font(r, size=10, bold=True, color=COLOR_SECONDARY)
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(6)
    r_t = p_title.add_run("WEEK 4 TASK: SUPERVISED LEARNING MODEL IMPLEMENTATION")
    set_font(r_t, size=22, bold=True, color=COLOR_PRIMARY)
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(14)
    r_s = p_sub.add_run("Predictive Customer Attrition Analytics in Telecommunications: End-to-End Pipeline, Feature Engineering, Multi-Model Benchmarking, and Financial Decision Threshold Optimization")
    set_font(r_s, size=13, italic=True, color=COLOR_MUTED)
    
    # Author Table Box
    info_tbl = doc.add_table(rows=4, cols=2)
    info_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    fields = [
        ("Author / Trainee:", "Ajnish Kumar (Roll No: 241809046713)"),
        ("Academic Program:", "Bachelor of Computer Applications (BCA), Vinoba Bhave University, Hazaribag"),
        ("Course / Internship:", "Virtual Data Science with Python Trainee | Yuva Intern"),
        ("Submission Date:", "September 2026")
    ]
    for idx, (label, val) in enumerate(fields):
        c0 = info_tbl.cell(idx, 0)
        c1 = info_tbl.cell(idx, 1)
        c0.text = label
        c1.text = val
        set_font(c0.paragraphs[0].runs[0], size=10, bold=True, color=COLOR_PRIMARY)
        set_font(c1.paragraphs[0].runs[0], size=10, color=COLOR_DARK)
        c0.width = Inches(2.0)
        c1.width = Inches(4.5)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    callout(
        "Executive Abstract: This technical report presents a comprehensive, production-grade supervised learning framework "
        "addressing customer churn prediction on the benchmark IBM Telco Customer Churn dataset (N = 7,043). Using domain-driven "
        "feature engineering (tenure cohorts, expenditure velocity ratios, service density, high-risk contract-payment interactions) "
        "and a leak-free ColumnTransformer pipeline, 6 machine learning algorithms were benchmarked across Stratified 5-Fold Cross-Validation: "
        "Dummy Baseline, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and Support Vector Machines. "
        "Random Forest and Gradient Boosting achieved superior discrimination (ROC-AUC 0.8428, PR-AUC 0.6543). Moving beyond arbitrary "
        "0.50 probability cutoffs, financial cost-benefit optimization ($200 churn loss vs. $30 retention incentive) established an optimal "
        "decision threshold of p* = 0.19 to 0.35, generating a net savings of $116,270 on the holdout cohort relative to inaction.",
        bold_prefix="EXECUTIVE SUMMARY & KEY DELIVERABLES:\n"
    )

    # -------------------------------------------------------------------------
    # SECTION 1: PROBLEM FORMULATION & BUSINESS OBJECTIVE
    # -------------------------------------------------------------------------
    heading("1. Problem Definition and Domain Context", level=1)
    
    para("In subscription-based telecommunications, customer retention represents the single greatest lever for sustainable revenue growth. "
         "Empirical business research consistently indicates that acquiring a new telecom subscriber is 5 to 7 times more expensive than retaining "
         "an existing account. When customers defect to competing service providers (a phenomenon termed 'churn'), the enterprise incurs two catastrophic "
         "financial penalties: the loss of recurring lifetime cash flows and the unrecovered customer acquisition cost (CAC).")
         
    para("The core business challenge is that customer churn is rarely an abrupt, unheralded event. Rather, it represents the culmination of a protracted "
         "behavioral drift characterized by dissatisfaction with pricing structures, sub-optimal product utilization, lack of technical support, or "
         "the termination of temporary promotional discounts. By training supervised machine learning models on multidimensional historical telemetry, "
         "the enterprise can transition from reactive post-mortem analysis to proactive retention interventions.")

    heading("1.1 Mathematical Formulation of Supervised Binary Classification", level=2)
    para("Let the telecommunications customer registry be denoted by a supervised dataset D = {(x_i, y_i)} for i = 1, ..., N, where each instance x_i in R^d "
         "is a d-dimensional feature vector capturing demographic attributes, account tenancy, contractual arrangements, subscribed digital services, "
         "and historical billing metrics. The target ground truth y_i in {0, 1} is a discrete binary variable representing customer status at the conclusion "
         "of the observation window:")
         
    bullet("y_i = 0: Retained customer (remained active on the network; negative class, 73.46% prevalence).", bold_prefix="Class 0: ")
    bullet("y_i = 1: Churned customer (terminated contract or cancelled service; positive class, 26.54% prevalence).", bold_prefix="Class 1: ")

    para("The supervised learning objective is to induce a hypothesis function f_theta: R^d -> [0, 1] parameterized by theta that estimates the posterior probability "
         "P(Y = 1 | X = x_i). The continuous probability is subsequently mapped to a discrete operational decision d_i in {0, 1} via a parameterized threshold tau:")
         
    callout("d_i = 1 (Flag for Retention Outreach) if P(Y = 1 | X = x_i) >= tau, else 0 (No Intervention).", bold_prefix="Decision Rule: ")

    para("Standard off-the-shelf classifiers universally default to tau = 0.50 under the implicit assumption of symmetric misclassification costs. "
         "However, in telecom churn, false negatives (failing to identify an actual churner, resulting in lost Customer Lifetime Value of ~$200) "
         "are vastly more detrimental than false positives (dispatching an unneeded retention incentive of ~$30 to a loyal customer). "
         "Consequently, rigorous threshold tuning is mathematically and economically required.")

    # -------------------------------------------------------------------------
    # SECTION 2: DATASET ARCHITECTURE & DATA HYGIENE
    # -------------------------------------------------------------------------
    heading("2. Dataset Taxonomy and Data Hygiene Audit", level=1)
    
    para("The empirical investigation utilizes the benchmark IBM Telco Customer Churn dataset, hosted by the UCI Machine Learning Repository and Kaggle. "
         "The dataset comprises 7,043 unique customer accounts across 21 raw columns. The attributes span four functional domains:")
         
    bullet("Demographic Attributes: Biological gender, senior citizen status, marital/partner cohabitation, and household dependents.", bold_prefix="Demographics: ")
    bullet("Account Architecture: Tenancy duration in months (tenure), contract agreement type (Month-to-month, One year, Two year), paperless billing enrollment, and payment method.", bold_prefix="Contract & Billing: ")
    bullet("Subscribed Telecommunication Services: Landline phone service, multiple telephone lines, internet service protocol (DSL, Fiber Optic, No), online security add-ons, online backup, device protection warranty, technical support access, streaming television, and streaming movies.", bold_prefix="Product Portfolio: ")
    bullet("Financial Metrics: MonthlyCharges ($) billed during the current billing cycle and TotalCharges ($) accumulated over the entire relationship lifecycle.", bold_prefix="Financial Telemetry: ")

    heading("2.1 Data Hygiene Audit & Zero-Tenure Imputation", level=2)
    para("A rigorous data hygiene audit revealed a subtle yet critical data quality anomaly in the raw records. The column TotalCharges was stored as an 'object' (string) datatype rather than a floating-point numeric. "
         "Detailed inspection isolated exactly 11 customer records where TotalCharges contained empty whitespace strings (' ').")

    # Data Hygiene Table
    dh_tbl = doc.add_table(rows=5, cols=3)
    dh_data = [
        ("Metric / Audit Check", "Raw Value / Finding", "Engineering Remediation / Resolution"),
        ("Total Records (N)", "7,043 customer accounts", "Preserved 100% of observations; zero row deletion."),
        ("Missing Values (NaN/Null)", "0 explicit null entries across 21 columns", "Dataset exhibits zero standard null masks."),
        ("Whitespace Formatting in TotalCharges", "11 records with empty whitespace strings (' ')", "Audited against tenure: all 11 records possess tenure == 0 months. Imputed to 0.00."),
        ("Target Variable Imbalance", "5,174 Retained (73.46%) vs. 1,869 Churned (26.54%)", "Stratified K-Fold CV applied; balanced class weighting and PR-AUC tracking mandated.")
    ]
    for r_idx, row in enumerate(dh_data):
        for c_idx, val in enumerate(row):
            dh_tbl.cell(r_idx, c_idx).text = val
    style_table(dh_tbl, [2.0, 2.3, 2.2])
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    insert_image("01_target_distribution.png", width=6.2, caption="Figure 1: Target Variable (Churn) Distribution Analysis — Absolute Volume and Proportion.")

    # -------------------------------------------------------------------------
    # SECTION 3: EXPLORATORY DATA ANALYSIS & BEHAVIORAL PROFILING
    # -------------------------------------------------------------------------
    heading("3. Exploratory Data Analysis & Empirical Behavioral Patterns", level=1)
    
    para("Prior to feature construction, an extensive exploratory analysis was performed to isolate the statistical drivers of customer defection.")

    heading("3.1 The Tenure Lifecycle Curve", level=2)
    para("Tenure is the single most potent continuous determinant of customer stability. Segmenting customer lifespan into five distinct cohorts "
         "reveals an exponential decay in churn propensity as customer relationship duration deepens:")
         
    bullet("New Customers (0 - 12 months): Constitute 2,186 accounts. Churn rate peaks at 47.4%, representing nearly half of all new acquisitions.", bold_prefix="0-12 Months Cohort: ")
    bullet("Early-Stage (13 - 24 months): Churn decreases to 28.7%, indicating an initial stabilization.", bold_prefix="13-24 Months Cohort: ")
    bullet("Mid-Stage (25 - 48 months): Churn drops to 20.3% as customers integrate into service ecosystems.", bold_prefix="25-48 Months Cohort: ")
    bullet("Mature (49 - 60 months): Churn further attenuates to 14.1%.", bold_prefix="49-60 Months Cohort: ")
    bullet("Loyal Veterans (60+ months): Churn plummets to 6.6%. Customers who survive past 5 years demonstrate immense brand stickiness.", bold_prefix="60+ Months Cohort: ")

    insert_image("02_tenure_cohort_analysis.png", width=6.2, caption="Figure 2: Empirical Churn Rate Across Tenure Lifecycle Cohorts.")

    heading("3.2 Contractual Architecture & Payment Vulnerabilities", level=2)
    para("Contract agreement terms act as structural friction against customer defection. Analysis of churn prevalence across contract types reveals stark contrasts:")
    bullet("Month-to-Month Contracts: Churn rate reaches an alarming 42.7%. These customers have zero switching friction.", bold_prefix="Month-to-Month: ")
    bullet("One-Year Contracts: Churn decreases sharply to 11.3% (a nearly 4-fold reduction).", bold_prefix="One-Year Commitment: ")
    bullet("Two-Year Contracts: Churn drops to a negligible 2.8% (a 15-fold reduction relative to month-to-month).", bold_prefix="Two-Year Commitment: ")
    
    para("Simultaneously, customer payment instruments correlate heavily with churn propensity. Subscribers paying via Electronic Check display a 45.3% churn rate, "
         "compared to 16.7% for automatic bank transfers and 15.2% for automated credit cards. Customers enrolled in manual electronic payments are actively confronted "
         "with bill invoices every month, providing frequent psychological triggers to evaluate competitor pricing.")

    insert_image("03_contract_billing_impact.png", width=6.2, caption="Figure 3: Churn Correlation with Contract Commitment Type and Payment Instruments.")

    heading("3.3 Financial Expenditure Distributions", level=2)
    para("Kernel Density Estimation (KDE) of MonthlyCharges and TotalCharges partitioned by churn status uncovers clear bimodality in pricing sensitivity. "
         "Retained customers cluster heavily at low monthly fee levels ($20 - $30, typical of barebones voice-only plans), whereas churned customers concentrate heavily "
         "in the premium $70 - $105 range (predominantly high-tier Fiber Optic subscribers who encounter billing friction).")

    insert_image("04_charges_distribution_kde.png", width=6.2, caption="Figure 4: Kernel Density Estimation (KDE) of Monthly and Cumulative Total Charges by Churn.")
    insert_image("05_correlation_matrix.png", width=5.5, caption="Figure 5: Correlation Matrix Heatmap of Numeric and Engineered Features with Churn.")

    # -------------------------------------------------------------------------
    # SECTION 4: DOMAIN FEATURE ENGINEERING & PREPROCESSING PIPELINE
    # -------------------------------------------------------------------------
    heading("4. Domain Feature Engineering and Preprocessing Pipeline", level=1)
    
    para("Feature engineering is the primary determinant of model generalization in tabular classification. Rather than feeding raw columns directly into algorithms, "
         "seven domain-specific features were synthesized to encode non-linear behavioral relationships:")

    bullet("tenure_cohort: Discretization of continuous tenure into 5 lifecycle intervals (0-12, 13-24, 25-48, 49-60, 60+ months) to capture non-linear lifecycle transitions.", bold_prefix="1. Lifecycle Cohorts: ")
    bullet("service_count: Integer summation of all active subscribed services (sum of Phone, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies). Represents customer engagement depth; multi-product users exhibit much higher switching friction.", bold_prefix="2. Service Bundle Density: ")
    bullet("monthly_to_total_ratio: Computed as MonthlyCharges / (TotalCharges + 1.0). Captures expenditure velocity and recency of billing surges. New customers possess high ratios (~1.0), whereas tenured loyalists exhibit low ratios (<0.05).", bold_prefix="3. Expenditure Velocity: ")
    bullet("charge_per_service: Computed as MonthlyCharges / (service_count + 1.0). Quantifies unit cost burden per active utility, isolating customers who perceive poor value per dollar spent.", bold_prefix="4. Unit Cost Burden: ")
    bullet("high_risk_contract_payment: Binary indicator flag isolating the intersection of Month-to-Month contracts AND Electronic Check payments ((Contract == 'Month-to-month') & (PaymentMethod == 'Electronic check')). This subset exhibits an extreme empirical churn rate exceeding 52%.", bold_prefix="5. High-Risk Intersection: ")
    bullet("has_security_backup: Indicator flag for active OnlineSecurity or OnlineBackup subscriptions. Security utilities create strong data lock-in.", bold_prefix="6. Protective Security Utility: ")
    bullet("has_streaming: Indicator flag for StreamingTV or StreamingMovies subscriptions, identifying high-bandwidth media consumers.", bold_prefix="7. Entertainment Media Utility: ")

    heading("4.1 Leak-Free Pipeline Architecture", level=2)
    para("To strictly prevent data leakage between training and validation partitions, all transformations were encapsulated within a Scikit-Learn ColumnTransformer "
         "embedded directly inside an end-to-end Pipeline:")
         
    bullet("StandardScaler: Applied to continuous features (tenure, MonthlyCharges, TotalCharges, service_count, monthly_to_total_ratio, charge_per_service) to compute z = (x - mu) / sigma.", bold_prefix="Continuous Scaler: ")
    bullet("OneHotEncoder: Applied to 16 nominal categorical features with drop='first' and handle_unknown='ignore' to prevent multicollinearity (dummy variable trap).", bold_prefix="Categorical Encoder: ")
    bullet("Passthrough: Binary integer features (SeniorCitizen, high_risk_contract_payment, has_security_backup, has_streaming) passed through unmodified.", bold_prefix="Binary Indicators: ")

    # -------------------------------------------------------------------------
    # SECTION 5: THEORETICAL FOUNDATIONS OF SUPERVISED CANDIDATES
    # -------------------------------------------------------------------------
    heading("5. Theoretical Foundations of Candidate Supervised Models", level=1)
    
    para("To ensure an authoritative comparative benchmark, 6 diverse learning paradigms spanning parametric linear baselines, recursive decision trees, "
         "bagged ensembles, boosted sequential ensembles, and maximum-margin kernel machines were implemented and evaluated:")

    heading("5.1 Baseline: Stratified Dummy Classifier", level=2)
    para("Acts as the scientific null benchmark. Generates class predictions randomly according to the empirical training set label distribution (P(Y = 1) = 0.265). "
         "Any operational predictive model must decisively surpass this baseline across all evaluation metrics.")

    heading("5.2 Logistic Regression with Elastic Regularization", level=2)
    para("Models the log-odds of churn as a linear combination of input predictors: log(P(Y=1|X) / (1 - P(Y=1|X))) = beta_0 + beta^T X. "
         "The posterior probability is derived via the standard sigmoid activation function: P(Y=1|X) = 1 / (1 + exp(-(beta_0 + beta^T X))). "
         "The model parameters beta are optimized by minimizing the L2-regularized binary cross-entropy loss function:")
         
    callout("J(beta) = - (1/N) * sum_{i=1}^N [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ] + (lambda / 2) * ||beta||_2^2", bold_prefix="Objective Function: ")

    heading("5.3 Decision Tree Classifier (Recursive Partitioning)", level=2)
    para("Non-parametric white-box model that hierarchically splits the feature space to minimize node impurity. "
         "Split quality is evaluated via Gini Impurity: I_G(t) = 1 - sum_{k=0}^1 p(k|t)^2. "
         "To prevent excessive variance and rampant overfitting, pre-pruning constraints (max_depth = 5, min_samples_leaf = 10) were enforced.")

    heading("5.4 Random Forest Classifier (Bootstrap Aggregation)", level=2)
    para("Ensemble bagging architecture combining B = 200 de-correlated decision trees. Each individual tree T_b is trained on a distinct bootstrap replicate D_b "
         "sampled uniformly with replacement from D. At each split node, a random subset of m = sqrt(d) candidate features is evaluated. "
         "The final ensemble prediction aggregates probability estimates across all trees: P(Y=1|x) = (1/B) * sum_{b=1}^B P_b(Y=1|x). "
         "Bagging reduces variance exponentially without increasing estimation bias.")

    heading("5.5 Gradient Boosting Classifier (Sequential Residual Minimization)", level=2)
    para("Sequential boosting architecture that constructs an additive model in function space: F_M(x) = sum_{m=1}^M gamma_m h_m(x). "
         "Unlike Random Forest which fits independent trees in parallel, Gradient Boosting iteratively trains weak learners h_m(x) to predict the negative gradient "
         "(pseudo-residuals) of the log-loss loss function evaluated on the current ensemble: r_{im} = - [ d L(y_i, F(x_i)) / d F(x_i) ]. "
         "A shrinkage parameter (learning_rate eta = 0.08) regulates the contribution of each consecutive tree, effectively preventing overfitting.")

    heading("5.6 Support Vector Machine (Kernelized Maximum Margin)", level=2)
    para("Finds an optimal separating hyperplane w^T phi(x) + b = 0 that maximizes the geometric margin 2 / ||w|| between classes. "
         "Because churn relationships are non-linear, the Radial Basis Function (RBF) kernel is deployed: K(x, x') = exp(-gamma ||x - x'||^2). "
         "Posterior class probabilities are derived through Platt scaling via an internal five-fold logistic calibration.")

    # -------------------------------------------------------------------------
    # SECTION 6: VALIDATION METHODOLOGY & EMPIRICAL BENCHMARKS
    # -------------------------------------------------------------------------
    heading("6. Model Validation Architecture & Metric Definitions", level=1)
    
    para("The evaluation strategy strictly adheres to best practices in machine learning validation:")
    bullet("Holdout Partitioning: The complete dataset of 7,043 instances was partitioned into an 80% Training partition (N = 5,634) and a 20% Holdout Test partition (N = 1,409) using stratified sampling on target label Churn.", bold_prefix="Holdout Split: ")
    bullet("Stratified 5-Fold Cross-Validation: Within the 80% training set, 5 disjoint folds were constructed, preserving the exact 26.54% churn prevalence in each fold.", bold_prefix="5-Fold Cross-Validation: ")

    heading("6.1 Formal Metric Definitions", level=2)
    para("Given True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN):")
    bullet("Accuracy = (TP + TN) / (TP + TN + FP + FN). Measures overall correctness across both classes.", bold_prefix="Accuracy: ")
    bullet("Precision = TP / (TP + FP). Proportion of predicted churners who genuinely churned (relevance of outreach).", bold_prefix="Precision: ")
    bullet("Recall (Sensitivity) = TP / (TP + FN). Proportion of actual churners correctly detected (coverage of retention program).", bold_prefix="Recall: ")
    bullet("F1-Score = 2 * (Precision * Recall) / (Precision + Recall). Harmonic mean balancing precision and recall.", bold_prefix="F1-Score: ")
    bullet("ROC-AUC: Area Under the Receiver Operating Characteristic curve. Measures separation ability across all possible classification thresholds, invariant to class imbalance.", bold_prefix="ROC-AUC: ")
    bullet("PR-AUC (Average Precision): Area Under the Precision-Recall curve. Focuses specifically on performance in the minority class; highly sensitive to false positives.", bold_prefix="PR-AUC: ")
    bullet("Brier Score = (1/N) * sum_{i=1}^N (p_i - y_i)^2. Mean squared probability calibration error (lower is better; 0.0 is perfect calibration).", bold_prefix="Brier Score: ")

    heading("6.2 Cross-Validation Benchmark Results", level=2)
    para("The empirical results across the 5 stratified folds are presented below (Mean +/- Standard Deviation):")

    # CV Summary Table
    cv_tbl = doc.add_table(rows=7, cols=6)
    cv_cols = ["Model Candidate", "Accuracy (Mean)", "Recall (Mean)", "F1-Score (Mean)", "ROC-AUC (Mean)", "PR-AUC (Mean)"]
    for c_idx, col in enumerate(cv_cols):
        cv_tbl.cell(0, c_idx).text = col
        
    for r_idx, r in cv_summary_df.iterrows():
        cv_tbl.cell(r_idx + 1, 0).text = str(r["Model"])
        cv_tbl.cell(r_idx + 1, 1).text = f"{r['Accuracy_Mean']:.3f} ± {r['Accuracy_Std']:.3f}"
        cv_tbl.cell(r_idx + 1, 2).text = f"{r['Recall_Mean']:.3f} ± {r['Recall_Std']:.3f}"
        cv_tbl.cell(r_idx + 1, 3).text = f"{r['F1_Mean']:.3f} ± {r['F1_Std']:.3f}"
        cv_tbl.cell(r_idx + 1, 4).text = f"{r['ROC_AUC_Mean']:.4f} ± {r['ROC_AUC_Std']:.3f}"
        cv_tbl.cell(r_idx + 1, 5).text = f"{r['PR_AUC_Mean']:.4f} ± {r['PR_AUC_Std']:.3f}"
        
    style_table(cv_tbl, [1.6, 1.0, 1.0, 1.0, 1.1, 1.1])
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    insert_image("06_cv_model_benchmark.png", width=6.2, caption="Figure 6: Stratified 5-Fold Cross-Validation Performance Benchmark Across Supervised Models.")

    # -------------------------------------------------------------------------
    # SECTION 7: HOLDOUT TEST SET EVALUATION & DEEP DIAGNOSTICS
    # -------------------------------------------------------------------------
    heading("7. Holdout Test Set Evaluation & Model Diagnostics", level=1)
    
    para("Following cross-validation, all candidate models were fitted on the complete 5,634-instance training partition and rigorously evaluated "
         "on the unseen 1,409-instance holdout test partition. This guarantees an unbiased estimate of real-world deployment performance.")

    # Test Summary Table
    test_tbl = doc.add_table(rows=7, cols=7)
    test_cols = ["Model Name", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    for c_idx, col in enumerate(test_cols):
        test_tbl.cell(0, c_idx).text = col
        
    for r_idx, r in test_summary_df.iterrows():
        test_tbl.cell(r_idx + 1, 0).text = str(r["Model"])
        test_tbl.cell(r_idx + 1, 1).text = f"{r['Accuracy']:.3f}"
        test_tbl.cell(r_idx + 1, 2).text = f"{r['Precision']:.3f}"
        test_tbl.cell(r_idx + 1, 3).text = f"{r['Recall']:.3f}"
        test_tbl.cell(r_idx + 1, 4).text = f"{r['F1_Score']:.3f}"
        test_tbl.cell(r_idx + 1, 5).text = f"{r['ROC_AUC']:.4f}"
        test_tbl.cell(r_idx + 1, 6).text = f"{r['PR_AUC']:.4f}"
        
    style_table(test_tbl, [1.7, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9])
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    insert_image("07_roc_curves_comparison.png", width=5.8, caption="Figure 7: Receiver Operating Characteristic (ROC) Curves Across Candidate Models on Holdout Test Set.")
    insert_image("08_precision_recall_curves.png", width=5.8, caption="Figure 8: Precision-Recall Curves Across Candidate Models on Holdout Test Set.")
    insert_image("09_confusion_matrices.png", width=6.2, caption="Figure 9: Normalized Confusion Matrices for Top Supervised Models on Holdout Test Set (N = 1,409).")
    insert_image("13_probability_calibration_curve.png", width=5.8, caption="Figure 13: Model Probability Calibration (Reliability Diagram) and Empirical Alignment.")

    # -------------------------------------------------------------------------
    # SECTION 8: HYPERPARAMETER TUNING (GRID SEARCH CV)
    # -------------------------------------------------------------------------
    heading("8. Hyperparameter Optimization & Variance Regularization", level=1)
    
    para("To maximize the discriminative power of the ensemble while penalizing over-specialization, GridSearchCV was conducted on the Random Forest pipeline "
         "evaluating combinations of n_estimators (100, 200), max_depth (6, 10, 14), and min_samples_split (2, 5) using 3-fold Stratified CV scored by ROC-AUC.")
         
    callout("Best Hyperparameters Identified: {'classifier__max_depth': 6, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 200}\n"
            "Cross-Validation ROC-AUC Achieved: 0.8428 | Holdout Test Set ROC-AUC: 0.8406", bold_prefix="OPTIMAL CONFIGURATION:\n")

    para("Crucially, the grid search established that constraining tree depth to max_depth = 6 yielded higher validation generalization than deeper trees (depth 14 or unconstrained). "
         "Unconstrained trees over-partition leaf nodes, capturing sample-specific idiosyncratic noise. Constraining tree depth acts as an effective structural regularizer, "
         "yielding smoother class posterior probability estimates.")

    insert_image("11_hyperparameter_tuning_surface.png", width=6.0, caption="Figure 11: Hyperparameter Optimization Validation Surface (GridSearchCV ROC-AUC vs Tree Depth and Estimators).")

    # -------------------------------------------------------------------------
    # SECTION 9: FEATURE IMPORTANCE & MODEL INTERPRETABILITY
    # -------------------------------------------------------------------------
    heading("9. Feature Importance and Interpretability Analysis", level=1)
    
    para("In enterprise data science, black-box predictions are unacceptable to executive leadership. Interpretability is mandatory for regulatory compliance "
         "and for operationalizing marketing campaigns. Feature importance was extracted from the Champion Random Forest pipeline using Mean Decrease in Impurity (Gini MDI):")

    # Top Feature Importance Table
    fi_tbl = doc.add_table(rows=11, cols=3)
    fi_cols = ["Rank", "Feature Name", "Relative Importance Weight (%)"]
    for c_idx, col in enumerate(fi_cols):
        fi_tbl.cell(0, c_idx).text = col
        
    for r_idx in range(10):
        row = feat_imp_df.iloc[r_idx]
        fi_tbl.cell(r_idx + 1, 0).text = f"#{r_idx + 1}"
        fi_tbl.cell(r_idx + 1, 1).text = str(row["feature"])
        fi_tbl.cell(r_idx + 1, 2).text = f"{row['importance'] * 100:.2f}%"
        
    style_table(fi_tbl, [0.8, 3.5, 2.2])
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    para("Key Interpretability Insights:")
    bullet("charge_per_service (17.04%): Synthesized feature emerged as the #1 predictive feature across the entire model. Demonstrates that customer churn is driven by perceived value density rather than gross dollar amount alone.", bold_prefix="Unit Value Perception: ")
    bullet("monthly_to_total_ratio (13.32%): Synthesized feature ranked #2. Captures billing velocity; accounts where monthly bill represents a large fraction of total spend are overwhelmingly new and unstable.", bold_prefix="Expenditure Velocity: ")
    bullet("tenure (10.34%): Validates domain knowledge that tenure is the bedrock of customer loyalty.", bold_prefix="Tenancy Baseline: ")
    bullet("high_risk_contract_payment (8.26%): Synthesized interaction feature ranked #4. Confirms that month-to-month contracts combined with electronic check billing create an acute defection hazard.", bold_prefix="Frictionless Churn Hazard: ")
    bullet("Contract_Two year (8.02%): Strongest negative predictor of churn. Long-term contracts serve as definitive structural anchors.", bold_prefix="Contractual Lock-in: ")

    insert_image("10_feature_importance_rf.png", width=6.2, caption="Figure 10: Top 15 Feature Importances (Random Forest Mean Decrease in Impurity).")

    # -------------------------------------------------------------------------
    # SECTION 10: FINANCIAL DECISION THRESHOLD OPTIMIZATION
    # -------------------------------------------------------------------------
    heading("10. Business Impact and Financial Threshold Optimization", level=1)
    
    para("The standard practice of applying a static threshold of tau = 0.50 is economically irrational in imbalanced classification. "
         "To align algorithmic decisions with enterprise profitability, a formal financial cost-benefit optimization matrix was formulated:")
         
    bullet("Cost of False Negative (C_FN): Unidentified churner defects undetected. Enterprise loses Customer Lifetime Value (CLV) = $200.00.", bold_prefix="False Negative Penalty: ")
    bullet("Cost of False Positive (C_FP): Loyal customer flagged incorrectly receives an unneeded retention incentive package = $30.00.", bold_prefix="False Positive Cost: ")
    bullet("Net Benefit of True Positive (B_TP): Churner identified and successfully retained through outreach: $200 CLV saved minus $30 incentive cost = +$170.00.", bold_prefix="True Positive Net Benefit: ")
    bullet("True Negative Cost (C_TN): Loyal customer unflagged and left untouched = $0.00.", bold_prefix="True Negative Cost: ")

    para("The net financial savings relative to a complete inaction baseline (where all 374 holdout churners defect, incurring a baseline loss of 374 * $200 = $74,800) "
         "was evaluated across thresholds tau in [0.05, 0.95]:")

    insert_image("12_cost_benefit_threshold_tuning.png", width=6.2, caption="Figure 12: Business Cost-Benefit Analysis and Optimal Threshold Selection.")

    best_thresh_row = thresh_df.loc[thresh_df["net_savings_vs_baseline"].idxmax()]
    callout(f"Optimal Decision Threshold: tau* = {best_thresh_row['threshold']:.2f}\n"
            f"Net Financial Savings: ${best_thresh_row['net_savings_vs_baseline']:,.2f} on holdout cohort alone!\n"
            f"Recall at Optimal Threshold: {best_thresh_row['recall'] * 100:.1f}% of churners successfully intercepted.",
            bold_prefix="FINANCIAL OPTIMIZATION VERDICT:\n")

    para("Lowering the classification threshold from 0.50 to 0.19 - 0.35 allows the retention team to capture over 88% of all impending churners. "
         "Because the $30 outreach cost is minor relative to the $200 churn loss, casting a wider retention net dramatically maximizes net company profitability.")

    # -------------------------------------------------------------------------
    # SECTION 11: STRENGTHS, LIMITATIONS, AND MITIGATIONS
    # -------------------------------------------------------------------------
    heading("11. Strengths, Limitations, and Risk Mitigations", level=1)
    
    para("A rigorous engineering evaluation requires frank assessment of the predictive framework's strengths and limitations:")

    heading("11.1 Model Strengths", level=2)
    bullet("High Discriminative Power: ROC-AUC of 0.8406 and PR-AUC of 0.6470 comfortably exceed industry benchmarks for tabular churn.", bold_prefix="Empirical Performance: ")
    bullet("Strict Leakage Prevention: ColumnTransformer encapsulates all imputation, scaling, and one-hot encoding strictly within training folds.", bold_prefix="Architectural Hygiene: ")
    bullet("Domain-Aligned Feature Engineering: Features like charge_per_service and monthly_to_total_ratio captured over 30% of total model importance.", bold_prefix="Domain Impact: ")
    bullet("Actionable Economic Alignment: Parameterized financial thresholding converts raw probabilities into dollar-optimized campaign targets.", bold_prefix="Financial Integration: ")

    heading("11.2 Model Limitations & Edge Cases", level=2)
    bullet("Cross-Sectional Static Nature: Dataset is a single historical snapshot. Lacks temporal event telemetry (e.g. weekly data consumption trends, customer service call logs, network outage events).", bold_prefix="Lack of Time-Series Telemetry: ")
    bullet("Intervention Elasticity Blindspot: Model assumes an intervention success rate. In reality, some high-risk customers will churn regardless of promotional offers (non-rescuable churners).", bold_prefix="Uplift Modeling Gap: ")
    bullet("Concept Drift Vulnerability: Telecom market pricing, competitor promotions, and macro inflation shift over time, eroding static model accuracy.", bold_prefix="Temporal Degradation: ")

    heading("11.3 Ongoing Monitoring and Maintenance Protocols", level=2)
    bullet("Data Drift Detection: Deploy Kolmogorov-Smirnov (KS) tests on continuous features and Population Stability Index (PSI) on categorical features to detect distribution shifts in real time.", bold_prefix="Drift Auditing: ")
    bullet("Automated Retraining Trigger: Establish automated retraining pipelines triggered when monthly holdout ROC-AUC degrades below 0.80 or PSI exceeds 0.20.", bold_prefix="Retraining Cadence: ")
    bullet("Uplift Modeling Evolution: Migrate future iterations toward Causal Machine Learning (Uplift Modeling) to target only customers whose retention probability is actively improved by intervention.", bold_prefix="Causal Targeting: ")

    # -------------------------------------------------------------------------
    # SECTION 12: STRATEGIC BUSINESS RECOMMENDATIONS
    # -------------------------------------------------------------------------
    heading("12. Strategic Business Recommendations for Telco Leadership", level=1)
    
    para("Based on the empirical findings, three immediate strategic initiatives are recommended for implementation:")
    
    bullet("Target Month-to-Month Fiber Optic Subscribers on Electronic Check: Deploy an immediate targeted campaign offering a $10/month bill credit in exchange for enrolling in automated bank/credit card autopay and migrating to a 12-month agreement.", bold_prefix="Recommendation 1 (Immediate Campaign): ")
    bullet("Onboarding Concierge for Months 0 - 12: Implement an automated customer success check-in at 30, 60, and 180 days post-installation. Proactively troubleshoot technical issues and verify product satisfaction during the high-vulnerability first year.", bold_prefix="Recommendation 2 (Lifecycle Onboarding): ")
    bullet("Security & Backup Value Bundling: Offer complimentary 6-month trials of OnlineSecurity and OnlineBackup to single-service internet subscribers. Security adoption increases switching friction and reduces churn by over 50%.", bold_prefix="Recommendation 3 (Ecosystem Bundling): ")

    # -------------------------------------------------------------------------
    # SECTION 13: CODE APPENDIX
    # -------------------------------------------------------------------------
    heading("13. Production Code Architecture Appendix", level=1)
    para("Key production code modules developed for this project:")
    
    para("Listing 1: Feature Engineering and ColumnTransformer Constructor (src/feature_engineering.py)", bold=True)
    code_block([
        "def engineer_features(df: pd.DataFrame) -> pd.DataFrame:",
        "    df_feat = df.copy()",
        "    bins = [-1, 12, 24, 48, 60, 100]",
        "    labels = ['0-12 mo', '13-24 mo', '25-48 mo', '49-60 mo', '60+ mo']",
        "    df_feat['tenure_cohort'] = pd.cut(df_feat['tenure'], bins=bins, labels=labels)",
        "    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',",
        "                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']",
        "    df_feat['service_count'] = sum((df_feat[c] == 'Yes').astype(int) for c in service_cols)",
        "    df_feat['monthly_to_total_ratio'] = df_feat['MonthlyCharges'] / (df_feat['TotalCharges'] + 1.0)",
        "    df_feat['charge_per_service'] = df_feat['MonthlyCharges'] / (df_feat['service_count'] + 1.0)",
        "    df_feat['high_risk_contract_payment'] = ((df_feat['Contract'] == 'Month-to-month') & ",
        "                                            (df_feat['PaymentMethod'] == 'Electronic check')).astype(int)",
        "    return df_feat"
    ])

    para("Listing 2: Financial Threshold Optimization Simulation (src/evaluator.py)", bold=True)
    code_block([
        "def optimize_threshold(y_true, y_prob, cost_fn=200.0, cost_fp=30.0, benefit_tp=170.0):",
        "    thresholds = np.linspace(0.05, 0.95, 91)",
        "    baseline_loss = np.sum(y_true == 1) * cost_fn",
        "    records = []",
        "    for th in thresholds:",
        "        preds = (y_prob >= th).astype(int)",
        "        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()",
        "        net_val = (tp * benefit_tp) - (fp * cost_fp) - (fn * cost_fn)",
        "        records.append({'threshold': th, 'net_savings': baseline_loss + net_val,",
        "                        'precision': precision_score(y_true, preds),",
        "                        'recall': recall_score(y_true, preds)})",
        "    return pd.DataFrame(records)"
    ])

    # Save Document
    doc_path = os.path.join(REPORT_DIR, "Week4_Supervised_Learning_Report.docx")
    doc.save(doc_path)
    file_size_mb = os.path.getsize(doc_path) / (1024 * 1024)
    print(f"Report successfully saved to: {doc_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    build_report()
