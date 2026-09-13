"""
generate_report.py
Generates the Week 2 EDA Report as a Word (.docx) file.
Author: Ajnish Kumar | August 2026
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR    = os.path.join(BASE_DIR, "visualizations")
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.15)
    section.right_margin  = Inches(1.15)

# ── Helper functions ──────────────────────────────────────────────────────────
def set_font(run, name="Calibri", size=11, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold      = bold
    run.italic    = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def heading(text, level=1, color=None):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.name = "Calibri"
        if color:
            run.font.color.rgb = RGBColor(*color)
    return p

def para(text="", bold=False, italic=False, indent=False, size=11):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Inches(0.3)
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic)
    return p

def code_block(lines):
    """Add a monospace code block."""
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent  = Inches(0.4)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(0)
        p.style = doc.styles["Normal"]
        run = p.add_run(line)
        run.font.name  = "Courier New"
        run.font.size  = Pt(9.5)
        run.font.color.rgb = RGBColor(30, 30, 30)

def insert_image(path, width=6.2, caption=None):
    if os.path.exists(path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(path, width=Inches(width))
        if caption:
            cp = doc.add_paragraph(caption)
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.paragraph_format.space_before = Pt(2)
            cp.paragraph_format.space_after  = Pt(10)
            for run in cp.runs:
                run.font.name   = "Calibri"
                run.font.size   = Pt(10)
                run.italic      = True
                run.font.color.rgb = RGBColor(80, 80, 80)
    else:
        para(f"[Image not found: {path}]", italic=True)

def bullet(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.4 + level * 0.2)
    run = p.add_run(text)
    set_font(run, size=11)
    return p

def table_2col(rows, header=True):
    tbl = doc.add_table(rows=len(rows), cols=2)
    tbl.style = "Table Grid"
    for i, row_data in enumerate(rows):
        cells = tbl.rows[i].cells
        cells[0].text = row_data[0]
        cells[1].text = row_data[1]
        for c in cells:
            for run in c.paragraphs[0].runs:
                run.font.name = "Calibri"
                run.font.size = Pt(10.5)
            if header and i == 0:
                for run in c.paragraphs[0].runs:
                    run.bold = True
    return tbl

# =============================================================================
# COVER PAGE
# =============================================================================
doc.add_paragraph()
doc.add_paragraph()

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_p.add_run("Week 2 Task Report")
run.font.name  = "Calibri"
run.font.size  = Pt(26)
run.bold       = True
run.font.color.rgb = RGBColor(31, 73, 125)

subtitle_p = doc.add_paragraph()
subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle_p.add_run("Exploratory Data Analysis and Visualization")
run.font.name  = "Calibri"
run.font.size  = Pt(16)
run.font.color.rgb = RGBColor(68, 114, 196)

doc.add_paragraph()
doc.add_paragraph()

meta = [
    ("Name",        "Ajnish Kumar"),
    ("Roll No",     "241809046713"),
    ("Course",      "Bachelor of Computer Applications (BCA)"),
    ("University",  "Vinoba Bhave University, Hazaribag"),
    ("Internship",  "Virtual Data Science with Python Trainee (Yuva Intern)"),
    ("Date",        "August 2026"),
]
for label, value in meta:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run(f"{label} : ")
    set_font(r1, size=12, bold=True, color=(31, 73, 125))
    r2 = p.add_run(value)
    set_font(r2, size=12)

doc.add_page_break()

# =============================================================================
# 1. INTRODUCTION
# =============================================================================
heading("1. Introduction", level=1)
para(
    "In Week 2, the focus shifted from raw data cleaning to something I personally find more exciting - "
    "actually looking at the data and trying to understand what story it is telling. Exploratory Data Analysis "
    "(EDA) is basically the process of digging into a dataset using statistics and visualizations to find "
    "patterns, relationships, and anything unusual before we move on to modelling."
)
para(
    "For this week, I chose the Titanic dataset. I know it is one of the most well-known datasets in data "
    "science, but that is precisely why I picked it - the human story behind the numbers is compelling, "
    "and it has the right mix of numeric and categorical features to demonstrate a wide variety of EDA techniques."
)
para(
    "The main goal this week was not just to produce graphs, but to actually think about what each graph means "
    "and why the patterns exist. I wanted to connect the visualizations to the real-world historical context "
    "of the Titanic disaster."
)

heading("1.1  What I Was Trying to Do", level=2)
for b in [
    "Understand the structure and distribution of the Titanic dataset",
    "Identify which features have the most impact on survival",
    "Create at least 10 well-annotated visualizations covering distributions, comparisons, correlations, and trends",
    "Interpret each visualization in the context of what actually happened on the Titanic",
    "Apply data transformations and feature engineering to enrich the analysis",
    "Perform statistical tests to validate whether observed patterns are significant or just random noise",
]:
    bullet(b)

heading("1.2  Libraries Used", level=2)
code_block([
    "import pandas as pd          # data loading, cleaning, aggregation",
    "import numpy as np           # numerical operations",
    "import matplotlib.pyplot as plt  # core plotting",
    "import seaborn as sns        # statistical visualizations",
    "from scipy import stats      # chi-square statistical tests",
])
doc.add_paragraph()

# =============================================================================
# 2. ABOUT THE DATASET
# =============================================================================
heading("2. About the Dataset", level=1)
para(
    "The Titanic dataset is one of the most famous datasets in machine learning. It contains passenger "
    "information from the RMS Titanic, which sank on April 15, 1912, after colliding with an iceberg. "
    "Of the 2,224 passengers and crew on board, only about 710 survived. The dataset I used is the "
    "version available through seaborn's built-in dataset loader (sns.load_dataset('titanic')), which "
    "is originally from Kaggle's Titanic: Machine Learning from Disaster competition."
)
para(
    "The dataset has 891 rows (passengers) and 15 columns. Each row represents one passenger."
)

heading("2.1  Column Overview", level=2)
tbl_rows = [
    ("Column", "Description"),
    ("survived",    "Target variable: 1 = survived, 0 = did not survive"),
    ("pclass",      "Passenger class: 1 = First, 2 = Second, 3 = Third"),
    ("sex",         "Gender of the passenger (male / female)"),
    ("age",         "Age in years (177 missing values)"),
    ("sibsp",       "Number of siblings / spouses on board"),
    ("parch",       "Number of parents / children on board"),
    ("fare",        "Ticket fare paid in British pounds"),
    ("embarked",    "Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton"),
    ("deck",        "Deck letter of cabin (77% missing - dropped)"),
    ("who",         "Categorization: man / woman / child"),
    ("alone",       "Boolean - whether passenger was travelling alone"),
]
table_2col(tbl_rows, header=True)
doc.add_paragraph()

para(
    "After the initial review I decided to drop 'deck' (77% missing), 'embark_town' (duplicates "
    "'embarked'), 'alive' (duplicates 'survived'), and 'class' (duplicates 'pclass'). This left 11 columns "
    "to work with."
)
doc.add_page_break()

# =============================================================================
# 3. LOADING THE DATASET
# =============================================================================
heading("3. Loading the Dataset", level=1)
para(
    "I loaded the dataset directly through seaborn's built-in dataset loader. This is very convenient "
    "because it downloads the data automatically and returns it as a pandas DataFrame - no separate CSV "
    "download needed."
)
code_block([
    "import seaborn as sns",
    "import pandas as pd",
    "",
    "# Load the Titanic dataset from seaborn's built-in collection",
    "df_raw = sns.load_dataset('titanic')",
    "",
    "print(f'Shape: {df_raw.shape}')  # Output: Shape: (891, 15)",
    "print(df_raw.head())",
    "print(df_raw.dtypes)",
])
para(
    "The first thing I always do after loading is check the shape, look at a few rows, and check the "
    "data types. From the dtypes output I could see that most columns are either int64, float64, or object "
    "(string). The 'who', 'class', 'alive', 'embark_town', and 'alone' columns were all object or category "
    "types that seemed redundant with other columns."
)
doc.add_page_break()

# =============================================================================
# 4. INITIAL DATA EXPLORATION
# =============================================================================
heading("4. Initial Data Exploration", level=1)

heading("4.1  Basic Statistics", level=2)
para(
    "The describe() method gives a quick statistical summary of all numeric columns. Here are the key "
    "observations I made from the output:"
)
code_block([
    "print(df_raw.describe().round(2))",
    "",
    "# Key observations from the output:",
    "# survived : mean = 0.38  -->  only 38.4% of passengers survived",
    "# pclass   : mean = 2.31  -->  majority were in 3rd class",
    "# age      : mean = 29.7, min = 0.42, max = 80  -->  wide age range",
    "# sibsp    : max = 8  -->  some passengers had very large families",
    "# fare     : mean = 32.20, max = 512.33  -->  huge variation in ticket price",
    "# fare     : 25% paid under 7.91, some paid over 500 - massive inequality",
])

heading("4.2  Missing Values", level=2)
para(
    "Checking for missing values is crucial. I did this for every column:"
)
code_block([
    "missing = df_raw.isnull().sum()",
    "miss_pct = (missing / len(df_raw) * 100).round(2)",
    "print(pd.DataFrame({'Count': missing, 'Pct%': miss_pct})[missing > 0])",
    "",
    "# Output:",
    "#          Count   Pct%",
    "# age        177  19.87  <-- significant, needs careful handling",
    "# embarked     2   0.22  <-- just 2 missing, easy to fill",
    "# deck       688  77.22  <-- too many missing, column dropped",
])
para(
    "The 'deck' column with 77% missing values is not useful - filling or imputing that many missing values "
    "would be guesswork. I dropped it completely. The 'age' column with about 20% missing is important and "
    "I handled it by filling with the median age grouped by passenger class and sex, which gives a more "
    "accurate estimate than just using the global median. The 2 missing 'embarked' values were filled with "
    "the mode (S = Southampton, which is the most common port)."
)

insert_image(
    os.path.join(VIZ_DIR, "fig01_missing_heatmap.png"),
    width=6.0,
    caption="Fig 1 - Missing Values Heatmap. Yellow cells indicate missing data. Deck has the most, age has moderate missing values."
)
doc.add_page_break()

# =============================================================================
# 5. DATA CLEANING AND TRANSFORMATION
# =============================================================================
heading("5. Data Cleaning and Transformation", level=1)
para(
    "After the initial exploration, I performed the following cleaning and transformation steps before "
    "creating visualizations. I want to document all of these clearly because they affect every analysis "
    "that follows."
)

heading("5.1  Dropping Redundant Columns", level=2)
code_block([
    "df = df_raw.copy()",
    "",
    "# Drop columns with too many missing values or that duplicate other columns",
    "df.drop(columns=['deck', 'embark_town', 'alive', 'class'], inplace=True)",
    "print(f'Shape after dropping: {df.shape}')  # (891, 11)",
])
para(
    "I always work on a copy (df_raw.copy()) to preserve the original data. Dropping the 4 redundant "
    "columns reduced the dataset from 15 to 11 columns, making it cleaner to work with."
)

heading("5.2  Filling Missing Age Values", level=2)
code_block([
    "# Fill missing age values with median grouped by passenger class and sex",
    "# This is more accurate than using the global median",
    "# e.g., the median age for 1st-class females is different from 3rd-class males",
    "",
    "df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(",
    "    lambda x: x.fillna(x.median())",
    ")",
    "",
    "print('Age nulls remaining:', df['age'].isnull().sum())  # Output: 0",
])
para(
    "Using group-based median is a much smarter imputation strategy. For example, the median age of a "
    "1st-class male passenger is quite different from a 3rd-class female passenger. Treating them the same "
    "by using a global median would introduce unnecessary noise."
)

heading("5.3  Feature Engineering", level=2)
para(
    "I created four new derived features that I thought would be more useful for analysis than the "
    "raw columns:"
)
code_block([
    "# 1. Family size - combining siblings/spouses and parents/children",
    "df['family_size'] = df['sibsp'] + df['parch']",
    "",
    "# 2. Is alone - was the passenger travelling alone?",
    "df['is_alone'] = (df['family_size'] == 0).astype(int)",
    "",
    "# 3. Age group - categorize ages into meaningful groups",
    "age_labels = ['Child(0-12)', 'Teen(13-18)', 'Young Adult(19-35)',",
    "              'Middle-Aged(36-60)', 'Senior(61+)']",
    "df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], labels=age_labels)",
    "",
    "# 4. Fare band - quartile-based fare grouping",
    "df['fare_band'] = pd.qcut(df['fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])",
    "",
    "print(f'Final shape: {df.shape}')  # (891, 15)",
])
para(
    "The 'family_size' column is more intuitive than keeping 'sibsp' and 'parch' separate. 'is_alone' "
    "gives a quick boolean flag. The 'age_group' and 'fare_band' categorizations allow me to do grouped "
    "bar chart comparisons which are much easier to interpret than continuous scatter plots."
)
doc.add_page_break()

# =============================================================================
# 6. EXPLORATORY DATA ANALYSIS AND VISUALIZATIONS
# =============================================================================
heading("6. Exploratory Data Analysis and Visualizations", level=1)
para(
    "Now comes the main part. I created 12 visualizations covering survival distribution, demographic "
    "breakdowns, feature distributions, correlations, and statistical patterns. I'll explain each one "
    "and interpret what it shows."
)

# Fig 2
heading("6.1  Overall Survival Distribution", level=2)
para(
    "The first thing to establish is the overall survival rate. Of 891 passengers in this dataset, "
    "only 342 (38.4%) survived, while 549 (61.6%) did not. This is consistent with historical records."
)
insert_image(
    os.path.join(VIZ_DIR, "fig02_survival_distribution.png"),
    width=6.0,
    caption="Fig 2 - Overall Survival Distribution. Left: percentage breakdown. Right: absolute counts."
)
para(
    "Interpretation: The majority of passengers did not survive. The pie chart makes the proportion "
    "immediately clear, while the bar chart on the right shows the absolute numbers. Having both representations "
    "in the same figure helps the reader understand both the scale and the ratio at once. The 38.4% survival "
    "rate reflects the chaos and limited lifeboat capacity during the Titanic disaster."
)

# Fig 3
heading("6.2  Survival by Sex", level=2)
para(
    "One of the most well-known facts about the Titanic is 'women and children first'. Let me verify "
    "this with data."
)
code_block([
    "sex_surv = df.groupby(['sex', 'survived']).size().unstack(fill_value=0)",
    "sex_pct  = sex_surv.div(sex_surv.sum(axis=1), axis=0) * 100",
    "print(sex_pct.round(1))",
    "",
    "# Output:",
    "# survived   Did Not Survive  Survived",
    "# sex",
    "# female          25.8%        74.2%",
    "# male            81.1%        18.9%",
])
insert_image(
    os.path.join(VIZ_DIR, "fig03_survival_by_sex.png"),
    width=6.0,
    caption="Fig 3 - Survival by Sex. Left: absolute counts. Right: survival rate percentages."
)
para(
    "Interpretation: This is striking. About 74% of female passengers survived compared to only 19% "
    "of male passengers. This strongly confirms the 'women and children first' evacuation protocol that "
    "was followed during the disaster. The crew and male passengers stepped back to allow women and children "
    "to board the limited number of lifeboats first. This is one of the strongest single-feature predictors "
    "of survival in the dataset."
)

# Fig 4
heading("6.3  Survival by Passenger Class", level=2)
para(
    "Passenger class (pclass) is a proxy for socioeconomic status. 1st class passengers were the "
    "wealthiest, and 3rd class passengers were mostly immigrants seeking a new life in America."
)
insert_image(
    os.path.join(VIZ_DIR, "fig04_survival_by_class.png"),
    width=6.0,
    caption="Fig 4 - Survival by Passenger Class. Left: counts. Right: survival rate per class."
)
para(
    "Interpretation: The survival rate drops dramatically from 1st to 3rd class. About 62.9% of 1st "
    "class passengers survived, compared to only 24.2% of 3rd class passengers. This reflects the "
    "real-world inequality of the time - 1st class cabins were on upper decks closer to the lifeboats, "
    "while 3rd class passengers were housed in the lower decks and had much farther to travel. Some "
    "historical accounts also suggest that 3rd class passengers were initially locked below decks during "
    "the evacuation."
)
doc.add_page_break()

# Fig 5
heading("6.4  Age Distribution", level=2)
para(
    "Let me look at the age distribution of all passengers and then split it by survival status."
)
code_block([
    "print(f'Mean age : {df[\"age\"].mean():.1f}')",
    "print(f'Median age : {df[\"age\"].median():.1f}')",
    "print(f'Age range : {df[\"age\"].min():.0f} to {df[\"age\"].max():.0f}')",
    "",
    "# Output:",
    "# Mean age  : 29.6",
    "# Median age: 28.0",
    "# Age range : 0 to 80",
])
insert_image(
    os.path.join(VIZ_DIR, "fig05_age_distribution.png"),
    width=6.0,
    caption="Fig 5 - Age Distribution. Left: all passengers with mean and median lines. Right: split by survival status."
)
para(
    "Interpretation: The age distribution is roughly right-skewed with most passengers in the 20-35 age "
    "range. The mean (29.6) is slightly above the median (28.0), which is typical for right-skewed "
    "distributions. Looking at the right panel, both survivors and non-survivors follow a similar overall "
    "distribution, but there is a noticeable peak of survivors in the 0-10 age range (children were "
    "prioritized). There is also a concentration of non-survivors in the 20-35 age range, which are "
    "mostly young adult male passengers who did not get priority."
)

# Fig 6
heading("6.5  Fare Distribution", level=2)
para(
    "Ticket fare is an interesting feature because it reflects both the passenger class and the specific "
    "cabin or ticket type."
)
insert_image(
    os.path.join(VIZ_DIR, "fig06_fare_distribution.png"),
    width=6.0,
    caption="Fig 6 - Fare Distribution. Left: box plot by class showing median and spread. Right: violin plot by survival."
)
para(
    "Interpretation: The box plot (left) clearly shows that 1st class passengers paid significantly "
    "higher fares - the median fare for 1st class is around 60 GBP, while 3rd class passengers paid "
    "a median of about 8 GBP. The box for 1st class is much wider, indicating huge variation even "
    "within the class (some paid over 500 GBP for premium cabins). "
    "The violin plot (right) shows that survivors generally paid higher fares, which aligns with the "
    "class effect - wealthy passengers in 1st class both paid more and survived at higher rates."
)
doc.add_page_break()

# Fig 7
heading("6.6  Survival by Age Group", level=2)
para(
    "I used the engineered 'age_group' column to compare survival rates across different life stages."
)
insert_image(
    os.path.join(VIZ_DIR, "fig07_survival_by_age_group.png"),
    width=6.0,
    caption="Fig 7 - Survival Rate by Age Group. Grouped bar chart showing percentage who survived vs did not survive in each age group."
)
para(
    "Interpretation: Children (0-12) had the highest survival rate at around 59%, which strongly "
    "reflects the 'children first' evacuation priority. Interestingly, Senior passengers (61+) also "
    "had a relatively higher survival rate, possibly because many were 1st class passengers. "
    "Young Adults (19-35) had one of the lowest survival rates, likely because this group is dominated "
    "by young adult males (the largest demographic) who were explicitly not prioritized for lifeboats."
)

# Fig 8
heading("6.7  Correlation Heatmap", level=2)
para(
    "A correlation heatmap helps identify which numeric features are linearly related to each other, "
    "especially which ones correlate with survival."
)
code_block([
    "num_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'is_alone']",
    "corr = df[num_cols].corr()",
    "print(corr['survived'].drop('survived').sort_values(ascending=False).round(3))",
    "",
    "# Output (correlations with 'survived'):",
    "# fare          0.257   <-- higher fare = more likely to survive",
    "# parch         0.082   <-- slight positive (travelling with parents/children)",
    "# family_size   0.016",
    "# sibsp        -0.035",
    "# is_alone     -0.194   <-- alone passengers had lower survival rate",
    "# age          -0.070   <-- older passengers slightly less likely to survive",
    "# pclass       -0.338   <-- higher class number = lower survival (strong!)",
])
insert_image(
    os.path.join(VIZ_DIR, "fig08_correlation_heatmap.png"),
    width=5.5,
    caption="Fig 8 - Correlation Heatmap. Lower triangle showing pairwise correlations. Blue = positive, Red = negative."
)
para(
    "Interpretation: The two strongest correlations with survival are pclass (negative, -0.34) and "
    "fare (positive, +0.26). This makes sense because class and fare are related - higher class = "
    "higher fare = better survival odds. The 'is_alone' feature has a notable negative correlation "
    "(-0.19), meaning passengers travelling alone were less likely to survive. pclass and fare have a "
    "strong negative correlation with each other (-0.55), confirming they are related variables."
)
doc.add_page_break()

# Fig 9
heading("6.8  Pairplot Analysis", level=2)
para(
    "A pairplot allows us to visualize pairwise relationships between multiple features simultaneously, "
    "coloured by survival status. This is a quick way to spot which feature combinations best separate "
    "survivors from non-survivors."
)
insert_image(
    os.path.join(VIZ_DIR, "fig09_pairplot.png"),
    width=6.0,
    caption="Fig 9 - Pairplot of key features, coloured by survival status (Red = Did Not Survive, Green = Survived)."
)
para(
    "Interpretation: The pairplot reveals several interesting things. The fare vs pclass scatter shows "
    "clear separation - survivors (green) are concentrated in higher fares and lower pclass numbers. "
    "The diagonal KDE plots show that fare has quite different distributions for survivors vs non-survivors. "
    "Age does not show strong separation, consistent with the lower correlation value we saw in the heatmap."
)

# Fig 10
heading("6.9  Embarkation Port and Family Size", level=2)
para(
    "Let me look at two more factors - the port of embarkation and whether travelling with family "
    "affected survival."
)
insert_image(
    os.path.join(VIZ_DIR, "fig10_embarkation_family.png"),
    width=6.2,
    caption="Fig 10 - Left: Survival rate by embarkation port. Right: Survival rate by family size."
)
para(
    "Interpretation (Embarkation): Cherbourg (C) passengers had the highest survival rate at around 55%, "
    "compared to Southampton (S) at about 34%. This is not because the port itself matters - it reflects "
    "the composition of passengers boarding at each port. Cherbourg had a higher proportion of 1st class "
    "passengers, which explains the higher survival rate. This is a classic example of a confounded "
    "relationship in data."
)
para(
    "Interpretation (Family Size): This is one of the more interesting findings. Solo passengers (family_size=0) "
    "had a relatively low survival rate (~30%). Passengers with small families (size 1-3) had much better "
    "survival rates, peaking at around 72% for family size of 3. But passengers with very large families (7+) "
    "had almost 0% survival. For small families, it's possible that family members helped each other to "
    "lifeboats. For very large families, they may have waited too long trying to find all their family members."
)
doc.add_page_break()

# Fig 11
heading("6.10  Class vs Sex Survival Heatmap", level=2)
para(
    "This is probably my favorite visualization because it combines two of the most powerful predictors "
    "in one compact heatmap."
)
insert_image(
    os.path.join(VIZ_DIR, "fig11_class_sex_heatmap.png"),
    width=5.0,
    caption="Fig 11 - Survival Rate (%) by Passenger Class and Sex. Green = high survival, Red = low survival."
)
para(
    "Interpretation: The pattern here is remarkable. 1st class females had a 96.8% survival rate - "
    "almost everyone survived. Even 2nd class females had 92.1% survival. However, 3rd class males "
    "had only 13.5% survival - a stark contrast. This heatmap makes it crystal clear that survival on "
    "the Titanic was determined by the combination of two factors: being female drastically increased "
    "your chances, and being in a higher class multiplied them further. For males, class still mattered "
    "somewhat (1st class males at 36.9% vs 3rd class at 13.5%) but gender was the dominant factor."
)

# Fig 12
heading("6.11  Fare Band and Survival", level=2)
para(
    "Finally, let me look at whether the ticket fare band (divided into quartiles) shows a clear "
    "trend with survival."
)
insert_image(
    os.path.join(VIZ_DIR, "fig12_fare_band_survival.png"),
    width=5.5,
    caption="Fig 12 - Survival Rate by Fare Band (quartile-based). Clear upward trend from Low to Very High fare."
)
para(
    "Interpretation: There is a very clear monotonic trend here. Passengers in the 'Low' fare band "
    "had only about 19.4% survival rate, while those in the 'Very High' band had 58.2% survival. "
    "This gradual increase confirms that fare (as a proxy for wealth and class) is strongly associated "
    "with survival. Higher fare passengers got better cabin locations, faster access to lifeboats, and "
    "possibly more personal assistance from crew."
)
doc.add_page_break()

# =============================================================================
# 7. STATISTICAL VALIDATION
# =============================================================================
heading("7. Statistical Validation (Chi-Square Tests)", level=1)
para(
    "Visual patterns can sometimes be misleading or just random variation. To verify that the "
    "relationships I observed are statistically real, I ran chi-square tests of independence for "
    "all categorical features against the survival variable."
)
code_block([
    "from scipy import stats",
    "",
    "def chi2t(col):",
    "    ct = pd.crosstab(df[col], df['survived'])",
    "    chi2, p, dof, _ = stats.chi2_contingency(ct)",
    "    sig = 'SIGNIFICANT' if p < 0.05 else 'NOT significant'",
    "    print(f'{col:15s}: chi2={chi2:.2f}, p={p:.4f}  -> {sig}')",
    "",
    "for col in ['sex', 'pclass', 'embarked', 'is_alone', 'fare_band', 'age_group']:",
    "    chi2t(col)",
    "",
    "# Results:",
    "# sex            : chi2=260.72, p=0.0000  -> SIGNIFICANT",
    "# pclass         : chi2=102.89, p=0.0000  -> SIGNIFICANT",
    "# embarked       : chi2=26.49,  p=0.0000  -> SIGNIFICANT",
    "# is_alone       : chi2=30.45,  p=0.0000  -> SIGNIFICANT",
    "# fare_band      : chi2=103.46, p=0.0000  -> SIGNIFICANT",
    "# age_group      : chi2=19.81,  p=0.0005  -> SIGNIFICANT",
])
para(
    "All six chi-square tests produced p-values well below 0.05 (the significance threshold). This means "
    "all the patterns I identified visually are statistically significant - they are not due to random "
    "chance. The strongest association is with sex (chi2=260.72), followed by fare_band (chi2=103.46) "
    "and pclass (chi2=102.89). The age_group result (p=0.0005) is also significant, though weaker."
)
doc.add_page_break()

# =============================================================================
# 8. KEY FINDINGS SUMMARY
# =============================================================================
heading("8. Key Findings Summary", level=1)
para(
    "After going through all the visualizations and statistical tests, here are the most important "
    "findings from this EDA:"
)

findings = [
    "Only 38.4% of passengers survived the Titanic disaster. This is the baseline we compare everything against.",
    "Sex was the most powerful predictor of survival. Women had a 74.2% survival rate vs only 18.9% for men, directly reflecting the 'women and children first' protocol.",
    "Passenger class was the second most powerful factor. 1st class: 62.9%, 2nd class: 47.3%, 3rd class: 24.2% survival rates. Wealth and cabin location were life-or-death factors.",
    "The combination of class and sex is devastating for 3rd-class males - only 13.5% survived, the lowest of any subgroup.",
    "Children (0-12) had the highest survival rate at ~59%, confirming children were also prioritized.",
    "Fare is strongly correlated with survival (not surprisingly, since it correlates with class).",
    "Passengers travelling alone had lower survival rates than those with small families (1-3 members).",
    "The Cherbourg boarding port appears to have higher survival rates, but this is a confounded effect explained by the higher proportion of 1st class passengers boarding there.",
    "All observed patterns were confirmed as statistically significant by chi-square tests (p < 0.05).",
]
for i, finding in enumerate(findings):
    p = doc.add_paragraph(style="List Number")
    run = p.add_run(finding)
    set_font(run, size=11)
doc.add_paragraph()

# =============================================================================
# 9. CHALLENGES I FACED
# =============================================================================
heading("9. Challenges I Faced", level=1)

heading("Challenge 1 - Deciding how to handle missing 'age' values", level=3)
para(
    "Age had 177 missing values (about 20%). I first tried filling with the global median, but then "
    "I realized that is too simplistic. A 1st-class elderly male passenger is very different from a "
    "3rd-class young female. So I used a grouped median approach - fill with the median age for each "
    "pclass-sex combination. This is a much more statistically sound imputation strategy."
)

heading("Challenge 2 - Overlapping labels on the age group bar chart", level=3)
para(
    "When I first plotted the survival rate by age group, the x-axis labels were overlapping because "
    "the group names are quite long. I experimented with rotating the labels and adjusting the figure "
    "width until the labels were readable. Small formatting issues like this take more time than you "
    "expect."
)

heading("Challenge 3 - Interpreting the embarkation result correctly", level=3)
para(
    "When I first saw that Cherbourg passengers had much higher survival rates, I initially thought "
    "maybe the port location or timing of departure had something to do with it. But after digging "
    "deeper into the class breakdown by embarkation port, I realized it was entirely explained by the "
    "composition of passengers. This was a good reminder that correlation does not mean causation, "
    "and every interesting result deserves a deeper look."
)

heading("Challenge 4 - The family size U-curve", level=3)
para(
    "The family size vs survival plot was unexpected. I expected a simple linear trend, but instead "
    "I found that medium-sized families (1-3) survived at much higher rates than both solo travellers "
    "and very large families. This required me to think about the real-world mechanics of what was "
    "happening. I concluded that small families could help each other, while very large families may "
    "have been paralyzed by trying to locate all their members before boarding lifeboats."
)
doc.add_page_break()

# =============================================================================
# 10. CONCLUSION
# =============================================================================
heading("10. Conclusion", level=1)
para(
    "This week's task was a really engaging exercise in using data to reconstruct a historical event. "
    "The Titanic dataset turned out to be an excellent choice - it has enough features to demonstrate "
    "multiple EDA techniques, and every finding connects to a real human story."
)
para(
    "The key takeaway from this EDA is that survival on the Titanic was not random. It was systematically "
    "influenced by gender, socioeconomic class, age, family structure, and even which port you boarded at. "
    "The 'women and children first' protocol was largely followed, and the physical layout of the ship "
    "(with 3rd class cabins deep in the hull) compounded the disadvantage of lower-class passengers."
)
para(
    "From a technical standpoint, I practised a wide range of EDA techniques: univariate distributions, "
    "bivariate comparisons, multivariate heatmaps, pairplots, chi-square tests, and feature engineering. "
    "I also learned to always ask 'why' when a pattern appears - the embarkation port result was a good "
    "reminder that confounding variables can mislead you if you don't investigate further."
)
para(
    "Next week I plan to use this cleaned dataset to build classification models (logistic regression, "
    "decision tree) and see how well we can predict survival. The features identified in this EDA "
    "- particularly sex, pclass, fare_band, and age_group - will be the first candidates for the model."
)

doc.add_paragraph()

# =============================================================================
# REFERENCES
# =============================================================================
heading("References", level=1)
refs = [
    "1. Kaggle Titanic Competition Dataset - https://www.kaggle.com/c/titanic",
    "2. Seaborn Documentation - https://seaborn.pydata.org/",
    "3. Pandas Documentation - https://pandas.pydata.org/docs/",
    "4. Matplotlib Documentation - https://matplotlib.org/stable/",
    "5. SciPy Stats Documentation - https://docs.scipy.org/doc/scipy/reference/stats.html",
    "6. Encyclopedia Titanica - https://www.encyclopedia-titanica.org/",
]
for ref in refs:
    p = doc.add_paragraph()
    run = p.add_run(ref)
    set_font(run, size=11)

doc.add_paragraph()
end_p = doc.add_paragraph()
end_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = end_p.add_run("- End of Report -")
set_font(run, size=12, italic=True, color=(100, 100, 100))

# =============================================================================
# SAVE
# =============================================================================
report_path = os.path.join(REPORT_DIR, "Week2_Titanic_EDA_Report.docx")
doc.save(report_path)
print(f"\nReport saved to: {report_path}")
print(f"File size: {os.path.getsize(report_path) / 1024:.1f} KB")
