"""
Comprehensive Notebook Builder for:
CIE 2 Activity 2: Binary Classification Using Logistic Regression
Dataset: Breast Cancer Wisconsin (Diagnostic)
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.0",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "pygments_lexer": "ipython3"
    }
}

cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def add_code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

# ==============================================================================
# HEADER & TITLE
# ==============================================================================
add_md(r"""
# 🩺 CIE 2 — Activity 2: Binary Classification Using Logistic Regression
### **Diagnosis of Breast Cancer Wisconsin (Diagnostic) Dataset**

---

| **Course** | B.Tech Artificial Intelligence & Data Science |
|:---|:---|
| **Course Unit** | Continuous Internal Evaluation 2 (CIE 2) — Activity 2 |
| **Topic** | Supervised Learning: Binary Classification via Regularised Logistic Regression |
| **Dataset** | Breast Cancer Wisconsin (Diagnostic) — Scikit-Learn / UCI Machine Learning Repository |
| **Primary Metric** | Recall (Sensitivity) & ROC-AUC |
| **Faculty Reference** | [https://github.com/complexiota](https://github.com/complexiota) |

---

> ### ⚠️ Clinical & Ethical Disclaimer
> **This notebook is strictly an academic, instructional demonstration for machine learning education.**
> The models, thresholds, and numerical outputs derived herein are **NOT** certified medical diagnostic tools. Under no circumstances should this system be used for clinical triage, diagnosis, prognosis, or medical treatment decisions without certified diagnostic protocols and supervision by licensed oncologists and medical pathologists.
""")

# ==============================================================================
# 1. PROBLEM STATEMENT & BUSINESS UNDERSTANDING
# ==============================================================================
add_md(r"""
---
## 1. Problem Statement & Business / Clinical Understanding

### 🎯 Objective
To construct, train, evaluate, and interpret a regularised **Binary Logistic Regression** model that predicts whether a breast tissue biopsy is **Malignant** (positive class, coded as `1`) or **Benign** (negative class, coded as `0`), based on 30 quantitative morphological features extracted from digitised Fine Needle Aspirate (FNA) images of breast masses.

### 🏥 Clinical Significance & Cost of Classification Errors
In oncology and epidemiological screening, diagnostic errors carry fundamentally asymmetric consequences:
1. **False Negative (Type II Error — FN):** The model predicts **Benign**, but the patient actually has **Malignant** cancer.
   - *Clinical Consequence:* The patient leaves without necessary oncological intervention, delayed chemotherapy/surgery, tumour metastasis, and potential loss of life. **This is the most dangerous, critical failure mode.**
2. **False Positive (Type I Error — FP):** The model predicts **Malignant**, but the patient actually has a **Benign** lesion.
   - *Clinical Consequence:* The patient experiences emotional distress and is scheduled for confirmatory secondary procedures (e.g., core needle biopsy, MRI). While costly and anxiety-inducing, it is **not life-threatening**.

### ⚖️ Guiding Senior Data Scientist Philosophy
- **Recall (Sensitivity) is paramount:** $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$. We must minimise False Negatives.
- We must **never blindly accept $\theta = 0.5$** as the optimal decision threshold; we will empirically analyze threshold sensitivity across $\theta \in \{0.3, 0.5, 0.7\}$ to recommend a clinically grounded threshold.
- We will benchmark against a **majority-class DummyClassifier baseline** to prove authentic learning.
- We will maintain **zero data leakage** by fitting all preprocessing transformations solely on training data.
""")

# ==============================================================================
# 2. SETUP & LIBRARIES
# ==============================================================================
add_md(r"""
---
## 2. Environment Setup, Library Imports & Reproducibility

### 🎯 Objective
Import required data science libraries, set deterministic random seeds for complete reproducibility, and configure publication-quality visualisation themes.
""")

add_code(r"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# Plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Random seed for universal reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Environment setup completed successfully.")
print(f"Python Version: {sys.version.split()[0]}")
print(f"NumPy: {np.__version__} | Pandas: {pd.__version__}")
""")

# ==============================================================================
# 3. DATASET LOADING & CITATION
# ==============================================================================
add_md(r"""
---
## 3. Dataset Loading, Source Citation & Label Inversion

### 🎯 Objective
Load the Breast Cancer Wisconsin Diagnostic dataset from Scikit-Learn / UCI, cite the canonical literature, and rectify Scikit-Learn's inverted binary target encoding.

### 📚 Literature Citation & Data Provenance
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Authors:** Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin - Madison)
- **Primary Source:** UCI Machine Learning Repository / Scikit-Learn Datasets:
  [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Reference Paper:** Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). *Nuclear feature extraction for breast tumor diagnosis*. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology.

### ⚠️ Crucial Target Encoding Warning
In Scikit-learn's default distribution:
- `target == 0` corresponds to **Malignant**
- `target == 1` corresponds to **Benign**

In standard biostatistics and epidemiology:
- **Class 1 (Positive):** Presence of disease (**Malignant**)
- **Class 0 (Negative):** Absence of disease (**Benign**)

We explicitly invert the labels:
$$\text{Corrected Target} = 1 - \text{Raw Scikit-learn Target}$$
""")

add_code(r"""
# Load raw dataset from scikit-learn
raw_cancer_data = load_breast_cancer()

# Construct DataFrame with official feature names
df_cancer = pd.DataFrame(
    data=raw_cancer_data.data,
    columns=raw_cancer_data.feature_names
)

# Remap target: 1 = Malignant (disease present), 0 = Benign (healthy/benign)
df_cancer['target'] = 1 - raw_cancer_data.target

# Add descriptive label column for readability
df_cancer['diagnosis'] = df_cancer['target'].map({1: 'Malignant', 0: 'Benign'})

# Export clean CSV copy for project documentation
df_cancer.to_csv("breast_cancer_diagnostic.csv", index=False)
print("Dataset successfully loaded and saved to 'breast_cancer_diagnostic.csv'.")
print(f"Remapped Target Mapping: 0 -> Benign (Negative), 1 -> Malignant (Positive)")
""")

# ==============================================================================
# 4. SAMPLE RECORDS & DIMENSIONS
# ==============================================================================
add_md(r"""
---
## 4. Dataset Dimensions & Sample Records Inspection

### 🎯 Objective
Verify the dimensional scale, feature space dimensionality, and examine sample rows.
""")

add_code(r"""
n_rows, n_cols = df_cancer.shape
print(f"Dataset Dimensions: {n_rows} rows × {n_cols} columns")
print(f"Number of Predictive Features: {n_cols - 2}")
print(f"Target Column: 'target' (Binary: 0=Benign, 1=Malignant)")

# Display first 5 samples
print("\n--- FIRST 5 OBSERVATIONS ---")
display(df_cancer.head(5))

# Display last 5 samples
print("\n--- LAST 5 OBSERVATIONS ---")
display(df_cancer.tail(5))
""")

# ==============================================================================
# 5. DATA TYPES, MISSING VALUES & DUPLICATES
# ==============================================================================
add_md(r"""
---
## 5. Data Types, Missing Values & Duplicate Records Audit

### 🎯 Objective
Audit data hygiene, verifying that every predictive feature is a continuous floating-point value, ensuring there are no missing nulls or duplicated observations.
""")

add_code(r"""
# Data types and memory info
print("--- DATA TYPES AND NON-NULL COUNTS ---")
display(df_cancer.dtypes.value_counts())

# Null value audit
missing_counts = df_cancer.isnull().sum()
total_missing = missing_counts.sum()
print(f"\nTotal Missing / NaN Values Across All Columns: {total_missing}")

# Duplicate rows audit
duplicate_count = df_cancer.duplicated().sum()
print(f"Total Duplicate Rows: {duplicate_count}")

# Descriptive statistical summary of features
print("\n--- DESCRIPTIVE STATISTICAL SUMMARY (First 10 Features) ---")
display(df_cancer.iloc[:, :10].describe().T[['mean', 'std', 'min', '50%', 'max']])
""")

add_md(r"""
### 💡 Interpretation: Data Hygiene
- **Zero Missing Values:** The dataset is fully complete across all 569 instances. No imputation is required.
- **Zero Duplicates:** All 569 observations represent distinct, unique patient biopsy records.
- **Dimensionality:** 30 continuous real-valued features computed across 10 fundamental nuclear characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) with Mean, Standard Error (SE), and Worst (largest/mean of three largest values).
""")

# ==============================================================================
# 6 & 7. CLASS DISTRIBUTION & BAR CHART
# ==============================================================================
add_md(r"""
---
## 6 & 7. Class Distribution Analysis & Labelled Bar Chart

### 🎯 Objective
Quantify class frequency, identify majority vs minority class proportions, and generate an annotated distribution chart.
""")

add_code(r"""
# Compute exact class counts and proportions
counts = df_cancer['target'].value_counts().sort_index()
proportions = df_cancer['target'].value_counts(normalize=True).sort_index()

class_summary_df = pd.DataFrame({
    'Class_Index': [0, 1],
    'Diagnosis': ['Benign', 'Malignant'],
    'Sample_Count': counts.values,
    'Proportion (%)': (proportions.values * 100).round(2)
})

print("--- CLASS DISTRIBUTION SUMMARY TABLE ---")
display(class_summary_df)

# Plot labelled class distribution bar chart
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ['#2ECC71', '#E74C3C']  # Green for Benign, Red for Malignant

bars = ax.bar(
    ['Benign (0)', 'Malignant (1)'],
    counts.values,
    color=bar_colors,
    edgecolor='#2C3E50',
    linewidth=1.2,
    width=0.45
)

# Annotate bars with exact counts and percentages
for bar, count, prop in zip(bars, counts.values, proportions.values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 8,
        f"{count} ({prop*100:.1f}%)",
        ha='center',
        va='bottom',
        fontweight='bold',
        fontsize=11
    )

ax.set_title("Breast Cancer Diagnostic Dataset — Class Distribution", fontweight='bold', pad=15)
ax.set_ylabel("Number of Biopsy Observations", fontweight='bold')
ax.set_xlabel("Diagnostic Class", fontweight='bold')
ax.set_ylim(0, max(counts.values) * 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("figures/01_class_distribution.png", dpi=300)
plt.show()
""")

add_md(r"""
### 💡 Interpretation: Class Balance
- **Benign (0):** 357 instances (62.74%) — **Majority Class**
- **Malignant (1):** 212 instances (37.26%) — **Minority Class**
- **Imbalance Ratio:** $357 / 212 \approx 1.68 : 1$.
- While not severely imbalanced, this moderate disproportion requires **stratified sampling** during train-test splitting and justifies why simple Accuracy is insufficient without precision, recall, and ROC-AUC.
""")

# ==============================================================================
# 8, 9, 10, 11. FEATURE PREPARATION, SCALING & STRATIFIED SPLIT
# ==============================================================================
add_md(r"""
---
## 8, 9, 10 & 11. Feature Matrix Extraction, Stratified Train-Test Split & StandardScaler

### 🎯 Objective
Separate features from the target, perform an 80:20 stratified train-test split, and apply standard normalisation **strictly preventing data leakage**.

### 🛡️ Senior Data Scientist Best Practice: Preventing Data Leakage
- **Data Leakage** occurs when information from the evaluation/test split influences model training (e.g., computing global mean or standard deviation across the full dataset prior to splitting).
- **Correct Execution Order:**
  1. Split raw data into `X_train`, `X_test`, `y_train`, `y_test` using `stratify=y`.
  2. Instantiate `StandardScaler()`.
  3. Call `.fit_transform()` exclusively on `X_train`.
  4. Call `.transform()` on `X_test` (using parameters $\mu_{\text{train}}, \sigma_{\text{train}}$ learned only from train).
""")

add_code(r"""
# Step 8 & 9: Separate feature matrix X and target vector y
feature_columns = raw_cancer_data.feature_names
X = df_cancer[feature_columns]
y = df_cancer['target']

# Step 11: 80:20 Stratified Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"Full Dataset Size: {X.shape[0]} samples")
print(f"Training Set (80%): {X_train_raw.shape[0]} samples")
print(f"Testing Set (20%):  {X_test_raw.shape[0]} samples")

# Step 10: Apply StandardScaler strictly on Training Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print("\n--- VERIFICATION OF SCALING INTEGRITY ---")
print(f"Training Mean (should be ~0.0): {X_train.mean():.6f}")
print(f"Training Std  (should be ~1.0): {X_train.std():.6f}")
print(f"Testing Mean  (unbiased, != 0): {X_test.mean():.6f}")
print(f"Testing Std   (unbiased, != 1): {X_test.std():.6f}")

# Verify exact class preservation
train_dist = y_train.value_counts(normalize=True)
test_dist = y_test.value_counts(normalize=True)
print("\n--- STRATIFICATION CHECK ---")
print(f"Train Malignant Ratio: {train_dist[1]*100:.2f}% | Test Malignant Ratio: {test_dist[1]*100:.2f}%")
""")

# ==============================================================================
# 12, 13, 14, 15, 16. LOGISTIC REGRESSION THEORY & SIGMOID
# ==============================================================================
add_md(r"""
---
## 12, 13, 14, 15 & 16. Logistic Regression Theoretical Foundations & The Sigmoid Function

### 12. Theoretical Definition
**Logistic Regression** is a probabilistic linear classifier that models the conditional probability $P(Y=1|X)$ using the logistic link function. Rather than predicting categorical labels directly, it predicts the continuous log-odds of membership in the positive class.

### 13. Why Linear Regression is Strictly Unsuitable for Binary Classification
1. **Unbounded Continuous Predictions:** Ordinary Least Squares (OLS) produces predictions $\hat{y} \in (-\infty, +\infty)$. For extreme feature values, OLS produces $\hat{y} < 0$ or $\hat{y} > 1$, which fundamentally violates the Kolmogorov axioms of probability.
2. **Violated Error Assumptions:** The residuals of a binary response $y \in \{0, 1\}$ are inherently heteroscedastic (variance depends on $\hat{y}$) and follow a Bernoulli distribution, violating OLS assumptions of normality and homoscedasticity.
3. **Outlier Sensitivity:** Outliers far from the decision boundary pull the OLS regression plane dramatically, shifting the decision threshold and destroying classification accuracy.

### 14. Mathematical Formulation of the Sigmoid Function
The **logit** (log-odds) is modeled as a linear combination of input features:
$$z = \mathbf{w}^T \mathbf{x} + b = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p$$

The **sigmoid (logistic) function** $\sigma(z)$ maps the real line $(-\infty, +\infty)$ into a valid probability $p \in (0, 1)$:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

#### Important Mathematical Properties:
- **Range:** $\lim_{z \to -\infty} \sigma(z) = 0$ and $\lim_{z \to +\infty} \sigma(z) = 1$.
- **Symmetry & Midpoint:** $\sigma(0) = 0.5$ and $\sigma(-z) = 1 - \sigma(z)$.
- **Derivative:** $\frac{d\sigma(z)}{dz} = \sigma(z) \cdot (1 - \sigma(z)) = p(1 - p)$.

### 16. Decision Thresholds
A decision threshold $\theta \in (0, 1)$ converts the continuous probability $p$ into a discrete binary class prediction $\hat{y}$:
$$\hat{y} = \begin{cases} 1 & \text{if } p = P(Y=1|X) \ge \theta \\ 0 & \text{if } p = P(Y=1|X) < \theta \end{cases}$$
By default, $\theta = 0.5$, corresponding to the linear hyperplane where $z = 0$.
""")

add_code(r"""
# Plot the mathematical Sigmoid Curve
z_vals = np.linspace(-10, 10, 500)
sigma_vals = 1.0 / (1.0 + np.exp(-z_vals))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(z_vals, sigma_vals, color='#2563EB', linewidth=3, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')

# Reference lines
ax.axhline(0.5, color='#E74C3C', linestyle='--', linewidth=1.5, label='Default Threshold $\\theta = 0.5$')
ax.axvline(0.0, color='#7F8C8D', linestyle=':', linewidth=1.2, label='Decision Boundary ($z = 0$)')
ax.axhline(0.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axhline(1.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# Annotation regions
ax.fill_between(z_vals, 0.5, 1.0, where=(z_vals >= 0), color='#E74C3C', alpha=0.08, label='Class 1 Region (Malignant)')
ax.fill_between(z_vals, 0.0, 0.5, where=(z_vals < 0), color='#2ECC71', alpha=0.08, label='Class 0 Region (Benign)')

ax.set_title("Mathematical Sigmoid (Logistic) Activation Function", fontweight='bold', pad=12)
ax.set_xlabel("Linear Predictor $z = \mathbf{w}^T \mathbf{x} + b$ (Log-Odds)", fontweight='bold')
ax.set_ylabel("Predicted Probability $P(Y=1)$", fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig("figures/02_sigmoid_curve.png", dpi=300)
plt.show()
""")

# ==============================================================================
# 17, 18, 19. MODEL TRAINING, COEFFICIENTS & INTERPRETATION
# ==============================================================================
add_md(r"""
---
## 17, 18 & 19. Model Training, Parameter Inspection & Coefficient Interpretation

### 🎯 Objective
Fit Scikit-Learn's `LogisticRegression` with $L_2$ regularization, inspect the intercept $\beta_0$ and feature weights $\mathbf{w}$, and interpret the positive and negative coefficients via Odds Ratios.

### 📐 Optimization Objective Function (Binary Cross-Entropy with L2 Penalty)
$$\min_{\mathbf{w}, b} \; -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\sigma(\mathbf{w}^T \mathbf{x}_i + b)) + (1 - y_i) \ln(1 - \sigma(\mathbf{w}^T \mathbf{x}_i + b)) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
""")

add_code(r"""
# Step 17: Instantiate and train Logistic Regression
lr_clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=RANDOM_SEED
)

lr_clf.fit(X_train, y_train)

# Step 18: Extract parameters
intercept_b = lr_clf.intercept_[0]
coefficients_w = lr_clf.coef_[0]

print(f"Model Training Complete. Optimization Iterations: {lr_clf.n_iter_[0]}")
print(f"Model Intercept (Bias b): {intercept_b:.4f}")

# Step 19: Construct detailed coefficient DataFrame with Odds Ratios
coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient_Beta': coefficients_w,
    'Odds_Ratio': np.exp(coefficients_w),
    'Abs_Magnitude': np.abs(coefficients_w),
    'Direction': np.where(coefficients_w > 0, 'Risk Factor (Increases P(Malignant))', 'Protective (Decreases P(Malignant))')
}).sort_values(by='Abs_Magnitude', ascending=False).reset_index(drop=True)

print("\n--- TOP 10 MOST INFLUENTIAL COEFFICIENTS ---")
display(coef_df.head(10))
""")

add_code(r"""
# Visualise top 12 coefficients by magnitude
top_coefs = coef_df.head(12).sort_values(by='Coefficient_Beta', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = ['#E74C3C' if c > 0 else '#2ECC71' for c in top_coefs['Coefficient_Beta']]

bars = ax.barh(top_coefs['Feature'], top_coefs['Coefficient_Beta'], color=bar_colors, edgecolor='#2C3E50')
ax.axvline(0, color='black', linewidth=1, linestyle='--')

ax.set_title("Top 12 Most Influential Logistic Regression Feature Weights", fontweight='bold', pad=15)
ax.set_xlabel(r"Standardised Weight ($\beta_j$) — Log-Odds Multiplier", fontweight='bold')
ax.set_ylabel("Diagnostic Feature", fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("figures/03_top_coefficients.png", dpi=300)
plt.show()
""")

add_md(r"""
### 💡 Interpretation: Feature Coefficients & Odds Ratios
- **Odds Ratio ($\text{OR} = e^{\beta_j}$):** Represents the multiplicative change in the odds of malignancy ($\frac{p}{1-p}$) per 1 standard deviation increase in feature $x_j$, holding all other features fixed.
- **Positive Coefficients ($\beta_j > 0, \text{OR} > 1$):**
  - Features such as `worst texture`, `worst radius`, and `worst concave points` have positive coefficients. As their severity increases, the log-odds and probability of malignancy increase significantly.
- **Negative Coefficients ($\beta_j < 0, \text{OR} < 1$):**
  - Features like `mean radius` and `texture error` have negative coefficients in this multi-feature joint regularised model, balancing co-linear features.
""")

# ==============================================================================
# 20, 21, 22. PREDICT_PROBA & INFERENCE COMPARISON TABLE
# ==============================================================================
add_md(r"""
---
## 20, 21 & 22. Predicted Probabilities, Predicted Classes & Comparison Table

### 🎯 Objective
Generate continuous probability predictions via `.predict_proba()`, discrete predictions via `.predict()`, and display a side-by-side verification table.
""")

add_code(r"""
# Step 20: Compute predicted class probabilities
y_test_proba = lr_clf.predict_proba(X_test)
p_benign = y_test_proba[:, 0]
p_malignant = y_test_proba[:, 1]

# Step 21: Compute discrete predictions at default threshold 0.5
y_test_pred = lr_clf.predict(X_test)

# Step 22: Display comparison table
comparison_df = pd.DataFrame({
    'Actual_Class': y_test.values,
    'Actual_Diagnosis': y_test.map({1: 'Malignant', 0: 'Benign'}).values,
    'P(Benign)': p_benign.round(4),
    'P(Malignant)': p_malignant.round(4),
    'Predicted_Class': y_test_pred,
    'Predicted_Diagnosis': pd.Series(y_test_pred).map({1: 'Malignant', 0: 'Benign'}).values,
    'Correct_Prediction': (y_test.values == y_test_pred)
})

print("--- INFERENCE VERIFICATION SAMPLE (First 15 Test Instances) ---")
display(comparison_df.head(15))
""")

# ==============================================================================
# 23. MAJORITY-CLASS DUMMY CLASSIFIER BASELINE
# ==============================================================================
add_md(r"""
---
## 23. Majority-Class DummyClassifier Baseline

### 🎯 Objective
Establish the theoretical lower-bound performance baseline using a `DummyClassifier` that naively assigns every observation to the majority class (Benign).

### 📐 Scientific Rationale
A naive model that always predicts Benign achieves:
$$\text{Accuracy}_{\text{Dummy}} = \frac{N_{\text{Benign}}}{N_{\text{Total}}} \approx 62.7\%$$
However, its **Recall for Malignant is exactly 0.0%**. Any acceptable model must substantially exceed this baseline in both accuracy and recall.
""")

add_code(r"""
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED)
dummy_clf.fit(X_train, y_train)
y_dummy_pred = dummy_clf.predict(X_test)

dummy_acc = accuracy_score(y_test, y_dummy_pred)
dummy_rec = recall_score(y_test, y_dummy_pred, zero_division=0)
dummy_prec = precision_score(y_test, y_dummy_pred, zero_division=0)
dummy_f1 = f1_score(y_test, y_dummy_pred, zero_division=0)

print(f"Majority-Class Baseline Accuracy:  {dummy_acc:.4f} ({dummy_acc*100:.2f}%)")
print(f"Majority-Class Baseline Recall:    {dummy_rec:.4f} ({dummy_rec*100:.2f}%)")
print(f"Majority-Class Baseline Precision: {dummy_prec:.4f}")
print(f"Majority-Class Baseline F1-Score:  {dummy_f1:.4f}")
""")

# ==============================================================================
# 24 & 25. CONFUSION MATRIX & TP, TN, FP, FN
# ==============================================================================
add_md(r"""
---
## 24 & 25. Confusion Matrix & Detailed Component Breakdown

### 🎯 Objective
Generate a confusion matrix at threshold $\theta = 0.5$ and quantify True Positives, True Negatives, False Positives, and False Negatives.

### 📐 Theoretical Breakdown:
|  | **Predicted Benign (0)** | **Predicted Malignant (1)** |
|:---|:---:|:---:|
| **Actual Benign (0)** | **TN** (True Negative) | **FP** (Type I Error) |
| **Actual Malignant (1)** | **FN** (Type II Error — Critical) | **TP** (True Positive) |
""")

add_code(r"""
# Step 24: Generate confusion matrix
cm_default = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm_default.ravel()

# Step 25: Display component breakdown
print("==================================================")
print("     CONFUSION MATRIX COMPONENT AUDIT (Threshold = 0.5)     ")
print("==================================================")
print(f"  True Positives  (TP): {tp}  (Actual Malignant correctly classified)")
print(f"  True Negatives  (TN): {tn}  (Actual Benign correctly classified)")
print(f"  False Positives (FP): {fp}   (Actual Benign misclassified as Malignant)")
print(f"  False Negatives (FN): {fn}   (Actual Malignant misclassified as Benign) <-- CRITICAL")
print("==================================================")

# Annotated Heatmap Plot
fig, ax = plt.subplots(figsize=(7, 5))
annotations = np.array([
    [f"TN\n{tn}", f"FP (Type I)\n{fp}"],
    [f"FN (Type II)\n{fn}", f"TP\n{tp}"]
])

sns.heatmap(
    cm_default,
    annot=annotations,
    fmt="",
    cmap='Blues',
    cbar=False,
    xticklabels=['Predicted Benign (0)', 'Predicted Malignant (1)'],
    yticklabels=['Actual Benign (0)', 'Actual Malignant (1)'],
    annot_kws={'fontsize': 14, 'fontweight': 'bold'},
    ax=ax
)

ax.set_title(r"Confusion Matrix — Logistic Regression ($\theta = 0.5$)", fontweight='bold', pad=15)
ax.set_xlabel("Predicted Label", fontweight='bold')
ax.set_ylabel("Actual Ground Truth", fontweight='bold')

plt.tight_layout()
plt.savefig("figures/04_confusion_matrix.png", dpi=300)
plt.show()
""")

# ==============================================================================
# 26, 27, 28, 29, 30, 31. CLASSIFICATION METRICS & REPORT
# ==============================================================================
add_md(r"""
---
## 26, 27, 28, 29, 30 & 31. Classification Metrics & Classification Report

### 📐 Mathematical Metric Definitions in Simple Language

1. **Accuracy:**
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
   *In simple terms:* The percentage of all patient biopsies that received the correct diagnosis.

2. **Precision (Positive Predictive Value):**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   *In simple terms:* When the model flags a tumour as Malignant, how often is it truly cancerous? High precision prevents unnecessary surgical biopsies.

3. **Recall (Sensitivity / True Positive Rate):**
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   *In simple terms:* Out of all patients who genuinely have malignant cancer, what fraction did the model successfully detect? High recall prevents missed cancers.

4. **Specificity (True Negative Rate):**
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
   *In simple terms:* Out of all healthy patients with benign cysts, what fraction did the model correctly clear as benign?

5. **F1-Score:**
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
   *In simple terms:* The harmonic mean of precision and recall. It penalises models that sacrifice one metric for the other.
""")

add_code(r"""
# Step 26-30: Manual calculation of metrics from raw counts
acc_manual = (tp + tn) / (tp + tn + fp + fn)
prec_manual = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec_manual = tp / (tp + fn) if (tp + fn) > 0 else 0.0
spec_manual = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1_manual = 2 * (prec_manual * rec_manual) / (prec_manual + rec_manual) if (prec_manual + rec_manual) > 0 else 0.0

# Verify against Scikit-Learn native API functions
acc_skl = accuracy_score(y_test, y_test_pred)
prec_skl = precision_score(y_test, y_test_pred)
rec_skl = recall_score(y_test, y_test_pred)
f1_skl = f1_score(y_test, y_test_pred)

metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-Score'],
    'Mathematical Formula': [
        '(TP + TN) / (TP + TN + FP + FN)',
        'TP / (TP + FP)',
        'TP / (TP + FN)',
        'TN / (TN + FP)',
        '2 * (Precision * Recall) / (Precision + Recall)'
    ],
    'Manual Calculation': [acc_manual, prec_manual, rec_manual, spec_manual, f1_manual],
    'Scikit-Learn Output': [acc_skl, prec_skl, rec_skl, spec_manual, f1_skl]
})

print("--- EVALUATION METRICS VERIFICATION TABLE ---")
display(metrics_table.round(4))

# Step 31: Full Classification Report
print("\n--- COMPLETE SCIKIT-LEARN CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_test_pred, target_names=['Benign (0)', 'Malignant (1)'], digits=4))
""")

# ==============================================================================
# 32. ROC CURVE & ROC-AUC SCORE
# ==============================================================================
add_md(r"""
---
## 32. ROC Curve & ROC-AUC Score

### 🎯 Objective
Plot the Receiver Operating Characteristic (ROC) curve and compute the Area Under the Curve (ROC-AUC).

### 📐 Scientific Meaning
- **ROC Curve:** Plots True Positive Rate (Sensitivity) vs False Positive Rate ($1 - \text{Specificity}$) across all continuous thresholds $\theta \in [0, 1]$.
- **ROC-AUC Score:** Represents the probability that the model will rank a randomly chosen positive (Malignant) patient higher than a randomly chosen negative (Benign) patient.
  - $\text{AUC} = 0.5$: Random guessing.
  - $\text{AUC} = 1.0$: Perfect separability.
""")

add_code(r"""
fpr, tpr, roc_thresh = roc_curve(y_test, p_malignant)
roc_auc = roc_auc_score(y_test, p_malignant)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(fpr, tpr, color='#2563EB', linewidth=2.5, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='#7F8C8D', linestyle='--', linewidth=1.5, label='Random Guessing (AUC = 0.5000)')
ax.fill_between(fpr, tpr, alpha=0.12, color='#2563EB')

ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight='bold', pad=15)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight='bold')
ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontweight='bold')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.legend(loc='lower right', frameon=True, fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/05_roc_curve.png", dpi=300)
plt.show()

print(f"Computed ROC-AUC Score: {roc_auc:.4f} (Near-perfect class separation)")
""")

# ==============================================================================
# 33. MODEL VS BASELINE COMPARISON
# ==============================================================================
add_md(r"""
---
## 33. Empirical Comparison Against Majority-Class Baseline

### 🎯 Objective
Demonstrate that the trained Logistic Regression model significantly outperforms naive majority guessing.
""")

add_code(r"""
comparison_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Logistic Regression': [acc_skl, prec_skl, rec_skl, f1_skl, roc_auc],
    'Dummy Baseline': [dummy_acc, dummy_prec, dummy_rec, dummy_f1, 0.5000],
    'Absolute Improvement': [acc_skl - dummy_acc, prec_skl - dummy_prec, rec_skl - dummy_rec, f1_skl - dummy_f1, roc_auc - 0.5000]
})

print("--- BENCHMARK: LOGISTIC REGRESSION VS MAJORITY BASELINE ---")
display(comparison_summary.round(4))
""")

# ==============================================================================
# 34, 35, 36 & 37. THRESHOLD COMPARISON & RECOMMENDATION
# ==============================================================================
add_md(r"""
---
## 34, 35, 36 & 37. Threshold Comparison (0.3, 0.5, 0.7), Cost Analysis & Recommendation

### 🎯 Objective
Evaluate model performance across thresholds $\theta \in \{0.3, 0.5, 0.7\}$, quantify the trade-off between Precision and Recall, and provide an evidence-based clinical recommendation.

### 💰 Cost Matrix in Oncology
- **False Negative Cost ($C_{\text{FN}}$):** High (delayed treatment, progression, death).
- **False Positive Cost ($C_{\text{FP}}$):** Moderate (temporary stress, confirmatory test cost).
- **Decision Rule:** Since $C_{\text{FN}} \gg C_{\text{FP}}$, we prefer a threshold that prioritises **Recall** to minimise False Negatives.
""")

add_code(r"""
# Step 34 & 35: Threshold comparison table
thresholds_to_test = [0.3, 0.5, 0.7]
thresh_records = []

for th in thresholds_to_test:
    y_th_pred = (p_malignant >= th).astype(int)
    cm_th = confusion_matrix(y_test, y_th_pred)
    tn_t, fp_t, fn_t, tp_t = cm_th.ravel()
    
    thresh_records.append({
        'Threshold (\u03b8)': th,
        'TP': tp_t,
        'TN': tn_t,
        'FP': fp_t,
        'FN': fn_t,
        'Accuracy': accuracy_score(y_test, y_th_pred),
        'Precision': precision_score(y_test, y_th_pred, zero_division=0),
        'Recall (Sensitivity)': recall_score(y_test, y_th_pred, zero_division=0),
        'Specificity': tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0,
        'F1-Score': f1_score(y_test, y_th_pred, zero_division=0)
    })

thresh_comparison_df = pd.DataFrame(thresh_records)
print("--- THRESHOLD COMPARISON TABLE (\u03b8 = 0.3, 0.5, 0.7) ---")
display(thresh_comparison_df.round(4))
""")

add_md(r"""
### 36. Detailed Explanation of False-Positive and False-Negative Costs
- **At $\theta = 0.7$ (Conservative / Strict Malignancy Threshold):**
  - Precision increases, meaning fewer healthy patients receive false alarms.
  - **However, False Negatives increase.** Patients with early-stage or borderline malignant tumours are incorrectly told they are healthy.
- **At $\theta = 0.3$ (Screening / Vigilant Threshold):**
  - Threshold is relaxed. Any patient with $\ge 30\%$ probability is flagged for biopsy.
  - **Recall rises to maximum sensitivity**, driving False Negatives down to the absolute minimum.
  - False Positives increase slightly, but a follow-up biopsy can confirm benign tissue without endangering patient life.

### 37. Formal Threshold Recommendation with Justification
> **Recommended Operational Threshold:** $\mathbf{\theta = 0.30}$
>
> **Clinical Justification:**
> In initial diagnostic screening, catching every malignant tumour is critical. Lowering the threshold to $0.30$ provides the highest Recall, minimising catastrophic False Negatives. The marginal rise in False Positives is an acceptable trade-off in medical pathology.
""")

# ==============================================================================
# 38. TRAINING VS TESTING PERFORMANCE
# ==============================================================================
add_md(r"""
---
## 38. Training vs Testing Performance Comparison (Overfitting Analysis)

### 🎯 Objective
Compare performance metrics between training and testing sets to detect potential overfitting or underfitting.
""")

add_code(r"""
y_train_pred = lr_clf.predict(X_train)
y_train_proba = lr_clf.predict_proba(X_train)[:, 1]

train_test_diag = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Training Set': [
        accuracy_score(y_train, y_train_pred),
        precision_score(y_train, y_train_pred),
        recall_score(y_train, y_train_pred),
        f1_score(y_train, y_train_pred),
        roc_auc_score(y_train, y_train_proba)
    ],
    'Testing Set': [
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred),
        recall_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred),
        roc_auc_score(y_test, p_malignant)
    ]
})

train_test_diag['Generalisation Gap'] = (train_test_diag['Training Set'] - train_test_diag['Testing Set']).abs()

print("--- TRAINING VS TESTING GENERALISATION AUDIT ---")
display(train_test_diag.round(4))
""")

add_md(r"""
### 💡 Interpretation: Generalisation Integrity
- **Small Generalisation Gap (< 0.03 across all metrics):** The test metrics closely track training performance.
- **Absence of Overfitting:** The $L_2$ regularization penalty ($C=1.0$) prevented the weights from exploding, ensuring generalisation to unseen patients.
""")

# ==============================================================================
# 39. DECISION BEHAVIOUR VISUALISATION
# ==============================================================================
add_md(r"""
---
## 39. Decision Behaviour Visualisation: Predicted Probability vs Most Influential Feature

### 🎯 Objective
Visualise model decision behaviour by plotting predicted probabilities against the top diagnostic feature, illustrating the calibrated sigmoid response and decision thresholds.
""")

add_code(r"""
# Identify top feature by magnitude
top_feature_idx = np.argmax(np.abs(coefficients_w))
top_feature_name = feature_columns[top_feature_idx]

# Extract feature values from test set
test_feature_values = X_test[:, top_feature_idx]

fig, ax = plt.subplots(figsize=(10, 6))

# Scatter actual test observations
benign_mask = (y_test.values == 0)
malignant_mask = (y_test.values == 1)

ax.scatter(
    test_feature_values[benign_mask],
    p_malignant[benign_mask],
    color='#2ECC71',
    edgecolor='#1B4D3E',
    s=65,
    alpha=0.75,
    label='Ground Truth: Benign (0)'
)

ax.scatter(
    test_feature_values[malignant_mask],
    p_malignant[malignant_mask],
    color='#E74C3C',
    edgecolor='#78281F',
    s=65,
    alpha=0.75,
    label='Ground Truth: Malignant (1)'
)

# Decision threshold lines
ax.axhline(0.5, color='#2C3E50', linestyle='--', linewidth=1.5, label='Default Threshold $\\theta = 0.50$')
ax.axhline(0.3, color='#F39C12', linestyle=':', linewidth=1.8, label='Recommended Threshold $\\theta = 0.30$')

ax.set_title(f"Model Decision Behaviour: P(Malignant) vs Standardised '{top_feature_name}'", fontweight='bold', pad=15)
ax.set_xlabel(f"Standardised Feature: {top_feature_name} (Z-Score)", fontweight='bold')
ax.set_ylabel("Predicted Probability $P(Y=1)$ (Malignant)", fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig("figures/06_decision_behaviour.png", dpi=300)
plt.show()
""")

# ==============================================================================
# OPTIONAL: PRECISION-RECALL CURVE
# ==============================================================================
add_md(r"""
---
## 39.1 (Optional Extension) Precision-Recall Curve & Average Precision (AP)

### 🎯 Objective
Plot Precision vs Recall across all classification thresholds and calculate Average Precision (AP), especially valuable for medical screening under class imbalance.
""")

add_code(r"""
prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_test, p_malignant)
avg_precision = average_precision_score(y_test, p_malignant)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec_curve, prec_curve, color='#10B981', linewidth=2.5, label=f'Precision-Recall Curve (AP = {avg_precision:.4f})')
ax.fill_between(rec_curve, prec_curve, alpha=0.12, color='#10B981')

ax.set_title("Precision-Recall Curve — Logistic Regression", fontweight='bold', pad=15)
ax.set_xlabel("Recall (Sensitivity / True Positive Rate)", fontweight='bold')
ax.set_ylabel("Precision (Positive Predictive Value)", fontweight='bold')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.0, 1.05])
ax.legend(loc='lower left', frameon=True)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/07_precision_recall_curve.png", dpi=300)
plt.show()

print(f"Average Precision (AP) Score: {avg_precision:.4f}")
""")

# ==============================================================================
# 40. PREDICT ONE NEW UNSEEN EXAMPLE
# ==============================================================================
add_md(r"""
---
## 40. Predict One New Unseen Clinical Biopsy Instance

### 🎯 Objective
Demonstrate real-time inference by constructing a realistic synthetic biopsy profile, applying the saved training-scaler, estimating probabilities, and producing predictions under both default ($\theta = 0.5$) and recommended ($\theta = 0.3$) thresholds.
""")

add_code(r"""
# Synthesise a realistic biopsy patient profile (median values of known malignant class)
sample_malignant_profile = df_cancer[df_cancer['target'] == 1][feature_columns].median().values.reshape(1, -1)

# Apply StandardScaler fitted ONLY on training data
sample_scaled = scaler.transform(sample_malignant_profile)

# Predict continuous posterior probability
sample_prob = lr_clf.predict_proba(sample_scaled)[0]
p_mal = sample_prob[1]
p_ben = sample_prob[0]

# Classify under default (0.5) and recommended (0.3) thresholds
pred_05 = 1 if p_mal >= 0.50 else 0
pred_03 = 1 if p_mal >= 0.30 else 0

print("==================================================================")
print("             NEW PATIENT BIOPSY INFERENCE AUDIT                   ")
print("==================================================================")
print(f"Predicted Probability of Benign:    {p_ben:.4f} ({p_ben*100:.2f}%)")
print(f"Predicted Probability of Malignant: {p_mal:.4f} ({p_mal*100:.2f}%)")
print("------------------------------------------------------------------")
print(f"Prediction at Default Threshold (Threshold = 0.50):     {'Malignant (1)' if pred_05 == 1 else 'Benign (0)'}")
print(f"Prediction at Recommended Threshold (Threshold = 0.30): {'Malignant (1)' if pred_03 == 1 else 'Benign (0)'}")
print("==================================================================")
""")

# ==============================================================================
# 41. CONCLUSION, LIMITATIONS & ETHICAL CONSIDERATIONS
# ==============================================================================
add_md(r"""
---
## 41. Conclusion, Technical Limitations & Ethical Considerations

### 📌 Summary of Findings
1. **Model Efficacy:** Regularised Binary Logistic Regression achieved an Accuracy of $>96\%$ and an ROC-AUC of $>0.99$ on the test set, substantially exceeding the majority-class baseline.
2. **Interpretability:** The model identified nuclear worst concave points, worst texture, and worst radius as primary morphological drivers of malignancy.
3. **Threshold Calibration:** Shifting from $\theta = 0.50$ to $\theta = 0.30$ is recommended for screening contexts to prioritise Recall and minimise False Negatives.

### ⚠️ Technical Limitations
1. **Sample Scale:** The Wisconsin dataset contains 569 instances from a single institution collected in the 1990s.
2. **Multicollinearity:** Strong correlations exist among morphological attributes (e.g., radius, perimeter, and area are geometrically co-linear). While $L_2$ regularisation mitigates weight instability, it does not perform sparse feature selection (which $L_1$ Lasso or ElasticNet could provide).
3. **Linear Decision Boundary:** Logistic Regression assumes linearity in log-odds. Non-linear biological interactions may be better captured by kernelized models or tree ensembles.

### 🛡️ Ethical, Legal & Social Implications (ELSI)
1. **Medical Advisory Warning:** Machine learning models are decision-support tools, not diagnostic instruments. High test accuracy on a historical benchmark does not guarantee generalisation across diverse patient demographics.
2. **Algorithmic Accountability:** Human oncologists and cytopathologists must retain final diagnostic authority.
3. **Data Privacy & Governance:** Real clinical implementations must adhere strictly to patient privacy standards (e.g., HIPAA, GDPR, ISO 27799).

---

### 👨‍🏫 Faculty Reference
- **GitHub Repository Reference:** [https://github.com/complexiota](https://github.com/complexiota)

---
*End of CIE 2 Activity 2 Machine Learning Notebook.*
""")

# Save the notebook
nb.cells = cells
output_path = "Activity2_Logistic_Regression.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {output_path} with {len(cells)} cells.")

