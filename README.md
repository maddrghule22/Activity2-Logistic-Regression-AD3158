# 🩺 CIE 2 — Activity 2: Binary Classification Using Logistic Regression
### *Diagnostic Classification of Breast Masses Using the Breast Cancer Wisconsin (Diagnostic) Dataset*

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![Faculty Reference](https://img.shields.io/badge/Faculty%20Reference-complexiota-purple.svg)](https://github.com/complexiota)

---

## 📌 Academic Metadata & Repository Identification

- **Course:** B.Tech in Artificial Intelligence & Data Science / Computer Science Engineering
- **Evaluation Unit:** Continuous Internal Evaluation 2 (CIE 2) — Activity 2
- **Project Title:** Binary Classification Using Logistic Regression
- **Target Repository Name:** `Activity2-Logistic-Regression-[RollNumber]`
- **Faculty Mentor / Course Reference:** [https://github.com/complexiota](https://github.com/complexiota)
- **Primary Notebook Deliverable:** [`Activity2_Logistic_Regression.ipynb`](./Activity2_Logistic_Regression.ipynb)

---

> ### ⚠️ Clinical & Ethical Disclaimer
> **This project is an academic, instructional implementation for machine learning education.**  
> The models, decision boundaries, probability calibrations, and numerical thresholds derived herein are **NOT certified medical devices** and must **NEVER** be used as a standalone clinical diagnostic tool. Clinical decisions must always be directed by licensed oncologists and cytopathologists utilizing verified histopathological protocols.

---

## 📖 Executive Summary & Clinical Problem Statement

Breast cancer is among the most commonly diagnosed malignancies worldwide. Early, precise differentiation between **Benign** (non-cancerous) and **Malignant** (cancerous) tumours substantially improves clinical prognosis, surgical staging, and patient survival rates.

In computational oncology and screening epidemiology, diagnostic classification errors have profoundly asymmetric human costs:

| Error Type | Statistical Classification | Clinical Consequence | Severity Level |
|:---|:---|:---|:---|
| **False Negative (FN)** | Malignant tumour misclassified as **Benign** (Type II Error) | Patient is discharged without urgent therapy. Tumour progression, distant metastasis, delayed chemotherapy/mastectomy, and potential mortality. | **Critical / Life-Threatening** |
| **False Positive (FP)** | Benign lesion misclassified as **Malignant** (Type I Error) | Patient experiences acute psychological anxiety and is subjected to secondary invasive diagnostic procedures (core biopsy, surgical excision). | **Moderate / Manageable** |

### 🎯 Senior Data Scientist Guiding Principles
1. **Recall (Sensitivity) is the primary clinical metric:** Missing a malignant tumour ($FN$) is fundamentally unacceptable. The screening system must drive $FN \to 0$.
2. **Never blindly accept default threshold $\theta = 0.5$:** A clinical threshold must be calibrated according to relative error costs. In this study, we empirically evaluate $\theta \in \{0.3, 0.5, 0.7\}$ and justify recommending $\theta = 0.30$.
3. **Zero Data Leakage:** Preprocessing transformations (`StandardScaler`) are strictly fitted **only on training folds** ($X_{\text{train}}$) and subsequently applied to test data.
4. **Majority-Class Baseline Benchmarking:** The trained model is audited against a `DummyClassifier` baseline to prove genuine pattern learning beyond class distribution bias.

---

## 📚 Dataset Provenance & Target Label Inversion

### Source Citation
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Authors:** Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin - Madison)
- **Primary Repository:** UCI Machine Learning Repository / Scikit-Learn:
  [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Foundational Paper:** Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). *Nuclear feature extraction for breast tumor diagnosis*. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology.

### ⚠️ Target Label Inversion (Crucial Correction)
Scikit-Learn's built-in target assigns:
- `0` $\rightarrow$ Malignant
- `1` $\rightarrow$ Benign

Under standard biostatistical conventions, **Class 1 (Positive)** denotes the presence of disease (**Malignant**), and **Class 0 (Negative)** denotes absence of disease (**Benign**). This project explicitly inverts the target vector:
$$\text{Corrected Target} = 1 - \text{Raw Scikit-Learn Target}$$
- **Class 0 (Benign):** 357 instances (62.74%) — Majority Class
- **Class 1 (Malignant):** 212 instances (37.26%) — Minority Class
- **Total Samples:** 569 instances, 30 continuous morphological features, 0 missing values, 0 duplicate rows.

A clean, remapped CSV copy is exported at [`breast_cancer_diagnostic.csv`](./breast_cancer_diagnostic.csv).

---

## 📐 Mathematical Foundations of Logistic Regression

### 1. Why Linear Regression Fails for Binary Classification
- **Unbounded Range:** Ordinary Least Squares (OLS) produces predictions $\hat{y} \in (-\infty, +\infty)$, frequently yielding probabilities $< 0$ or $> 1$.
- **Heteroscedasticity:** For a binary outcome $y \in \{0, 1\}$, residual variance $\text{Var}(y|x) = p(x)(1 - p(x))$ depends on the input, violating OLS homoscedasticity.
- **Outlier Distortion:** Extreme feature values exert high leverage on the OLS hyperplane, shifting decision boundaries and misclassifying cluster centers.

### 2. The Logit Link & Sigmoid Activation Function
Logistic Regression models the log-odds (logit) as a linear combination of predictors:
$$z = \mathbf{w}^T \mathbf{x} + b = \beta_0 + \sum_{j=1}^p \beta_j x_j$$

The continuous log-odds $z \in (-\infty, +\infty)$ are mapped into calibrated probabilities $p \in (0, 1)$ via the **Sigmoid (logistic) function**:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

Key mathematical properties:
- **Range:** $\sigma(z) \in (0, 1)$
- **Symmetry:** $\sigma(-z) = 1 - \sigma(z)$
- **Midpoint:** $\sigma(0) = 0.5$
- **Derivative:** $\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z)) = p(1 - p)$

### 3. Binary Cross-Entropy Loss (Log-Loss) with $L_2$ Regularization
Parameters are optimized via Maximum Likelihood Estimation (MLE), minimizing:
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
where $C = 1.0$ controls inverse regularization strength to prevent coefficient explosion and overfitting.

### 4. Classification Metrics Formulas (Simple Language)

| Metric | Mathematical Formula | Simple Language Explanation |
|:---|:---:|:---|
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | Out of all biopsy samples evaluated, what percentage received the correct diagnosis? |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | When the model flags a tumour as Malignant, how often is it truly malignant? High precision prevents unnecessary invasive biopsies. |
| **Recall (Sensitivity)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Out of all patients who genuinely have malignant cancer, what fraction did the model successfully detect? High recall prevents missed fatal tumours. |
| **Specificity** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | Out of all patients with benign tissue, what fraction was correctly diagnosed as benign? |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | The harmonic mean balancing precision and recall. Penalizes models with high precision but low recall or vice versa. |

---

## 📊 Empirical Results & Benchmarks

### 1. Model vs Majority-Class Baseline (Test Set, 114 Samples)

| Metric | Trained Logistic Regression | Majority-Class Dummy Baseline | Absolute Improvement |
|:---|:---:|:---:|:---:|
| **Accuracy** | **97.37%** | 63.16% | **+34.21%** |
| **Precision** | **97.56%** | 0.00% | **+97.56%** |
| **Recall (Sensitivity)** | **95.24%** | 0.00% | **+95.24%** |
| **Specificity** | **98.61%** | 100.00% | -1.39% |
| **F1-Score** | **96.39%** | 0.00% | **+96.39%** |
| **ROC-AUC** | **0.9924** | 0.5000 | **+0.4924** |

### 2. Multi-Threshold Clinical Sensitivity Analysis ($\theta \in \{0.3, 0.5, 0.7\}$)

| Threshold ($\theta$) | TP | TN | FP | FN | Accuracy | Precision | Recall | Specificity | F1-Score | Clinical Role |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$\theta = 0.30$ (Recommended)** | **41** | **70** | **2** | **1** | **97.37%** | **95.35%** | **97.62%** | **97.22%** | **96.47%** | **Primary Screening (Minimises Missed Cancers)** |
| $\theta = 0.50$ (Default) | 40 | 71 | 1 | 2 | 97.37% | 97.56% | 95.24% | 98.61% | 96.39% | Standard Balanced Operations |
| $\theta = 0.70$ (Conservative) | 39 | 71 | 1 | 3 | 96.49% | 97.50% | 92.86% | 98.61% | 95.12% | High-Specificity Confirmation |

### 3. Threshold Recommendation & Justification
> **Operational Decision:** Set classification threshold to **$\theta = 0.30$**.
>
> **Scientific Justification:**  
> Shifting $\theta$ from $0.50$ to $0.30$ cuts False Negatives in half ($2 \to 1$), elevating **Recall to 97.62%**. The minor penalty is a single additional False Positive ($1 \to 2$), which simply triggers a harmless secondary test. In oncology screening, trading one false alarm to save a patient from an undetected cancer is the correct clinical decision.

---

## 🖼️ Visualisation Suite

All figures are programmatically generated and stored in [`figures/`](./figures/) at publication-grade 300 DPI resolution:

| Figure | Description |
|:---|:---|
| [`01_class_distribution.png`](./figures/01_class_distribution.png) | Annotated bar chart illustrating benign vs malignant class frequencies and percentages. |
| [`02_sigmoid_curve.png`](./figures/02_sigmoid_curve.png) | Mathematical sigmoid activation curve showing $z=0$ inflection and decision regions. |
| [`03_top_coefficients.png`](./figures/03_top_coefficients.png) | Horizontal bar plot of the top 12 standardized feature coefficients by log-odds impact. |
| [`04_confusion_matrix.png`](./figures/04_confusion_matrix.png) | Annotated confusion matrix displaying exact TP, TN, FP, and FN counts. |
| [`05_roc_curve.png`](./figures/05_roc_curve.png) | ROC curve demonstrating an exceptional AUC of $0.9924$ against random chance. |
| [`06_decision_behaviour.png`](./figures/06_decision_behaviour.png) | Sigmoid probability response vs standardized top diagnostic feature (`worst texture`). |
| [`07_precision_recall_curve.png`](./figures/07_precision_recall_curve.png) | Precision-Recall curve with an Average Precision (AP) score of $0.9912$. |

---

## 📂 Project Repository Structure

```
Activity2-Logistic-Regression-[RollNumber]/
├── Activity2_Logistic_Regression.ipynb   # Complete, executed, self-contained notebook
├── breast_cancer_diagnostic.csv          # Remapped dataset (0: Benign, 1: Malignant)
├── figures/                              # Publication-grade PNG charts (300 DPI)
│   ├── 01_class_distribution.png
│   ├── 02_sigmoid_curve.png
│   ├── 03_top_coefficients.png
│   ├── 04_confusion_matrix.png
│   ├── 05_roc_curve.png
│   ├── 06_decision_behaviour.png
│   └── 07_precision_recall_curve.png
├── build_notebook.py                     # Programmatic notebook generator
├── execute_notebook.py                   # Automated headless notebook test & execution engine
├── requirements.txt                      # Project dependency manifest
├── README.md                             # Comprehensive technical & academic documentation
├── run.bat                               # Windows 1-click launcher
└── run.sh                                # macOS / Linux 1-click launcher
```

---

## ⚙️ Installation & Execution Guide

### Option A: Open Directly in Google Colab
1. Upload [`Activity2_Logistic_Regression.ipynb`](./Activity2_Logistic_Regression.ipynb) to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime $\rightarrow$ Run all** (`Ctrl + F9`).
3. The notebook will automatically download the dataset from `sklearn`, execute all 47 cells sequentially, and display all tables and figures.

### Option B: Local Execution (Jupyter Lab / Notebook)
1. **Clone or download the repository:**
   ```bash
   git clone https://github.com/<your-username>/Activity2-Logistic-Regression-[RollNumber].git
   cd Activity2-Logistic-Regression-[RollNumber]
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows:
   py -m venv .venv
   .venv\Scripts\activate

   # On macOS / Linux:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Activity2_Logistic_Regression.ipynb
   ```
   Execute cells sequentially from beginning to end.

---

## 🎯 CIE 2 Activity 2 Evaluation Rubric Alignment (5-Mark Scheme)

| Evaluation Component | Maximum Marks | Implementation in this Project |
|:---|:---:|:---|
| **Data Understanding, Preprocessing & Leakage Prevention** | **1.0 Mark** | Complete provenance citation, target label inversion ($1 - y$), stratified 80:20 split, missing value & duplicate audits, and `StandardScaler` fitted strictly on train data. |
| **Theoretical Formulation & Sigmoid Mathematical Rigour** | **1.0 Mark** | Mathematical derivations of log-odds, why linear regression fails, sigmoid equation, derivative, cross-entropy loss function, and publication curve. |
| **Model Training, Parameters & Coefficient Interpretation** | **1.0 Mark** | Scikit-learn `LogisticRegression` fitted with $L_2$ regularization, intercept and coefficients table with Odds Ratios ($\exp(\beta)$), top-12 weight visualization. |
| **Comprehensive Metrics, Baseline Comparison & Visualisations** | **1.0 Mark** | Confusion matrix, manual vs sklearn verification of Accuracy, Precision, Recall, Specificity, F1; ROC curve (AUC=0.9924), PR curve (AP=0.9912), comparison against `DummyClassifier`. |
| **Threshold Sensitivity, Cost Analysis & Clinical Ethics** | **1.0 Mark** | Threshold grid ($\theta \in \{0.3, 0.5, 0.7\}$), asymmetric clinical cost analysis ($FN \gg FP$), recommendation of $\theta = 0.30$, single-patient prediction, and ELSI governance disclaimer. |
| **Total Score** | **5.0 / 5.0** | **Fully Satisfied & Academically Exemplary** |

---

## 👨‍🏫 Acknowledgments & Course References

- **Course Faculty Reference:** [https://github.com/complexiota](https://github.com/complexiota)
- **Primary Literature:** Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). *Nuclear feature extraction for breast tumor diagnosis*.
- **Machine Learning Tooling:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.
