# 🎓 Smart Student Performance & Placement Prediction System
### *An End-to-End Supervised Machine Learning Platform using Regularized Logistic Regression*

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary & Abstract
In modern higher education institutions, campus placement is a defining metric of academic quality and student professional success. Timely identification of students who may face challenges during recruitment drives allows university placement departments and academic counselors to provide targeted coding bootcamps, soft-skill training, and academic interventions.

This project delivers an interactive, industry-standard, production-grade Machine Learning decision support platform built on **regularized Logistic Regression**. The system evaluates student academic indicators (`CGPA`, `Academic Backlogs`) and industry competency metrics (`Programming Proficiency`, `Communication Score`, `Aptitude`, `Internships`, `Projects`, `Technical Certifications`, and `Work Experience`).

By encapsulating all data transformations in a Scikit-Learn `Pipeline` (`ColumnTransformer`), the system **strictly eliminates data leakage**. It provides real-time threshold optimization, interactive 2D decision boundary visualization, multi-model benchmarking against tree-based ensembles, and persistence via an embedded SQLite database.

---

## 🚀 Key System Features

### 1. Data Management & Preprocessing
* **Sample Placement Dataset:** Built-in curated dataset of 1,200+ student observations with realistic academic correlations.
* **Custom Data Ingestion:** Upload custom CSV files with automatic schema detection and summary statistics.
* **Data Sanitization:** Automatic deduplication, median imputation for numerical features, and mode imputation for categorical attributes.
* **Zero Data Leakage:** Preprocessing statistics are calculated exclusively on training folds via Scikit-Learn `Pipeline`.

### 2. Exploratory Data Analysis (EDA)
* **Univariate Distributions:** Histograms with marginal boxplots categorized by placement status.
* **Correlation Heatmap:** Interactive Pearson correlation matrix revealing multi-collinearity and predictive traits.
* **Bivariate Interaction:** Interactive scatter plots with dynamic X and Y axis selectors.
* **Academic Driver Analysis:** Cross-tabulations and distributions for CGPA, internships, projects, and backlogs.
* **Outlier Detection:** Interquartile Range (IQR) boxplots highlighting extreme observations.

### 3. Model Training & Optimization
* **Leakage-Free Pipeline:** Preprocessing and estimator bundled together.
* **Hyperparameter Controls:** Interactive tuning for regularization penalty ($L_1$ Lasso, $L_2$ Ridge), inverse regularization strength $C$, and solvers (`lbfgs`, `liblinear`, `saga`).
* **Automated Grid Search:** Built-in `GridSearchCV` mode for automated hyperparameter optimization.
* **Stratified 5-Fold Cross-Validation:** Full out-of-fold performance evaluation ($Mean \pm Std$).

### 4. Comprehensive Model Evaluation
* **Full Metric Suite:** Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, and Binary Cross-Entropy Loss.
* **Interactive Confusion Matrix:** Color-coded True Positive, False Positive, True Negative, and False Negative counts.
* **Interactive Classification Threshold Slider:** Real-time dynamic recalculation of all metrics and confusion matrices as the threshold shifts between $0.05$ and $0.95$.
* **ROC & Precision-Recall Curves:** Publication-ready diagnostic curves with baseline comparisons.

### 5. Interactive Inference Engine
* **Student Input Form:** Sliders and inputs for CGPA, programming, communication, aptitude, projects, internships, certifications, backlogs, and work experience.
* **Instant Classification:** "Likely to be Placed" vs. "Placement Unlikely (At Risk)" with confidence percentages.
* **Probability Gauge:** Animated meter visualizing placement odds relative to the chosen decision threshold.
* **Odds-Ratio Feature Breakdown:** Local explanation calculating the exact log-odds contribution ($\beta_i \cdot x_i$) for each attribute.
* **Actionable Student Recommendations:** Automatic tailored remedial suggestions (e.g., clearing backlogs, increasing coding scores).

### 6. Model Comparison & Advanced Diagnostics
* **Empirical Benchmarking:** Direct side-by-side comparison with **Decision Tree** and **Random Forest** classifiers.
* **Feature Importance Analysis:** Graphical visualization of Logistic Regression coefficients ($\beta$) and Odds Ratios ($\exp(\beta)$).
* **2D Decision Boundary Surface:** Renders 2D probability contours with decision isoclines for any two selected features against real student data points.

### 7. SQLite Prediction History Audit
* **Audit Trail:** Automatic logging of every student inference query with timestamps, features, predictions, and applied thresholds.
* **Search & Filter:** Query history by student roll number or filter by placement status.
* **Data Export:** Instant CSV export of prediction history.

### 8. Academic Viva Voce Guide & Documentation
* In-app reference manual covering mathematical derivations, loss functions, gradient descent updates, and the **Top 10 Viva Examination Q&A**.

---

## 🛠️ Technology Stack
* **Language:** Python 3.11+ / 3.12
* **Machine Learning:** Scikit-Learn, NumPy, SciPy
* **Data Manipulation:** Pandas
* **Interactive Visualization:** Plotly Express & Plotly Graph Objects
* **Static Graphics:** Matplotlib, Seaborn
* **Web UI Framework:** Streamlit
* **Model Serialization:** Joblib
* **Database & Persistence:** SQLite3
* **Testing Suite:** Pytest

---

## 📂 Project Architecture

```
logistic-regression/
├── .venv/                              # Python Virtual Environment
├── assets/                             # Pre-generated publication figures
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── pr_curve.png
│   └── roc_curve.png
├── data/
│   └── student_placement_dataset.csv   # Curated 1,200 student placement dataset
├── database/
│   └── prediction_history.db          # SQLite persistent storage for predictions
├── models/
│   └── placement_logistic_model.joblib # Serialized Scikit-learn Pipeline
├── src/
│   ├── __init__.py
│   ├── data_generator.py              # Synthetic realistic dataset generator
│   ├── preprocessor.py                # ColumnTransformer & train-test split
│   ├── model_engine.py                # Estimator training, CV, GridSearch, comparison
│   ├── evaluator.py                   # Metrics, confusion matrix, threshold curves
│   ├── db_manager.py                  # SQLite CRUD operations
│   ├── ui_components.py               # Custom modern CSS & metric cards
│   ├── visualizer.py                  # Plotly interactive visualizations
│   └── pages_views/                   # Modular views for all 9 dashboard pages
│       ├── view_overview.py
│       ├── view_dataset.py
│       ├── view_eda.py
│       ├── view_training.py
│       ├── view_evaluation.py
│       ├── view_prediction.py
│       ├── view_comparison.py
│       ├── view_history.py
│       └── view_documentation.py
├── tests/                             # Pytest test suite
│   ├── test_db.py
│   ├── test_model.py
│   └── test_preprocessor.py
├── app.py                             # Main Streamlit dashboard application
├── generate_plots.py                  # Static plot export script
├── train_model.py                     # Standalone CLI training script
├── requirements.txt                   # Production package dependencies
├── run.bat                            # 1-Click launcher for Windows
├── run.sh                             # 1-Click launcher for Linux/macOS
└── README.md                          # Comprehensive documentation
```

---

## 📐 Mathematical Overview of Logistic Regression

### 1. Log-Odds (Logit)
Logistic regression models the log-odds of the positive class ($Y=1$, Placed) as a linear function of $n$ predictors:
$$\ln\left(\frac{p}{1 - p}\right) = z = \beta_0 + \sum_{i=1}^n \beta_i x_i$$

### 2. Sigmoid Activation Function
Inverting the logit transformation yields the sigmoid function, restricting output probabilities to $p \in (0, 1)$:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + \exp\left(-\left(\beta_0 + \sum_{i=1}^n \beta_i x_i\right)\right)}$$

### 3. Binary Cross-Entropy Loss
Model parameters $\beta$ are estimated via Maximum Likelihood Estimation (MLE), minimizing:
$$\mathcal{L}(\beta) = -\frac{1}{N} \sum_{j=1}^N \left[ y_j \ln(p_j) + (1 - y_j) \ln(1 - p_j) \right] + \lambda \cdot \mathcal{R}(\beta)$$
where:
* **$L_2$ Regularization (Ridge):** $\mathcal{R}(\beta) = \frac{1}{2}\sum_{i=1}^n \beta_i^2$ (shrinks weights smoothly, handles correlated features)
* **$L_1$ Regularization (Lasso):** $\mathcal{R}(\beta) = \sum_{i=1}^n |\beta_i|$ (enforces coefficient sparsity for automated feature selection)
* $\lambda = \frac{1}{C}$

### 4. Gradient Descent Update
$$\beta_i \leftarrow \beta_i - \eta \cdot \frac{1}{N} \sum_{j=1}^N (\sigma(z_j) - y_j) x_{ji}$$

---

## 📊 Empirical Baseline Performance

Trained on 1,200 student records with an 80/20 stratified split:

| Metric | Score (Test Set) | Academic Significance |
| :--- | :--- | :--- |
| **Accuracy** | **89.2%** | High overall classification accuracy |
| **Precision** | **87.4%** | Low false-positive rate; prevents inaccurate placement expectations |
| **Recall (Sensitivity)** | **85.6%** | High capture rate of genuinely placeable students |
| **F1 Score** | **86.5%** | Robust harmonic balance between Precision and Recall |
| **ROC-AUC** | **0.968** | Exceptional class separability across all classification thresholds |
| **Log-Loss** | **0.222** | Well-calibrated, confident posterior probability estimation |

---

## ⚙️ Installation & Execution Guide

### Prerequisites
* Python 3.11 or Python 3.12 installed on your system.
* Git installed (optional).

### Option A: Windows 1-Click Launch
Double-click `run.bat` in the project root folder. It will automatically initialize the virtual environment, install dependencies, and launch the dashboard.

### Option B: Manual Setup (Windows / macOS / Linux)

1. **Clone or navigate to the repository directory:**
   ```bash
   cd "logistic regression"
   ```

2. **Create and activate a virtual environment:**
   * **Windows (Command Prompt / PowerShell):**
     ```cmd
     py -m venv .venv
     .venv\Scripts\activate
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Run tests to verify setup:**
   ```bash
   pytest tests
   ```

5. **(Optional) Retrain model via CLI:**
   ```bash
   python train_model.py
   ```

6. **Launch the Streamlit ML Dashboard:**
   ```bash
   streamlit run app.py
   ```
   The dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 🎓 Viva Voce Quick Revision Guide

1. **Why use Logistic Regression instead of Linear Regression for classification?**
   * *Linear regression outputs continuous predictions $(-\infty, +\infty)$ that can violate probability bounds $[0, 1]$. In contrast, Logistic Regression applies the Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to map outputs to valid probabilities.*
2. **What does the Odds Ratio ($e^{\beta_i}$) mean?**
   * *An Odds Ratio $> 1.0$ indicates that as feature $x_i$ increases by 1 unit, the odds of placement increase multiplicatively. An Odds Ratio $< 1.0$ (e.g., for backlogs) indicates a reduction in placement odds.*
3. **What is Data Leakage and how is it prevented?**
   * *Data leakage occurs when test set distribution parameters (e.g., means, variances, min/max) leak into training preprocessing. It is prevented here using a Scikit-Learn `Pipeline` which fits scalers only on training folds.*
4. **Why adjust the classification threshold?**
   * *The default threshold is $0.50$. For high-stakes recruiting, increasing the threshold (e.g., $0.65$) increases Precision. For early academic counseling, lowering the threshold (e.g., $0.40$) increases Recall, ensuring fewer struggling students are missed.*
5. **How does L1 differ from L2 regularization?**
   * *L2 (Ridge) penalizes squared weights and shrinks them smoothly. L1 (Lasso) penalizes absolute weights and drives non-informative features to zero, performing automatic feature selection.*

---

## 📜 License
This project is open-source under the MIT License. Developed for academic presentation, B.Tech capstone, and viva demonstration.
