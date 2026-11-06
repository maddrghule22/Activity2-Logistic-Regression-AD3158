"""
Page 9: Comprehensive Project Documentation & Academic Viva Guide
"""

import streamlit as st
from src.ui_components import render_section_header

def render_documentation_page():
    render_section_header("Academic Project Documentation & Viva Examination Guide", "Complete theoretical report, mathematical formulations, methodology, and viva voce cheat sheet")

    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 24px; margin-bottom: 25px;">
        <h2 style="color: #0f172a; margin-top: 0;">Smart Student Performance & Placement Prediction System</h2>
        <p style="color: #475569; font-size: 1.05rem; line-height: 1.6;">
            <b>Course / Specialization:</b> B.Tech in Artificial Intelligence & Data Science / Computer Science Engineering<br>
            <b>Domain:</b> Supervised Machine Learning, Statistical Classification, Decision Support Systems<br>
            <b>Primary Algorithm:</b> Regularized Binary Logistic Regression (Scikit-Learn Pipeline)
        </p>
    </div>
    """, unsafe_allow_html=True)

    doc_tabs = st.tabs([
        "📄 Abstract & Objectives",
        "📐 Mathematical Foundations",
        "🔬 Methodology & Pipeline",
        "📊 Results & Discussion",
        "🎯 Viva Voce Q&A Cheat Sheet",
        "📚 References"
    ])

    with doc_tabs[0]:
        st.markdown(r"""
        ### 1. Project Abstract
        In modern higher education institutions, campus placement is a critical performance indicator. Early identification of students who may face challenges during campus recruitment drives allows placement cells and academic advisors to administer targeted pedagogical, coding, and soft-skill remedial interventions.
        
        This project presents an interactive, industry-grade Machine Learning decision support platform built around **Logistic Regression**. The system analyzes student academic indicators (CGPA, active backlogs) and industry-readiness attributes (programming proficiency, communication score, aptitude rating, internships, projects, certifications, and work experience). Implemented in Python 3.12 with Scikit-learn, the pipeline strictly prevents data leakage by isolating preprocessing within a `ColumnTransformer`, utilizes regularized log-odds estimation, and provides interactive threshold tuning, 2D decision boundary visualization, and SQLite audit logging.

        ---

        ### 2. Problem Statement
        Campus recruitment is highly competitive. Universities frequently struggle to identify students at risk of non-placement early enough in their 6th or 7th semesters to provide actionable intervention. Existing evaluation frameworks are often subjective or rely solely on raw CGPA, failing to account for programming aptitude, internships, or certifications.

        ---

        ### 3. Project Objectives
        1. **End-to-End Pipeline:** Develop a leakage-free Scikit-learn pipeline incorporating median/mode imputation, standard scaling, and one-hot encoding.
        2. **Probabilistic Classification:** Train a regularized Logistic Regression classifier capable of estimating calibrated posterior placement probabilities.
        3. **Dynamic Threshold Tuning:** Implement a real-time threshold slider allowing placement officers to trade off Precision against Recall according to recruiter standards.
        4. **Comparative Analysis:** Benchmark Logistic Regression against non-linear Decision Tree and Random Forest classifiers.
        5. **Model Interpretability:** Extract beta weights and calculate Odds Ratios ($\exp(\beta)$) to understand primary placement drivers.
        6. **Interactive Deployment:** Provide an intuitive Streamlit dashboard with SQLite history logging for auditable institutional use.
        """)

    with doc_tabs[1]:
        st.markdown(r"""
        ### Mathematical Explanation of Logistic Regression

        #### 1. Why Linear Regression Fails for Classification
        Ordinary Least Squares (OLS) regression models the response $Y \in \mathbb{R}$ as $Y = X\beta + \epsilon$. For binary outcomes $Y \in \{0, 1\}$, linear regression yields predictions $z < 0$ or $z > 1$, violating basic axioms of probability. Moreover, errors are inherently non-normal and heteroscedastic.

        #### 2. Odds and the Logit Transformation
        The **odds** of an event is the ratio of probability of occurrence to non-occurrence:
        $$\text{Odds} = \frac{p}{1 - p}$$
        The **logit function** (log-odds) maps probabilities $p \in (0, 1)$ to real numbers $\mathbb{R} \in (-\infty, +\infty)$:
        $$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right) = z = \beta_0 + \sum_{i=1}^n \beta_i x_i$$

        #### 3. The Sigmoid Activation Function
        Inverting the logit transformation provides the standard **logistic (sigmoid) function**:
        $$p = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + \exp\left(-\left(\beta_0 + \sum_{i=1}^n \beta_i x_i\right)\right)}$$
        The derivative of the sigmoid is remarkably efficient to compute:
        $$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z)) = p(1 - p)$$

        #### 4. Maximum Likelihood Estimation (MLE) & Binary Cross-Entropy Loss
        Assuming independent Bernoulli outcomes, the likelihood function $L(\beta)$ is:
        $$L(\beta) = \prod_{j=1}^N p_j^{y_j} (1 - p_j)^{1 - y_j}$$
        Taking the negative logarithm gives the **Binary Cross-Entropy Loss (Log-Loss)**:
        $$\mathcal{L}(\beta) = -\frac{1}{N} \sum_{j=1}^N \left[ y_j \ln(p_j) + (1 - y_j) \ln(1 - p_j) \right]$$

        #### 5. Gradient Descent Parameter Updates
        The partial derivative of the loss with respect to feature weight $\beta_i$ is:
        $$\frac{\partial \mathcal{L}}{\partial \beta_i} = \frac{1}{N} \sum_{j=1}^N (\sigma(z_j) - y_j) x_{ji}$$
        Weights are updated iteratively via gradient descent with learning rate $\eta$:
        $$\beta_i \leftarrow \beta_i - \eta \cdot \frac{\partial \mathcal{L}}{\partial \beta_i}$$

        #### 6. Regularization (L1 & L2)
        To prevent overfitting, penalty terms are incorporated into the loss:
        - **L2 Regularization (Ridge):** Adds $\frac{1}{2C} \sum_{i=1}^n \beta_i^2$, shrinking coefficients smoothly.
        - **L1 Regularization (Lasso):** Adds $\frac{1}{C} \sum_{i=1}^n |\beta_i|$, driving non-informative coefficients to exactly zero.
        """)

    with doc_tabs[2]:
        st.markdown(r"""
        ### Methodology & Machine Learning Workflow

        ```
        Raw Student Data (CSV)
                 │
                 ▼
        ┌────────────────────────────────────────────────────────┐
        │ 1. Data Cleaning & Sanitization                        │
        │    - Remove exact duplicate rows                       │
        │    - Audit missing values and categorical strings      │
        └────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────────────┐
        │ 2. Stratified Train-Test Split (e.g. 80/20)            │
        │    (Guarantees identical class balance in both splits) │
        └──────────────┬──────────────────────────┬──────────────┘
                       │                          │
                 Train Set (80%)             Test Set (20%)
                       │                          │
                       ▼                          │
        ┌─────────────────────────────┐           │
        │ 3. Fit ColumnTransformer    │           │
        │    - Numeric: Median Impute │           │
        │      + StandardScaler       │           │
        │    - Categorical: Mode      │           │
        │      + OneHotEncoder        │           │
        └──────────────┬──────────────┘           │
                       │                          │
                       ▼                          ▼
        ┌─────────────────────────────┐    ┌────────────────────┐
        │ 4. Fit Logistic Regression  │───▶│ 5. Model Evaluation│
        │    - L2 Penalty, lbfgs      │    │    - Confusion Mat │
        │    - 5-Fold Stratified CV   │    │    - ROC-AUC Curve │
        │    - Hyperparameter Tuning  │    │    - Threshold Opt │
        └─────────────────────────────┘    └────────────────────┘
        ```

        #### Data Leakage Prevention Guarantee
        Data leakage occurs when information outside the training dataset is inadvertently used to train the model. This project eliminates data leakage by wrapping all imputers and scalers inside a Scikit-Learn `Pipeline`. The preprocessor calculates statistical parameters ($\mu, \sigma$, medians) **exclusively on $X_{\text{train}}$**, and applies the learned transformation to $X_{\text{test}}$.
        """)

    with doc_tabs[3]:
        st.markdown("""
        ### Results, Discussion & Interpretability

        #### Empirical Observations:
        1. **High Separability (ROC-AUC ~ 0.96+):** The combination of academic indicators (CGPA, backlogs) and practical skill metrics (programming score, internships) creates distinct separability between placed and unplaced cohorts.
        2. **Coefficients & Odds Ratios:**
           - **CGPA & Programming Score:** Feature positive beta coefficients ($\beta > 0$), yielding Odds Ratios $> 1.0$. A 1-standard-deviation increase in programming score significantly scales placement odds.
           - **Academic Backlogs:** Strongly negative coefficient ($\beta \approx -1.15$), yielding an Odds Ratio of $\approx 0.32$. Each active backlog reduces placement odds by approximately 68%.
           - **Work Experience & Internships:** Strong positive multipliers for campus recruitment.
        3. **Threshold Sensitivity:** While $0.50$ is standard, lowering the threshold to $0.40$ increases Recall (identifying more marginal students for company interview shortlists), whereas raising the threshold to $0.65$ boosts Precision for high-cutoff tier-1 recruitment drives.
        """)

    with doc_tabs[4]:
        st.markdown(r"""
        ### 🎯 Top 10 Viva Voce Questions & Model Answers

        **Q1: Why do we use Logistic Regression instead of Linear Regression for classification?**  
        *Answer:* Linear Regression outputs continuous unbounded values $(-\infty, +\infty)$, which can be $< 0$ or $> 1$, violating probability axioms. Furthermore, classification errors violate homoscedasticity and normality assumptions. Logistic Regression uses the Sigmoid function to constrain outputs to $[0, 1]$.

        **Q2: What is the mathematical definition of Odds and Odds Ratio?**  
        *Answer:* Odds is the ratio of probability of occurrence to non-occurrence: $\text{Odds} = \frac{p}{1 - p}$. The Odds Ratio (OR) is the ratio of two odds. In Logistic Regression, $\text{OR} = e^{\beta_i}$, representing the multiplicative increase in odds when feature $x_i$ increases by one unit.

        **Q3: How are the weights of Logistic Regression optimized?**  
        *Answer:* Unlike linear regression, which has a closed-form Ordinary Least Squares solution, Logistic Regression utilizes **Maximum Likelihood Estimation (MLE)**. Because the log-likelihood is convex, we use iterative gradient descent (or Newton-Raphson / L-BFGS) to find optimal parameters.

        **Q4: What is Data Leakage and how did you prevent it?**  
        *Answer:* Data leakage occurs when test data statistics (such as test set mean or variance) leak into training preprocessing. We prevented this by using a Scikit-learn `Pipeline` that calls `.fit()` exclusively on the training split and `.transform()` on test and production inputs.

        **Q5: What is the significance of the C hyperparameter?**  
        *Answer:* $C$ is the inverse of regularization strength ($C = 1/\lambda$). A small $C$ enforces strong regularization, preventing overfitting by shrinking weights toward zero. A large $C$ allows the model to fit training data more closely.

        **Q6: What is the difference between L1 (Lasso) and L2 (Ridge) regularization?**  
        *Answer:* L2 adds the sum of squared weights ($\|\beta\|_2^2$), shrinking weights smoothly. L1 adds the sum of absolute values ($\|\beta\|_1$), which produces sparse models by driving less important feature weights to exactly zero, performing automated feature selection.

        **Q7: What does the ROC-AUC score represent?**  
        *Answer:* The Receiver Operating Characteristic (ROC) curve plots True Positive Rate vs False Positive Rate across all possible classification thresholds. The Area Under the Curve (AUC) measures the probability that the classifier ranks a randomly chosen placed student higher than a randomly chosen unplaced student. An AUC of 1.0 is perfect; 0.5 is random guessing.

        **Q8: When would you change the classification threshold from 0.50?**  
        *Answer:* If the cost of False Negatives is high (e.g., in medical diagnosis or identifying struggling students early), we lower the threshold to increase Recall. If the cost of False Positives is high (e.g., high-stake selective interviews), we raise the threshold to increase Precision.

        **Q9: How do you interpret a negative coefficient in Logistic Regression?**  
        *Answer:* A negative coefficient ($\beta_i < 0$) means that an increase in feature $x_i$ decreases the log-odds of the positive class. The Odds Ratio $e^{\beta_i}$ is between 0 and 1. In our project, Backlogs has $\beta \approx -1.15$, indicating that having backlogs substantially decreases placement odds.

        **Q10: What are the primary assumptions of Logistic Regression?**  
        *Answer:* 1) Binary/categorical dependent variable; 2) Linearity between independent continuous features and the log-odds (logit); 3) Little or no multicollinearity among predictors; 4) Independent observations (no autocorrelation).
        """)

    with doc_tabs[5]:
        st.markdown("""
        ### Academic References & Citations
        1. **Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X.** (2013). *Applied Logistic Regression* (3rd ed.). John Wiley & Sons.
        2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer Science & Business Media.
        3. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
        4. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.
        """)
