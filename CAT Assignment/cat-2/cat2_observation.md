# CAT Assignment 2: Observation & Experimental Report
**Course:** ICS1502 -- Introduction to Machine Learning  
**Degree & Branch:** M. Tech (Integrated) Computer Science & Engineering (Semester V)  
**Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
**Student Name:** Danusu K (Danush K)  
**Register Number:** 3122247001013  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  
**Date:** 14-10-2026 / 18-10-2026  

---

## 1. Executive Summary & Assignment Scope
This experimental study performs a rigorous comparative evaluation of **Linear Classification Models** (Logistic Regression with and without $\mathcal{L}_2$ regularization) against **Neural Network Architectures** (Single-Layer Perceptron and Deep Multi-Layer Perceptron) using the **UCI Banknote Authentication Dataset** (1,372 instances, 4 continuous Haar Wavelet features).

The study addresses all six formal rubrics:
1. **Linear Model Baseline:** Quantitative evaluation of unregularized vs. $\mathcal{L}_2$-penalized Logistic Regression.
2. **Regularization Dynamics:** Continuous sweep across regularization strength $\lambda \in [10^{-4}, 10^4]$ and weight shrinkage trajectories.
3. **Decision Boundary Geometry:** Analytical derivation and 3D visualization of the separating hyperplane $w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$, alongside 2D posterior probability contour slices.
4. **Neural Network Benchmarking:** Deep MLP ($4 \to 16 \to 8 \to 1$) vs. Single-Layer Perceptron (SLP) vs. Logistic Regression across accuracy, precision, recall, F1, ROC-AUC, and parameter capacity.
5. **Dimensionality Reduction via PCA:** Spectral decomposition of the standardized covariance matrix, explained variance ratios, and critical evaluation of classifier degradation in reduced subspaces ($k=2, k=3$).
6. **Statistical Significance Testing:** 10-Fold Stratified Cross-Validation benchmarked with One-Way ANOVA, Tukey HSD post-hoc test, non-parametric Friedman Test, Iman-Davenport adjusted $F$-statistic, and Nemenyi Critical Difference analysis.
7. **Robustness to Outliers & Deployment Synthesis:** Systematic perturbation experiment assessing boundary stability, leading to an actionable deployment recommendation for banking hardware.

---

## 2. Dataset Overview & Exploratory Data Analysis
- **Total Samples:** $N = 1372$
- **Genuine Banknotes ($y=0$):** 762 samples (55.54%)
- **Counterfeit Banknotes ($y=1$):** 610 samples (44.46%)
- **Input Features ($D=4$):**
  1. $x_1$ (`variance`): Continuous Haar Wavelet variance ($\mu = 0.434, \sigma = 2.843$)
  2. $x_2$ (`skewness`): Continuous Haar Wavelet skewness ($\mu = 1.922, \sigma = 5.869$)
  3. $x_3$ (`curtosis`): Continuous Haar Wavelet kurtosis ($\mu = 1.398, \sigma = 4.310$)
  4. $x_4$ (`entropy`): Image Shannon entropy ($\mu = -1.192, \sigma = 2.101$)

### Pearson Feature Correlations:
- `variance` vs `class`: $r = -0.725$ (Strong negative correlation; counterfeit notes exhibit markedly lower wavelet variance).
- `skewness` vs `class`: $r = -0.445$ (Moderate negative correlation).
- `curtosis` vs `class`: $r = +0.156$ (Weak positive correlation).
- `entropy` vs `class`: $r = -0.027$ (Near zero linear correlation).
- Multicollinearity: `skewness` and `curtosis` exhibit negative correlation ($r = -0.787$), reflecting complementary moment properties of Wavelet packet subbands.

---

## 3. Experimental Results Summary

### Task 1: Linear Model Baseline (With vs. Without $\mathcal{L}_2$ Regularization)
Train: $N_{\text{train}} = 1097$ (80%), Test: $N_{\text{test}} = 275$ (20%), Stratified Split.

| Metric | Unregularized LR ($C = \infty$) | $\mathcal{L}_2$ Regularized LR ($C = 1.0$) | Delta / Effect |
| :--- | :---: | :---: | :---: |
| **Train Accuracy** | **99.09%** | 98.45% | Slight drop due to shrinkage constraint |
| **Test Accuracy** | **98.55%** | 97.09% | Unregularized slightly higher on clean data |
| **Precision (Class 1)** | 96.83% | 93.85% | +2.98% for unregularized |
| **Recall (Class 1)** | **100.00%** | **100.00%** | Zero false negatives across both models |
| **F1-Score (Class 1)** | **98.39%** | 96.83% | High harmonic mean |
| **ROC-AUC** | **1.0000** | 0.9999 | Near-perfect separation |
| **Binary Cross-Entropy**| **0.0213** | 0.0541 | Unregularized achieves sharper probabilities |
| **Weight L2 Norm $\|\mathbf{w}\|_2$** | **36.65** | **8.02** | **78.1% weight shrinkage under $\mathcal{L}_2$** |
| **Learned Weights $[w_1, w_2, w_3, w_4]$** | $[-19.76, -22.73, -20.85, -1.30]$ | $[-4.67, -4.81, -4.40, +0.26]$ | Regularization suppresses extreme coefficient explosion |
| **Bias Term $b$** | $-9.49$ | $-1.40$ | Balanced intercept |

---

### Task 2: Regularization Sweep over $\lambda = 1/C$
- **Tested Range:** $\lambda \in [10^{-4}, 10^{4}]$ across 100 logarithmic points.
- **Optimal Regularization Strength:** $\lambda^* = 1.0 \times 10^{-4}$ ($C = 10,000$), yielding **Test Accuracy = 98.55%**.
- **Regime Analysis:**
  1. **Underfitting Regime ($\lambda > 10^1$, $C < 0.1$):**
     - Weights are heavily penalized ($\|\mathbf{w}\|_2 \to 0$).
     - At $\lambda = 10^4$, accuracy plummets to **55.5%** (the naive majority class baseline).
     - Cross-entropy loss escalates to $0.693$ ($\ln 2$), indicating complete loss of discriminatory capability.
  2. **Overfitting / Ill-Conditioned Regime ($\lambda < 10^{-4}$, $C \to \infty$):**
     - Because genuine and counterfeit notes are largely linearly separable, the unregularized cross-entropy loss drives $\|\mathbf{w}\| \to \infty$ to push sigmoid outputs $\sigma(z) \to 1.0$.
     - $\mathcal{L}_2$ regularization stabilizes the Hessian matrix $\mathbf{H} = \mathbf{X}^T \mathbf{S} \mathbf{X} + \lambda \mathbf{I}$, preventing numerical overflow and bounding model confidence.

---

### Task 3: 3D Decision Hyperplane & Class Separability
Selected Features: $x_1$ (`variance`), $x_2$ (`skewness`), $x_3$ (`curtosis`).
- **3D Logistic Model Test Accuracy:** **97.09%** (only 8 misclassifications out of 275 held-out samples using just 3 features).
- **Separating Hyperplane Equation:**
  $$-4.563 \cdot x_1 - 4.988 \cdot x_2 - 4.416 \cdot x_3 - 1.448 = 0$$
  $$x_3 = -1.033 \cdot x_1 - 1.129 \cdot x_2 - 0.328$$
- **Separability Interpretation:**
  The classes occupy distinct spatial clusters in $\mathbb{R}^3$. Genuine banknotes cluster in the positive half-space ($x_1 > 0, x_2 > 0$), while forged banknotes concentrate in the negative variance/skewness sector. The linear hyperplane provides clear class separability, with minimal sample overlap confined to the boundary margin where wavelet transform moments decay.

---

### Task 4: Neural Network vs. Linear Model Benchmark

| Model Architecture | Train Acc | Test Acc | Precision | Recall | F1-Score | ROC-AUC | Parameters | Generalization Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression ($\mathcal{L}_2$)** | 98.45% | 97.09% | 93.85% | 100.00% | 96.83% | 0.9999 | **5** | 1.36% |
| **Single-Layer NN (SLP)** | 98.45% | 97.09% | 93.85% | 100.00% | 96.83% | 1.0000 | **5** | 1.36% |
| **Multi-Layer MLP ($4 \to 16 \to 8 \to 1$)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 225 | **0.00%** |

#### Key Analytical Takeaways:
1. **Mathematical Equivalence:** The Single-Layer Perceptron (SLP) trained with PyTorch Adam converges to the exact same test accuracy and F1-score as scikit-learn's Logistic Regression, validating that both minimize binary cross-entropy over a linear combination of inputs.
2. **Non-Linear Expressivity:** The Deep MLP achieves **100.00% test accuracy** (zero errors on all 275 held-out samples). The two hidden layers with ReLU non-linearities synthesize piecewise planar decision boundaries that perfectly resolve the narrow boundary manifold that linear models slightly misclassify.
3. **Generalization Gap:** The MLP exhibits zero generalization gap ($\Delta_{\text{gen}} = 0.00\%$), indicating that the model is not overfitting despite having 45 times more parameters than the linear model.

---

### Task 5: Principal Component Analysis (PCA) Dimensionality Reduction

#### Spectral Decomposition:
- Standardized Covariance Matrix $\mathbf{\Sigma} \in \mathbb{R}^{4 \times 4}$.
- **Eigenvalues:** $\lambda_1 = 2.170, \lambda_2 = 1.310, \lambda_3 = 0.353, \lambda_4 = 0.171$.
- **Explained Variance Ratio (EVR):**
  - $\text{PC}_1$: $54.19\%$
  - $\text{PC}_2$: $32.72\%$ ($\text{Cumulative: } 86.91\%$)
  - $\text{PC}_3$: $8.82\%$ ($\text{Cumulative: } 95.73\%$)
  - $\text{PC}_4$: $4.27\%$ ($\text{Cumulative: } 100.00\%$)

#### Classification Performance Across Subspaces:

| Feature Space | Dimensions | Retained Variance | Logistic Regression Acc | Multi-Layer MLP Acc |
| :--- | :---: | :---: | :---: | :---: |
| **Original Space** | **4D** | **100.00%** | **97.09%** | **100.00%** |
| **PCA Reduced ($k=3$)** | 3D | 95.73% | 90.55% (-6.54%) | 95.27% (-4.73%) |
| **PCA Reduced ($k=2$)** | 2D | 86.91% | 79.64% (-17.45%) | 85.45% (-14.55%) |

#### Critical Interpretation of PCA:
> **Core Finding:** PCA severely degrades classification performance on this dataset.
- **Unsupervised vs. Supervised Disconnect:** PCA computes orthogonal directions of maximal variance without regard to class labels $y$.
- **Discriminative Signal Loss:** The 3rd and 4th principal components carry only $13.09\%$ of total variance, but encode high-frequency Wavelet kurtosis and entropy fluctuations that uniquely distinguish genuine intaglio printing from counterfeit ink-jet reproductions.
- **Dimensionality Fallacy:** When the original feature space is already extremely small ($D=4$), there is no "Curse of Dimensionality" to mitigate. Dimensionality reduction only discards vital discriminative information.

---

### Task 6: Statistical Significance Testing

#### 10-Fold Stratified Cross-Validation Summary:
- **Logistic (Unregularized):** Mean = $99.20\% \pm 1.11\%$
- **Logistic ($\mathcal{L}_2$):** Mean = $98.10\% \pm 1.30\%$
- **SLP (Single-Layer):** Mean = $98.18\% \pm 1.20\%$
- **MLP (Multi-Layer):** Mean = **$100.00\% \pm 0.00\%$**
- **Logistic (PCA $k=2$):** Mean = $75.81\% \pm 3.39\%$
- **MLP (PCA $k=2$):** Mean = $81.78\% \pm 4.08\%$

---

#### 1. One-Way ANOVA (Parametric Test)
- **Null Hypothesis ($H_0$):** $\mu_1 = \mu_2 = \mu_3 = \mu_4 = \mu_5 = \mu_6$ (all models have identical mean cross-validation accuracy).
- **Alternative Hypothesis ($H_1$):** At least one model has a significantly different mean accuracy.
- **Assumptions Check:**
  - Levene's Test for Homoscedasticity: $F = 4.62, p = 1.02 \times 10^{-3} < 0.05$ (Variances are unequal / heteroscedastic, confirming that non-parametric rank tests are essential).
- **ANOVA Table:**
  - $SS_{\text{between}} = 0.5577, \quad df_{\text{between}} = 5, \quad MS_{\text{between}} = 0.1115$
  - $SS_{\text{within}} = 0.0292, \quad df_{\text{within}} = 54, \quad MS_{\text{within}} = 0.000541$
  - **$F$-Statistic:** **$206.02$**
  - **$p$-Value:** **$7.06 \times 10^{-34} \ll 0.001$**
  - **Conclusion:** Overwhelmingly reject $H_0$. Performance differences across model regimes are highly statistically significant.

#### Post-Hoc Tukey HSD ($\alpha = 0.05$):
- $\text{MLP (Full 4D)}$ vs $\text{Logistic (L2)}$: Difference = $+1.90\%, p_{\text{adj}} = 0.461$ (Difference is not statistically significant at 5% level).
- $\text{MLP (Full 4D)}$ vs $\text{Logistic (Unreg)}$: Difference = $+0.80\%, p_{\text{adj}} = 0.942$ (Difference is not statistically significant).
- $\text{All Original 4D Models}$ vs $\text{PCA (k=2) Models}$: Difference $> 16.3\%, p_{\text{adj}} = 0.000 < 0.001$ (**Statistically significant degradation confirmed!**).

---

#### 2. Friedman Test (Non-Parametric Paired Rank Test)
- **Null Hypothesis ($H_0$):** All models perform identically across folds; mean ranks are equal.
- **Mean Ranks (1 = Best, 6 = Worst):**
  1. **Multi-Layer MLP:** **1.30**
  2. **Logistic (Unreg):** 2.25
  3. **Single-Layer NN (SLP):** 3.15
  4. **Logistic ($\mathcal{L}_2$):** 3.30
  5. **MLP (PCA $k=2$):** 5.00
  6. **Logistic (PCA $k=2$):** 6.00
- **Friedman Chi-Square:** $\chi_F^2 = 46.79, \quad p = 6.28 \times 10^{-9} \ll 0.001$.
- **Iman-Davenport Adjusted $F_F$:**
  $$F_F = \frac{(N-1)\chi_F^2}{N(K-1) - \chi_F^2} = 131.00 \sim F(5, 45), \quad p = 1.21 \times 10^{-25} \ll 0.001$$
- **Nemenyi Post-Hoc Critical Difference:**
  $$CD = q_{\alpha=0.05} \sqrt{\frac{K(K+1)}{6N}} = 2.850 \sqrt{\frac{6 \times 7}{60}} = \mathbf{2.3845}$$
- **Post-Hoc Rank Analysis:**
  - $|\text{Rank}_{\text{MLP}} - \text{Rank}_{\text{Logistic(PCA)}}| = |1.30 - 6.00| = 4.70 > CD$ (**Significant superiority**).
  - $|\text{Rank}_{\text{MLP}} - \text{Rank}_{\text{MLP(PCA)}}| = |1.30 - 5.00| = 3.70 > CD$ (**Significant superiority**).
  - $|\text{Rank}_{\text{MLP}} - \text{Rank}_{\text{Logistic(Unreg)}}| = |1.30 - 2.25| = 0.95 < CD$ (Statistically equivalent).
  - $|\text{Rank}_{\text{MLP}} - \text{Rank}_{\text{Logistic(L2)}}| = |1.30 - 3.30| = 2.00 < CD$ (Statistically equivalent).

---

## 4. Robustness to Outliers (Perturbation Experiment)
When injecting leverage outliers ($2\%$ to $20\%$ contamination by shifting feature values by $+4\sigma$ and flipping labels):
- **Logistic Regression:** Degrades smoothly from $97.1\%$ down to $76.2\%$ at $20\%$ contamination. The linear decision boundary is vulnerable to high-leverage boundary points that pull the hyperplane.
- **Deep MLP:** Retains $>98\%$ accuracy up to $8\%$ contamination due to non-linear ReLU saturation and multi-layer distributed representations, but overfits to noise beyond $15\%$ contamination.

---

## 5. Industrial Deployment Tradeoff & Final Recommendation

| Evaluation Dimension | Logistic Regression ($\mathcal{L}_2$) | Multi-Layer MLP ($4 \to 16 \to 8 \to 1$) | Winner |
| :--- | :--- | :--- | :---: |
| **Accuracy / F1** | 98.10% CV mean (97.09% test) | **100.00% CV mean (100.00% test)** | MLP |
| **Statistical Significance** | Statistically comparable to MLP ($p > 0.05$, diff $< CD$) | Top-ranked ($R = 1.30$) | Tie |
| **Interpretability** | **White-box odds ratios ($e^{w_j}$), exact linear weights** | Black-box distributed weights | Logistic |
| **Parameter Count** | **5 parameters** | 225 parameters ($45\times$ larger) | Logistic |
| **Inference Latency** | **$0.02\, \mu\text{s}$ (4 MAC operations)** | $1.85\, \mu\text{s}$ ($90\times$ slower) | Logistic |
| **Robustness to Contamination** | Linear hyperplane tilt | **Higher resilience up to 8% contamination** | MLP |
| **Regulatory & Audit Compliance** | **Full explainability under Basel III / financial standards** | Opaque decision manifold | Logistic |

### Final Engineering Recommendation:
For **real-world Automated Teller Machines (ATMs) and high-speed currency sorting hardware** (processing 40 banknotes/second on embedded microcontrollers):
> **Regularized Logistic Regression ($\mathcal{L}_2$) is the superior engineering solution.**
While the Deep MLP achieves a perfect 100% accuracy, the difference between 98.1% and 100% is **statistically non-significant under the post-hoc Nemenyi test ($CD = 2.38$) and Tukey HSD ($p = 0.46$)**. Conversely, Logistic Regression provides guaranteed real-time execution in 4 multiply-accumulate (MAC) operations, zero vulnerability to uninterpretable black-box edge cases, and verifiable audit compliance demanded by banking regulators.
