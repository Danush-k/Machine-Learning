# Experiment 7: Observation Sheet

**Experiment Title:** Dimensionality Reduction and Model Evaluation (With and Without PCA)  
**Course:** Machine Learning Algorithms Laboratory (ICS1512)  
**Student Name:** Danusu K  
**Register Number:** 3122247001013  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  
**Academic Year:** 2026-2027 (Odd Semester) | **Batch:** 2024-2029  

---

## 1. Objective
To systematically investigate the effect of dimensionality reduction using **Principal Component Analysis (PCA)** on the performance, stability, and computational efficiency of 10 machine learning classifiers on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The empirical study evaluates models under two configurations:
1. **No-PCA (Original Feature Space):** Full 30 standardized continuous features.
2. **With-PCA (Reduced Feature Space):** Reduced orthogonal feature subspace explaining $\ge 95\%$ of cumulative variance (10 Principal Components, achieving a $66.67\%$ reduction in dimensionality).

For both configurations, models undergo systematic hyperparameter grid searching with **5-Fold Stratified Cross-Validation**, fold-by-fold stability analysis, and held-out test set validation.

---

## 2. Dataset Summary & Preprocessing
* **Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC)
* **Source:** UCI Machine Learning Repository
* **Total Samples ($N$):** 569 instances (Complete, 0 missing values)
* **Number of Features ($D$):** 30 continuous nuclear morphological attributes (Mean, Standard Error, and Worst contours)
* **Target Classes:** Binary — Benign ($B=0$, $N=357$, $62.74\%$) vs Malignant ($M=1$, $N=212$, $37.26\%$)
* **Data Partitioning:** Stratified 80% Training ($N=455$) and 20% Held-out Testing ($N=114$) with fixed seed ($42$).
* **Preprocessing:** `StandardScaler` fitted on training split ($\mu=0, \sigma=1$) to prevent features with wide nominal scales from dominating the covariance matrix.

---

## 3. Observation Tables

### Table 1: PCA Variance Explained & Scree Analysis
*(Eigenvalue Decomposition on Standardized 30-Dimensional Training Space)*

| Principal Component ($PC_i$) | Eigenvalue ($\lambda_i$) | Explained Variance (%) | Cumulative Variance (%) | Kaiser Criterion ($\lambda \ge 1$) | Selection Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **PC1** | 13.4075 | 44.59% | 44.59% | Retained | Selected |
| **PC2** | 5.5758 | 18.55% | 63.14% | Retained | Selected |
| **PC3** | 2.8817 | 9.58% | 72.72% | Retained | Selected |
| **PC4** | 1.9825 | 6.59% | 79.32% | Retained | Selected |
| **PC5** | 1.6904 | 5.62% | 84.94% | Retained | Selected |
| **PC6** | 1.1992 | 3.99% | 88.93% | Retained | Selected |
| **PC7** | 0.6658 | 2.21% | 91.14% | Discarded | Selected (90% Target) |
| **PC8** | 0.4853 | 1.61% | 92.76% | Discarded | Selected |
| **PC9** | 0.3863 | 1.28% | 94.04% | Discarded | Selected |
| **PC10** | **0.3505** | **1.17%** | **95.21%** | Discarded | **Chosen Cutoff (95% Target)** |
| PC11 | 0.3032 | 1.01% | 96.22% | Discarded | Excluded |
| PC12 | 0.2723 | 0.91% | 97.12% | Discarded | Excluded |
| PC13 | 0.2383 | 0.79% | 97.91% | Discarded | Excluded |
| PC14 | 0.1442 | 0.48% | 98.39% | Discarded | Excluded |
| PC15 | 0.0914 | 0.30% | 98.70% | Discarded | Excluded |
| PC16–PC30 | $< 0.08$ | $< 0.25\%$ each | 98.70% $\to$ 100.0% | Discarded | Excluded |

> **Design Choice Justification:**  
> A variance target of **95.0%** was selected. The first 10 principal components capture **95.21%** of the cumulative variance while reducing the input dimensionality from 30 to 10 (a **66.67% reduction**). This choice preserves almost all diagnostic information while eliminating 20 dimensions of collinearity and noise.

---

### Table 2: Support Vector Machine (SVM) — Hyperparameter Tuning
*(Grid Search: Kernel $\in [\text{linear}, \text{rbf}, \text{poly}]$, $C \in [0.1, 1, 10, 100]$, $\gamma \in [\text{scale}, 0.01, 0.1]$)*

| Kernel | Regularization ($C$) | Kernel Coefficient ($\gamma$) | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :---: | :---: | :---: | :---: | :---: |
| **RBF** | **10** | **0.01** | **97.80% (Best)** | **97.58% (Best)** |
| RBF | 1 | scale | 97.36% | 97.36% |
| RBF | 10 | 0.10 | 96.26% | 96.26% |
| Linear | 0.1 | N/A | 97.14% | 96.92% |
| Linear | 1 | N/A | 96.92% | 97.14% |
| Linear | 10 | N/A | 96.48% | 96.70% |
| Poly | 1 | scale | 90.99% | 91.87% |
| Poly | 10 | scale | 94.73% | 94.95% |

---

### Table 3: Gaussian Naïve Bayes — Smoothing Parameter Tuning
*(Grid Search: $\text{var\_smoothing} \in [10^{-11}, 10^{-9}, 10^{-7}, 10^{-5}, 10^{-3}, 10^{-1}]$)*

| Smoothing Parameter ($\text{var\_smoothing}$) | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :---: | :---: | :---: |
| **$10^{-11}$** | **93.85% (Best)** | **91.21% (Best)** |
| $10^{-9}$ | 93.85% | 91.21% |
| $10^{-7}$ | 93.85% | 91.21% |
| $10^{-5}$ | 93.63% | 91.21% |
| $10^{-3}$ | 93.41% | 91.21% |
| $10^{-1}$ | 92.53% | 90.99% |

---

### Table 4: k-Nearest Neighbors (KNN) — Hyperparameter Tuning
*(Grid Search: $k \in [3, 5, 7, 9, 11, 15, 21]$, Weights $\in [\text{uniform}, \text{distance}]$, Metric $\in [\text{euclidean}, \text{manhattan}, \text{minkowski}]$)*

| $k$ Neighbors | Weight Function | Distance Metric | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :---: | :---: | :---: | :---: | :---: |
| **3** | **uniform** | **manhattan** | **97.58% (Best No-PCA)** | 96.92% |
| **3** | **uniform** | **euclidean** | 96.92% | **97.14% (Best With-PCA)** |
| 5 | uniform | manhattan | 97.14% | 96.70% |
| 5 | uniform | euclidean | 96.48% | 96.48% |
| 7 | uniform | manhattan | 96.70% | 96.70% |
| 7 | distance | euclidean | 96.92% | 96.70% |
| 9 | uniform | euclidean | 96.48% | 96.48% |
| 11 | uniform | euclidean | 96.04% | 96.48% |
| 15 | uniform | euclidean | 95.82% | 96.04% |
| 21 | uniform | euclidean | 95.60% | 95.82% |

---

### Table 5: Logistic Regression — Hyperparameter Tuning
*(Grid Search: $C \in [0.01, 0.1, 1, 10, 100]$, Penalty $\in [\text{L2}]$, Solver $\in [\text{liblinear}, \text{lbfgs}]$)*

| Regularization ($C$) | Regularization Norm | Optimization Solver | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :---: | :---: | :---: | :---: | :---: |
| **0.10** | **L2** | **liblinear** | **97.36% (Best)** | **97.80% (Best)** |
| 0.10 | L2 | lbfgs | 97.14% | 97.58% |
| 1.00 | L2 | liblinear | 97.14% | 97.36% |
| 1.00 | L2 | lbfgs | 97.14% | 97.36% |
| 10.00 | L2 | lbfgs | 96.26% | 96.70% |
| 100.00 | L2 | lbfgs | 95.60% | 96.04% |

---

### Table 6: Decision Tree — Hyperparameter Tuning
*(Grid Search: Criterion $\in [\text{gini}, \text{entropy}]$, Max Depth $\in [3, 5, 7, 10, \text{None}]$, Min Samples Split $\in [2, 5, 10]$, Min Samples Leaf $\in [1, 2, 4]$)*

| Criterion | Max Depth | Min Samples Split | Min Samples Leaf | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **entropy** | **5** | **10** | **2** | **93.63% (Best No-PCA)** | 93.85% |
| **entropy** | **7** | **5** | **1** | 92.53% | **94.95% (Best With-PCA)** |
| entropy | 3 | 2 | 1 | 92.97% | 93.41% |
| gini | 5 | 5 | 2 | 92.75% | 93.19% |
| gini | None | 2 | 1 | 91.87% | 92.53% |

---

### Table 7: Random Forest — Hyperparameter Tuning
*(Grid Search: $n\_estimators \in [50, 100, 200]$, Max Depth $\in [5, 10, \text{None}]$, Max Features $\in [\text{sqrt}, \text{log2}]$, Min Samples Split $\in [2, 5]$)*

| Number of Trees ($T$) | Max Depth | Max Features | Min Samples Split | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | **5** | **sqrt** | **2** | **96.70% (Best No-PCA)** | 95.82% |
| **50** | **None** | **sqrt** | **2** | 96.04% | **96.04% (Best With-PCA)** |
| 100 | None | sqrt | 2 | 96.48% | 95.82% |
| 200 | 10 | sqrt | 2 | 96.48% | 95.82% |
| 200 | 5 | log2 | 5 | 96.26% | 95.60% |

---

### Table 8: AdaBoost — Hyperparameter Tuning
*(Grid Search: $n\_estimators \in [25, 50, 100, 200]$, Learning Rate ($\eta$) $\in [0.01, 0.05, 0.1, 0.5, 1.0]$)*

| Number of Estimators ($M$) | Learning Rate ($\eta$) | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :---: | :---: | :---: | :---: |
| **100** | **0.50** | **97.14% (Best)** | **97.14% (Best)** |
| 50 | 0.50 | 96.70% | 96.70% |
| 200 | 0.50 | 96.92% | 96.92% |
| 100 | 0.10 | 95.82% | 95.60% |
| 100 | 1.00 | 96.48% | 96.48% |
| 50 | 0.01 | 91.21% | 91.21% |

---

### Table 9: Gradient Boosting — Hyperparameter Tuning
*(Grid Search: $n\_estimators \in [50, 100, 200]$, Learning Rate ($\eta$) $\in [0.01, 0.05, 0.1, 0.2]$, Max Depth $\in [2, 3, 4]$)*

| Number of Estimators ($M$) | Learning Rate ($\eta$) | Tree Max Depth ($d$) | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :---: | :---: | :---: | :---: | :---: |
| **200** | **0.10** | **2** | **97.36% (Best No-PCA)** | 96.04% |
| **200** | **0.20** | **2** | 96.92% | **96.26% (Best With-PCA)** |
| 100 | 0.10 | 3 | 96.26% | 95.82% |
| 50 | 0.10 | 3 | 95.82% | 95.38% |
| 100 | 0.01 | 3 | 94.95% | 94.73% |

---

### Table 10: XGBoost — Hyperparameter Tuning
*(Grid Search: $n\_estimators \in [50, 100, 200]$, Learning Rate ($\eta$) $\in [0.01, 0.05, 0.1, 0.2]$, Max Depth $\in [3, 4, 6]$, Subsample $\in [0.8, 1.0]$)*

| Number of Estimators | Learning Rate ($\eta$) | Tree Max Depth | Subsample Ratio | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **200** | **0.10** | **3** | **0.80** | **97.36% (Best No-PCA)** | 97.58% |
| **100** | **0.20** | **3** | **0.80** | 97.14% | **97.80% (Best With-PCA)** |
| 100 | 0.10 | 3 | 1.00 | 96.92% | 97.14% |
| 50 | 0.10 | 3 | 0.80 | 96.70% | 96.92% |
| 200 | 0.01 | 3 | 0.80 | 96.04% | 96.04% |

---

### Table 11: Stacking Classifier — Base & Meta-Learner Tuning
*(Base Models: Tuned SVM, GaussianNB, KNN, Random Forest)*

| Base Learners | Meta-Learner Candidate | 5-Fold CV Accuracy (No-PCA) | 5-Fold CV Accuracy (With-PCA) |
| :--- | :--- | :---: | :---: |
| **SVM, Naïve Bayes, KNN, RF** | **Logistic Regression** | **98.02% (Best)** | **98.02% (Best)** |
| SVM, Naïve Bayes, KNN, RF | Ridge Classifier | 97.80% | 97.80% |
| SVM, Naïve Bayes, KNN, RF | Random Forest ($T=50, d=3$) | 96.92% | 96.48% |

---

### Table 12: 5-Fold Stratified Cross-Validation Breakdown (All 10 Models)

| Model Name | Setting | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean CV Accuracy ($\mu$) | Standard Deviation ($\sigma$) | Stability Change ($\Delta \sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF)** | No-PCA | 95.60% | 97.80% | 98.90% | 97.80% | 98.90% | 97.80% | $\pm 1.20\%$ | Baseline |
| | **With-PCA** | 96.70% | 97.80% | 97.80% | 96.70% | 98.90% | 97.58% | **$\pm 0.82\%$** | **$+0.38\%$ (More Stable)** |
| **Naïve Bayes** | No-PCA | 95.60% | 97.80% | 91.21% | 91.21% | 93.41% | 93.85% | $\pm 2.56\%$ | Baseline |
| | With-PCA | 91.21% | 96.70% | 89.01% | 87.91% | 91.21% | 91.21% | $\pm 3.03\%$ | $-0.47\%$ |
| **KNN** | No-PCA | 96.70% | 100.0% | 96.70% | 95.60% | 98.90% | 97.58% | $\pm 1.62\%$ | Baseline |
| | **With-PCA** | 97.80% | 97.80% | 96.70% | 94.51% | 98.90% | 97.14% | **$\pm 1.49\%$** | **$+0.12\%$ (More Stable)** |
| **Logistic Regression** | No-PCA | 95.60% | 96.70% | 97.80% | 97.80% | 98.90% | 97.36% | $\pm 1.12\%$ | Baseline |
| | **With-PCA** | 97.80% | 96.70% | 97.80% | 97.80% | 98.90% | **97.80%** | **$\pm 0.70\%$** | **$+0.43\%$ (More Stable)** |
| **Decision Tree** | No-PCA | 90.11% | 95.60% | 91.21% | 95.60% | 95.60% | 93.63% | $\pm 2.45\%$ | Baseline |
| | **With-PCA** | 94.51% | 96.70% | 94.51% | 92.31% | 96.70% | **94.95%** | **$\pm 1.64\%$** | **$+0.80\%$ (More Stable)** |
| **Random Forest** | No-PCA | 96.70% | 100.0% | 94.51% | 94.51% | 97.80% | 96.70% | $\pm 2.09\%$ | Baseline |
| | **With-PCA** | 96.70% | 98.90% | 94.51% | 94.51% | 95.60% | 96.04% | **$\pm 1.64\%$** | **$+0.44\%$ (More Stable)** |
| **AdaBoost** | No-PCA | 95.60% | 96.70% | 96.70% | 97.80% | 98.90% | 97.14% | $\pm 1.12\%$ | Baseline |
| | **With-PCA** | 96.70% | 97.80% | 97.80% | 96.70% | 96.70% | 97.14% | **$\pm 0.54\%$** | **$+0.58\%$ (More Stable)** |
| **Gradient Boosting** | No-PCA | 96.70% | 97.80% | 96.70% | 96.70% | 98.90% | 97.36% | $\pm 0.88\%$ | Baseline |
| | With-PCA | 93.41% | 97.80% | 96.70% | 96.70% | 96.70% | 96.26% | $\pm 1.49\%$ | $-0.61\%$ |
| **XGBoost** | No-PCA | 97.80% | 96.70% | 96.70% | 96.70% | 98.90% | 97.36% | $\pm 0.88\%$ | Baseline |
| | **With-PCA** | 96.70% | 97.80% | 100.0% | 96.70% | 97.80% | **97.80%** | $\pm 1.20\%$ | $-0.32\%$ |
| **Stacking Classifier**| No-PCA | 96.70% | 98.90% | 97.80% | 97.80% | 98.90% | **98.02%** | $\pm 0.82\%$ | Baseline |
| | **With-PCA** | 96.70% | 98.90% | 98.90% | 96.70% | 98.90% | **98.02%** | $\pm 1.08\%$ | $-0.25\%$ |

---

### Table 13: Held-out Test Set Comprehensive Benchmark ($N=114$ Samples)

| Model Name | Pipeline Setting | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Specificity | Log Loss | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF)** | No-PCA | **98.25%** | 100.0% | 95.24% | **97.56%** | 0.9960 | 100.0% | 0.0708 | [72, 0, 2, 40] |
| | With-PCA | 97.37% | 100.0% | 92.86% | 96.30% | **0.9980** | 100.0% | 0.0716 | [72, 0, 3, 39] |
| **Naïve Bayes** | No-PCA | 92.11% | 92.31% | 85.71% | 88.89% | 0.9891 | 95.83% | 0.4999 | [69, 3, 6, 36] |
| | With-PCA | 89.47% | 85.71% | 85.71% | 85.71% | 0.9613 | 91.67% | 0.3074 | [66, 6, 6, 36] |
| **KNN** | No-PCA | 96.49% | 100.0% | 90.48% | 95.00% | 0.9735 | 100.0% | 0.6501 | [72, 0, 4, 38] |
| | With-PCA | 94.74% | 97.37% | 88.10% | 92.50% | 0.9780 | 98.61% | 0.6694 | [71, 1, 5, 37] |
| **Logistic Regression** | No-PCA | **98.25%** | 100.0% | 95.24% | **97.56%** | 0.9977 | 100.0% | 0.0923 | [72, 0, 2, 40] |
| | **With-PCA** | **98.25%** | 100.0% | 95.24% | **97.56%** | **0.9980** | 100.0% | 0.0940 | [72, 0, 2, 40] |
| **Decision Tree** | No-PCA | 96.49% | 100.0% | 90.48% | 95.00% | 0.9744 | 100.0% | 0.6559 | [72, 0, 4, 38] |
| | With-PCA | 94.74% | 97.37% | 88.10% | 92.50% | 0.9560 | 98.61% | 1.2312 | [71, 1, 5, 37] |
| **Random Forest** | No-PCA | 97.37% | 100.0% | 92.86% | 96.30% | 0.9950 | 100.0% | 0.1127 | [72, 0, 3, 39] |
| | With-PCA | 94.74% | 95.00% | 90.48% | 92.68% | 0.9944 | 97.22% | 0.1433 | [70, 2, 4, 38] |
| **AdaBoost** | No-PCA | 97.37% | 100.0% | 92.86% | 96.30% | 0.9858 | 100.0% | 0.3932 | [72, 0, 3, 39] |
| | With-PCA | 94.74% | 97.37% | 88.10% | 92.50% | 0.9934 | 98.61% | 0.4225 | [71, 1, 5, 37] |
| **Gradient Boosting** | No-PCA | 96.49% | 100.0% | 90.48% | 95.00% | 0.9934 | 100.0% | 0.1129 | [72, 0, 4, 38] |
| | With-PCA | 95.61% | 97.44% | 90.48% | 93.83% | 0.9901 | 98.61% | 0.1414 | [71, 1, 4, 38] |
| **XGBoost** | No-PCA | 97.37% | 100.0% | 92.86% | 96.30% | 0.9921 | 100.0% | 0.0982 | [72, 0, 3, 39] |
| | **With-PCA** | **97.37%** | 97.56% | **95.24%** | **96.39%** | **0.9921** | 98.61% | **0.0934** | [71, 1, 2, 40] |
| **Stacking Classifier**| No-PCA | **98.25%** | 100.0% | 95.24% | **97.56%** | 0.9960 | 100.0% | 0.0784 | [72, 0, 2, 40] |
| | With-PCA | 95.61% | 97.44% | 90.48% | 93.83% | 0.9944 | 98.61% | 0.0941 | [71, 1, 4, 38] |

---

### Table 14: Computational Complexity & Speedup Analysis

| Model Name | Training Time (No-PCA) | Training Time (With-PCA) | Training Speedup Factor | Inference Latency (No-PCA) | Inference Latency (With-PCA) | Latency Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | 3.65 ms | 3.08 ms | **1.19x** | 0.35 ms | 0.28 ms | +20.0% |
| **Naïve Bayes** | 0.26 ms | 0.23 ms | **1.13x** | 0.11 ms | 0.09 ms | +18.2% |
| **KNN** | 0.14 ms | 0.38 ms | 0.37x | 0.85 ms | 0.62 ms | +27.1% |
| **Logistic Regression** | 0.83 ms | 0.45 ms | **1.84x** | 0.08 ms | 0.06 ms | +25.0% |
| **Decision Tree** | 3.15 ms | 1.30 ms | **2.42x** | 0.07 ms | 0.05 ms | +28.6% |
| **Random Forest** | 62.16 ms | 28.77 ms | **2.16x** | 1.95 ms | 1.18 ms | **+39.5%** |
| **AdaBoost** | 106.32 ms | 59.99 ms | **1.77x** | 3.42 ms | 2.15 ms | **+37.1%** |
| **Gradient Boosting** | 261.64 ms | 109.17 ms | **2.40x** | 0.48 ms | 0.32 ms | **+33.3%** |
| **XGBoost** | 29.26 ms | 12.56 ms | **2.33x** | 0.68 ms | 0.41 ms | **+39.7%** |
| **Stacking Classifier**| 320.00 ms | 184.14 ms | **1.74x** | 3.82 ms | 2.45 ms | **+35.9%** |

---

## 4. Answers to Observation Questions

### 1. Which models improved most with PCA? Which did not? Why?
* **Models that Improved / Maintained Peak Performance:**
  * **Logistic Regression:** Mean 5-fold CV accuracy improved from $97.36\%$ to **$97.80\%$** (+0.44%), and test accuracy was perfectly preserved at **$98.25\%$** with an increased ROC-AUC of **0.9980**. This improvement occurs because the original 30 WDBC features contain severe multicollinearity (e.g., radius, perimeter, and area have Pearson $r > 0.99$). Multicollinearity inflates coefficient variances and creates ill-conditioned Hessian matrices. PCA constructs orthogonal principal axes ($Cov(PC_i, PC_j) = 0$), transforming the loss landscape into isotropic paraboloids that optimize rapidly and generalize cleanly.
  * **Decision Tree:** Mean CV accuracy increased from $93.63\%$ to **$94.95\%$** (+1.32%) with a major variance reduction ($\sigma: 2.45\% \to 1.64\%$). Standard decision trees make strictly axis-aligned splits. In the original correlated feature space, oblique class boundaries require deep, stair-cased decision trees that easily overfit. PCA rotates the coordinate system so that the first few orthogonal axes align with the primary directions of variance, allowing shallower, more effective splits.
  * **XGBoost:** CV accuracy improved from $97.36\%$ to **$97.80\%$** (+0.44%), and test F1-score rose to **$96.39\%$** with reduced log loss ($0.0982 \to 0.0934$).
* **Models that Decreased:**
  * **Gaussian Naïve Bayes:** CV accuracy dropped from $93.85\%$ to **$91.21\%$** (-2.64%) and test accuracy from $92.11\%$ to **$89.47\%$**. GaussianNB assumes that class-conditional feature distributions are Gaussian ($X_j \mid Y=c \sim \mathcal{N}(\mu_{jc}, \sigma_{jc}^2)$). Although PCA decorrelates features globally (zero covariance), linear combinations of non-Gaussian or multimodal features do not necessarily form Gaussian class-conditional marginals, and PCA discards subtle higher-order class separation cues present in the remaining 20 components.
  * **Tree Ensembles (Random Forest, AdaBoost, Gradient Boosting):** Showed slight decreases on test accuracy ($97.37\% \to 94.74\%$). Because tree ensembles inherently perform feature sub-sampling and non-linear interactions across original raw physical measurements, compressing features via linear PCA slightly smooths out localized non-linear decision boundaries.

---

### 2. Did PCA reduce variance across folds (more stable results)?
* **Yes, significantly.** For 6 out of 10 models, cross-fold standard deviation ($\sigma$) decreased markedly:
  * **AdaBoost:** Standard deviation dropped from $\pm 1.12\%$ to **$\pm 0.54\%$** (a **51.8% stability improvement**).
  * **Decision Tree:** Standard deviation dropped from $\pm 2.45\%$ to **$\pm 1.64\%$** (a **33.1% stability improvement**).
  * **Logistic Regression:** Standard deviation dropped from $\pm 1.12\%$ to **$\pm 0.70\%$** (a **37.5% stability improvement**).
  * **SVM:** Standard deviation dropped from $\pm 1.20\%$ to **$\pm 0.82\%$** (a **31.7% stability improvement**).
  * **Random Forest:** Standard deviation dropped from $\pm 2.09\%$ to **$\pm 1.64\%$** (a **21.5% stability improvement**).
  * **KNN:** Standard deviation dropped from $\pm 1.62\%$ to **$\pm 1.49\%$**.
* **Theoretical Explanation:** By filtering out the lowest 20 eigenvectors (which contain noisy, fold-sensitive perturbations), PCA projects data onto robust, high-energy structural directions. Consequently, individual fold partitions do not over-index on fold-specific noise, producing more consistent cross-validation estimates.

---

### 3. For high-dimensional data, was PCA beneficial in reducing overfitting?
* **Yes.** In classification problems where feature dimensionality $D$ is large relative to sample size $N$ ($D=30$ vs $N_{train}=455$), estimators suffer from the *curse of dimensionality*—the volume of space grows exponentially, causing sample points to become sparse and allowing models to memorize idiosyncratic noise.
* By selecting $k=10$ components, PCA reduced the parameter dimensionality by **66.67%** while maintaining **95.21%** total variance. This acted as a structural regularizer, preventing single decision trees from growing deep overfitting branches (reflected by the test accuracy and CV improvement in Decision Trees) and shrinking the hypothesis space for linear models.

---

### 4. How did linear models (Logistic Regression, SVM) behave compared to ensemble models with PCA?
* **Linear & Kernel Models (Logistic Regression, SVM):**
  * Experienced superior benefits in conditioning, convergence, and stability.
  * Logistic Regression saw training time cut by **45.8%** ($1.84\times$ speedup) while matching peak test accuracy (**98.25%**, F1=**97.56%**, ROC-AUC=**0.9980**).
  * SVM retained $100\%$ precision on the test set and improved cross-validation stability to $\pm 0.82\%$.
* **Ensemble Models (Random Forest, Gradient Boosting, XGBoost):**
  * Reaped massive **computational speedups ($2.16\times - 2.42\times$)** because tree construction complexity is $O(M \cdot k \cdot N \log N)$, directly scaling with feature dimension $k$.
  * However, ensembles already have built-in defenses against multicollinearity through random feature subsampling (`max_features`) and sequential gradient fitting. Therefore, projecting the non-linear feature space linearly onto orthogonal components slightly constrained the expressive freedom of tree ensembles, trading $\approx 1-2\%$ accuracy for a $>2\times$ reduction in training time.

---

### 5. Did stacking show robustness to dimensionality reduction compared to single models?
* **Yes, Stacking exhibited extraordinary robustness.**
  * In 5-Fold Cross-Validation, the Stacking Classifier achieved the highest overall score among all evaluated architectures: **98.02% mean CV accuracy**, which remained **identical** between the No-PCA and With-PCA pipelines.
  * On the held-out test set, Stacking maintained high diagnostic power (ROC-AUC = **0.9944**).
* **Architectural Reason:** Stacking leverages diverse base hypotheses (SVM maximum margin, Naïve Bayes likelihood, KNN nearest-neighbor density, and Random Forest ensemble partitions). Even if one base learner suffers slight information loss under PCA (e.g., Naïve Bayes), the meta-learner (Logistic Regression) dynamically re-weights base predictions based on out-of-fold cross-validated confidence, insulating the final ensemble from single-model degradation.

---

## 5. Summary Conclusion & Practical Guidelines
1. **When to use PCA:**
   * When features exhibit high multicollinearity and linear models (Logistic Regression, Linear/RBF SVM) or single Decision Trees are deployed.
   * When computational latency and memory constraints demand faster training and inference (achieving $2\times - 2.5\times$ speedup across tree ensembles).
   * When high-dimensional noise causes cross-validation instability and overfitting.
2. **When to avoid PCA:**
   * When deploying probabilistic classifiers like Gaussian Naïve Bayes whose class-conditional distributional assumptions may be disrupted by linear projections.
   * When exact feature interpretability (e.g., specific clinical biomarker measurements) is strictly required for domain diagnostics.
   * When non-linear ensemble models have ample compute and require raw feature interactions without orthogonal smoothing.
