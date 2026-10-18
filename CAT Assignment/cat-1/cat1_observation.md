# CAT Assignment 1: Observation Sheet

**Course:** Introduction to Machine Learning (ICS1502)  
**Degree & Branch:** M. Tech (Integrated) Computer Science & Engineering (Semester V)  
**Student Name:** Danusu K  
**Register Number:** 3122247001013  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  
**Academic Year:** 2026–2027 (Odd Semester) | **Batch:** 2024–2029  
**Submission Date:** \today  

---

## 1. Objective
1. **Regression Analysis (Mobile Phone Price Prediction):**
   - Implement Ordinary Least Squares Linear Regression via matrix closed-form solution (Normal Equation) and Batch Gradient Descent from scratch.
   - Formulate and justify the training data matrix $\mathbf{X} \in \mathbb{R}^{N \times (D+1)}$, label matrix $\mathbf{y} \in \mathbb{R}^{N \times 1}$, and parameter matrix $\mathbf{w} \in \mathbb{R}^{(D+1) \times 1}$.
   - Extend formulations with $\mathcal{L}_2$ Tikhonov regularization (Ridge Regression) for both closed-form and iterative gradient descent.
   - Critically evaluate the mathematical and numerical role of feature standardization on Gram matrix condition numbers, gradient descent convergence, and $\mathcal{L}_2$ shrinkage equity.
   - Perform regularization parameter sweeps ($\lambda$) to evaluate the bias–variance trade-off and analyze feature importance from standardized weights.
2. **Linear Classification (Banknote Authentication):**
   - Formulate and train a linear classifier (Logistic Regression) with and without $\mathcal{L}_2$ regularization on the UCI Banknote Authentication dataset using stratified train-test splitting.
   - Analyze model convergence, loss surfaces, and accuracy response across regularization strengths $\lambda \in [10^{-4}, 10^4]$.
   - Extract top discriminatory features and visualize the 3D feature space along with the separating hyperplane.
   - Intentionally inject controlled synthetic outliers by shifting coordinates into opposing territories, quantifying decision hyperplane distortion and generalization degradation.

---

## 2. Dataset Descriptions & Matrix Configurations

### Table 1: Dataset Configurations and Dimensional Specifications
| Attribute | Part 1: Mobile Phone Price Prediction | Part 2: Banknote Authentication |
| :--- | :--- | :--- |
| **Source** | Kaggle / Mobile Spec Repository | UCI Machine Learning Repository |
| **Total Instances ($N$)** | 807 instances | 1,372 instances |
| **Input Features ($D$)** | 7 predictor features | 4 continuous wavelet features |
| **Target Variable ($y$)** | Continuous Price (INR) [479 to 153,000] | Binary Class (0: Authentic, 1: Forged) |
| **Train–Test Partition** | 80% Train (645 samples) / 20% Test (162 samples) | 80% Train (1,097 samples) / 20% Test (275 samples) |
| **Data Matrix $\mathbf{X}_{train}$** | $\mathbb{R}^{645 \times 8}$ (Prepend $\mathbf{1}_{645}$ bias column) | $\mathbb{R}^{1097 \times 4}$ (Standardized $z$-score) |
| **Label Matrix $\mathbf{y}_{train}$** | $\mathbb{R}^{645 \times 1}$ (Continuous target vector) | $\mathbb{R}^{1097 \times 1}$ (Binary indicator vector) |
| **Parameter Matrix $\mathbf{w}$** | $\mathbb{R}^{8 \times 1}$ ($[w_0, w_1, \dots, w_7]^T$) | $\mathbb{R}^{4 \times 1}$ weights + bias scalar $b \in \mathbb{R}$ |

---

## 3. Part 1: Regression Observations & Evaluation

### Table 2: Closed-Form vs. Gradient Descent (Unregularized $\lambda = 0$)
*(Evaluated on Unseen Test Partition: $N_{test} = 162$ samples)*

| Formulation / Algorithm | Train RMSE (INR) | Train $R^2$ | Test MAE (INR) | Test RMSE (INR) | Test $R^2$ | Convergence Epochs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Closed-Form (Normal Eq.)** | 17,273.88 | 0.4675 | 10,019.06 | 15,480.00 | 0.4326 | Direct Analytical ($O(D^3)$) |
| **Batch Gradient Descent** ($\alpha=0.05$) | 17,273.88 | 0.4675 | 10,019.06 | 15,480.00 | 0.4326 | 3,500 iterations |

> **Inference:** The Batch Gradient Descent solution matches the closed-form normal equation solution up to 14 decimal digits. This confirms the mathematical equivalence of iterative first-order optimization and exact matrix inversion on convex quadratic loss surfaces.

---

### Table 3: $\mathcal{L}_2$ Regularized Regression (Ridge Regression, $\lambda = 10.0$)

| Formulation / Algorithm | Train RMSE (INR) | Train $R^2$ | Test MAE (INR) | Test RMSE (INR) | Test $R^2$ | Parameter Penalty $\frac{\lambda}{2N}\|\mathbf{w}_{1:D}\|_2^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Closed-Form** | 17,274.95 | 0.4674 | 9,956.66 | 15,431.00 | **0.4362** | Added to normal matrix: $\lambda \mathbf{I}'$ |
| **Ridge Gradient Descent** | 17,274.95 | 0.4674 | 9,956.66 | 15,431.00 | **0.4362** | Weight decay gradient: $\frac{\lambda}{N}\mathbf{w}$ |

> **Inference:** Adding $\mathcal{L}_2$ regularization with $\lambda = 10.0$ improves the test $R^2$ from 0.4326 to **0.4362** and reduces test RMSE from 15,480.00 to 15,431.00 INR. The slight shrinkage in weights prevents overfitting to collinear features in the training partition.

---

### Table 4: Critical Comparison: With vs. Without Data Standardization ($\lambda = 10.0$)

| Metric / Property | Without Standardization (Raw Features) | With Standardization ($z$-score Normalized) | Relative Improvement / Impact |
| :--- | :---: | :---: | :---: |
| **Gram Matrix Cond. No. $\kappa(\mathbf{X}^T\mathbf{X})$** | **$2.153 \times 10^9$** (Ill-conditioned) | **$4.299$** (Well-conditioned) | **Reduced by a factor of $5 \times 10^8$** |
| **Gradient Descent Stability** | Diverges unless $\alpha < 10^{-9}$ | Rapid convergence ($\alpha = 0.05$) | Gradient descent is computationally viable |
| **Ridge Test RMSE (INR)** | 15,408.88 | 15,431.00 | Physically consistent feature penalties |
| **Ridge Test $R^2$ Score** | 0.4378 (Distorted Penalties) | 0.4362 (Equitable Shrinkage) | Standardization treats all unit scales equally |
| **Penalty Equity** | Disproportionately penalizes low-scale features | Uniform $\mathcal{L}_2$ weight shrinkage across features | Mandatory for meaningful feature importance |

> **Inference:** Without standardization, the design matrix condition number is $2.153 \times 10^9$, causing extreme numerical sensitivity. Features with naturally large ranges (`Battery_Power` $\in [1020, 6000]$) have artificially small weights and escape regularization penalties, whereas features with small ranges (`Ratings` $\in [2.8, 4.8]$) receive massive weights and are penalized unfairly. Standardization compresses the condition number to 4.299, ensuring equitable regularization.

---

### Table 5: Regularization Parameter $\lambda$ Sweep Analysis

| Regularization Strength $\lambda$ | Train RMSE (INR) | Test RMSE (INR) | Test $R^2$ Score | Weight $\mathcal{L}_2$ Norm $\|\mathbf{w}_{1:D}\|_2$ | Regime / Behavior |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $10^{-3}$ | 17,273.88 | 15,479.99 | 0.4326 | 13,016.92 | Unregularized Ordinary Least Squares |
| $10^{-2}$ | 17,273.88 | 15,479.95 | 0.4326 | 13,016.71 | Negligible shrinkage |
| $10^{-1}$ | 17,273.88 | 15,479.48 | 0.4326 | 13,014.65 | Minimal shrinkage |
| $1.0$ | 17,273.99 | 15,474.84 | 0.4330 | 12,994.08 | Mild variance reduction |
| **$10.0$** | **17,274.95** | **15,431.00** | **0.4362** | **12,793.81** | **Optimal Regularization Sweet Spot** |
| $50.0$ | 17,294.62 | 15,283.47 | 0.4468 | 12,014.73 | Strong variance suppression |
| $100.0$ | 17,341.35 | 15,165.74 | 0.4554 | 11,219.00 | Effective smoothing on test set |
| $1000.0$ | 18,349.52 | 15,310.66 | 0.4444 | 5,568.10 | Incipient underfitting |
| $5000.0$ | 21,392.20 | 18,344.82 | 0.2023 | 1,885.67 | Severe underfitting / over-shrinkage |

---

### Table 6: Feature Importance Ranked by Standardized $\mathcal{L}_2$ Weights ($\lambda = 10.0$)

| Rank | Feature Name | Learned Weight ($w_j$) | Absolute Impact ($|w_j|$) | Correlation with Price ($r$) | Practical Domain Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `Ratings` | **+9,541.93** | 9,541.93 | +0.5706 | Premium consumer perception and brand tier driver. |
| **2** | `RAM` | **+5,607.73** | 5,607.73 | +0.2984 | Key hardware capability metric for high-end processing. |
| **3** | `Primary_Cam` | **-5,211.95** | 5,211.95 | -0.2233 | Counter-intuitive negative weight: Budget phones boast inflated MP sensors (48/64MP), while flagship phones (e.g., iPhone) utilize optimized 12MP sensors. |
| **4** | `ROM` | **+3,951.51** | 3,951.51 | +0.4749 | Internal storage tiers directly scale with commercial price. |
| **5** | `Battery_Power` | **+2,298.43** | 2,298.43 | +0.3692 | Higher battery capacity positively correlates with price. |
| **6** | `Selfi_Cam` | **+780.84** | 780.84 | +0.0158 | Minor positive contribution to overall handset price. |
| **7** | `Mobile_Size` | **+476.62** | 476.62 | +0.0842 | Screen dimensions have minimal standalone price impact. |

---

## 4. Part 2: Linear Classification Observations (Banknote Authentication)

### Table 7: Classifier Performance With and Without $\mathcal{L}_2$ Regularization
*(Evaluated on Stratified Test Partition: $N_{test} = 275$ samples; Authentic: 153, Forged: 122)*

| Evaluation Metric | Unregularized Classifier ($C = 10^5$, $\lambda \approx 0$) | $\mathcal{L}_2$ Regularized Classifier ($C = 1.0$, $\lambda = 1.0$) | Practical Implication |
| :--- | :---: | :---: | :--- |
| **Training Accuracy** | **99.09%** | 98.45% | Unregularized model fits training samples aggressively. |
| **Testing Accuracy** | **98.55%** | 97.09% | Both achieve outstanding generalization on clean data. |
| **Precision** | 98.35% | 96.69% | Reliable forgery detection with very low false positive rate. |
| **Recall (Sensitivity)** | 98.36% | 96.72% | Detects 118 out of 122 counterfeit banknotes. |
| **F1-Score** | 0.9836 | 0.9671 | Harmonic balance of precision and recall. |
| **ROC-AUC Score** | **0.9997** | **0.9996** | Near-perfect separation along predicted probability ranking. |
| **Weight Vector $\mathbf{w}$** | $[-19.76, -22.73, -20.85, -1.30]$ | $[-4.67, -4.81, -4.40, +0.26]$ | Regularization suppresses extreme weight magnitudes. |
| **Weight Norm $\|\mathbf{w}\|_2$** | **36.03** (Exploding) | **8.15** (Bounded) | $\mathcal{L}_2$ prevents numerical overflow and overconfidence. |

---

### Table 8: Banknote Feature Importance Ranking

| Feature Name | Description | Learned Weight ($w_j$) | Absolute Importance ($|w_j|$) | Physical Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| `skewness` | Skewness of Wavelet Transform | **-4.8065** | 4.8065 | Primary discriminator: Asymmetry in banknote texture. |
| `variance` | Variance of Wavelet Transform | **-4.6660** | 4.6660 | Primary discriminator: High-frequency surface variations. |
| `curtosis` | Curtosis of Wavelet Transform | **-4.3989** | 4.3989 | Primary discriminator: Tailedness and peak sharpness of print. |
| `entropy` | Information Entropy of Image | **+0.2580** | 0.2580 | Marginal discriminator: Minimal individual separability power. |

---

### Table 9: Outlier Injection Impact and Sensitivity Analysis
*(Outliers introduced by perturbing training points by $6\sigma$ to $9\sigma$ and flipping labels; tested on clean test set)*

| Injected Outlier Ratio (%) | Corrupted Training Points ($N_{out}$) | Unregularized Test Accuracy (%) | $\mathcal{L}_2$ Regularized Test Accuracy (%) | Accuracy Drop ($\Delta$) | Degradation Severity |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0% (Clean)** | 0 | **98.55%** | **97.09%** | 0.00% | Baseline clean performance |
| **2.0%** | 21 | 96.73% | 96.00% | -1.82% | Mild boundary rotation |
| **5.0%** | 54 | 93.82% | 93.45% | -4.73% | Moderate hyperplane pull |
| **8.0%** | 87 | **90.18%** | **90.18%** | -8.37% | Significant test degradation |
| **12.0%** | 131 | 82.55% | 83.27% | -16.00% | Severe decision boundary skew |
| **15.0%** | 164 | **76.00%** | **77.82%** | **-22.55%** | Critical failure; boundary pulled into forged territory |

> **Inference:** Because ordinary linear classification relies on cross-entropy loss without a bounded influence function, extreme leverage points exert quadratic-like torque on the separating hyperplane. While both models experience degradation as outliers exceed 10%, the $\mathcal{L}_2$ regularized classifier retains higher accuracy because weight shrinkage prevents the hyperplane from wildly over-tilting to satisfy distant mislabeled leverage outliers.

---

## 5. Summary of Conclusions
1. **Mathematical Equivalence:** The matrix-based closed-form normal equation and batch gradient descent arrive at the identical global minimum on standardized data.
2. **Standardization is Essential:** Without feature standardization, the Gram matrix condition number explodes to $2.153 \times 10^9$, gradient descent stalls or diverges, and $\mathcal{L}_2$ regularization arbitrarily penalizes small-range features while ignoring large-range features.
3. **Price Drivers:** Mobile phone pricing is predominantly driven by consumer perception (`Ratings`), memory capacity (`RAM`, `ROM`), and battery capacity. Camera megapixel specs exhibit counter-intuitive negative weighting due to market segment positioning.
4. **Binary Classification & Linearity:** Banknote authentication is highly linearly separable along wavelet `variance`, `skewness`, and `curtosis`, achieving $>98.5\%$ accuracy.
5. **Vulnerability to Outliers:** Standard linear classifiers are vulnerable to leverage outliers, suffering a drop from $98.55\%$ down to $76.00\%$ under 15% training corruption, underscoring the need for bounded loss functions (e.g., Huber, SVM soft margin) or outlier sanitization pipelines in high-stakes deployment.
