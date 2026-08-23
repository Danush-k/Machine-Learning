# Experiment 6: Observation Sheet

**Experiment Title:** Bagging, Boosting, and Stacked Ensemble Models  
**Course:** Machine Learning Algorithms Laboratory (ICS1512)  
**Student Name:** Danusu K  
**Register Number:** 3122247001013  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  

---

## Aim
To implement and evaluate **Bagging**, **Boosting** (AdaBoost and Gradient Boosting), and **Stacked Ensemble** classifiers on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, perform hyperparameter tuning using 5-Fold Stratified Cross-Validation, compare model performance across standard classification metrics, and analyze the empirical impact of ensemble paradigms on the bias–variance trade-off.

---

## Observation Tables

### Table 1: Bagging Hyperparameter Evaluation
*(Base Estimator: Decision Tree, $K=5$ Fold Stratified Cross-Validation)*

| $n\_estimators$ | $max\_samples$ | Avg CV Accuracy (%) | Avg CV F1 Score |
| :---: | :---: | :---: | :---: |
| 5 | 0.40 | 94.73% | 0.9261 |
| 5 | 0.60 | 94.95% | 0.9312 |
| 5 | 0.80 | 94.51% | 0.9262 |
| 5 | 1.00 | 95.82% | 0.9435 |
| 10 | 0.40 | 95.38% | 0.9352 |
| 10 | 0.60 | 94.95% | 0.9285 |
| 10 | 0.80 | 94.51% | 0.9226 |
| 10 | 1.00 | 96.04% | 0.9459 |
| **25** | **1.00** | **97.14%** | **0.9621** |
| 25 | 0.80 | 96.92% | 0.9585 |
| 25 | 0.60 | 96.48% | 0.9525 |
| 25 | 0.40 | 95.82% | 0.9425 |
| 50 | 0.80 | 96.48% | 0.9526 |
| 50 | 1.00 | 96.04% | 0.9467 |
| 50 | 0.40 | 96.04% | 0.9454 |
| 50 | 0.60 | 95.82% | 0.9424 |
| 100 | 0.40 | 96.26% | 0.9490 |
| 100 | 1.00 | 95.82% | 0.9431 |
| 100 | 0.80 | 95.60% | 0.9399 |
| 100 | 0.60 | 95.38% | 0.9370 |
| 150 | 1.00 | 96.26% | 0.9490 |
| 150 | 0.40 | 96.04% | 0.9459 |
| 150 | 0.80 | 95.82% | 0.9434 |
| 150 | 0.60 | 95.82% | 0.9431 |

---

### Table 2: Boosting Hyperparameter Evaluation
*(Algorithm: Gradient Boosting Classifier, $K=5$ Fold Stratified Cross-Validation)*

| $n\_estimators$ | $learning\_rate$ ($\eta$) | Avg CV Accuracy (%) | Avg CV F1 Score |
| :---: | :---: | :---: | :---: |
| 10 | 0.01 | 92.53% | 0.8906 |
| 10 | 0.05 | 95.16% | 0.9324 |
| 10 | 0.10 | 95.82% | 0.9427 |
| 10 | 0.50 | 95.60% | 0.9392 |
| 10 | 1.00 | 94.73% | 0.9298 |
| 25 | 0.01 | 94.29% | 0.9192 |
| 25 | 0.05 | 96.04% | 0.9463 |
| 25 | 0.10 | 96.26% | 0.9490 |
| 25 | 0.50 | 96.26% | 0.9490 |
| 25 | 1.00 | 95.38% | 0.9388 |
| 50 | 0.01 | 94.95% | 0.9294 |
| 50 | 0.05 | 96.70% | 0.9554 |
| 50 | 0.10 | 96.26% | 0.9496 |
| 50 | 0.50 | 96.70% | 0.9546 |
| 50 | 1.00 | 95.60% | 0.9409 |
| 100 | 0.01 | 95.16% | 0.9324 |
| 100 | 0.05 | 96.48% | 0.9525 |
| **100** | **0.10** | **96.70%** | **0.9553** |
| 100 | 0.50 | 96.70% | 0.9543 |
| 100 | 1.00 | 95.82% | 0.9443 |
| 200 | 0.01 | 95.60% | 0.9392 |
| **200** | **0.05** | **96.92%** | **0.9581** |
| **200** | **0.10** | **96.92%** | **0.9582** |
| 200 | 0.50 | 96.26% | 0.9490 |
| 200 | 1.00 | 95.82% | 0.9443 |

---

### Table 3: Stacked Ensemble Evaluation
*(Heterogeneous Base Learners & Meta-Learner, $K=5$ Fold Cross-Validation)*

| Base Models | Meta Learner | Avg CV Accuracy (%) | Avg CV F1 Score |
| :--- | :--- | :---: | :---: |
| **SVM (RBF), Naïve Bayes, Decision Tree** | **Ridge Classifier** | **96.92%** | **0.9581** |
| **SVM (RBF), Naïve Bayes, Decision Tree** | **Random Forest** | **96.48%** | **0.9520** |
| **SVM (RBF), Decision Tree** | **Logistic Regression** | **96.48%** | **0.9518** |
| **SVM (RBF), Naïve Bayes, Decision Tree** | **Logistic Regression** | **96.26%** | **0.9490** |
| SVM (RBF), Naïve Bayes | Logistic Regression | 95.82% | 0.9429 |
| Naïve Bayes, Decision Tree | Logistic Regression | 93.63% | 0.9122 |

---

### Table 4: Performance Comparison of Ensemble Models
*(Evaluated on Held-out 20% Test Set, $N=114$ Samples)*

| Model | Accuracy (%) | Precision | Recall | F1 Score | ROC-AUC | 5-Fold CV Accuracy ($\mu \pm \sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Baseline)** | 92.98% | 1.0000 | 0.8095 | 0.8947 | 0.9563 | 94.07% $\pm$ 2.15% |
| **Bagging Classifier** | **97.37%** | **1.0000** | **0.9286** | **0.9630** | **0.9901** | **96.48% $\pm$ 1.46%** |
| **AdaBoost Classifier** | 96.49% | 1.0000 | 0.9048 | 0.9500 | 0.9944 | 96.04% $\pm$ 2.26% |
| **Gradient Boosting** | 96.49% | 1.0000 | 0.9048 | 0.9500 | **0.9947** | 96.70% $\pm$ 1.39% |
| **Stacked Ensemble** | 96.49% | 1.0000 | 0.9048 | 0.9500 | **0.9947** | 96.26% $\pm$ 1.79% |

---

### 5. Fold-by-Fold Cross-Validation Performance ($K=5$)

| Fold | Decision Tree | Bagging | AdaBoost | Gradient Boosting | Stacked Ensemble |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 93.41% | 97.80% | 98.90% | 97.80% | 96.70% |
| **Fold 2** | 95.60% | 97.80% | 95.60% | 97.80% | 97.80% |
| **Fold 3** | 90.11% | 94.51% | 93.41% | 94.51% | 93.41% |
| **Fold 4** | 95.60% | 95.60% | 94.51% | 96.70% | 95.60% |
| **Fold 5** | 95.60% | 96.70% | 97.80% | 96.70% | 97.80% |
| **Mean ($\mu$)** | **94.07%** | **96.48%** | **96.04%** | **96.70%** | **96.26%** |
| **Std Dev ($\sigma$)** | **$\pm 2.15\%$** | **$\pm 1.46\%$** | **$\pm 2.26\%$** | **$\pm 1.39\%$** | **$\pm 1.79\%$** |

---

### 6. Bias–Variance Decomposition (100 Bootstrap Iterations)

| Model | 0-1 Bias | Variance | Total Expected Error |
| :--- | :---: | :---: | :---: |
| **Decision Tree (Baseline)** | 0.0439 | 0.0580 | 0.0736 |
| **Bagging Classifier** | 0.0351 | 0.0242 | 0.0500 |
| **AdaBoost Classifier** | 0.0351 | 0.0125 | 0.0357 |
| **Gradient Boosting** | 0.0351 | 0.0187 | 0.0425 |
| **Stacked Ensemble** | 0.0351 | 0.0165 | 0.0377 |

---

## Observation Questions & Detailed Answers

### 1. How does Bagging reduce variance?
* **Theoretical Mechanism:** Bagging (Bootstrap Aggregating) trains $B$ independent base models $h_1, h_2, \dots, h_B$ on bootstrap resamples $D_b \sim D$ drawn uniformly with replacement. When individual base models (e.g., deep unpruned Decision Trees) exhibit high variance $\sigma^2$ and pairwise correlation $\rho$, the variance of their ensemble mean $\bar{h}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B h_b(\mathbf{x})$ is:
  $$\text{Var}(\bar{h}(\mathbf{x})) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
  As $B$ increases, the second term $\frac{1-\rho}{B}\sigma^2 \to 0$, effectively eliminating sample-dependent fluctuation noise.
* **Empirical Validation:** In our experiment, the baseline Decision Tree exhibited a high variance of **0.0580**. Bagging reduced this variance to **0.0242** (a **58.3% variance reduction**), shrinking the cross-validation standard deviation from $\pm 2.15\%$ down to $\pm 1.46\%$ and boosting test accuracy from $92.98\%$ to $97.37\%$.

### 2. How does Boosting address model bias?
* **Theoretical Mechanism:** Boosting (AdaBoost / Gradient Boosting) employs a sequential, adaptive learning scheme where base models (typically shallow trees or stumps with high bias and low variance) are trained sequentially:
  * **AdaBoost** increases the sample weight $w_i^{(m+1)} = w_i^{(m)} \exp(-\alpha_m y_i h_m(\mathbf{x}_i))$ of previously misclassified points, forcing subsequent estimators to specialize in difficult boundary regions.
  * **Gradient Boosting** iteratively models pseudo-residuals $r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}}$, performing gradient descent in function space to minimize empirical loss.
* **Empirical Validation:** Sequential residual minimization successfully reduced model bias from **0.0439** (Decision Tree) to **0.0351**, elevating the diagnostic ROC-AUC to an exceptional **0.9947** and eliminating 4 false-negative diagnostic errors compared to the single decision tree.

### 3. Why does Stacking benefit from heterogeneous models?
* **Complementary Hypothesis Spaces:** Homogeneous ensembles (Bagging and Boosting) combine identical model classes (trees). In contrast, Stacking combines fundamentally distinct functional representations:
  1. **Support Vector Machine (RBF kernel):** Constructs a maximal-margin separating hyperplane in non-linear kernel Hilbert space.
  2. **Gaussian Naïve Bayes:** Evaluates conditional feature probabilities assuming class-conditional Gaussian distributions.
  3. **Decision Tree:** Partitions the feature space orthogonally via axis-parallel splits.
* **Orthogonal Error Distributions:** Because these algorithms make misclassification errors in different regions of the feature space (low inter-model prediction correlation), the meta-learner (Logistic Regression) learns optimal weighting coefficients ($w_{\text{SVM}} = 1.34$, $w_{\text{NB}} = 0.58$, $w_{\text{DT}} = 0.22$) to produce robust out-of-fold generalizations ($96.49\%$ test accuracy, $0.9947$ ROC-AUC).

### 4. Which ensemble method performed best and why?
* **Clinical Performance:** The **Bagging Classifier** achieved the best overall test accuracy (**97.37%**) and F1-score (**0.9630**), correctly classifying 39 out of 42 malignant tumors (Recall = 92.86%, Precision = 100.0%, 0 False Positives, only 3 False Negatives).
* **Diagnostic Discrimination:** **Gradient Boosting** and **Stacked Ensemble** achieved the highest ROC-AUC (**0.9947** and **0.9944**) and lowest bootstrap expected error (**0.0357** – **0.0425**).
* **Conclusion:** For clinical diagnostic deployment on WDBC, **Bagging** is the best model due to its superior sensitivity (minimizing dangerous false negatives) and maximum overall accuracy.

---

## Learning Outcomes
* Mastered the theoretical foundations and implementation differences among **Bagging**, **Boosting**, and **Stacking**.
* Demonstrated how **Bootstrap Aggregation** stabilizes variance in high-capacity non-parametric estimators.
* Analyzed sequential functional gradient descent in **Gradient Boosting** and adaptive reweighting in **AdaBoost** for bias reduction.
* Implemented cross-validated meta-feature generation in **Stacked Ensembles** using heterogeneous base learners.
* Quantified the empirical **Bias–Variance trade-off** and validated model reliability using 5-Fold Stratified Cross-Validation.
