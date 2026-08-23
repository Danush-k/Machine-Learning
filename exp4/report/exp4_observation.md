# Experiment 4: Observation Sheet

**Experiment Title:** Binary Classification using Linear and Kernel-Based Models  
**Course:** Machine Learning Algorithms Laboratory (ICS1512)  
**Student Name:** Danusu K  
**Register Number:** 3122247001013  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  

---

## Aim
To implement and evaluate binary classification models using **Logistic Regression** and **Support Vector Machines (SVM)** on the Spambase dataset, optimize their hyperparameters using Grid Search and Randomized Search, compare SVM kernel functions, and analyze their performance and computational trade-offs.

---

## Observation Tables

### 1. Hyperparameter Tuning Results
| Model | Search Method | Best Parameters | Best CV Accuracy |
| :--- | :--- | :--- | :---: |
| **Logistic Regression** | Grid Search | `{'C': 100, 'penalty': 'l1', 'solver': 'liblinear'}` | 0.9261 (92.61%) |
| **Logistic Regression** | Randomized Search | `{'solver': 'liblinear', 'penalty': 'l1', 'C': 100}` | 0.9261 (92.61%) |
| **SVM** | Grid Search | `{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}` | 0.9356 (93.56%) |
| **SVM** | Randomized Search | `{'kernel': 'linear', 'C': 1}` | 0.9274 (92.74%) |

---

### 2. Logistic Regression Performance
| Metric | Value |
| :--- | :---: |
| **Accuracy** | 0.9262 (92.62%) |
| **Precision** | 0.9202 (92.02%) |
| **Recall** | 0.8898 (88.98%) |
| **F1 Score** | 0.9048 (90.48%) |
| **Training Time (s)** | 0.018 s (Fit) / 6.63 s (Grid Search Total) |

---

### 3. SVM Kernel-wise Performance
| Kernel | Accuracy | F1 Score | Training Time (s) |
| :--- | :---: | :---: | :---: |
| **Linear** | 0.9294 (92.94%) | 0.9093 | 0.8263 s |
| **Polynomial** | 0.7796 (77.96%) | 0.6220 | 0.8498 s |
| **RBF** | 0.9273 (92.73%) | 0.9055 | 0.5436 s |
| **Sigmoid** | 0.8849 (88.49%) | 0.8528 | 0.4244 s |

---

### 4. K-Fold Cross-Validation Results ($K = 5$)
| Fold | Logistic Regression | SVM |
| :--- | :---: | :---: |
| **Fold 1** | 0.9416 (94.16%) | 0.9443 (94.43%) |
| **Fold 2** | 0.9253 (92.53%) | 0.9416 (94.16%) |
| **Fold 3** | 0.9348 (93.48%) | 0.9348 (93.48%) |
| **Fold 4** | 0.9117 (91.17%) | 0.9307 (93.07%) |
| **Fold 5** | 0.9171 (91.71%) | 0.9266 (92.66%) |
| **Average** | **0.9261 (92.61%)** | **0.9356 (93.56%)** |

---

### 5. Comparative Analysis
| Criterion | Logistic Regression | SVM |
| :--- | :---: | :---: |
| **Accuracy** | High (92.61% CV / 92.62% Test) | High (93.56% CV / 92.94% Test) |
| **Model Complexity** | Low | High |
| **Training Time** | Low | High |
| **Interpretability** | High | Low |

---

## Observations
* **Identify the best-performing classifier:**  
  The Support Vector Machine (SVM) with an **RBF kernel** achieved the highest 5-fold cross-validation accuracy of **93.56%** and an F1-score of **0.9055**, closely followed by Linear SVM (**92.94%** test accuracy).
* **Explain the impact of regularization:**  
  In Logistic Regression, $L_1$ regularization performed automatic feature selection by zeroing out uninformative weights, with $C = 100$ providing optimal balance. In SVM, $C = 10$ enforced an optimal penalty on margin violations, maximizing accuracy without overfitting.
* **Discuss kernel behavior in SVM:**  
  The **RBF kernel** effectively modeled non-linear interactions with fast training ($0.54\text{ s}$). The **Linear kernel** performed exceptionally well ($92.94\%$), showing the dataset has strong linear separability. The **Polynomial kernel** underperformed ($77.96\%$) due to high feature dimensionality, and the **Sigmoid kernel** yielded lower accuracy ($88.49\%$).
* **Comment on bias–variance trade-off:**  
  Small $C$ values ($C \le 0.1$) caused high bias (underfitting). Optimized parameters ($C=100$ for LR, $C=10$ for SVM) produced low bias and low variance across all 5 cross-validation folds ($\sigma = \pm 0.0071$ for SVM).

---

## Learning Outcomes
* Understand the core theoretical and practical differences between probabilistic classifiers (**Logistic Regression**) and margin-based classifiers (**SVM**).
* Master hyperparameter tuning workflows using **Grid Search** and **Randomized Search** with stratified $K$-fold cross-validation.
* Understand the role and behavior of different **SVM kernel functions** (Linear, Polynomial, RBF, Sigmoid) on multi-dimensional data.
* Gain hands-on experience in controlling model complexity through **regularization** ($L_1/L_2$ penalties and $C$ parameter) to balance the bias–variance trade-off.
