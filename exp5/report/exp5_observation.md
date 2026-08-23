# Experiment 5: Observation Sheet

**Experiment Title:** Decision Tree and Random Forest: A Comparative Classification Study  
**Course:** Machine Learning Algorithms Laboratory (ICS1512)  
**Student Name:** Danusu K  
**Register Number:** 3122247001013  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  

---

## Aim
To implement and evaluate a single **Decision Tree** classifier and a **Random Forest** ensemble classifier on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, optimize their hyperparameters using 5-Fold Stratified Cross-Validation, analyze overfitting and tree pruning, and evaluate the bias–variance trade-off and feature importance.

---

## Observation Tables

### 1. Hyperparameter Tuning Results
| Model | Search Method | Best Parameters | Best CV Accuracy |
| :--- | :--- | :--- | :---: |
| **Decision Tree** | Grid Search (5-Fold CV) | `{'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 2}` | **94.07%** (0.9407) |
| **Random Forest** | Grid Search (5-Fold CV) | `{'n_estimators': 25, 'max_depth': 12, 'max_features': 0.5, 'bootstrap': True}` | **97.58%** (0.9758) |

---

### 2. Decision Tree Performance (Optimal Model on Test Set)
| Metric | Value |
| :--- | :---: |
| **Accuracy** | 0.9298 (92.98%) |
| **Precision** | 1.0000 (100.0%) |
| **Recall** | 0.8095 (80.95%) |
| **F1 Score** | 0.8947 (89.47%) |
| **ROC-AUC** | 0.9563 (95.63%) |
| **Training Time (s)** | 0.0028 s (Fit) / 3.45 s (Tuning Total) |

---

### 3. Random Forest Performance (Optimal Model on Test Set)
| Metric | Value |
| :--- | :---: |
| **Accuracy** | 0.9737 (97.37%) |
| **Precision** | 1.0000 (100.0%) |
| **Recall** | 0.9286 (92.86%) |
| **F1 Score** | 0.9630 (96.30%) |
| **ROC-AUC** | 0.9902 (99.02%) |
| **Training Time (s)** | 0.0245 s (Fit) / 34.40 s (Tuning Total) |

---

### 4. K-Fold Cross-Validation Results ($K = 5$)
| Fold | Decision Tree | Random Forest |
| :--- | :---: | :---: |
| **Fold 1** | 0.9341 (93.41%) | 0.9670 (96.70%) |
| **Fold 2** | 0.9560 (95.60%) | 1.0000 (100.0%) |
| **Fold 3** | 0.9011 (90.11%) | 0.9780 (97.80%) |
| **Fold 4** | 0.9560 (95.60%) | 0.9451 (94.51%) |
| **Fold 5** | 0.9560 (95.60%) | 0.9890 (98.90%) |
| **Average** | **0.9407 (94.07%)** | **0.9758 (97.58%)** |
| **Standard Deviation ($\sigma$)** | **$\pm 2.15\%$** | **$\pm 1.89\%$** |

---

### 5. Comparative Analysis
| Criterion | Decision Tree | Random Forest |
| :--- | :---: | :---: |
| **Accuracy** | High (94.07% CV / 92.98% Test) | Very High (97.58% CV / 97.37% Test) |
| **Model Complexity** | Low (Single tree, depth = 4) | High (Ensemble of 25 trees) |
| **Training Time** | Low (0.0028 s) | Moderate (0.0245 s) |
| **Interpretability** | High (White-box if-then rules) | Low (Black-box ensemble voting) |

---

## Observations
* **Identify the best-performing classifier:**  
  The **Random Forest** ensemble is the best-performing model, achieving **97.58%** cross-validation accuracy, **97.37%** test accuracy, and an exceptional **0.9902 ROC-AUC**, substantially outperforming the single Decision Tree (**92.98%** accuracy).
* **Explain the impact of tree depth and pruning:**  
  Unpruned decision trees overfit the training data (reaching 100.0% training accuracy at depth $d \ge 8$), but suffer lower validation scores due to high variance. Pre-pruning the single tree at **$\text{max\_depth} = 4$** prevented noise memorization and maximized cross-validation performance.
* **Discuss bagging and feature subspace selection:**  
  Bootstrap aggregation (bagging) reduced sample variance by averaging across 25 independent trees. Random feature subspace selection ($\text{max\_features} = 0.5$) de-correlated individual trees ($\rho \downarrow$), driving the total ensemble variance down without inflating model bias.
* **Comment on bias–variance trade-off:**  
  The single Decision Tree had higher variance and missed 8 malignant tumors (False Negatives = 8, $\text{Recall} = 80.95\%$). Random Forest reduced false negatives to only 3 (False Negatives = 3, $\text{Recall} = 92.86\%$) with lower cross-validation variance ($\sigma = \pm 1.89\%$), proving superior clinical reliability.

---

## Learning Outcomes
* Understand recursive binary splitting and impurity reduction criteria (**Gini Impurity** and **Entropy / Information Gain**) in Decision Trees.
* Master ensemble learning principles: **Bootstrap Aggregation (Bagging)** and **Random Feature Subspaces** in Random Forests.
* Understand how hyperparameters (**maximum depth**, **minimum samples split**, **minimum samples leaf**) control tree complexity and prevent overfitting.
* Evaluate the trade-off between **model interpretability** (white-box decision trees) and **predictive power / variance reduction** (ensemble random forests).
