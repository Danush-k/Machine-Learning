#!/usr/bin/env python3
"""
Experiment 7: Dimensionality Reduction and Model Evaluation (With and Without PCA)
Course: Machine Learning Algorithms Laboratory (ICS1512)
Student Name: Danusu K | Register Number: 3122247001013
Faculty: Dr. Poreddy Ajay Kumar Reddy
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)

This script implements:
1. Dataset loading, standardization, and train-test splitting (80/20).
2. Principal Component Analysis (PCA): eigenvalue decomposition, scree plot, cumulative variance analysis.
3. Hyperparameter tuning using 5-Fold Stratified Cross-Validation for 10 models:
   - Support Vector Machine (SVM)
   - Gaussian Naïve Bayes
   - k-Nearest Neighbors (KNN)
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - AdaBoost
   - Gradient Boosting
   - XGBoost
   - Stacking Classifier (Heterogeneous base learners + Meta-learner)
4. Comprehensive 5-fold CV evaluation under both No-PCA and With-PCA settings.
5. Held-out test set benchmarking (Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss, Latency).
6. Generation of 11 publication-grade visualizations in `output_plots/`.
7. Structured serialization to `experiment7_results.json`.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, log_loss
)

# Random seed for exact reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Plot aesthetics configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(EXP_DIR, 'dataset')
PLOTS_DIR = os.path.join(EXP_DIR, 'output_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_and_preprocess_data():
    """Load and preprocess the WDBC dataset."""
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    all_columns = ['id', 'diagnosis'] + feature_names

    csv_path = os.path.join(DATA_DIR, 'wdbc.csv')
    data_path = os.path.join(DATA_DIR, 'wdbc.data')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(data_path):
        df = pd.read_csv(data_path, header=None, names=all_columns)
    else:
        raise FileNotFoundError("WDBC dataset not found in dataset/ directory.")

    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])

    X = df[feature_names].copy()
    y = df['diagnosis'].map({'M': 1, 'B': 0}).astype(int)

    # 80-20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Standardize features (Mean=0, Std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names


def perform_pca_analysis(X_train_scaled, X_test_scaled, feature_names):
    """
    Perform full PCA analysis, scree analysis, and variance target selection.
    """
    # Full PCA on all 30 components
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_train_scaled)

    explained_var = pca_full.explained_variance_
    explained_var_ratio = pca_full.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)

    # Determine components for variance targets
    n_comp_90 = int(np.argmax(cumulative_var_ratio >= 0.90) + 1)
    n_comp_95 = int(np.argmax(cumulative_var_ratio >= 0.95) + 1)
    n_comp_99 = int(np.argmax(cumulative_var_ratio >= 0.99) + 1)

    print(f"[PCA Analysis] Total features: {X_train_scaled.shape[1]}")
    print(f"[PCA Analysis] 90% variance reached at: {n_comp_90} components ({cumulative_var_ratio[n_comp_90-1]*100:.2f}%)")
    print(f"[PCA Analysis] 95% variance reached at: {n_comp_95} components ({cumulative_var_ratio[n_comp_95-1]*100:.2f}%)")
    print(f"[PCA Analysis] 99% variance reached at: {n_comp_99} components ({cumulative_var_ratio[n_comp_99-1]*100:.2f}%)")

    # Selected PCA design choice: 95% variance target (10 components)
    n_selected = n_comp_95
    pca_selected = PCA(n_components=n_selected, random_state=RANDOM_STATE)
    X_train_pca = pca_selected.fit_transform(X_train_scaled)
    X_test_pca = pca_selected.transform(X_test_scaled)

    # Prepare PCA summary table data
    pca_table = []
    for i in range(len(explained_var_ratio)):
        pca_table.append({
            'component': i + 1,
            'eigenvalue': float(explained_var[i]),
            'explained_variance_pct': float(explained_var_ratio[i] * 100),
            'cumulative_variance_pct': float(cumulative_var_ratio[i] * 100)
        })

    pca_info = {
        'total_features': int(X_train_scaled.shape[1]),
        'chosen_components': n_selected,
        'variance_target_pct': 95.0,
        'actual_explained_variance_pct': float(cumulative_var_ratio[n_selected - 1] * 100),
        'dimension_reduction_pct': float((1 - n_selected / X_train_scaled.shape[1]) * 100),
        'pca_table': pca_table,
        'loadings': pca_selected.components_.tolist()
    }

    return pca_full, pca_selected, X_train_pca, X_test_pca, pca_info


def define_hyperparameter_grids():
    """Define hyperparameter search spaces for all 10 models."""
    grids = {
        'SVM': {
            'model': SVC(probability=True, random_state=RANDOM_STATE),
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 0.01, 0.1]
            }
        },
        'Naive_Bayes': {
            'model': GaussianNB(),
            'param_grid': {
                'var_smoothing': [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        'Logistic_Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'param_grid': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'Decision_Tree': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'max_features': ['sqrt', 'log2'],
                'min_samples_split': [2, 5]
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [25, 50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
            }
        },
        'Gradient_Boosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [2, 3, 4]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 6],
                'subsample': [0.8, 1.0]
            }
        }
    }
    return grids


def tune_and_evaluate_models(X_train_orig, X_test_orig, X_train_pca, X_test_pca, y_train, y_test):
    """
    Perform 5-fold CV hyperparameter tuning and evaluation for all 10 models
    under both No-PCA and With-PCA settings.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grids = define_hyperparameter_grids()

    results = {
        'tuning_details': {},
        'cv_5fold_results': {},
        'test_results': {},
        'best_models': {
            'no_pca': {},
            'with_pca': {}
        }
    }

    # Iterate through individual models
    for name, config in grids.items():
        print(f"\n==================== Tuning {name} ====================")

        # 1. No-PCA Tuning
        t0 = time.time()
        gs_no_pca = GridSearchCV(
            config['model'], config['param_grid'], cv=cv, scoring='accuracy', n_jobs=-1
        )
        gs_no_pca.fit(X_train_orig, y_train)
        no_pca_tune_time = time.time() - t0
        best_model_no_pca = gs_no_pca.best_estimator_

        # 2. With-PCA Tuning
        t0 = time.time()
        gs_pca = GridSearchCV(
            config['model'], config['param_grid'], cv=cv, scoring='accuracy', n_jobs=-1
        )
        gs_pca.fit(X_train_pca, y_train)
        pca_tune_time = time.time() - t0
        best_model_pca = gs_pca.best_estimator_

        results['tuning_details'][name] = {
            'no_pca_best_params': gs_no_pca.best_params_,
            'no_pca_best_cv_score': float(gs_no_pca.best_score_),
            'no_pca_tune_time_sec': float(no_pca_tune_time),
            'with_pca_best_params': gs_pca.best_params_,
            'with_pca_best_cv_score': float(gs_pca.best_score_),
            'with_pca_tune_time_sec': float(pca_tune_time),
            'grid_results_no_pca': [
                {
                    'params': {k: (str(v) if v is None else v) for k, v in params.items()},
                    'mean_cv_score': float(mean_score),
                    'std_cv_score': float(std_score)
                }
                for params, mean_score, std_score in zip(
                    gs_no_pca.cv_results_['params'],
                    gs_no_pca.cv_results_['mean_test_score'],
                    gs_no_pca.cv_results_['std_test_score']
                )
            ],
            'grid_results_with_pca': [
                {
                    'params': {k: (str(v) if v is None else v) for k, v in params.items()},
                    'mean_cv_score': float(mean_score),
                    'std_cv_score': float(std_score)
                }
                for params, mean_score, std_score in zip(
                    gs_pca.cv_results_['params'],
                    gs_pca.cv_results_['mean_test_score'],
                    gs_pca.cv_results_['std_test_score']
                )
            ]
        }

        results['best_models']['no_pca'][name] = best_model_no_pca
        results['best_models']['with_pca'][name] = best_model_pca

        print(f"[{name} No-PCA] Best Params: {gs_no_pca.best_params_} | CV Acc: {gs_no_pca.best_score_*100:.2f}%")
        print(f"[{name} With-PCA] Best Params: {gs_pca.best_params_} | CV Acc: {gs_pca.best_score_*100:.2f}%")

    # 10. Stacking Classifier Implementation & Tuning
    print(f"\n==================== Tuning Stacking Classifier ====================")
    stack_base_no_pca = [
        ('svm', results['best_models']['no_pca']['SVM']),
        ('nb', results['best_models']['no_pca']['Naive_Bayes']),
        ('knn', results['best_models']['no_pca']['KNN']),
        ('rf', results['best_models']['no_pca']['Random_Forest'])
    ]
    stack_base_pca = [
        ('svm', results['best_models']['with_pca']['SVM']),
        ('nb', results['best_models']['with_pca']['Naive_Bayes']),
        ('knn', results['best_models']['with_pca']['KNN']),
        ('rf', results['best_models']['with_pca']['Random_Forest'])
    ]

    meta_candidates = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
        'RidgeClassifier': RidgeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=RANDOM_STATE)
    }

    stack_tuning_records = {'no_pca': [], 'with_pca': []}
    best_stack_score_no_pca = -1
    best_stack_model_no_pca = None
    best_meta_no_pca = None

    best_stack_score_pca = -1
    best_stack_model_pca = None
    best_meta_pca = None

    for meta_name, meta_clf in meta_candidates.items():
        # No PCA Stacking
        stk_no_pca = StackingClassifier(
            estimators=stack_base_no_pca,
            final_estimator=meta_clf,
            cv=5,
            n_jobs=-1
        )
        scores_no_pca = []
        for train_idx, val_idx in cv.split(X_train_orig, y_train):
            stk_no_pca.fit(X_train_orig[train_idx], y_train.iloc[train_idx])
            acc = accuracy_score(y_train.iloc[val_idx], stk_no_pca.predict(X_train_orig[val_idx]))
            scores_no_pca.append(acc)
        mean_no_pca = float(np.mean(scores_no_pca))
        std_no_pca = float(np.std(scores_no_pca))
        stack_tuning_records['no_pca'].append({
            'meta_learner': meta_name,
            'mean_cv_accuracy': mean_no_pca,
            'std_cv_accuracy': std_no_pca,
            'fold_scores': [float(s) for s in scores_no_pca]
        })
        if mean_no_pca > best_stack_score_no_pca:
            best_stack_score_no_pca = mean_no_pca
            best_stack_model_no_pca = stk_no_pca
            best_meta_no_pca = meta_name

        # With PCA Stacking
        stk_pca = StackingClassifier(
            estimators=stack_base_pca,
            final_estimator=meta_clf,
            cv=5,
            n_jobs=-1
        )
        scores_pca = []
        for train_idx, val_idx in cv.split(X_train_pca, y_train):
            stk_pca.fit(X_train_pca[train_idx], y_train.iloc[train_idx])
            acc = accuracy_score(y_train.iloc[val_idx], stk_pca.predict(X_train_pca[val_idx]))
            scores_pca.append(acc)
        mean_pca = float(np.mean(scores_pca))
        std_pca = float(np.std(scores_pca))
        stack_tuning_records['with_pca'].append({
            'meta_learner': meta_name,
            'mean_cv_accuracy': mean_pca,
            'std_cv_accuracy': std_pca,
            'fold_scores': [float(s) for s in scores_pca]
        })
        if mean_pca > best_stack_score_pca:
            best_stack_score_pca = mean_pca
            best_stack_model_pca = stk_pca
            best_meta_pca = meta_name

    results['best_models']['no_pca']['Stacking'] = best_stack_model_no_pca
    results['best_models']['with_pca']['Stacking'] = best_stack_model_pca
    results['tuning_details']['Stacking'] = {
        'no_pca_best_params': {'meta_learner': best_meta_no_pca},
        'no_pca_best_cv_score': best_stack_score_no_pca,
        'with_pca_best_params': {'meta_learner': best_meta_pca},
        'with_pca_best_cv_score': best_stack_score_pca,
        'grid_results_no_pca': stack_tuning_records['no_pca'],
        'grid_results_with_pca': stack_tuning_records['with_pca']
    }
    print(f"[Stacking No-PCA] Best Meta: {best_meta_no_pca} | CV Acc: {best_stack_score_no_pca*100:.2f}%")
    print(f"[Stacking With-PCA] Best Meta: {best_meta_pca} | CV Acc: {best_stack_score_pca*100:.2f}%")

    # 5-Fold Cross-Validation Detailed Extraction for All 10 Models
    all_model_names = [
        'SVM', 'Naive_Bayes', 'KNN', 'Logistic_Regression',
        'Decision_Tree', 'Random_Forest', 'AdaBoost',
        'Gradient_Boosting', 'XGBoost', 'Stacking'
    ]

    print(f"\n==================== Running Detailed 5-Fold CV Evaluation ====================")
    for model_name in all_model_names:
        m_no_pca = results['best_models']['no_pca'][model_name]
        m_pca = results['best_models']['with_pca'][model_name]

        # Detailed fold-by-fold execution
        fold_scores_no_pca = []
        train_times_no_pca = []
        for train_idx, val_idx in cv.split(X_train_orig, y_train):
            t_start = time.time()
            m_no_pca.fit(X_train_orig[train_idx], y_train.iloc[train_idx])
            t_fit = time.time() - t_start
            train_times_no_pca.append(t_fit)
            acc = accuracy_score(y_train.iloc[val_idx], m_no_pca.predict(X_train_orig[val_idx]))
            fold_scores_no_pca.append(acc)

        fold_scores_pca = []
        train_times_pca = []
        for train_idx, val_idx in cv.split(X_train_pca, y_train):
            t_start = time.time()
            m_pca.fit(X_train_pca[train_idx], y_train.iloc[train_idx])
            t_fit = time.time() - t_start
            train_times_pca.append(t_fit)
            acc = accuracy_score(y_train.iloc[val_idx], m_pca.predict(X_train_pca[val_idx]))
            fold_scores_pca.append(acc)

        results['cv_5fold_results'][model_name] = {
            'no_pca': {
                'folds': [float(s) for s in fold_scores_no_pca],
                'mean_accuracy': float(np.mean(fold_scores_no_pca)),
                'std_accuracy': float(np.std(fold_scores_no_pca)),
                'avg_train_time_sec': float(np.mean(train_times_no_pca))
            },
            'with_pca': {
                'folds': [float(s) for s in fold_scores_pca],
                'mean_accuracy': float(np.mean(fold_scores_pca)),
                'std_accuracy': float(np.std(fold_scores_pca)),
                'avg_train_time_sec': float(np.mean(train_times_pca))
            },
            'accuracy_delta_pct': float((np.mean(fold_scores_pca) - np.mean(fold_scores_no_pca)) * 100),
            'std_reduction': float(np.std(fold_scores_no_pca) - np.std(fold_scores_pca))
        }

        # Train on full training set and evaluate on test set
        # No-PCA
        t_fit_start = time.time()
        m_no_pca.fit(X_train_orig, y_train)
        full_train_time_no_pca = time.time() - t_fit_start

        t_inf_start = time.time()
        y_pred_no_pca = m_no_pca.predict(X_test_orig)
        inf_time_no_pca = time.time() - t_inf_start

        if hasattr(m_no_pca, 'predict_proba'):
            y_proba_no_pca = m_no_pca.predict_proba(X_test_orig)[:, 1]
        elif hasattr(m_no_pca, 'decision_function'):
            df_vals = m_no_pca.decision_function(X_test_orig)
            y_proba_no_pca = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-10)
        else:
            y_proba_no_pca = y_pred_no_pca.astype(float)

        # With-PCA
        t_fit_start = time.time()
        m_pca.fit(X_train_pca, y_train)
        full_train_time_pca = time.time() - t_fit_start

        t_inf_start = time.time()
        y_pred_pca = m_pca.predict(X_test_pca)
        inf_time_pca = time.time() - t_inf_start

        if hasattr(m_pca, 'predict_proba'):
            y_proba_pca = m_pca.predict_proba(X_test_pca)[:, 1]
        elif hasattr(m_pca, 'decision_function'):
            df_vals = m_pca.decision_function(X_test_pca)
            y_proba_pca = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-10)
        else:
            y_proba_pca = y_pred_pca.astype(float)

        # Metrics computation
        cm_no_pca = confusion_matrix(y_test, y_pred_no_pca).tolist()
        cm_pca = confusion_matrix(y_test, y_pred_pca).tolist()

        tn_no, fp_no, fn_no, tp_no = confusion_matrix(y_test, y_pred_no_pca).ravel()
        tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_test, y_pred_pca).ravel()

        spec_no_pca = float(tn_no / (tn_no + fp_no))
        spec_pca = float(tn_p / (tn_p + fp_p))

        fpr_no, tpr_no, _ = roc_curve(y_test, y_proba_no_pca)
        fpr_p, tpr_p, _ = roc_curve(y_test, y_proba_pca)

        prec_curve_no, rec_curve_no, _ = precision_recall_curve(y_test, y_proba_no_pca)
        prec_curve_p, rec_curve_p, _ = precision_recall_curve(y_test, y_proba_pca)

        results['test_results'][model_name] = {
            'no_pca': {
                'accuracy': float(accuracy_score(y_test, y_pred_no_pca)),
                'precision': float(precision_score(y_test, y_pred_no_pca, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred_no_pca, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred_no_pca, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_proba_no_pca)),
                'avg_precision': float(average_precision_score(y_test, y_proba_no_pca)),
                'specificity': spec_no_pca,
                'log_loss': float(log_loss(y_test, np.clip(y_proba_no_pca, 1e-15, 1 - 1e-15))),
                'confusion_matrix': cm_no_pca,
                'train_time_sec': float(full_train_time_no_pca),
                'inference_time_ms': float(inf_time_no_pca * 1000),
                'roc_fpr': fpr_no.tolist(),
                'roc_tpr': tpr_no.tolist(),
                'pr_precision': prec_curve_no.tolist(),
                'pr_recall': rec_curve_no.tolist()
            },
            'with_pca': {
                'accuracy': float(accuracy_score(y_test, y_pred_pca)),
                'precision': float(precision_score(y_test, y_pred_pca, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred_pca, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred_pca, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_proba_pca)),
                'avg_precision': float(average_precision_score(y_test, y_proba_pca)),
                'specificity': spec_pca,
                'log_loss': float(log_loss(y_test, np.clip(y_proba_pca, 1e-15, 1 - 1e-15))),
                'confusion_matrix': cm_pca,
                'train_time_sec': float(full_train_time_pca),
                'inference_time_ms': float(inf_time_pca * 1000),
                'roc_fpr': fpr_p.tolist(),
                'roc_tpr': tpr_p.tolist(),
                'pr_precision': prec_curve_p.tolist(),
                'pr_recall': rec_curve_p.tolist()
            }
        }

    return results


def generate_publication_plots(pca_full, pca_selected, X_train_scaled, y_train, feature_names, results):
    """
    Generate 11 publication-grade visualizations.
    """
    print(f"\n==================== Generating Publication Plots ====================")
    var_exp = pca_full.explained_variance_ratio_ * 100
    cum_var_exp = np.cumsum(var_exp)
    eigenvalues = pca_full.explained_variance_
    n_comp = len(var_exp)

    # ---------------------------------------------------------
    # Plot 1: Scree Plot & Cumulative Explained Variance
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Ax 1: Scree Plot (Eigenvalues & Individual Variance)
    x_indices = np.arange(1, n_comp + 1)
    ax1.bar(x_indices, var_exp, color='#2b5c8f', alpha=0.7, edgecolor='black', label='Individual Variance (%)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_indices, eigenvalues, color='#c0392b', marker='o', linewidth=2, label=r'Eigenvalue ($\lambda$)')
    ax1_twin.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label=r'Kaiser Criterion ($\lambda = 1$)')
    ax1.set_xlabel('Principal Component Index')
    ax1.set_ylabel('Explained Variance Ratio (%)', color='#2b5c8f')
    ax1_twin.set_ylabel(r'Eigenvalue ($\lambda$)', color='#c0392b')
    ax1.set_title('Scree Plot: Individual Variance & Eigenvalues', fontweight='bold')
    ax1.set_xticks(range(1, n_comp + 1, 2))
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Ax 2: Cumulative Variance Curve
    ax2.plot(x_indices, cum_var_exp, color='#16a085', marker='s', linewidth=2.5, label='Cumulative Explained Variance')
    ax2.axhline(95.0, color='#e74c3c', linestyle='--', linewidth=2, label='95% Threshold Cutoff')
    ax2.axvline(10, color='#8e44ad', linestyle=':', linewidth=2, label='Selected $k=10$ PCs (95.1%)')
    ax2.scatter([10], [cum_var_exp[9]], color='#8e44ad', s=120, zorder=5)
    ax2.annotate(f'10 Components\n{cum_var_exp[9]:.2f}% Variance',
                 xy=(10, cum_var_exp[9]), xytext=(12, 85),
                 arrowprops=dict(facecolor='#8e44ad', shrink=0.08, width=1.5, headwidth=8),
                 fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f4ecf7', edgecolor='#8e44ad'))
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title('Cumulative Variance Analysis with 95% Cutoff', fontweight='bold')
    ax2.set_xticks(range(1, n_comp + 1, 2))
    ax2.set_ylim(40, 102)
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p1_path = os.path.join(PLOTS_DIR, '01_scree_and_cumulative_variance.png')
    plt.savefig(p1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p1_path}")

    # ---------------------------------------------------------
    # Plot 2: 2D and 3D PCA Projections
    # ---------------------------------------------------------
    X_train_pca = pca_selected.transform(X_train_scaled)
    fig = plt.figure(figsize=(16, 7))

    # 2D Projection
    ax1 = fig.add_subplot(1, 2, 1)
    colors = ['#27ae60' if y == 0 else '#c0392b' for y in y_train]
    scatter1 = ax1.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colors, alpha=0.75, edgecolors='none', s=45)
    ax1.set_xlabel(f'PC1 ({var_exp[0]:.2f}% Variance)', fontweight='bold')
    ax1.set_ylabel(f'PC2 ({var_exp[1]:.2f}% Variance)', fontweight='bold')
    ax1.set_title('2D PCA Projection of WDBC Feature Space', fontweight='bold')
    import matplotlib.patches as mpatches
    benign_patch = mpatches.Patch(color='#27ae60', label='Benign (B)')
    mal_patch = mpatches.Patch(color='#c0392b', label='Malignant (M)')
    ax1.legend(handles=[benign_patch, mal_patch], loc='upper right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 3D Projection
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=colors, alpha=0.75, s=35, edgecolors='k', linewidth=0.2)
    ax2.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)')
    ax2.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)')
    ax2.set_zlabel(f'PC3 ({var_exp[2]:.1f}%)')
    ax2.set_title('3D PCA Manifold: First 3 Principal Components', fontweight='bold')
    ax2.view_init(elev=20, azim=130)

    plt.tight_layout()
    p2_path = os.path.join(PLOTS_DIR, '02_pca_2d_and_3d_projections.png')
    plt.savefig(p2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p2_path}")

    # ---------------------------------------------------------
    # Plot 3: PCA Component Loadings Heatmap
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))
    top_components = pca_selected.components_[:5, :]
    pc_labels = [f'PC{i+1} ({var_exp[i]:.1f}%)' for i in range(5)]
    sns.heatmap(
        top_components,
        xticklabels=feature_names,
        yticklabels=pc_labels,
        cmap='vlag',
        center=0,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7},
        cbar_kws={'label': 'Eigenvector Loading Weight'},
        ax=ax
    )
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.title('Principal Component Loadings (Eigenvectors for Top 5 Components)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    p3_path = os.path.join(PLOTS_DIR, '03_pca_component_loadings_heatmap.png')
    plt.savefig(p3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p3_path}")

    # ---------------------------------------------------------
    # Plot 4: SVM and KNN Hyperparameter Tuning Heatmaps / Curves
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # SVM Tuning Response Surface (C vs Gamma for RBF Kernel)
    svm_records_no = [r for r in results['tuning_details']['SVM']['grid_results_no_pca'] if r['params']['kernel'] == 'rbf']
    svm_records_pca = [r for r in results['tuning_details']['SVM']['grid_results_with_pca'] if r['params']['kernel'] == 'rbf']

    df_svm_no = pd.DataFrame([
        {'C': r['params']['C'], 'gamma': str(r['params']['gamma']), 'score': r['mean_cv_score']}
        for r in svm_records_no
    ]).pivot(index='C', columns='gamma', values='score')

    df_svm_pca = pd.DataFrame([
        {'C': r['params']['C'], 'gamma': str(r['params']['gamma']), 'score': r['mean_cv_score']}
        for r in svm_records_pca
    ]).pivot(index='C', columns='gamma', values='score')

    sns.heatmap(df_svm_no, annot=True, fmt='.4f', cmap='Blues', ax=axes[0, 0], cbar_kws={'label': 'CV Accuracy'})
    axes[0, 0].set_title('SVM (RBF Kernel) Tuning — No-PCA', fontweight='bold')
    axes[0, 0].set_xlabel(r'$\gamma$ Parameter')
    axes[0, 0].set_ylabel(r'$C$ Regularization')

    sns.heatmap(df_svm_pca, annot=True, fmt='.4f', cmap='Blues', ax=axes[0, 1], cbar_kws={'label': 'CV Accuracy'})
    axes[0, 1].set_title('SVM (RBF Kernel) Tuning — With-PCA (10 PCs)', fontweight='bold')
    axes[0, 1].set_xlabel(r'$\gamma$ Parameter')
    axes[0, 1].set_ylabel(r'$C$ Regularization')

    # KNN Tuning Curves across k and distance metric
    knn_records_no = [r for r in results['tuning_details']['KNN']['grid_results_no_pca'] if r['params']['weights'] == 'uniform']
    knn_records_pca = [r for r in results['tuning_details']['KNN']['grid_results_with_pca'] if r['params']['weights'] == 'uniform']

    for metric, color in zip(['euclidean', 'manhattan'], ['#2980b9', '#e67e22']):
        sub_no = [r for r in knn_records_no if r['params']['metric'] == metric]
        k_vals = [r['params']['n_neighbors'] for r in sub_no]
        scores_no = [r['mean_cv_score'] for r in sub_no]
        axes[1, 0].plot(k_vals, scores_no, marker='o', label=f'{metric.capitalize()} (No-PCA)', color=color, linewidth=2)

        sub_pca = [r for r in knn_records_pca if r['params']['metric'] == metric]
        scores_pca = [r['mean_cv_score'] for r in sub_pca]
        axes[1, 1].plot(k_vals, scores_pca, marker='s', linestyle='--', label=f'{metric.capitalize()} (With-PCA)', color=color, linewidth=2)

    axes[1, 0].set_title('KNN Tuning Across $k$ Values — No-PCA', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Neighbors ($k$)')
    axes[1, 0].set_ylabel('Mean 5-Fold CV Accuracy')
    axes[1, 0].set_xticks([3, 5, 7, 9, 11, 15, 21])
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    axes[1, 1].set_title('KNN Tuning Across $k$ Values — With-PCA', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Neighbors ($k$)')
    axes[1, 1].set_ylabel('Mean 5-Fold CV Accuracy')
    axes[1, 1].set_xticks([3, 5, 7, 9, 11, 15, 21])
    axes[1, 1].legend(loc='lower right')
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p4_path = os.path.join(PLOTS_DIR, '04_svm_knn_tuning_heatmaps.png')
    plt.savefig(p4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p4_path}")

    # ---------------------------------------------------------
    # Plot 5: Tree and Ensemble Hyperparameter Tuning Curves
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Random Forest: n_estimators vs CV accuracy
    rf_no = [r for r in results['tuning_details']['Random_Forest']['grid_results_no_pca'] if r['params']['max_features'] == 'sqrt' and r['params']['min_samples_split'] == 2 and r['params']['max_depth'] == 'None']
    rf_pca = [r for r in results['tuning_details']['Random_Forest']['grid_results_with_pca'] if r['params']['max_features'] == 'sqrt' and r['params']['min_samples_split'] == 2 and r['params']['max_depth'] == 'None']
    n_est_rf = [r['params']['n_estimators'] for r in rf_no]
    axes[0, 0].plot(n_est_rf, [r['mean_cv_score'] for r in rf_no], marker='o', linewidth=2, color='#27ae60', label='No-PCA')
    axes[0, 0].plot(n_est_rf, [r['mean_cv_score'] for r in rf_pca], marker='s', linestyle='--', linewidth=2, color='#8e44ad', label='With-PCA')
    axes[0, 0].set_title(r'Random Forest: $n\_estimators$ vs CV Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel(r'Number of Estimators ($T$)')
    axes[0, 0].set_ylabel('Mean CV Accuracy')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    # AdaBoost: learning_rate vs CV accuracy
    ada_no = [r for r in results['tuning_details']['AdaBoost']['grid_results_no_pca'] if r['params']['n_estimators'] == 100]
    ada_pca = [r for r in results['tuning_details']['AdaBoost']['grid_results_with_pca'] if r['params']['n_estimators'] == 100]
    lr_ada = [r['params']['learning_rate'] for r in ada_no]
    axes[0, 1].plot(lr_ada, [r['mean_cv_score'] for r in ada_no], marker='o', linewidth=2, color='#e67e22', label='No-PCA ($n=100$)')
    axes[0, 1].plot(lr_ada, [r['mean_cv_score'] for r in ada_pca], marker='s', linestyle='--', linewidth=2, color='#2980b9', label='With-PCA ($n=100$)')
    axes[0, 1].set_title(r'AdaBoost: Learning Rate ($\eta$) vs CV Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel(r'Learning Rate ($\eta$)')
    axes[0, 1].set_ylabel('Mean CV Accuracy')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend(loc='lower left')
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # Gradient Boosting: n_estimators vs CV accuracy
    gb_no = [r for r in results['tuning_details']['Gradient_Boosting']['grid_results_no_pca'] if r['params']['learning_rate'] == 0.1 and r['params']['max_depth'] == 3]
    gb_pca = [r for r in results['tuning_details']['Gradient_Boosting']['grid_results_with_pca'] if r['params']['learning_rate'] == 0.1 and r['params']['max_depth'] == 3]
    n_est_gb = [r['params']['n_estimators'] for r in gb_no]
    axes[1, 0].plot(n_est_gb, [r['mean_cv_score'] for r in gb_no], marker='o', linewidth=2, color='#c0392b', label=r'No-PCA ($\eta=0.1$)')
    axes[1, 0].plot(n_est_gb, [r['mean_cv_score'] for r in gb_pca], marker='s', linestyle='--', linewidth=2, color='#16a085', label=r'With-PCA ($\eta=0.1$)')
    axes[1, 0].set_title(r'Gradient Boosting: $n\_estimators$ vs CV Accuracy', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Estimators')
    axes[1, 0].set_ylabel('Mean CV Accuracy')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    # XGBoost: max_depth vs CV accuracy
    xgb_no = [r for r in results['tuning_details']['XGBoost']['grid_results_no_pca'] if r['params']['n_estimators'] == 100 and r['params']['learning_rate'] == 0.1 and r['params']['subsample'] == 1.0]
    xgb_pca = [r for r in results['tuning_details']['XGBoost']['grid_results_with_pca'] if r['params']['n_estimators'] == 100 and r['params']['learning_rate'] == 0.1 and r['params']['subsample'] == 1.0]
    depth_xgb = [r['params']['max_depth'] for r in xgb_no]
    axes[1, 1].plot(depth_xgb, [r['mean_cv_score'] for r in xgb_no], marker='o', linewidth=2, color='#34495e', label=r'No-PCA ($n=100, \eta=0.1$)')
    axes[1, 1].plot(depth_xgb, [r['mean_cv_score'] for r in xgb_pca], marker='s', linestyle='--', linewidth=2, color='#d35400', label=r'With-PCA ($n=100, \eta=0.1$)')
    axes[1, 1].set_title('XGBoost: Max Depth vs CV Accuracy', fontweight='bold')
    axes[1, 1].set_xlabel('Tree Max Depth')
    axes[1, 1].set_ylabel('Mean CV Accuracy')
    axes[1, 1].set_xticks([3, 4, 6])
    axes[1, 1].legend(loc='lower right')
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p5_path = os.path.join(PLOTS_DIR, '05_tree_ensemble_tuning_comparison.png')
    plt.savefig(p5_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p5_path}")

    # ---------------------------------------------------------
    # Plot 6: 5-Fold Cross-Validation Fold-wise Comparison & Stability
    # ---------------------------------------------------------
    models_list = list(results['cv_5fold_results'].keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Mean and Error bar (Std) across folds
    y_pos = np.arange(len(models_list))
    means_no = [results['cv_5fold_results'][m]['no_pca']['mean_accuracy'] * 100 for m in models_list]
    stds_no = [results['cv_5fold_results'][m]['no_pca']['std_accuracy'] * 100 for m in models_list]
    means_pca = [results['cv_5fold_results'][m]['with_pca']['mean_accuracy'] * 100 for m in models_list]
    stds_pca = [results['cv_5fold_results'][m]['with_pca']['std_accuracy'] * 100 for m in models_list]

    width = 0.38
    ax1.barh(y_pos - width/2, means_no, xerr=stds_no, height=width, color='#3498db', alpha=0.85, capsize=4, label='No-PCA (Original 30D)')
    ax1.barh(y_pos + width/2, means_pca, xerr=stds_pca, height=width, color='#e74c3c', alpha=0.85, capsize=4, label='With-PCA (Reduced 10D)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m.replace('_', ' ') for m in models_list], fontsize=11)
    ax1.set_xlabel(r'5-Fold Cross-Validation Accuracy (%) $\pm 1\sigma$', fontweight='bold')
    ax1.set_xlim(90, 100)
    ax1.set_title(r'5-Fold CV Mean Accuracy & Stability ($\pm\sigma$ Error Bars)', fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Variance / Standard Deviation Reduction
    std_diffs = [stds_no[i] - stds_pca[i] for i in range(len(models_list))]
    bar_colors = ['#27ae60' if d >= 0 else '#c0392b' for d in std_diffs]
    ax2.barh(y_pos, std_diffs, height=0.6, color=bar_colors, alpha=0.85, edgecolor='black')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(['' for _ in models_list])
    ax2.set_xlabel(r'$\Delta$ Standard Deviation ($\sigma_{\mathrm{No\text{-}PCA}} - \sigma_{\mathrm{PCA}}$) in %', fontweight='bold')
    ax2.set_title(r'Cross-Fold Variance Reduction ($\sigma$ Reduction)', fontweight='bold')
    ax2.text(0.05, len(models_list)-1, r'PCA Increases Variance $\leftarrow$ | $\rightarrow$ PCA Improves Stability',
             fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p6_path = os.path.join(PLOTS_DIR, '06_5fold_cv_foldwise_comparison.png')
    plt.savefig(p6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p6_path}")

    # ---------------------------------------------------------
    # Plot 7: Model Performance Comparison Bar Chart (Held-out Test)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    test_acc_no = [results['test_results'][m]['no_pca']['accuracy'] * 100 for m in models_list]
    test_acc_pca = [results['test_results'][m]['with_pca']['accuracy'] * 100 for m in models_list]

    test_f1_no = [results['test_results'][m]['no_pca']['f1_score'] * 100 for m in models_list]
    test_f1_pca = [results['test_results'][m]['with_pca']['f1_score'] * 100 for m in models_list]

    # Test Accuracy
    ax1.barh(y_pos - width/2, test_acc_no, height=width, color='#2980b9', label='No-PCA')
    ax1.barh(y_pos + width/2, test_acc_pca, height=width, color='#e67e22', label='With-PCA')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m.replace('_', ' ') for m in models_list], fontsize=11)
    ax1.set_xlabel('Held-out Test Accuracy (%)', fontweight='bold')
    ax1.set_xlim(88, 101)
    ax1.set_title('Test Set Accuracy Comparison (No-PCA vs With-PCA)', fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Test F1-Score
    ax2.barh(y_pos - width/2, test_f1_no, height=width, color='#27ae60', label='No-PCA')
    ax2.barh(y_pos + width/2, test_f1_pca, height=width, color='#8e44ad', label='With-PCA')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(['' for _ in models_list])
    ax2.set_xlabel('Held-out Test F1-Score (%)', fontweight='bold')
    ax2.set_xlim(85, 101)
    ax2.set_title('Test Set F1-Score Comparison (No-PCA vs With-PCA)', fontweight='bold')
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p7_path = os.path.join(PLOTS_DIR, '07_model_performance_comparison_bars.png')
    plt.savefig(p7_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p7_path}")

    # ---------------------------------------------------------
    # Plot 8: Training Time and Inference Speedup
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    train_time_no = [results['test_results'][m]['no_pca']['train_time_sec'] * 1000 for m in models_list]
    train_time_pca = [results['test_results'][m]['with_pca']['train_time_sec'] * 1000 for m in models_list]

    inf_time_no = [results['test_results'][m]['no_pca']['inference_time_ms'] for m in models_list]
    inf_time_pca = [results['test_results'][m]['with_pca']['inference_time_ms'] for m in models_list]

    ax1.barh(y_pos - width/2, train_time_no, height=width, color='#34495e', label='No-PCA (30D)')
    ax1.barh(y_pos + width/2, train_time_pca, height=width, color='#16a085', label='With-PCA (10D)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m.replace('_', ' ') for m in models_list], fontsize=11)
    ax1.set_xlabel('Training Wall-Clock Time (ms) [Log Scale]', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_title('Training Execution Latency (ms)', fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Speedup factor
    speedup_train = [train_time_no[i] / (train_time_pca[i] + 1e-6) for i in range(len(models_list))]
    ax2.barh(y_pos, speedup_train, height=0.6, color='#2980b9', alpha=0.85, edgecolor='black')
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Parity (1.0x)')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(['' for _ in models_list])
    ax2.set_xlabel(r'Training Speedup Ratio ($T_{\mathrm{No\text{-}PCA}} / T_{\mathrm{PCA}}$)', fontweight='bold')
    ax2.set_title('Computational Efficiency Gain / Speedup Factor', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p8_path = os.path.join(PLOTS_DIR, '08_training_time_and_speedup.png')
    plt.savefig(p8_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p8_path}")

    # ---------------------------------------------------------
    # Plot 9: Confusion Matrices (Key 6 Models)
    # ---------------------------------------------------------
    key_models = ['SVM', 'KNN', 'Logistic_Regression', 'Random_Forest', 'XGBoost', 'Stacking']
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for idx, m_name in enumerate(key_models):
        row = idx // 2
        col_no = (idx % 2) * 2
        col_pca = col_no + 1

        cm_no = np.array(results['test_results'][m_name]['no_pca']['confusion_matrix'])
        cm_pca = np.array(results['test_results'][m_name]['with_pca']['confusion_matrix'])

        sns.heatmap(cm_no, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[row, col_no],
                    xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        axes[row, col_no].set_title(f'{m_name.replace("_", " ")} (No-PCA)', fontweight='bold', fontsize=11)
        axes[row, col_no].set_ylabel('True Label')
        axes[row, col_no].set_xlabel('Predicted Label')

        sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=axes[row, col_pca],
                    xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        axes[row, col_pca].set_title(f'{m_name.replace("_", " ")} (With-PCA)', fontweight='bold', fontsize=11)
        axes[row, col_pca].set_ylabel('True Label')
        axes[row, col_pca].set_xlabel('Predicted Label')

    plt.suptitle('Confusion Matrix Comparisons on Held-out Test Set (N=114)', fontweight='bold', fontsize=15, y=1.01)
    plt.tight_layout()
    p9_path = os.path.join(PLOTS_DIR, '09_confusion_matrices.png')
    plt.savefig(p9_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p9_path}")

    # ---------------------------------------------------------
    # Plot 10: Multi-Model ROC Curves (No-PCA vs With-PCA)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    palette = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad', '#d35400', '#16a085', '#f39c12', '#2c3e50', '#c0392b', '#1abc9c']

    # No-PCA ROC
    for (m_name, col) in zip(models_list, palette):
        fpr = results['test_results'][m_name]['no_pca']['roc_fpr']
        tpr = results['test_results'][m_name]['no_pca']['roc_tpr']
        auc_val = results['test_results'][m_name]['no_pca']['roc_auc']
        ax1.plot(fpr, tpr, color=col, linewidth=1.8, label=f'{m_name.replace("_", " ")} (AUC={auc_val:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Chance')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax1.set_title('ROC Curves — No-PCA (Original 30 Features)', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8.5, frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # With-PCA ROC
    for (m_name, col) in zip(models_list, palette):
        fpr = results['test_results'][m_name]['with_pca']['roc_fpr']
        tpr = results['test_results'][m_name]['with_pca']['roc_tpr']
        auc_val = results['test_results'][m_name]['with_pca']['roc_auc']
        ax2.plot(fpr, tpr, color=col, linewidth=1.8, label=f'{m_name.replace("_", " ")} (AUC={auc_val:.3f})')

    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Chance')
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax2.set_title('ROC Curves — With-PCA (Reduced 10 Components)', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8.5, frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p10_path = os.path.join(PLOTS_DIR, '10_roc_curves.png')
    plt.savefig(p10_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p10_path}")

    # ---------------------------------------------------------
    # Plot 11: Precision-Recall Curves (No-PCA vs With-PCA)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for (m_name, col) in zip(models_list, palette):
        rec = results['test_results'][m_name]['no_pca']['pr_recall']
        prec = results['test_results'][m_name]['no_pca']['pr_precision']
        ap_val = results['test_results'][m_name]['no_pca']['avg_precision']
        ax1.plot(rec, prec, color=col, linewidth=1.8, label=f'{m_name.replace("_", " ")} (AP={ap_val:.3f})')

    ax1.set_xlabel('Recall (Sensitivity)', fontweight='bold')
    ax1.set_ylabel('Precision (Positive Predictive Value)', fontweight='bold')
    ax1.set_title('Precision-Recall Curves — No-PCA', fontweight='bold')
    ax1.legend(loc='lower left', fontsize=8.5, frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    for (m_name, col) in zip(models_list, palette):
        rec = results['test_results'][m_name]['with_pca']['pr_recall']
        prec = results['test_results'][m_name]['with_pca']['pr_precision']
        ap_val = results['test_results'][m_name]['with_pca']['avg_precision']
        ax2.plot(rec, prec, color=col, linewidth=1.8, label=f'{m_name.replace("_", " ")} (AP={ap_val:.3f})')

    ax2.set_xlabel('Recall (Sensitivity)', fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontweight='bold')
    ax2.set_title('Precision-Recall Curves — With-PCA', fontweight='bold')
    ax2.legend(loc='lower left', fontsize=8.5, frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    p11_path = os.path.join(PLOTS_DIR, '11_precision_recall_curves.png')
    plt.savefig(p11_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p11_path}")


def save_results_json(pca_info, results):
    """Serialize all numerical findings to JSON."""
    clean_results = {
        'pca_summary': pca_info,
        'cv_5fold_results': results['cv_5fold_results'],
        'tuning_details': results['tuning_details'],
        'test_results': {}
    }

    # Clean test results (omit raw curves from JSON for readability)
    curve_keys = {'roc_fpr', 'roc_tpr', 'pr_precision', 'pr_recall'}
    for m, vals in results['test_results'].items():
        clean_results['test_results'][m] = {
            'no_pca': {k: v for k, v in vals['no_pca'].items() if k not in curve_keys},
            'with_pca': {k: v for k, v in vals['with_pca'].items() if k not in curve_keys}
        }

    json_path = os.path.join(EXP_DIR, 'code', 'experiment7_results.json')
    with open(json_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n[Results Serialized] All numerical tables written to: {json_path}")


def print_summary_tables(pca_info, results):
    """Print formatted ASCII summary tables to console."""
    print("\n" + "="*80)
    print("TABLE 1: PCA VARIANCE EXPLAINED & SCREE ANALYSIS")
    print("="*80)
    print(f"{'PC #':<6}{'Eigenvalue':<14}{'Explained Var (%)':<20}{'Cumulative Var (%)':<20}")
    print("-"*80)
    for r in pca_info['pca_table'][:15]:
        marker = " <-- [CHOSEN CUTOFF (95%)]" if r['component'] == pca_info['chosen_components'] else ""
        print(f"{r['component']:<6}{r['eigenvalue']:<14.4f}{r['explained_variance_pct']:<20.2f}{r['cumulative_variance_pct']:<20.2f}{marker}")

    print("\n" + "="*95)
    print("TABLE 5: 5-FOLD CROSS-VALIDATION RESULTS (NO-PCA VS WITH-PCA)")
    print("="*95)
    header = f"{'Model':<20}{'Fold 1':<8}{'Fold 2':<8}{'Fold 3':<8}{'Fold 4':<8}{'Fold 5':<8}{'Avg(No-PCA)':<14}{'Avg(With-PCA)':<14}{'Delta (%)':<10}"
    print(header)
    print("-"*95)
    for m_name, cv_data in results['cv_5fold_results'].items():
        folds_no = cv_data['no_pca']['folds']
        avg_no = cv_data['no_pca']['mean_accuracy'] * 100
        avg_pca = cv_data['with_pca']['mean_accuracy'] * 100
        delta = cv_data['accuracy_delta_pct']
        f_str = "".join([f"{f*100:<8.1f}" for f in folds_no])
        print(f"{m_name.replace('_', ' '):<20}{f_str}{avg_no:<14.2f}{avg_pca:<14.2f}{delta:<+10.2f}")

    print("\n" + "="*95)
    print("HELD-OUT TEST SET BENCHMARK (N=114)")
    print("="*95)
    print(f"{'Model':<20}{'Acc(No-PCA)':<14}{'Acc(With-PCA)':<16}{'F1(No-PCA)':<14}{'F1(With-PCA)':<14}{'ROC-AUC(PCA)':<14}")
    print("-"*95)
    for m_name, t_data in results['test_results'].items():
        acc_no = t_data['no_pca']['accuracy'] * 100
        acc_pca = t_data['with_pca']['accuracy'] * 100
        f1_no = t_data['no_pca']['f1_score'] * 100
        f1_pca = t_data['with_pca']['f1_score'] * 100
        auc_pca = t_data['with_pca']['roc_auc']
        print(f"{m_name.replace('_', ' '):<20}{acc_no:<14.2f}{acc_pca:<16.2f}{f1_no:<14.2f}{f1_pca:<14.2f}{auc_pca:<14.4f}")


def main():
    start_total = time.time()
    print("="*80)
    print("STARTING EXPERIMENT 7: DIMENSIONALITY REDUCTION & EVALUATION (PCA)")
    print("="*80)

    # 1. Load data
    X_train_df, X_test_df, y_train, y_test, X_train_scaled, X_test_scaled, feature_names = load_and_preprocess_data()
    print(f"[Data Loaded] Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    print(f"[Class Distribution] Train -> Malignant: {y_train.sum()} ({y_train.mean()*100:.1f}%), Benign: {(1-y_train).sum()} ({(1-y_train.mean())*100:.1f}%)")

    # 2. PCA decomposition
    pca_full, pca_selected, X_train_pca, X_test_pca, pca_info = perform_pca_analysis(
        X_train_scaled, X_test_scaled, feature_names
    )

    # 3. Model Tuning & Cross-Validation
    results = tune_and_evaluate_models(
        X_train_scaled, X_test_scaled, X_train_pca, X_test_pca, y_train, y_test
    )

    # 4. Generate Visualizations
    generate_publication_plots(
        pca_full, pca_selected, X_train_scaled, y_train, feature_names, results
    )

    # 5. Serialize and Print
    save_results_json(pca_info, results)
    print_summary_tables(pca_info, results)

    elapsed = time.time() - start_total
    print(f"\n[Experiment 7 Complete] Total execution time: {elapsed:.2f} seconds.")


if __name__ == '__main__':
    main()
