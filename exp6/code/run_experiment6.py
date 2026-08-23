#!/usr/bin/env python3
"""
Experiment 6: Bagging, Boosting, and Stacked Ensemble Models
Course: Machine Learning Algorithms Laboratory (ICS1512)
Student Name: Danusu K | Register Number: 3122247001013
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)

This script implements:
1. Dataset loading, preprocessing, and EDA.
2. Bagging Classifier with hyperparameter tuning (Table 1).
3. Boosting Classifiers (AdaBoost & Gradient Boosting) with hyperparameter tuning (Table 2).
4. Stacked Ensemble with heterogeneous base learners and meta-learners (Table 3).
5. Comprehensive performance evaluation on test set & 5-Fold Stratified CV (Table 4).
6. Bias-Variance Decomposition across models.
7. Generation of 10 high-resolution publication-quality visualization figures.
8. Serialization of all numerical results to JSON.
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

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, log_loss
)

# Set random seed for exact reproducibility
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
    
    # Clean dataframe if column names have different casing
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
    
    return df, feature_names, X, y, X_train, X_test, y_train, y_test


def plot_eda(df, feature_names, y):
    """Generate EDA figures for class distribution and correlation."""
    # Plot 1: Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    counts = y.value_counts().rename({0: 'Benign (B)', 1: 'Malignant (M)'})
    colors = ['#2b5c8f', '#d95f02']
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(y)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 6,
                f'{int(height)} ({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax.set_title('WDBC Dataset: Class Distribution (569 Samples)', pad=15, fontweight='bold')
    ax.set_ylabel('Sample Count')
    ax.set_ylim(0, 420)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_class_distribution.png'), bbox_inches='tight')
    plt.close()
    
    # Plot 2: Top 15 Feature Correlations with Malignancy
    corrs = df[feature_names].apply(lambda col: col.corr(y)).abs().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.barplot(x=corrs.values, y=corrs.index, palette='viridis', ax=ax, edgecolor='black', linewidth=0.8)
    for i, val in enumerate(corrs.values):
        ax.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    ax.set_title('Top 15 Nuclear Features Correlated with Malignancy (|r|)', pad=15, fontweight='bold')
    ax.set_xlabel('Absolute Pearson Correlation Coefficient (|r|)')
    ax.set_xlim(0, 0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_feature_correlation.png'), bbox_inches='tight')
    plt.close()


def evaluate_bagging(X_train, y_train, cv):
    """Explore Bagging hyperparameter search space (Table 1)."""
    n_estimators_list = [5, 10, 25, 50, 100, 150]
    max_samples_list = [0.4, 0.6, 0.8, 1.0]
    
    results = []
    grid_matrix_acc = np.zeros((len(n_estimators_list), len(max_samples_list)))
    
    for i, n_est in enumerate(n_estimators_list):
        for j, m_samp in enumerate(max_samples_list):
            clf = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                n_estimators=n_est,
                max_samples=m_samp,
                max_features=1.0,
                bootstrap=True,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            cv_res = cross_validate(clf, X_train, y_train, cv=cv, scoring=['accuracy', 'f1'])
            acc_mean = cv_res['test_accuracy'].mean()
            f1_mean = cv_res['test_f1'].mean()
            grid_matrix_acc[i, j] = acc_mean
            
            results.append({
                'n_estimators': n_est,
                'max_samples': m_samp,
                'avg_cv_acc': acc_mean * 100,
                'avg_cv_f1': f1_mean
            })
            
    # Plot 3: Bagging Hyperparameter Tuning
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Heatmap
    sns.heatmap(grid_matrix_acc * 100, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'{s:.1f}' for s in max_samples_list],
                yticklabels=n_estimators_list, ax=ax1, cbar_kws={'label': 'Avg CV Accuracy (%)'})
    ax1.set_title('Bagging: 5-Fold CV Accuracy (%) Heatmap', fontweight='bold')
    ax1.set_xlabel('Max Samples Fraction')
    ax1.set_ylabel('Number of Estimators (n_estimators)')
    
    # Line plot: Accuracy vs n_estimators across max_samples
    for j, m_samp in enumerate(max_samples_list):
        ax2.plot(n_estimators_list, grid_matrix_acc[:, j] * 100, marker='o', linewidth=2, label=f'max_samples = {m_samp}')
    ax2.set_title('Bagging: Accuracy vs Number of Trees', fontweight='bold')
    ax2.set_xlabel('Number of Estimators (n_estimators)')
    ax2.set_ylabel('Avg 5-Fold CV Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_bagging_hyperparameter_tuning.png'), bbox_inches='tight')
    plt.close()
    
    return results


def evaluate_boosting(X_train, y_train, cv):
    """Explore Boosting hyperparameter search spaces (Table 2)."""
    n_estimators_list = [10, 25, 50, 100, 200]
    learning_rate_list = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    results = []
    gb_matrix_acc = np.zeros((len(n_estimators_list), len(learning_rate_list)))
    ada_matrix_acc = np.zeros((len(n_estimators_list), len(learning_rate_list)))
    
    for i, n_est in enumerate(n_estimators_list):
        for j, lr in enumerate(learning_rate_list):
            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=3,
                random_state=RANDOM_STATE
            )
            cv_res_gb = cross_validate(gb, X_train, y_train, cv=cv, scoring=['accuracy', 'f1'])
            gb_acc = cv_res_gb['test_accuracy'].mean()
            gb_f1 = cv_res_gb['test_f1'].mean()
            gb_matrix_acc[i, j] = gb_acc
            
            # AdaBoost
            ada = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
                n_estimators=n_est,
                learning_rate=lr,
                random_state=RANDOM_STATE
            )
            cv_res_ada = cross_validate(ada, X_train, y_train, cv=cv, scoring=['accuracy', 'f1'])
            ada_acc = cv_res_ada['test_accuracy'].mean()
            ada_f1 = cv_res_ada['test_f1'].mean()
            ada_matrix_acc[i, j] = ada_acc
            
            results.append({
                'algorithm': 'Gradient Boosting',
                'n_estimators': n_est,
                'learning_rate': lr,
                'avg_cv_acc': gb_acc * 100,
                'avg_cv_f1': gb_f1
            })
            
    # Plot 4: Boosting Hyperparameter Tuning
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    sns.heatmap(gb_matrix_acc * 100, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'{lr:.2f}' for lr in learning_rate_list],
                yticklabels=n_estimators_list, ax=ax1, cbar_kws={'label': 'Avg CV Accuracy (%)'})
    ax1.set_title('Gradient Boosting: 5-Fold CV Accuracy (%)', fontweight='bold')
    ax1.set_xlabel('Learning Rate (η)')
    ax1.set_ylabel('Number of Estimators (n_estimators)')
    
    for j, lr in enumerate(learning_rate_list):
        ax2.plot(n_estimators_list, gb_matrix_acc[:, j] * 100, marker='s', linewidth=2, label=f'η = {lr}')
    ax2.set_title('Gradient Boosting: Convergence vs Estimators', fontweight='bold')
    ax2.set_xlabel('Number of Estimators (n_estimators)')
    ax2.set_ylabel('Avg 5-Fold CV Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_boosting_hyperparameter_tuning.png'), bbox_inches='tight')
    plt.close()
    
    return results


def evaluate_stacking(X_train, y_train, cv):
    """Explore Stacked Ensemble with multiple base models & meta-learners (Table 3)."""
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='rbf', C=1.0, random_state=RANDOM_STATE))
    ])
    nb = GaussianNB()
    dt = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    
    configurations = [
        {
            'name': 'SVM + Naïve Bayes + Decision Tree',
            'base_models_desc': 'SVM, Naïve Bayes, Decision Tree',
            'estimators': [('svm', svm_pipe), ('nb', nb), ('dt', dt)],
            'meta_name': 'Logistic Regression',
            'meta_clf': LogisticRegression(random_state=RANDOM_STATE)
        },
        {
            'name': 'SVM + Naïve Bayes + Decision Tree (Ridge Meta)',
            'base_models_desc': 'SVM, Naïve Bayes, Decision Tree',
            'estimators': [('svm', svm_pipe), ('nb', nb), ('dt', dt)],
            'meta_name': 'Ridge Classifier',
            'meta_clf': RidgeClassifier(random_state=RANDOM_STATE)
        },
        {
            'name': 'SVM + Naïve Bayes + Decision Tree (RF Meta)',
            'base_models_desc': 'SVM, Naïve Bayes, Decision Tree',
            'estimators': [('svm', svm_pipe), ('nb', nb), ('dt', dt)],
            'meta_name': 'Random Forest',
            'meta_clf': RandomForestClassifier(n_estimators=25, max_depth=3, random_state=RANDOM_STATE)
        },
        {
            'name': 'SVM + Decision Tree',
            'base_models_desc': 'SVM, Decision Tree',
            'estimators': [('svm', svm_pipe), ('dt', dt)],
            'meta_name': 'Logistic Regression',
            'meta_clf': LogisticRegression(random_state=RANDOM_STATE)
        },
        {
            'name': 'SVM + Naïve Bayes',
            'base_models_desc': 'SVM, Naïve Bayes',
            'estimators': [('svm', svm_pipe), ('nb', nb)],
            'meta_name': 'Logistic Regression',
            'meta_clf': LogisticRegression(random_state=RANDOM_STATE)
        },
        {
            'name': 'Naïve Bayes + Decision Tree',
            'base_models_desc': 'Naïve Bayes, Decision Tree',
            'estimators': [('nb', nb), ('dt', dt)],
            'meta_name': 'Logistic Regression',
            'meta_clf': LogisticRegression(random_state=RANDOM_STATE)
        }
    ]
    
    results = []
    
    for config in configurations:
        stack_clf = StackingClassifier(
            estimators=config['estimators'],
            final_estimator=config['meta_clf'],
            cv=cv,
            n_jobs=-1
        )
        cv_res = cross_validate(stack_clf, X_train, y_train, cv=cv, scoring=['accuracy', 'f1'])
        acc_mean = cv_res['test_accuracy'].mean()
        f1_mean = cv_res['test_f1'].mean()
        
        results.append({
            'base_models': config['base_models_desc'],
            'meta_learner': config['meta_name'],
            'avg_cv_acc': acc_mean * 100,
            'avg_cv_f1': f1_mean
        })
        
    # Plot 5: Stacking Meta-Learner Weights & Prediction Correlation
    opt_stack = StackingClassifier(
        estimators=[('svm', svm_pipe), ('nb', nb), ('dt', dt)],
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        cv=cv,
        n_jobs=-1
    )
    opt_stack.fit(X_train, y_train)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    meta_coefs = opt_stack.final_estimator_.coef_[0]
    base_names = ['SVM (RBF)', 'Naïve Bayes', 'Decision Tree']
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    bars = ax1.bar(base_names, meta_coefs, color=colors, edgecolor='black', width=0.45)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + (0.05 if h >= 0 else -0.15),
                 f'{h:.3f}', ha='center', va='bottom' if h >= 0 else 'top', fontweight='bold')
    ax1.set_title('Meta-Learner (Logistic Regression) Feature Weights', fontweight='bold')
    ax1.set_ylabel('Weight / Coefficient Magnitude')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    svm_pipe.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    
    preds_df = pd.DataFrame({
        'SVM': svm_pipe.predict_proba(X_train)[:, 1],
        'Naïve Bayes': nb.predict_proba(X_train)[:, 1],
        'Decision Tree': dt.predict_proba(X_train)[:, 1]
    })
    corr_matrix = preds_df.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', vmin=0.5, vmax=1.0, ax=ax2)
    ax2.set_title('Base Learner Prediction Diversity (Correlation)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_stacking_architecture_and_weights.png'), bbox_inches='tight')
    plt.close()
    
    return results


def perform_bias_variance_decomposition(models, X_train, y_train, X_test, y_test, n_bootstraps=100):
    """Empirically estimate bias, variance, and expected loss using bootstrap sampling."""
    bv_results = {}
    
    X_tr_mat = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_tr_mat = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_te_mat = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_te_mat = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    n_samples = len(X_tr_mat)
    n_test = len(X_te_mat)
    
    for name, clf in models.items():
        all_preds = np.zeros((n_bootstraps, n_test))
        
        for b in range(n_bootstraps):
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_tr_mat[boot_idx]
            y_boot = y_tr_mat[boot_idx]
            
            clf_clone = sklearn_clone(clf)
            clf_clone.fit(X_boot, y_boot)
            all_preds[b, :] = clf_clone.predict(X_te_mat)
            
        main_pred = (np.mean(all_preds, axis=0) >= 0.5).astype(int)
        expected_loss = np.mean(all_preds != y_te_mat[np.newaxis, :])
        bias = np.mean(main_pred != y_te_mat)
        variance = np.mean(all_preds != main_pred[np.newaxis, :])
        
        bv_results[name] = {
            'expected_loss': float(expected_loss),
            'bias': float(bias),
            'variance': float(variance)
        }
        
    names = list(bv_results.keys())
    biases = [bv_results[m]['bias'] for m in names]
    variances = [bv_results[m]['variance'] for m in names]
    losses = [bv_results[m]['expected_loss'] for m in names]
    
    x = np.arange(len(names))
    width = 0.28
    
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    rects1 = ax.bar(x - width, biases, width, label='Squared Bias / 0-1 Bias', color='#e41a1c', edgecolor='black')
    rects2 = ax.bar(x, variances, width, label='Variance', color='#377eb8', edgecolor='black')
    rects3 = ax.bar(x + width, losses, width, label='Total Expected Error', color='#4daf4a', edgecolor='black')
    
    ax.set_ylabel('Error / Decomposition Component')
    ax.set_title('Bias–Variance Decomposition Across Ensemble Architectures', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, max(losses) * 1.35)
    
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.001,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_bias_variance_decomposition.png'), bbox_inches='tight')
    plt.close()
    
    return bv_results


def evaluate_final_models(X_train, y_train, X_test, y_test, cv):
    """Train optimal configurations, evaluate on test set, run 5-Fold CV, and create plots."""
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, kernel='rbf', C=1.0, random_state=RANDOM_STATE))
    ])
    nb = GaussianNB()
    dt_base = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    
    models = {
        'Decision Tree (Baseline)': DecisionTreeClassifier(
            criterion='entropy', max_depth=4, min_samples_split=2, min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        'Bagging Classifier': BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=50,
            max_samples=0.8,
            max_features=1.0,
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'AdaBoost Classifier': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE
        ),
        'Stacked Ensemble': StackingClassifier(
            estimators=[('svm', svm_pipe), ('nb', nb), ('dt', dt_base)],
            final_estimator=LogisticRegression(random_state=RANDOM_STATE),
            cv=cv,
            n_jobs=-1
        )
    }
    
    table4_results = []
    cv_fold_scores = {name: [] for name in models.keys()}
    test_metrics = {}
    fitted_models = {}
    
    # 5-Fold Cross Validation Fold-by-Fold
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        for name, clf in models.items():
            clf_clone = sklearn_clone(clf)
            clf_clone.fit(X_tr, y_tr)
            fold_acc = accuracy_score(y_va, clf_clone.predict(X_va))
            cv_fold_scores[name].append(fold_acc)
            
    # Test set evaluation
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        fitted_models[name] = clf
        
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = clf.decision_function(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp)
        loss = log_loss(y_test, y_prob) if hasattr(clf, "predict_proba") else 0.0
        
        cv_mean = np.mean(cv_fold_scores[name]) * 100
        cv_std = np.std(cv_fold_scores[name]) * 100
        
        test_metrics[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'roc_auc': float(auc),
            'specificity': float(spec),
            'log_loss': float(loss),
            'fit_time_sec': float(fit_time),
            'confusion_matrix': cm.tolist(),
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'fold_scores': [float(s * 100) for s in cv_fold_scores[name]]
        }
        
        table4_results.append({
            'model': name,
            'test_acc': acc * 100,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'cv_acc_mean': cv_mean,
            'cv_acc_std': cv_std
        })
        
    # Plot 7: 5-Fold Cross-Validation Comparison
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    box_data = [np.array(cv_fold_scores[name]) * 100 for name in models.keys()]
    bp = ax.boxplot(box_data, patch_artist=True, labels=[n.replace(' Classifier', '') for n in models.keys()])
    colors = ['#aec7e8', '#1f77b4', '#ffbb78', '#ff7f0e', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        
    for i, name in enumerate(models.keys()):
        mean_val = np.mean(cv_fold_scores[name]) * 100
        std_val = np.std(cv_fold_scores[name]) * 100
        ax.scatter(i + 1, mean_val, color='darkred', s=60, zorder=5)
        ax.text(i + 1, mean_val + 0.4, f'{mean_val:.2f}±{std_val:.2f}%', ha='center', fontweight='bold', fontsize=9)
        
    ax.set_title('5-Fold Stratified Cross-Validation Accuracy Distribution', pad=15, fontweight='bold')
    ax.set_ylabel('CV Accuracy (%)')
    ax.set_ylim(88, 101)
    plt.xticks(rotation=15, ha='right', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_kfold_cv_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # Plot 8: Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), dpi=300)
    for idx, (name, clf) in enumerate(fitted_models.items()):
        cm = confusion_matrix(y_test, clf.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                    annot_kws={'fontsize': 13, 'fontweight': 'bold'})
        axes[idx].set_title(name.replace(' Classifier', ''), fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Predicted Label')
        if idx == 0:
            axes[idx].set_ylabel('True Label')
        axes[idx].set_xticklabels(['Benign (0)', 'Malignant (1)'])
        axes[idx].set_yticklabels(['Benign (0)', 'Malignant (1)'])
    plt.suptitle('Test Set Confusion Matrices Comparison (N=114)', y=1.05, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_confusion_matrices.png'), bbox_inches='tight')
    plt.close()
    
    # Plot 9: Multi-Model ROC Curves
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
    palette = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    
    for idx, (name, clf) in enumerate(fitted_models.items()):
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', color=palette[idx], linewidth=2.2)
        
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Chance (AUC = 0.5000)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', pad=15, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.03])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '09_roc_curves.png'), bbox_inches='tight')
    plt.close()
    
    # Plot 10: Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
    for idx, (name, clf) in enumerate(fitted_models.items()):
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = clf.decision_function(X_test)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec_curve, prec_curve, label=f'{name} (AP = {ap:.4f})', color=palette[idx], linewidth=2.2)
        
    ax.set_title('Precision-Recall Curves (Clinical Malignancy Detection)', pad=15, fontweight='bold')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (Positive Predictive Value)')
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.7, 1.03])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '10_precision_recall_curves.png'), bbox_inches='tight')
    plt.close()
    
    return table4_results, test_metrics, models


def sklearn_clone(clf):
    """Utility to clone sklearn estimators including StackingClassifier."""
    from sklearn.base import clone
    try:
        return clone(clf)
    except Exception:
        if isinstance(clf, StackingClassifier):
            return StackingClassifier(
                estimators=clf.estimators,
                final_estimator=clone(clf.final_estimator),
                cv=clf.cv,
                n_jobs=-1
            )
        return clone(clf)


def main():
    print("=" * 75)
    print("EXPERIMENT 6: BAGGING, BOOSTING, AND STACKED ENSEMBLE MODELS")
    print("=" * 75)
    
    # 1. Load Data
    print("\n[Step 1] Loading and Preprocessing WDBC Dataset...")
    df, feature_names, X, y, X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Class Distribution: {np.sum(y == 0)} Benign (0), {np.sum(y == 1)} Malignant (1).")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # 2. EDA
    print("\n[Step 2] Generating EDA Visualizations...")
    plot_eda(df, feature_names, y)
    print("  -> Saved 01_class_distribution.png and 02_feature_correlation.png")
    
    # 3. Stratified 5-Fold CV Setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # 4. Bagging Hyperparameter Evaluation (Table 1)
    print("\n[Step 3] Evaluating Bagging Classifier Hyperparameters (Table 1)...")
    bagging_results = evaluate_bagging(X_train, y_train, cv)
    print("  -> Top Bagging Configurations:")
    sorted_bagging = sorted(bagging_results, key=lambda x: x['avg_cv_acc'], reverse=True)
    for res in sorted_bagging[:5]:
        print(f"     n_estimators={res['n_estimators']}, max_samples={res['max_samples']} => CV Acc: {res['avg_cv_acc']:.2f}%, F1: {res['avg_cv_f1']:.4f}")
        
    # 5. Boosting Hyperparameter Evaluation (Table 2)
    print("\n[Step 4] Evaluating Boosting Classifiers Hyperparameters (Table 2)...")
    boosting_results = evaluate_boosting(X_train, y_train, cv)
    print("  -> Top Gradient Boosting Configurations:")
    sorted_boosting = sorted(boosting_results, key=lambda x: x['avg_cv_acc'], reverse=True)
    for res in sorted_boosting[:5]:
        print(f"     n_estimators={res['n_estimators']}, learning_rate={res['learning_rate']} => CV Acc: {res['avg_cv_acc']:.2f}%, F1: {res['avg_cv_f1']:.4f}")
        
    # 6. Stacked Ensemble Evaluation (Table 3)
    print("\n[Step 5] Evaluating Stacked Ensemble Configurations (Table 3)...")
    stacking_results = evaluate_stacking(X_train, y_train, cv)
    for res in stacking_results:
        print(f"     Base: [{res['base_models']}] | Meta: {res['meta_learner']} => CV Acc: {res['avg_cv_acc']:.2f}%, F1: {res['avg_cv_f1']:.4f}")
        
    # 7. Final Models Evaluation (Table 4)
    print("\n[Step 6] Evaluating Optimal Models on Held-out Test Set (Table 4)...")
    table4_results, test_metrics, final_models = evaluate_final_models(X_train, y_train, X_test, y_test, cv)
    
    print("\n" + "=" * 80)
    print("TABLE 4: PERFORMANCE COMPARISON OF ENSEMBLE MODELS")
    print("=" * 80)
    print(f"{'Model':<30} | {'Acc (%)':<8} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'5-Fold CV (%)':<15}")
    print("-" * 96)
    for r in table4_results:
        print(f"{r['model']:<30} | {r['test_acc']:<8.2f} | {r['precision']:<10.4f} | {r['recall']:<8.4f} | {r['f1_score']:<8.4f} | {r['roc_auc']:<8.4f} | {r['cv_acc_mean']:.2f} ± {r['cv_acc_std']:.2f}%")
    print("=" * 80)
    
    # 8. Bias-Variance Decomposition
    print("\n[Step 7] Running Bias-Variance Decomposition (100 Bootstrap Iterations)...")
    bv_results = perform_bias_variance_decomposition(final_models, X_train, y_train, X_test, y_test, n_bootstraps=100)
    for m_name, bv in bv_results.items():
        print(f"     {m_name:<26} => Bias: {bv['bias']:.4f} | Variance: {bv['variance']:.4f} | Expected Loss: {bv['expected_loss']:.4f}")
        
    # 9. Save Complete Results to JSON
    output_json_path = os.path.join(EXP_DIR, 'code', 'experiment6_results.json')
    complete_results = {
        'experiment': 'Experiment 6: Bagging, Boosting, and Stacked Ensemble Models',
        'dataset': 'Wisconsin Diagnostic Breast Cancer (WDBC)',
        'samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'bagging_table1': bagging_results,
        'boosting_table2': boosting_results,
        'stacking_table3': stacking_results,
        'comparison_table4': table4_results,
        'test_metrics_detail': test_metrics,
        'bias_variance_decomposition': bv_results
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(complete_results, f, indent=4)
        
    print(f"\n[Step 8] All experimental metrics serialized to {output_json_path}")
    print(f"All 10 figures successfully generated in {PLOTS_DIR}")
    print("=" * 75)
    print("EXPERIMENT 6 COMPLETED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == '__main__':
    main()
