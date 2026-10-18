"""
CAT Assignment 1 - Comprehensive Implementation Script
ICS1502: Introduction to Machine Learning
Department of Computer Science and Engineering, SSN College of Engineering
Student: Danusu K (3122247001013)
Faculty: Dr. Poreddy Ajay Kumar Reddy

Part 1: Regression on Mobile Phone Price Prediction Dataset
Part 2: Linear Classification on Banknote Authentication Dataset
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

# Set plotting style for academic presentation
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

results_dict = {}

print("=" * 80)
print("CAT-1 EXPERIMENT: REGRESSION AND CLASSIFICATION PIPELINE")
print("=" * 80)

# ==============================================================================
# PART 1: REGRESSION (Mobile Phone Price Prediction)
# ==============================================================================
print("\n>>> [1/2] RUNNING PART 1: REGRESSION ANALYSIS...")

mobile_path = os.path.join(DATASET_DIR, "Mobile-Price-Prediction (CAT-1-Ass).csv")
df_mobile = pd.read_csv(mobile_path)

print(f"Loaded Mobile Price Dataset: {df_mobile.shape[0]} samples, {df_mobile.shape[1]} features.")
# Handling missing values via median imputation
imputation_values = {}
for col in df_mobile.columns:
    if df_mobile[col].isnull().any():
        med_val = df_mobile[col].median()
        imputation_values[col] = float(med_val)
        df_mobile[col] = df_mobile[col].fillna(med_val)

feature_cols = [c for c in df_mobile.columns if c != 'Price']
X_raw = df_mobile[feature_cols].values
y_raw = df_mobile['Price'].values.reshape(-1, 1)

# Divide into Train and Test sets (80-20 split, seed=42)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Construct Design Matrices (Prepend column of 1s for bias term)
N_tr, D = X_train_raw.shape
X_train_b_unscaled = np.hstack([np.ones((N_tr, 1)), X_train_raw])
X_test_b_unscaled = np.hstack([np.ones((X_test_raw.shape[0], 1)), X_test_raw])

X_train_b_scaled = np.hstack([np.ones((N_tr, 1)), X_train_scaled])
X_test_b_scaled = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

# Gram matrices and condition numbers
cond_unscaled = float(np.linalg.cond(X_train_b_unscaled.T @ X_train_b_unscaled))
cond_scaled = float(np.linalg.cond(X_train_b_scaled.T @ X_train_b_scaled))
print(f"Condition Number (Unscaled X^T X): {cond_unscaled:.3e}")
print(f"Condition Number (Scaled X^T X):   {cond_scaled:.3f}")

# 1. Closed-Form Solution (Unregularized)
# w* = (X^T X)^-1 X^T y
w_cf_scaled = np.linalg.inv(X_train_b_scaled.T @ X_train_b_scaled) @ (X_train_b_scaled.T @ y_train)
y_pred_train_cf = X_train_b_scaled @ w_cf_scaled
y_pred_test_cf = X_test_b_scaled @ w_cf_scaled

cf_metrics = {
    'train_mse': float(mean_squared_error(y_train, y_pred_train_cf)),
    'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train_cf))),
    'train_mae': float(mean_absolute_error(y_train, y_pred_train_cf)),
    'train_r2': float(r2_score(y_train, y_pred_train_cf)),
    'test_mse': float(mean_squared_error(y_test, y_pred_test_cf)),
    'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test_cf))),
    'test_mae': float(mean_absolute_error(y_test, y_pred_test_cf)),
    'test_r2': float(r2_score(y_test, y_pred_test_cf)),
}
print(f"Closed-Form Test RMSE: {cf_metrics['test_rmse']:.2f}, R2: {cf_metrics['test_r2']:.4f}")

# 2. Gradient Descent Implementation
def gradient_descent_linear_regression(X, y, alpha=0.05, epochs=3000, l2=0.0):
    N, D_dim = X.shape
    w = np.zeros((D_dim, 1))
    cost_history = []
    I_reg = np.eye(D_dim)
    I_reg[0, 0] = 0.0 # Bias w0 is not penalized
    for epoch in range(epochs):
        pred = X @ w
        error = pred - y
        grad = (1.0 / N) * (X.T @ error) + (l2 / N) * (I_reg @ w)
        w -= alpha * grad
        cost = (1.0 / (2.0 * N)) * np.sum(error ** 2) + (l2 / (2.0 * N)) * np.sum(w[1:] ** 2)
        cost_history.append(float(cost))
    return w, cost_history

w_gd_scaled, cost_history_gd = gradient_descent_linear_regression(
    X_train_b_scaled, y_train, alpha=0.05, epochs=3500, l2=0.0
)
y_pred_train_gd = X_train_b_scaled @ w_gd_scaled
y_pred_test_gd = X_test_b_scaled @ w_gd_scaled

gd_metrics = {
    'train_mse': float(mean_squared_error(y_train, y_pred_train_gd)),
    'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train_gd))),
    'train_mae': float(mean_absolute_error(y_train, y_pred_train_gd)),
    'train_r2': float(r2_score(y_train, y_pred_train_gd)),
    'test_mse': float(mean_squared_error(y_test, y_pred_test_gd)),
    'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test_gd))),
    'test_mae': float(mean_absolute_error(y_test, y_pred_test_gd)),
    'test_r2': float(r2_score(y_test, y_pred_test_gd)),
}
print(f"Gradient Descent Test RMSE: {gd_metrics['test_rmse']:.2f}, R2: {gd_metrics['test_r2']:.4f}")

# 4. L2 Regularization (Ridge) Closed-Form and Gradient Descent
best_lambda = 10.0
I_reg = np.eye(X_train_b_scaled.shape[1])
I_reg[0, 0] = 0.0

w_ridge_cf = np.linalg.inv(X_train_b_scaled.T @ X_train_b_scaled + best_lambda * I_reg) @ (X_train_b_scaled.T @ y_train)
y_pred_train_ridge_cf = X_train_b_scaled @ w_ridge_cf
y_pred_test_ridge_cf = X_test_b_scaled @ w_ridge_cf

ridge_cf_metrics = {
    'train_mse': float(mean_squared_error(y_train, y_pred_train_ridge_cf)),
    'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train_ridge_cf))),
    'train_mae': float(mean_absolute_error(y_train, y_pred_train_ridge_cf)),
    'train_r2': float(r2_score(y_train, y_pred_train_ridge_cf)),
    'test_mse': float(mean_squared_error(y_test, y_pred_test_ridge_cf)),
    'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test_ridge_cf))),
    'test_mae': float(mean_absolute_error(y_test, y_pred_test_ridge_cf)),
    'test_r2': float(r2_score(y_test, y_pred_test_ridge_cf)),
}

w_ridge_gd, cost_history_ridge_gd = gradient_descent_linear_regression(
    X_train_b_scaled, y_train, alpha=0.05, epochs=3500, l2=best_lambda
)
y_pred_train_ridge_gd = X_train_b_scaled @ w_ridge_gd
y_pred_test_ridge_gd = X_test_b_scaled @ w_ridge_gd

ridge_gd_metrics = {
    'train_mse': float(mean_squared_error(y_train, y_pred_train_ridge_gd)),
    'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train_ridge_gd))),
    'train_mae': float(mean_absolute_error(y_train, y_pred_train_ridge_gd)),
    'train_r2': float(r2_score(y_train, y_pred_train_ridge_gd)),
    'test_mse': float(mean_squared_error(y_test, y_pred_test_ridge_gd)),
    'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test_ridge_gd))),
    'test_mae': float(mean_absolute_error(y_test, y_pred_test_ridge_gd)),
    'test_r2': float(r2_score(y_test, y_pred_test_ridge_gd)),
}
print(f"Ridge Closed-Form Test RMSE: {ridge_cf_metrics['test_rmse']:.2f}, R2: {ridge_cf_metrics['test_r2']:.4f}")
print(f"Ridge GD Test RMSE:          {ridge_gd_metrics['test_rmse']:.2f}, R2: {ridge_gd_metrics['test_r2']:.4f}")

# 5. Comparison: With vs Without Data Standardization under L2
# Unstandardized Ridge Closed-Form
w_ridge_unscaled = np.linalg.pinv(X_train_b_unscaled.T @ X_train_b_unscaled + best_lambda * I_reg) @ (X_train_b_unscaled.T @ y_train)
y_pred_test_ridge_unscaled = X_test_b_unscaled @ w_ridge_unscaled
ridge_unscaled_metrics = {
    'test_mse': float(mean_squared_error(y_test, y_pred_test_ridge_unscaled)),
    'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test_ridge_unscaled))),
    'test_mae': float(mean_absolute_error(y_test, y_pred_test_ridge_unscaled)),
    'test_r2': float(r2_score(y_test, y_pred_test_ridge_unscaled)),
}

# 6. Lambda Sweep Evaluation
lambda_list = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0]
sweep_results = []
pred_by_lambda = {}
for lam in lambda_list:
    w_lam = np.linalg.inv(X_train_b_scaled.T @ X_train_b_scaled + lam * I_reg) @ (X_train_b_scaled.T @ y_train)
    y_tr_p = X_train_b_scaled @ w_lam
    y_te_p = X_test_b_scaled @ w_lam
    tr_rmse = np.sqrt(mean_squared_error(y_train, y_tr_p))
    te_rmse = np.sqrt(mean_squared_error(y_test, y_te_p))
    te_r2 = r2_score(y_test, y_te_p)
    sweep_results.append({
        'lambda': lam,
        'train_rmse': float(tr_rmse),
        'test_rmse': float(te_rmse),
        'test_r2': float(te_r2),
        'weights': [float(x) for x in w_lam.ravel()]
    })
    pred_by_lambda[lam] = y_te_p

# 7. Feature Importance Analysis
weights_ridge_std = w_ridge_cf[1:].ravel()
feat_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Weight': weights_ridge_std,
    'Abs_Weight': np.abs(weights_ridge_std)
}).sort_values(by='Abs_Weight', ascending=False)
print("\nFeature Importance from L2 Weights (Standardized):")
print(feat_importance_df.to_string(index=False))

# Store Part 1 Results
results_dict['part1_regression'] = {
    'closed_form': cf_metrics,
    'gradient_descent': gd_metrics,
    'ridge_closed_form': ridge_cf_metrics,
    'ridge_gradient_descent': ridge_gd_metrics,
    'ridge_unscaled': ridge_unscaled_metrics,
    'condition_numbers': {'unscaled': cond_unscaled, 'scaled': cond_scaled},
    'best_lambda': best_lambda,
    'lambda_sweep': sweep_results,
    'feature_importance': feat_importance_df.to_dict(orient='records')
}

# ==============================================================================
# PART 2: LINEAR CLASSIFICATION (Banknote Authentication)
# ==============================================================================
print("\n>>> [2/2] RUNNING PART 2: LINEAR CLASSIFICATION ANALYSIS...")

bank_path = os.path.join(DATASET_DIR, "data_banknote_authentication.txt")
bank_cols = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df_bank = pd.read_csv(bank_path, header=None, names=bank_cols)

X_bank = df_bank[['variance', 'skewness', 'curtosis', 'entropy']].values
y_bank = df_bank['class'].values

# 1. Stratified Train-Test Split (80-20, seed=42)
X_bank_tr, X_bank_te, y_bank_tr, y_bank_te = train_test_split(
    X_bank, y_bank, test_size=0.20, random_state=42, stratify=y_bank
)

scaler_bank = StandardScaler()
X_bank_tr_s = scaler_bank.fit_transform(X_bank_tr)
X_bank_te_s = scaler_bank.transform(X_bank_te)

# 2. Fit Classification Models: Unregularized vs L2 Regularized
# Unregularized (Large C = 1e5)
clf_unreg = LogisticRegression(C=1e5, solver='lbfgs', max_iter=2000, random_state=42)
clf_unreg.fit(X_bank_tr_s, y_bank_tr)
y_pred_te_unreg = clf_unreg.predict(X_bank_te_s)
y_prob_te_unreg = clf_unreg.predict_proba(X_bank_te_s)[:, 1]

# L2 Regularized (C = 1.0)
clf_l2 = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=42)
clf_l2.fit(X_bank_tr_s, y_bank_tr)
y_pred_te_l2 = clf_l2.predict(X_bank_te_s)
y_prob_te_l2 = clf_l2.predict_proba(X_bank_te_s)[:, 1]

def evaluate_classifier(clf, X_tr, y_tr, X_te, y_te):
    y_pr_tr = clf.predict(X_tr)
    y_pr_te = clf.predict(X_te)
    y_pb_te = clf.predict_proba(X_te)[:, 1]
    return {
        'train_acc': float(accuracy_score(y_tr, y_pr_tr)),
        'test_acc': float(accuracy_score(y_te, y_pr_te)),
        'precision': float(precision_score(y_te, y_pr_te)),
        'recall': float(recall_score(y_te, y_pr_te)),
        'f1': float(f1_score(y_te, y_pr_te)),
        'roc_auc': float(roc_auc_score(y_te, y_pb_te)),
        'confusion_matrix': confusion_matrix(y_te, y_pr_te).tolist(),
        'weights': [float(w) for w in clf.coef_[0]],
        'intercept': float(clf.intercept_[0])
    }

metrics_unreg = evaluate_classifier(clf_unreg, X_bank_tr_s, y_bank_tr, X_bank_te_s, y_bank_te)
metrics_l2 = evaluate_classifier(clf_l2, X_bank_tr_s, y_bank_tr, X_bank_te_s, y_bank_te)

print(f"Unregularized - Train Acc: {metrics_unreg['train_acc']:.4f}, Test Acc: {metrics_unreg['test_acc']:.4f}")
print(f"L2 Regularized - Train Acc: {metrics_l2['train_acc']:.4f}, Test Acc: {metrics_l2['test_acc']:.4f}")

# 3. Accuracy vs Lambda Sweep
lambda_cls_sweep = np.logspace(-4, 4, 30)
Cs = 1.0 / lambda_cls_sweep
cls_sweep_results = []
for lam, c in zip(lambda_cls_sweep, Cs):
    m = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, random_state=42)
    m.fit(X_bank_tr_s, y_bank_tr)
    tr_acc = accuracy_score(y_bank_tr, m.predict(X_bank_tr_s))
    te_acc = accuracy_score(y_bank_te, m.predict(X_bank_te_s))
    w_norm = np.linalg.norm(m.coef_[0])
    cls_sweep_results.append({
        'lambda': float(lam),
        'C': float(c),
        'train_acc': float(tr_acc),
        'test_acc': float(te_acc),
        'weight_norm': float(w_norm)
    })

# 4. Feature Importance for Banknote
bank_features = ['variance', 'skewness', 'curtosis', 'entropy']
bank_importance_df = pd.DataFrame({
    'Feature': bank_features,
    'Weight': clf_l2.coef_[0],
    'Abs_Weight': np.abs(clf_l2.coef_[0])
}).sort_values(by='Abs_Weight', ascending=False)
print("\nBanknote Feature Importance:")
print(bank_importance_df.to_string(index=False))

# 5 & 6. Intentionally Introduce Outliers
np.random.seed(42)
outlier_ratios = [0.0, 0.02, 0.05, 0.08, 0.12, 0.15]
outlier_experiments = []

# Selected single representative outlier dataset (8% corruption)
N_bank_tr = len(y_bank_tr)
n_outliers_rep = int(0.08 * N_bank_tr)
idx_outliers = np.random.choice(N_bank_tr, n_outliers_rep, replace=False)
X_bank_tr_out = X_bank_tr_s.copy()
y_bank_tr_out = y_bank_tr.copy()

# Shift coordinates by 6 to 9 standard deviations and flip the label
shift_vectors = np.random.uniform(6.0, 9.0, size=(n_outliers_rep, 4)) * np.random.choice([-1, 1], size=(n_outliers_rep, 4))
X_bank_tr_out[idx_outliers] += shift_vectors
y_bank_tr_out[idx_outliers] = 1 - y_bank_tr_out[idx_outliers]

# Train models on outlier-corrupted data
clf_out_unreg = LogisticRegression(C=1e5, solver='lbfgs', max_iter=2000, random_state=42)
clf_out_unreg.fit(X_bank_tr_out, y_bank_tr_out)

clf_out_l2 = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=42)
clf_out_l2.fit(X_bank_tr_out, y_bank_tr_out)

metrics_out_unreg = evaluate_classifier(clf_out_unreg, X_bank_tr_out, y_bank_tr_out, X_bank_te_s, y_bank_te)
metrics_out_l2 = evaluate_classifier(clf_out_l2, X_bank_tr_out, y_bank_tr_out, X_bank_te_s, y_bank_te)

print(f"\n[8% Outliers] Unregularized Test Acc on Clean Data: {metrics_out_unreg['test_acc']:.4f}")
print(f"[8% Outliers] L2 Regularized Test Acc on Clean Data: {metrics_out_l2['test_acc']:.4f}")

# Multi-ratio evaluation for sensitivity analysis
for ratio in outlier_ratios:
    n_out = int(ratio * N_bank_tr)
    X_cur = X_bank_tr_s.copy()
    y_cur = y_bank_tr.copy()
    if n_out > 0:
        idxs = np.random.choice(N_bank_tr, n_out, replace=False)
        shifts = np.random.uniform(6.0, 9.0, size=(n_out, 4)) * np.random.choice([-1, 1], size=(n_out, 4))
        X_cur[idxs] += shifts
        y_cur[idxs] = 1 - y_cur[idxs]
    
    m_un = LogisticRegression(C=1e5, solver='lbfgs', max_iter=2000, random_state=42).fit(X_cur, y_cur)
    m_l2 = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=42).fit(X_cur, y_cur)
    
    outlier_experiments.append({
        'ratio': ratio,
        'unreg_test_acc': float(accuracy_score(y_bank_te, m_un.predict(X_bank_te_s))),
        'l2_test_acc': float(accuracy_score(y_bank_te, m_l2.predict(X_bank_te_s))),
        'unreg_weights': [float(x) for x in m_un.coef_[0]],
        'l2_weights': [float(x) for x in m_l2.coef_[0]]
    })

results_dict['part2_classification'] = {
    'unregularized': metrics_unreg,
    'l2_regularized': metrics_l2,
    'lambda_sweep': cls_sweep_results,
    'feature_importance': bank_importance_df.to_dict(orient='records'),
    'outlier_metrics_8pct': {
        'unregularized': metrics_out_unreg,
        'l2_regularized': metrics_out_l2
    },
    'outlier_sensitivity': outlier_experiments
}

# ==============================================================================
# VISUALIZATION GENERATION (12 High-Resolution Diagnostic Figures)
# ==============================================================================
print("\n>>> GENERATING ALL 12 DIAGNOSTIC PLOTS...")

# ------------------------------------------------------------------------------
# Plot 1: Regression Target Distribution and Feature Correlations
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_mobile['Price'], kde=True, ax=axes[0], color='#1f77b4', bins=35)
axes[0].set_title('Mobile Phone Price Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Price (INR)', fontsize=11)
axes[0].set_ylabel('Sample Count', fontsize=11)
axes[0].axvline(df_mobile['Price'].median(), color='red', linestyle='--', label=f'Median: {df_mobile["Price"].median():,.0f}')
axes[0].legend()

corr_matrix = df_mobile.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', ax=axes[1], cbar=True)
axes[1].set_title('Feature Correlation Matrix with Price', fontsize=13, fontweight='bold')
plt.tight_layout()
p1_path = os.path.join(OUTPUT_DIR, '01_regression_target_and_correlations.png')
plt.savefig(p1_path, dpi=300)
plt.close()
print("Saved:", p1_path)

# ------------------------------------------------------------------------------
# Plot 2: Gradient Descent Cost Convergence
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
epochs_axis = np.arange(1, len(cost_history_gd) + 1)
ax.plot(epochs_axis[:1000], cost_history_gd[:1000], label='Unregularized GD (alpha=0.05)', color='#1f77b4', lw=2)
ax.plot(epochs_axis[:1000], cost_history_ridge_gd[:1000], label=f'L2 Regularized GD (lambda={best_lambda}, alpha=0.05)', color='#d62728', linestyle='--', lw=2)
ax.set_title('Gradient Descent Cost Convergence Curve (First 1,000 Iterations)', fontsize=13, fontweight='bold')
ax.set_xlabel('Iteration Index (Epoch)', fontsize=11)
ax.set_ylabel('Mean Squared Cost J(w)', fontsize=11)
ax.set_yscale('log')
ax.legend(fontsize=10)
plt.tight_layout()
p2_path = os.path.join(OUTPUT_DIR, '02_gd_convergence_curves.png')
plt.savefig(p2_path, dpi=300)
plt.close()
print("Saved:", p2_path)

# ------------------------------------------------------------------------------
# Plot 3: Predicted vs Actual on Test Data (Closed-Form vs Gradient Descent)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
max_val = max(np.max(y_test), np.max(y_pred_test_cf), np.max(y_pred_test_gd)) * 1.05

# Closed Form
axes[0].scatter(y_test, y_pred_test_cf, alpha=0.6, color='#2ca02c', edgecolors='k', s=45)
axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal Fit (y = x)')
axes[0].set_title(f'Closed-Form Solution\n(R² = {cf_metrics["test_r2"]:.4f}, RMSE = {cf_metrics["test_rmse"]:,.1f})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual Mobile Price (INR)', fontsize=11)
axes[0].set_ylabel('Predicted Mobile Price (INR)', fontsize=11)
axes[0].legend()
axes[0].set_xlim(0, max_val)
axes[0].set_ylim(0, max_val)

# Gradient Descent
axes[1].scatter(y_test, y_pred_test_gd, alpha=0.6, color='#1f77b4', edgecolors='k', s=45)
axes[1].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal Fit (y = x)')
axes[1].set_title(f'Gradient Descent Solution\n(R² = {gd_metrics["test_r2"]:.4f}, RMSE = {gd_metrics["test_rmse"]:,.1f})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Actual Mobile Price (INR)', fontsize=11)
axes[1].set_ylabel('Predicted Mobile Price (INR)', fontsize=11)
axes[1].legend()
axes[1].set_xlim(0, max_val)
axes[1].set_ylim(0, max_val)

plt.tight_layout()
p3_path = os.path.join(OUTPUT_DIR, '03_pred_vs_actual_unregularized.png')
plt.savefig(p3_path, dpi=300)
plt.close()
print("Saved:", p3_path)

# ------------------------------------------------------------------------------
# Plot 4: Predicted vs Actual with L2 Regularization (Closed-Form vs GD)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred_test_ridge_cf, alpha=0.6, color='#9467bd', edgecolors='k', s=45)
axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal Fit (y = x)')
axes[0].set_title(f'L2 Ridge Closed-Form (λ = {best_lambda})\n(R² = {ridge_cf_metrics["test_r2"]:.4f}, RMSE = {ridge_cf_metrics["test_rmse"]:,.1f})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual Mobile Price (INR)', fontsize=11)
axes[0].set_ylabel('Predicted Mobile Price (INR)', fontsize=11)
axes[0].legend()
axes[0].set_xlim(0, max_val)
axes[0].set_ylim(0, max_val)

axes[1].scatter(y_test, y_pred_test_ridge_gd, alpha=0.6, color='#8c564b', edgecolors='k', s=45)
axes[1].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal Fit (y = x)')
axes[1].set_title(f'L2 Ridge Gradient Descent (λ = {best_lambda})\n(R² = {ridge_gd_metrics["test_r2"]:.4f}, RMSE = {ridge_gd_metrics["test_rmse"]:,.1f})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Actual Mobile Price (INR)', fontsize=11)
axes[1].set_ylabel('Predicted Mobile Price (INR)', fontsize=11)
axes[1].legend()
axes[1].set_xlim(0, max_val)
axes[1].set_ylim(0, max_val)

plt.tight_layout()
p4_path = os.path.join(OUTPUT_DIR, '04_pred_vs_actual_ridge.png')
plt.savefig(p4_path, dpi=300)
plt.close()
print("Saved:", p4_path)

# ------------------------------------------------------------------------------
# Plot 5: Standardization Comparison (Condition Number and Residuals)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Condition Number Bar Chart (log scale)
bars = axes[0].bar(['Unstandardized Data', 'Standardized Data'], [cond_unscaled, cond_scaled], color=['#d62728', '#2ca02c'], width=0.5)
axes[0].set_yscale('log')
axes[0].set_ylabel('Condition Number of Design Matrix X^T X (log scale)', fontsize=11)
axes[0].set_title('Gram Matrix Condition Number Comparison', fontsize=12, fontweight='bold')
for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval * 2.0, f'{yval:.2e}', ha='center', va='bottom', fontweight='bold')

# Test RMSE Comparison
rmse_vals = [ridge_unscaled_metrics['test_rmse'], ridge_cf_metrics['test_rmse']]
bars2 = axes[1].bar(['Ridge Without Standardization', 'Ridge With Standardization'], rmse_vals, color=['#ff7f0e', '#1f77b4'], width=0.5)
axes[1].set_ylabel('Test RMSE (INR)', fontsize=11)
axes[1].set_title('L2 Regularized Model Test RMSE Comparison', fontsize=12, fontweight='bold')
for bar in bars2:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval * 1.02, f'{yval:,.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
p5_path = os.path.join(OUTPUT_DIR, '05_standardization_comparison.png')
plt.savefig(p5_path, dpi=300)
plt.close()
print("Saved:", p5_path)

# ------------------------------------------------------------------------------
# Plot 6: Predicted vs Actual for Different Lambda Values
# ------------------------------------------------------------------------------
selected_lambdas = [0.01, 10.0, 100.0, 1000.0]
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for i, lam in enumerate(selected_lambdas):
    y_pred_lam = pred_by_lambda[lam]
    r2_lam = r2_score(y_test, y_pred_lam)
    rmse_lam = np.sqrt(mean_squared_error(y_test, y_pred_lam))
    axes[i].scatter(y_test, y_pred_lam, alpha=0.6, color=sns.color_palette("viridis", 4)[i], edgecolors='k', s=40)
    axes[i].plot([0, max_val], [0, max_val], 'r--', lw=1.8, label='Ideal Fit')
    axes[i].set_title(f'λ = {lam} | Test R² = {r2_lam:.4f} | RMSE = {rmse_lam:,.1f}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Actual Mobile Price', fontsize=10)
    axes[i].set_ylabel('Predicted Price', fontsize=10)
    axes[i].set_xlim(0, max_val)
    axes[i].set_ylim(0, max_val)
    axes[i].legend(loc='upper left')

plt.suptitle('Predicted vs Actual Mobile Prices across Regularization Parameter λ', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
p6_path = os.path.join(OUTPUT_DIR, '06_ridge_lambda_sweep_and_pred_actual.png')
plt.savefig(p6_path, dpi=300)
plt.close()
print("Saved:", p6_path)

# ------------------------------------------------------------------------------
# Plot 7: Feature Importance from Standardized L2 Weights
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
colors = ['#2ca02c' if w > 0 else '#d62728' for w in feat_importance_df['Weight']]
y_pos = np.arange(len(feat_importance_df))
ax.barh(y_pos, feat_importance_df['Weight'], color=colors, align='center', edgecolor='black', alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(feat_importance_df['Feature'], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Standardized Ridge Regression Coefficient (Weight)', fontsize=11)
ax.set_title('Feature Importance and Coefficient Impact on Mobile Price (λ = 10)', fontsize=13, fontweight='bold')
ax.axvline(0, color='black', lw=1)

for i, v in enumerate(feat_importance_df['Weight']):
    ha = 'left' if v >= 0 else 'right'
    offset = 120 if v >= 0 else -120
    ax.text(v + offset, i, f'{v:,.1f}', va='center', ha=ha, fontsize=9.5, fontweight='bold')

plt.tight_layout()
p7_path = os.path.join(OUTPUT_DIR, '07_regression_feature_importance_weights.png')
plt.savefig(p7_path, dpi=300)
plt.close()
print("Saved:", p7_path)

# ------------------------------------------------------------------------------
# Plot 8: Banknote Classification EDA and Distributions
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x='class', data=df_bank, ax=axes[0], palette=['#1f77b4', '#ff7f0e'])
axes[0].set_title('Banknote Class Distribution (0: Authentic, 1: Forged)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Class Label', fontsize=11)
axes[0].set_ylabel('Sample Count', fontsize=11)
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())} ({p.get_height()/len(df_bank)*100:.1f}%)',
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)

sns.scatterplot(x='variance', y='skewness', hue='class', data=df_bank, palette=['#1f77b4', '#ff7f0e'], alpha=0.7, ax=axes[1])
axes[1].set_title('Wavelet Variance vs. Skewness by Authenticity', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Variance of Wavelet Transform', fontsize=11)
axes[1].set_ylabel('Skewness of Wavelet Transform', fontsize=11)

plt.tight_layout()
p8_path = os.path.join(OUTPUT_DIR, '08_classification_eda_and_pairplot.png')
plt.savefig(p8_path, dpi=300)
plt.close()
print("Saved:", p8_path)

# ------------------------------------------------------------------------------
# Plot 9: Training and Test Accuracy vs Lambda
# ------------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))
lams = [x['lambda'] for x in cls_sweep_results]
tr_accs = [x['train_acc'] * 100.0 for x in cls_sweep_results]
te_accs = [x['test_acc'] * 100.0 for x in cls_sweep_results]
w_norms = [x['weight_norm'] for x in cls_sweep_results]

ax1.plot(lams, tr_accs, 'o-', color='#1f77b4', label='Training Accuracy (%)', lw=2)
ax1.plot(lams, te_accs, 's--', color='#2ca02c', label='Testing Accuracy (%)', lw=2)
ax1.set_xscale('log')
ax1.set_xlabel('Regularization Parameter λ (log scale)', fontsize=11)
ax1.set_ylabel('Classification Accuracy (%)', fontsize=11)
ax1.set_title('Training and Test Accuracy vs Regularization Parameter λ', fontsize=13, fontweight='bold')
ax1.set_ylim(85, 101)

ax2 = ax1.twinx()
ax2.plot(lams, w_norms, '^-', color='#d62728', alpha=0.5, label='Weight L2 Norm ||w||_2', lw=1.5)
ax2.set_ylabel('Weight L2 Norm ||w||_2', color='#d62728', fontsize=11)
ax2.tick_params(axis='y', labelcolor='#d62728')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

plt.tight_layout()
p9_path = os.path.join(OUTPUT_DIR, '09_classification_acc_and_loss_vs_lambda.png')
plt.savefig(p9_path, dpi=300)
plt.close()
print("Saved:", p9_path)

# ------------------------------------------------------------------------------
# Plot 10: Confusion Matrices and ROC Curves
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Confusion Matrix for L2 Model
cm = confusion_matrix(y_bank_te, y_pred_te_l2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Authentic (0)', 'Forged (1)'],
            yticklabels=['Authentic (0)', 'Forged (1)'])
axes[0].set_title(f'Confusion Matrix (L2 Classifier)\nAccuracy = {metrics_l2["test_acc"]*100:.2f}%', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=11)
axes[0].set_ylabel('True Label', fontsize=11)

# ROC Curves
fpr_unreg, tpr_unreg, _ = roc_curve(y_bank_te, y_prob_te_unreg)
fpr_l2, tpr_l2, _ = roc_curve(y_bank_te, y_prob_te_l2)
axes[1].plot(fpr_unreg, tpr_unreg, color='#1f77b4', lw=2, label=f'Unregularized (AUC = {metrics_unreg["roc_auc"]:.4f})')
axes[1].plot(fpr_l2, tpr_l2, color='#2ca02c', lw=2, linestyle='--', label=f'L2 Regularized (AUC = {metrics_l2["roc_auc"]:.4f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance Baseline (AUC = 0.50)')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
axes[1].set_xlabel('False Positive Rate', fontsize=11)
axes[1].set_ylabel('True Positive Rate', fontsize=11)
axes[1].legend(loc='lower right', fontsize=10)

plt.tight_layout()
p10_path = os.path.join(OUTPUT_DIR, '10_classification_confusion_matrices_and_roc.png')
plt.savefig(p10_path, dpi=300)
plt.close()
print("Saved:", p10_path)

# ------------------------------------------------------------------------------
# Plot 11: 3D Visualization using Three Important Features
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

top3_cols = ['variance', 'skewness', 'curtosis']
idx_auth = (y_bank == 0)
idx_forg = (y_bank == 1)

# Plot actual samples
ax.scatter(X_bank[idx_auth, 0], X_bank[idx_auth, 1], X_bank[idx_auth, 2],
           c='#1f77b4', marker='o', alpha=0.55, edgecolors='k', s=25, label='Authentic Banknotes (0)')
ax.scatter(X_bank[idx_forg, 0], X_bank[idx_forg, 1], X_bank[idx_forg, 2],
           c='#d62728', marker='^', alpha=0.55, edgecolors='k', s=25, label='Forged Banknotes (1)')

# Fit a 3D boundary plane based on the top 3 features
w_top3 = clf_l2.coef_[0][:3]
b_top3 = clf_l2.intercept_[0]

x1_grid = np.linspace(X_bank[:, 0].min(), X_bank[:, 0].max(), 20)
x2_grid = np.linspace(X_bank[:, 1].min(), X_bank[:, 1].max(), 20)
X1, X2 = np.meshgrid(x1_grid, x2_grid)

# Decision boundary: w1*x1 + w2*x2 + w3*x3 + b = 0 => x3 = -(w1*x1 + w2*x2 + b) / w3
# Note: since weights are for scaled data, compute directly or illustrate approximate plane
x3_surf = -(w_top3[0] * ((X1 - scaler_bank.mean_[0])/scaler_bank.scale_[0]) +
            w_top3[1] * ((X2 - scaler_bank.mean_[1])/scaler_bank.scale_[1]) + b_top3) / w_top3[2]
x3_surf = x3_surf * scaler_bank.scale_[2] + scaler_bank.mean_[2]

ax.plot_surface(X1, X2, x3_surf, alpha=0.25, color='gray')

ax.set_title('3D Feature Space & Linear Decision Hyperplane\n(Variance, Skewness, Curtosis)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Variance (Wavelet)', fontsize=10, labelpad=10)
ax.set_ylabel('Skewness (Wavelet)', fontsize=10, labelpad=10)
ax.set_zlabel('Curtosis (Wavelet)', fontsize=10, labelpad=10)
ax.legend(loc='upper right', fontsize=10)
ax.view_init(elev=22, azim=-55)

plt.tight_layout()
p11_path = os.path.join(OUTPUT_DIR, '11_banknote_3d_feature_scatter.png')
plt.savefig(p11_path, dpi=300)
plt.close()
print("Saved:", p11_path)

# ------------------------------------------------------------------------------
# Plot 12: Outlier Injection Impact and Sensitivity Comparison
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Scatter of Clean vs Corrupted Training Points (Variance vs Skewness)
axes[0].scatter(X_bank_tr_s[:, 0], X_bank_tr_s[:, 1], c=y_bank_tr, cmap='coolwarm', alpha=0.3, s=20, label='Clean Train Data')
axes[0].scatter(X_bank_tr_out[idx_outliers, 0], X_bank_tr_out[idx_outliers, 1],
                color='black', marker='X', s=70, label=f'Injected Outliers (8% = {n_outliers_rep} pts)')
axes[0].set_title('Shifted Outliers Injected into Training Space', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Standardized Variance', fontsize=11)
axes[0].set_ylabel('Standardized Skewness', fontsize=11)
axes[0].legend(loc='lower left')

# Accuracy Degradation Across Outlier Percentage
ratios_pct = [x['ratio'] * 100 for x in outlier_experiments]
unreg_accs = [x['unreg_test_acc'] * 100 for x in outlier_experiments]
l2_accs = [x['l2_test_acc'] * 100 for x in outlier_experiments]

axes[1].plot(ratios_pct, unreg_accs, 'o-', color='#d62728', lw=2, label='Unregularized Classifier')
axes[1].plot(ratios_pct, l2_accs, 's--', color='#1f77b4', lw=2, label='L2 Regularized Classifier')
axes[1].set_title('Generalization Accuracy Degradation vs Outlier Ratio', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Percentage of Injected Training Outliers (%)', fontsize=11)
axes[1].set_ylabel('Test Accuracy on Clean Unseen Data (%)', fontsize=11)
axes[1].set_ylim(50, 102)
axes[1].legend(loc='lower left', fontsize=10)

plt.tight_layout()
p12_path = os.path.join(OUTPUT_DIR, '12_outlier_injection_impact_comparison.png')
plt.savefig(p12_path, dpi=300)
plt.close()
print("Saved:", p12_path)

# ==============================================================================
# SAVE RESULTS JSON
# ==============================================================================
results_json_path = os.path.join(BASE_DIR, 'code', 'cat1_results.json')
with open(results_json_path, 'w') as f:
    json.dump(results_dict, f, indent=4)
print(f"\nSaved structured experimental metrics to: {results_json_path}")
print("=" * 80)
print("EXPERIMENT EXECUTION COMPLETE SUCCESSFULLY!")
print("=" * 80)
