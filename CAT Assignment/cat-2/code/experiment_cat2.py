#!/usr/bin/env python3
"""
CAT Assignment 2: Comprehensive Machine Learning Experimentation
Course: ICS1502 - Introduction to Machine Learning
Dataset: UCI Banknote Authentication Dataset
Author: Danush K (Reg. No. 3122247001013)
Institution: Sri Sivasubramaniya Nadar College of Engineering, Chennai
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings('ignore')

import scipy.stats as stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, log_loss
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Style configuration for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'banknote_authentication.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'output_plots')
RESULTS_DIR = os.path.join(BASE_DIR, 'code')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================================================================
# 1. DATA INGESTION AND EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print(">>> Loading and Preprocessing Banknote Dataset...")
df = pd.read_csv(DATA_PATH)
feature_names = ['variance', 'skewness', 'curtosis', 'entropy']
target_name = 'class'

X = df[feature_names].values
y = df[target_name].values

# Stratified Split: 80% Train, 20% Test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# Feature Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Save EDA Plot: Pairplot, Violins, Correlation Heatmap
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (A) Correlation Heatmap
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, ax=axes[0, 0],
            linewidths=1, linecolor='white', square=True)
axes[0, 0].set_title("(a) Pearson Correlation Matrix", fontweight='bold')

# (B) Feature Distributions by Class
df_melted = pd.melt(df, id_vars=['class'], value_vars=feature_names,
                    var_name='Feature', value_name='Value')
sns.violinplot(x='Feature', y='Value', hue='class', data=df_melted,
               split=True, inner="quart", palette={0: "#2b5c8f", 1: "#d95f02"}, ax=axes[0, 1])
axes[0, 1].set_title("(b) Standardized Feature Densities by Class (0: Genuine, 1: Forged)", fontweight='bold')
axes[0, 1].legend(title='Banknote Class', labels=['Genuine (0)', 'Forged (1)'])

# (C) 2D Scatter: Variance vs Skewness
scatter = axes[1, 0].scatter(df['variance'], df['skewness'], c=df['class'],
                             cmap='coolwarm', alpha=0.75, edgecolors='k', linewidth=0.5, s=35)
axes[1, 0].set_xlabel("Variance of Wavelet Transform")
axes[1, 0].set_ylabel("Skewness of Wavelet Transform")
axes[1, 0].set_title("(c) Primary Discriminant Plane: Variance vs Skewness", fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Genuine (0)', 'Forged (1)'])

# (D) Class Balance Barplot
counts = df['class'].value_counts()
bars = axes[1, 1].bar(['Genuine (0)', 'Forged (1)'], counts, color=['#2b5c8f', '#d95f02'], width=0.5, edgecolor='black')
for bar in bars:
    yval = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval} ({yval/len(df)*100:.1f}%)",
                    ha='center', va='bottom', fontweight='bold')
axes[1, 1].set_ylim(0, 900)
axes[1, 1].set_ylabel("Number of Samples")
axes[1, 1].set_title("(d) Class Distribution Balance (Total N=1372)", fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '01_eda_distributions_and_correlations.png'))
plt.close(fig)
print("Saved: 01_eda_distributions_and_correlations.png")

# ==============================================================================
# 2. TASK 1: LINEAR MODEL BASELINE (LOGISTIC REGRESSION WITH & WITHOUT L2)
# ==============================================================================
print("\n>>> Task 1: Training Linear Classification Models (With & Without L2)...")

# Unregularized Logistic Regression (C = 1e6 or penalty=None)
lr_unreg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=SEED)
lr_unreg.fit(X_train, y_train)

# L2 Regularized Logistic Regression (Standard C = 1.0)
lr_l2 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=SEED)
lr_l2.fit(X_train, y_train)

# Predictions
y_train_pred_unreg = lr_unreg.predict(X_train)
y_test_pred_unreg = lr_unreg.predict(X_test)
y_test_prob_unreg = lr_unreg.predict_proba(X_test)[:, 1]

y_train_pred_l2 = lr_l2.predict(X_train)
y_test_pred_l2 = lr_l2.predict(X_test)
y_test_prob_l2 = lr_l2.predict_proba(X_test)[:, 1]

metrics_task1 = {
    "Unregularized": {
        "Train_Accuracy": float(accuracy_score(y_train, y_train_pred_unreg)),
        "Test_Accuracy": float(accuracy_score(y_test, y_test_pred_unreg)),
        "Precision": float(precision_score(y_test, y_test_pred_unreg)),
        "Recall": float(recall_score(y_test, y_test_pred_unreg)),
        "F1_Score": float(f1_score(y_test, y_test_pred_unreg)),
        "ROC_AUC": float(roc_auc_score(y_test, y_test_prob_unreg)),
        "Log_Loss": float(log_loss(y_test, y_test_prob_unreg)),
        "Weights": lr_unreg.coef_[0].tolist(),
        "Intercept": float(lr_unreg.intercept_[0]),
        "Weight_L2_Norm": float(np.linalg.norm(lr_unreg.coef_[0]))
    },
    "L2_Regularized (C=1.0)": {
        "Train_Accuracy": float(accuracy_score(y_train, y_train_pred_l2)),
        "Test_Accuracy": float(accuracy_score(y_test, y_test_pred_l2)),
        "Precision": float(precision_score(y_test, y_test_pred_l2)),
        "Recall": float(recall_score(y_test, y_test_pred_l2)),
        "F1_Score": float(f1_score(y_test, y_test_pred_l2)),
        "ROC_AUC": float(roc_auc_score(y_test, y_test_prob_l2)),
        "Log_Loss": float(log_loss(y_test, y_test_prob_l2)),
        "Weights": lr_l2.coef_[0].tolist(),
        "Intercept": float(lr_l2.intercept_[0]),
        "Weight_L2_Norm": float(np.linalg.norm(lr_l2.coef_[0]))
    }
}

print(json.dumps(metrics_task1, indent=2))

# Plot Baseline ROC & Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix Unregularized
cm_unreg = confusion_matrix(y_test, y_test_pred_unreg)
sns.heatmap(cm_unreg, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Genuine (0)', 'Forged (1)'], yticklabels=['Genuine (0)', 'Forged (1)'])
axes[0].set_title(f"Unregularized Logistic Regression\nTest Acc: {metrics_task1['Unregularized']['Test_Accuracy']*100:.2f}%", fontweight='bold')
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Confusion Matrix L2 Regularized
cm_l2 = confusion_matrix(y_test, y_test_pred_l2)
sns.heatmap(cm_l2, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=['Genuine (0)', 'Forged (1)'], yticklabels=['Genuine (0)', 'Forged (1)'])
axes[1].set_title(f"L2 Regularized (C=1.0)\nTest Acc: {metrics_task1['L2_Regularized (C=1.0)']['Test_Accuracy']*100:.2f}%", fontweight='bold')
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

# ROC Curves
fpr_unreg, tpr_unreg, _ = roc_curve(y_test, y_test_prob_unreg)
fpr_l2, tpr_l2, _ = roc_curve(y_test, y_test_prob_l2)
axes[2].plot(fpr_unreg, tpr_unreg, label=f"Unregularized (AUC = {metrics_task1['Unregularized']['ROC_AUC']:.4f})", color='#d95f02', lw=2)
axes[2].plot(fpr_l2, tpr_l2, label=f"L2 Regularized (AUC = {metrics_task1['L2_Regularized (C=1.0)']['ROC_AUC']:.4f})", color='#2b5c8f', lw=2, linestyle='--')
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.6)
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curves: Baseline Linear Classifiers", fontweight='bold')
axes[2].legend(loc='lower right')

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '02_logistic_regression_baseline.png'))
plt.close(fig)
print("Saved: 02_logistic_regression_baseline.png")

# ==============================================================================
# 3. TASK 2: REGULARIZATION EFFECT AS A FUNCTION OF LAMBDA
# ==============================================================================
print("\n>>> Task 2: Evaluating Regularization Parameter Sweep (Lambda)...")
# In scikit-learn, C = 1 / lambda
# We sweep lambda from 1e-4 to 1e4 (C from 1e4 down to 1e-4)
lambdas = np.logspace(-4, 4, 100)
C_values = 1.0 / lambdas

train_accs = []
test_accs = []
train_losses = []
test_losses = []
weights_history = []
weight_norms = []

for C in C_values:
    clf = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=2000, random_state=SEED)
    clf.fit(X_train, y_train)
    
    y_tr_pred = clf.predict(X_train)
    y_te_pred = clf.predict(X_test)
    y_tr_prob = clf.predict_proba(X_train)
    y_te_prob = clf.predict_proba(X_test)
    
    train_accs.append(accuracy_score(y_train, y_tr_pred))
    test_accs.append(accuracy_score(y_test, y_te_pred))
    train_losses.append(log_loss(y_train, y_tr_prob))
    test_losses.append(log_loss(y_test, y_te_prob))
    weights_history.append(clf.coef_[0].copy())
    weight_norms.append(np.linalg.norm(clf.coef_[0]))

weights_history = np.array(weights_history)

# Find optimal lambda
best_idx = np.argmax(test_accs)
best_lambda = lambdas[best_idx]
print(f"Optimal Lambda: {best_lambda:.4e} (Test Accuracy = {test_accs[best_idx]*100:.2f}%)")

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# (A) Train vs Test Accuracy vs Lambda
axes[0].semilogx(lambdas, train_accs, label='Training Accuracy', color='#2b5c8f', lw=2.2)
axes[0].semilogx(lambdas, test_accs, label='Test Accuracy', color='#d95f02', lw=2.2, linestyle='--')
axes[0].axvline(best_lambda, color='green', linestyle=':', label=f'Optimal $\\lambda^* = {best_lambda:.2e}$')
axes[0].set_xlabel(r"Regularization Strength $\lambda = 1/C$ (Log Scale)")
axes[0].set_ylabel("Classification Accuracy")
axes[0].set_title(r"(a) Accuracy vs. Regularization Strength $\lambda$", fontweight='bold')
axes[0].legend(loc='lower left')

# (B) Cross-Entropy Loss vs Lambda
axes[1].semilogx(lambdas, train_losses, label='Train Log Loss', color='#2b5c8f', lw=2.2)
axes[1].semilogx(lambdas, test_losses, label='Test Log Loss', color='#d95f02', lw=2.2, linestyle='--')
axes[1].axvline(best_lambda, color='green', linestyle=':', label=f'Optimal $\\lambda^*$')
axes[1].set_xlabel(r"Regularization Strength $\lambda = 1/C$ (Log Scale)")
axes[1].set_ylabel("Binary Cross-Entropy Loss")
axes[1].set_title(r"(b) Loss Landscape vs. Regularization Strength $\lambda$", fontweight='bold')
axes[1].legend(loc='upper left')

# (C) Weight Shrinkage Paths
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for j, f_name in enumerate(feature_names):
    axes[2].semilogx(lambdas, weights_history[:, j], label=f"$w_{{{f_name}}}$", color=colors[j], lw=2)
axes[2].semilogx(lambdas, weight_norms, label=r"$\|\mathbf{w}\|_2$ (Total Norm)", color='black', lw=2.5, linestyle='-.')
axes[2].set_xlabel(r"Regularization Strength $\lambda = 1/C$ (Log Scale)")
axes[2].set_ylabel("Coefficient Magnitude")
axes[2].set_title(r"(c) $L_2$ Regularization Paths (Weight Shrinkage)", fontweight='bold')
axes[2].legend(loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '03_regularization_path_and_weights.png'))
plt.close(fig)
print("Saved: 03_regularization_path_and_weights.png")

# ==============================================================================
# 4. TASK 3: DECISION BOUNDARY VISUALIZATION IN 3D & 2D SLICES
# ==============================================================================
print("\n>>> Task 3: Visualizing Decision Boundaries in 3D & 2D...")
# Select top 3 features: variance (0), skewness (1), curtosis (2)
feat_3d_indices = [0, 1, 2]
feat_3d_names = [feature_names[i] for i in feat_3d_indices]

X_train_3d = X_train[:, feat_3d_indices]
X_test_3d = X_test[:, feat_3d_indices]

clf_3d = LogisticRegression(penalty='l2', C=1.0, random_state=SEED)
clf_3d.fit(X_train_3d, y_train)

acc_3d = accuracy_score(y_test, clf_3d.predict(X_test_3d))
w_3d = clf_3d.coef_[0]
b_3d = clf_3d.intercept_[0]
print(f"3D Model (Variance, Skewness, Curtosis) Test Accuracy: {acc_3d*100:.2f}%")
print(f"3D Hyperplane Equation: {w_3d[0]:.3f}*x1 + {w_3d[1]:.3f}*x2 + {w_3d[2]:.3f}*x3 + {b_3d:.3f} = 0")

fig = plt.figure(figsize=(18, 6))

# (A) 3D Decision Surface
ax3d = fig.add_subplot(1, 3, 1, projection='3d')
idx0 = np.where(y_test == 0)[0]
idx1 = np.where(y_test == 1)[0]

ax3d.scatter(X_test_3d[idx0, 0], X_test_3d[idx0, 1], X_test_3d[idx0, 2],
             c='#2b5c8f', label='Genuine (0)', alpha=0.7, edgecolors='k', s=30)
ax3d.scatter(X_test_3d[idx1, 0], X_test_3d[idx1, 1], X_test_3d[idx1, 2],
             c='#d95f02', label='Forged (1)', alpha=0.7, edgecolors='k', s=30)

# Create meshgrid for hyperplane: x3 = -(w0*x1 + w1*x2 + b) / w2
x1_range = np.linspace(X_test_3d[:, 0].min(), X_test_3d[:, 0].max(), 20)
x2_range = np.linspace(X_test_3d[:, 1].min(), X_test_3d[:, 1].max(), 20)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
X3_grid = -(w_3d[0] * X1_grid + w_3d[1] * X2_grid + b_3d) / w_3d[2]

# Clip grid to reasonable limits for visualization
ax3d.plot_surface(X1_grid, X2_grid, X3_grid, alpha=0.35, color='gray', edgecolor='none')
ax3d.set_xlabel('Variance (std)')
ax3d.set_ylabel('Skewness (std)')
ax3d.set_zlabel('Curtosis (std)')
ax3d.set_title(f"(a) 3D Separating Hyperplane\nTest Accuracy: {acc_3d*100:.2f}%", fontweight='bold')
ax3d.legend(loc='upper right')
ax3d.view_init(elev=20, azim=135)

# (B) 2D Decision Contour: Variance vs Skewness (Setting Curtosis = 0, Entropy = 0)
ax2d_1 = fig.add_subplot(1, 3, 2)
clf_2d = LogisticRegression(penalty='l2', C=1.0, random_state=SEED)
clf_2d.fit(X_train[:, [0, 1]], y_train)

x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = clf_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

contour = ax2d_1.contourf(xx, yy, Z, levels=25, cmap='RdBu_r', alpha=0.7)
ax2d_1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5)
ax2d_1.scatter(X_test[idx0, 0], X_test[idx0, 1], c='#2b5c8f', edgecolors='k', s=35, label='Genuine (0)')
ax2d_1.scatter(X_test[idx1, 0], X_test[idx1, 1], c='#d95f02', edgecolors='k', s=35, label='Forged (1)')
ax2d_1.set_xlabel("Variance (Standardized)")
ax2d_1.set_ylabel("Skewness (Standardized)")
ax2d_1.set_title("(b) 2D Decision Contour (Variance vs Skewness)", fontweight='bold')
ax2d_1.legend(loc='lower left')

# (C) 2D Decision Contour: Variance vs Curtosis
ax2d_2 = fig.add_subplot(1, 3, 3)
clf_2d_k = LogisticRegression(penalty='l2', C=1.0, random_state=SEED)
clf_2d_k.fit(X_train[:, [0, 2]], y_train)

x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
y_min, y_max = X_test[:, 2].min() - 0.5, X_test[:, 2].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z_k = clf_2d_k.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

ax2d_2.contourf(xx, yy, Z_k, levels=25, cmap='RdBu_r', alpha=0.7)
ax2d_2.contour(xx, yy, Z_k, levels=[0.5], colors='black', linewidths=2.5)
ax2d_2.scatter(X_test[idx0, 0], X_test[idx0, 2], c='#2b5c8f', edgecolors='k', s=35, label='Genuine (0)')
ax2d_2.scatter(X_test[idx1, 0], X_test[idx1, 2], c='#d95f02', edgecolors='k', s=35, label='Forged (1)')
ax2d_2.set_xlabel("Variance (Standardized)")
ax2d_2.set_ylabel("Curtosis (Standardized)")
ax2d_2.set_title("(c) 2D Decision Contour (Variance vs Curtosis)", fontweight='bold')
ax2d_2.legend(loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '04_decision_boundary_3d_and_2d.png'))
plt.close(fig)
print("Saved: 04_decision_boundary_3d_and_2d.png")

# ==============================================================================
# 5. TASK 4: NEURAL NETWORK COMPARISON (SINGLE-LAYER & MULTI-LAYER MLP)
# ==============================================================================
print("\n>>> Task 4: Fitting Single-Layer & Multi-Layer Neural Networks...")

# PyTorch Datasets and Loaders
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Architecture 1: Single-Layer Perceptron (Logistic Sigmoid Unit)
class SingleLayerNN(nn.Module):
    def __init__(self, in_features=4):
        super(SingleLayerNN, self).__init__()
        self.fc = nn.Linear(in_features, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Architecture 2: Multi-Layer Perceptron (4 -> 16 -> 8 -> 1)
class MultiLayerNN(nn.Module):
    def __init__(self, in_features=4):
        super(MultiLayerNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def train_neural_net(model, train_loader, test_loader, epochs=100, lr=0.01, weight_decay=1e-4):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            preds = (outputs >= 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_x, val_y in test_loader:
                val_out = model(val_x)
                v_loss = criterion(val_out, val_y)
                val_loss += v_loss.item() * val_x.size(0)
                v_preds = (val_out >= 0.5).float()
                correct_val += (v_preds == val_y).sum().item()
                total_val += val_x.size(0)
                
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss / total_val)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(correct_val / total_val)
        
    return history

slp_model = SingleLayerNN(in_features=4)
history_slp = train_neural_net(slp_model, train_loader, test_loader, epochs=100, lr=0.01)

mlp_model = MultiLayerNN(in_features=4)
history_mlp = train_neural_net(mlp_model, train_loader, test_loader, epochs=100, lr=0.01)

# Evaluation
slp_model.eval()
with torch.no_grad():
    y_test_slp_prob = slp_model(torch.FloatTensor(X_test)).numpy().ravel()
    y_test_slp_pred = (y_test_slp_prob >= 0.5).astype(int)
    y_train_slp_pred = (slp_model(torch.FloatTensor(X_train)).numpy().ravel() >= 0.5).astype(int)

mlp_model.eval()
with torch.no_grad():
    y_test_mlp_prob = mlp_model(torch.FloatTensor(X_test)).numpy().ravel()
    y_test_mlp_pred = (y_test_mlp_prob >= 0.5).astype(int)
    y_train_mlp_pred = (mlp_model(torch.FloatTensor(X_train)).numpy().ravel() >= 0.5).astype(int)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

metrics_models = {
    "Logistic Regression (L2)": {
        "Train_Acc": float(accuracy_score(y_train, y_train_pred_l2)),
        "Test_Acc": float(accuracy_score(y_test, y_test_pred_l2)),
        "Precision": float(precision_score(y_test, y_test_pred_l2)),
        "Recall": float(recall_score(y_test, y_test_pred_l2)),
        "F1_Score": float(f1_score(y_test, y_test_pred_l2)),
        "ROC_AUC": float(roc_auc_score(y_test, y_test_prob_l2)),
        "Parameters": 5,
        "Generalization_Gap": float(accuracy_score(y_train, y_train_pred_l2) - accuracy_score(y_test, y_test_pred_l2))
    },
    "Single-Layer NN (SLP)": {
        "Train_Acc": float(accuracy_score(y_train, y_train_slp_pred)),
        "Test_Acc": float(accuracy_score(y_test, y_test_slp_pred)),
        "Precision": float(precision_score(y_test, y_test_slp_pred)),
        "Recall": float(recall_score(y_test, y_test_slp_pred)),
        "F1_Score": float(f1_score(y_test, y_test_slp_pred)),
        "ROC_AUC": float(roc_auc_score(y_test, y_test_slp_prob)),
        "Parameters": count_parameters(slp_model),
        "Generalization_Gap": float(accuracy_score(y_train, y_train_slp_pred) - accuracy_score(y_test, y_test_slp_pred))
    },
    "Multi-Layer MLP (4-16-8-1)": {
        "Train_Acc": float(accuracy_score(y_train, y_train_mlp_pred)),
        "Test_Acc": float(accuracy_score(y_test, y_test_mlp_pred)),
        "Precision": float(precision_score(y_test, y_test_mlp_pred)),
        "Recall": float(recall_score(y_test, y_test_mlp_pred)),
        "F1_Score": float(f1_score(y_test, y_test_mlp_pred)),
        "ROC_AUC": float(roc_auc_score(y_test, y_test_mlp_prob)),
        "Parameters": count_parameters(mlp_model),
        "Generalization_Gap": float(accuracy_score(y_train, y_train_mlp_pred) - accuracy_score(y_test, y_test_mlp_pred))
    }
}

print("\nModel Performance Comparison:")
print(pd.DataFrame(metrics_models).T.to_string())

# Plot Learning Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_slp['train_loss'], label='SLP Train Loss', color='#1f77b4', linestyle=':')
axes[0].plot(history_slp['val_loss'], label='SLP Val Loss', color='#1f77b4', lw=2)
axes[0].plot(history_mlp['train_loss'], label='MLP Train Loss', color='#d62728', linestyle=':')
axes[0].plot(history_mlp['val_loss'], label='MLP Val Loss', color='#d62728', lw=2)
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].set_title("(a) Training & Validation Loss Curves", fontweight='bold')
axes[0].legend()

axes[1].plot(history_slp['train_acc'], label='SLP Train Acc', color='#1f77b4', linestyle=':')
axes[1].plot(history_slp['val_acc'], label='SLP Val Acc', color='#1f77b4', lw=2)
axes[1].plot(history_mlp['train_acc'], label='MLP Train Acc', color='#d62728', linestyle=':')
axes[1].plot(history_mlp['val_acc'], label='MLP Val Acc', color='#d62728', lw=2)
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Classification Accuracy")
axes[1].set_title("(b) Training & Validation Accuracy Progression", fontweight='bold')
axes[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '05_neural_network_architectures_and_learning_curves.png'))
plt.close(fig)
print("Saved: 05_neural_network_architectures_and_learning_curves.png")

# Bar Chart Metrics Comparison
fig, ax = plt.subplots(figsize=(10, 5))
comp_df = pd.DataFrame(metrics_models).T[['Test_Acc', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']]
comp_df.plot(kind='bar', ax=ax, colormap='viridis', width=0.75, edgecolor='black')
ax.set_ylim(0.90, 1.02)
ax.set_ylabel("Score")
ax.set_title("Comparative Evaluation: Linear Model vs SLP vs Deep MLP", fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontweight='bold')
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '06_linear_vs_neural_performance_comparison.png'))
plt.close(fig)
print("Saved: 06_linear_vs_neural_performance_comparison.png")

# ==============================================================================
# 6. TASK 5: DIMENSIONALITY REDUCTION VIA PCA
# ==============================================================================
print("\n>>> Task 5: Principal Component Analysis (PCA) Dimensionality Reduction...")

pca_full = PCA(n_components=4, random_state=SEED)
pca_full.fit(X_train)

eigenvalues = pca_full.explained_variance_
evr = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr)

print(f"Eigenvalues: {eigenvalues}")
print(f"Explained Variance Ratio: {evr}")
print(f"Cumulative EVR: {cum_evr}")

# PCA Projections
pca_2d = PCA(n_components=2, random_state=SEED)
X_train_pca2 = pca_2d.fit_transform(X_train)
X_test_pca2 = pca_2d.transform(X_test)

pca_3d = PCA(n_components=3, random_state=SEED)
X_train_pca3 = pca_3d.fit_transform(X_train)
X_test_pca3 = pca_3d.transform(X_test)

# Train Classifiers in Reduced Spaces
# Models on 2D PCA
lr_pca2 = LogisticRegression(penalty='l2', C=1.0, random_state=SEED).fit(X_train_pca2, y_train)
acc_lr_pca2 = accuracy_score(y_test, lr_pca2.predict(X_test_pca2))

mlp_pca2 = MultiLayerNN(in_features=2)
train_loader_pca2 = DataLoader(TensorDataset(torch.FloatTensor(X_train_pca2), torch.FloatTensor(y_train).unsqueeze(1)), batch_size=32, shuffle=True)
test_loader_pca2 = DataLoader(TensorDataset(torch.FloatTensor(X_test_pca2), torch.FloatTensor(y_test).unsqueeze(1)), batch_size=64, shuffle=False)
train_neural_net(mlp_pca2, train_loader_pca2, test_loader_pca2, epochs=80, lr=0.01)
mlp_pca2.eval()
with torch.no_grad():
    acc_mlp_pca2 = accuracy_score(y_test, (mlp_pca2(torch.FloatTensor(X_test_pca2)).numpy().ravel() >= 0.5).astype(int))

# Models on 3D PCA
lr_pca3 = LogisticRegression(penalty='l2', C=1.0, random_state=SEED).fit(X_train_pca3, y_train)
acc_lr_pca3 = accuracy_score(y_test, lr_pca3.predict(X_test_pca3))

mlp_pca3 = MultiLayerNN(in_features=3)
train_loader_pca3 = DataLoader(TensorDataset(torch.FloatTensor(X_train_pca3), torch.FloatTensor(y_train).unsqueeze(1)), batch_size=32, shuffle=True)
test_loader_pca3 = DataLoader(TensorDataset(torch.FloatTensor(X_test_pca3), torch.FloatTensor(y_test).unsqueeze(1)), batch_size=64, shuffle=False)
train_neural_net(mlp_pca3, train_loader_pca3, test_loader_pca3, epochs=80, lr=0.01)
mlp_pca3.eval()
with torch.no_grad():
    acc_mlp_pca3 = accuracy_score(y_test, (mlp_pca3(torch.FloatTensor(X_test_pca3)).numpy().ravel() >= 0.5).astype(int))

pca_comparison_results = {
    "Original 4D": {
        "Logistic Regression": float(metrics_models["Logistic Regression (L2)"]["Test_Acc"]),
        "Multi-Layer MLP": float(metrics_models["Multi-Layer MLP (4-16-8-1)"]["Test_Acc"])
    },
    "PCA 3 Components": {
        "Logistic Regression": float(acc_lr_pca3),
        "Multi-Layer MLP": float(acc_mlp_pca3)
    },
    "PCA 2 Components": {
        "Logistic Regression": float(acc_lr_pca2),
        "Multi-Layer MLP": float(acc_mlp_pca2)
    }
}

print("\nPCA Subspace vs Original Space Performance:")
print(pd.DataFrame(pca_comparison_results).to_string())

# Plot PCA Scree, Loadings, and 2D Projection
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (A) Scree Plot
bars = axes[0].bar(range(1, 5), evr*100, color='#2b5c8f', alpha=0.85, edgecolor='black', label='Individual EVR')
axes[0].plot(range(1, 5), cum_evr*100, color='#d95f02', marker='o', lw=2.2, label='Cumulative EVR')
for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
axes[0].set_xticks(range(1, 5))
axes[0].set_xticklabels([f"PC{i}" for i in range(1, 5)])
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("(a) PCA Scree Plot & Cumulative Variance", fontweight='bold')
axes[0].legend(loc='center right')
axes[0].set_ylim(0, 115)

# (B) PCA Loadings Heatmap
loadings = pd.DataFrame(pca_full.components_.T, index=feature_names, columns=[f"PC{i}" for i in range(1, 5)])
sns.heatmap(loadings, annot=True, fmt=".3f", cmap="coolwarm", center=0, cbar=True, ax=axes[1],
            linewidths=1, linecolor='white')
axes[1].set_title("(b) Principal Component Feature Loadings Matrix", fontweight='bold')

# (C) 2D PCA Projection
scatter = axes[2].scatter(X_test_pca2[:, 0], X_test_pca2[:, 1], c=y_test,
                          cmap='coolwarm', alpha=0.8, edgecolors='k', s=40)
axes[2].set_xlabel(f"PC1 ({evr[0]*100:.1f}% Var)")
axes[2].set_ylabel(f"PC2 ({evr[1]*100:.1f}% Var)")
axes[2].set_title(f"(c) Test Data in 2D PCA Subspace (Total: {cum_evr[1]*100:.1f}% Var)", fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[2])
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Genuine (0)', 'Forged (1)'])

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '07_pca_scree_and_variance_loadings.png'))
plt.close(fig)
print("Saved: 07_pca_scree_and_variance_loadings.png")

# Plot Original vs PCA Performance
fig, ax = plt.subplots(figsize=(8, 5))
pd.DataFrame(pca_comparison_results).plot(kind='bar', ax=ax, colormap='magma', edgecolor='black', width=0.7)
ax.set_ylim(0.70, 1.05)
ax.set_ylabel("Test Accuracy")
ax.set_title("Effect of Dimensionality Reduction: Original 4D vs Reduced PCA Spaces", fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontweight='bold')
ax.legend(title='Feature Space', loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '08_original_vs_pca_model_comparison.png'))
plt.close(fig)
print("Saved: 08_original_vs_pca_model_comparison.png")

# ==============================================================================
# 7. ROBUSTNESS TO OUTLIERS EXPERIMENT
# ==============================================================================
print("\n>>> Testing Outlier Robustness (Perturbation Experiment)...")
contamination_rates = [0.0, 0.02, 0.05, 0.08, 0.12, 0.15, 0.20]
lr_outlier_accs = []
mlp_outlier_accs = []

for contam in contamination_rates:
    X_train_contam = X_train.copy()
    y_train_contam = y_train.copy()
    
    if contam > 0:
        n_outliers = int(contam * len(X_train))
        outlier_indices = np.random.choice(len(X_train), size=n_outliers, replace=False)
        # Flip label or inject extreme leverage points in opposite class region
        X_train_contam[outlier_indices] += np.random.normal(loc=4.0, scale=1.5, size=(n_outliers, 4))
        y_train_contam[outlier_indices] = 1 - y_train_contam[outlier_indices]
        
    # Fit Logistic Regression
    clf_lr_c = LogisticRegression(penalty='l2', C=1.0, max_iter=2000, random_state=SEED)
    clf_lr_c.fit(X_train_contam, y_train_contam)
    lr_outlier_accs.append(accuracy_score(y_test, clf_lr_c.predict(X_test)))
    
    # Fit MLP
    mlp_c = MultiLayerNN(in_features=4)
    ds_c = TensorDataset(torch.FloatTensor(X_train_contam), torch.FloatTensor(y_train_contam).unsqueeze(1))
    dl_c = DataLoader(ds_c, batch_size=32, shuffle=True)
    train_neural_net(mlp_c, dl_c, test_loader, epochs=50, lr=0.01)
    mlp_c.eval()
    with torch.no_grad():
        mlp_acc = accuracy_score(y_test, (mlp_c(torch.FloatTensor(X_test)).numpy().ravel() >= 0.5).astype(int))
    mlp_outlier_accs.append(mlp_acc)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.array(contamination_rates)*100, lr_outlier_accs, marker='o', color='#2b5c8f', lw=2.2, label='Logistic Regression (L2)')
ax.plot(np.array(contamination_rates)*100, mlp_outlier_accs, marker='s', color='#d62728', lw=2.2, label='Multi-Layer MLP')
ax.set_xlabel("Outlier Contamination Rate (%) in Training Data")
ax.set_ylabel("Test Accuracy on Clean Data")
ax.set_title("Robustness to Outliers: Sensitivity of Decision Boundaries", fontweight='bold')
ax.legend(loc='lower left')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '09_robustness_to_outliers_perturbation.png'))
plt.close(fig)
print("Saved: 09_robustness_to_outliers_perturbation.png")

# ==============================================================================
# 8. TASK 6: STATISTICAL SIGNIFICANCE TESTING (ANOVA & FRIEDMAN TEST)
# ==============================================================================
print("\n>>> Task 6: Performing 10-Fold Stratified Cross-Validation for Statistical Tests...")

# We define 6 distinct model configurations to benchmark:
# 1. Logistic Regression (Unregularized)
# 2. Logistic Regression (L2 Reg, C=1.0)
# 3. Single-Layer NN (SLP)
# 4. Multi-Layer MLP (4-16-8-1)
# 5. Logistic Regression with PCA (k=2)
# 6. Multi-Layer MLP with PCA (k=2)

model_names = [
    "Logistic (Unreg)",
    "Logistic (L2)",
    "SLP (Single-Layer)",
    "MLP (Multi-Layer)",
    "Logistic (PCA k=2)",
    "MLP (PCA k=2)"
]

# Run 10-fold Stratified CV
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
cv_scores = {name: [] for name in model_names}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr_f, X_val_f = X[train_idx], X[val_idx]
    y_tr_f, y_val_f = y[train_idx], y[val_idx]
    
    sc_f = StandardScaler()
    X_tr_f = sc_f.fit_transform(X_tr_f)
    X_val_f = sc_f.transform(X_val_f)
    
    # 1. Logistic Unreg
    m1 = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, random_state=SEED)
    m1.fit(X_tr_f, y_tr_f)
    cv_scores["Logistic (Unreg)"].append(accuracy_score(y_val_f, m1.predict(X_val_f)))
    
    # 2. Logistic L2
    m2 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=SEED)
    m2.fit(X_tr_f, y_tr_f)
    cv_scores["Logistic (L2)"].append(accuracy_score(y_val_f, m2.predict(X_val_f)))
    
    # PyTorch DataLoaders
    tr_ds = TensorDataset(torch.FloatTensor(X_tr_f), torch.FloatTensor(y_tr_f).unsqueeze(1))
    val_ds = TensorDataset(torch.FloatTensor(X_val_f), torch.FloatTensor(y_val_f).unsqueeze(1))
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # 3. SLP
    m3 = SingleLayerNN(in_features=4)
    train_neural_net(m3, tr_loader, val_loader, epochs=60, lr=0.01)
    m3.eval()
    with torch.no_grad():
        preds3 = (m3(torch.FloatTensor(X_val_f)).numpy().ravel() >= 0.5).astype(int)
    cv_scores["SLP (Single-Layer)"].append(accuracy_score(y_val_f, preds3))
    
    # 4. MLP
    m4 = MultiLayerNN(in_features=4)
    train_neural_net(m4, tr_loader, val_loader, epochs=60, lr=0.01)
    m4.eval()
    with torch.no_grad():
        preds4 = (m4(torch.FloatTensor(X_val_f)).numpy().ravel() >= 0.5).astype(int)
    cv_scores["MLP (Multi-Layer)"].append(accuracy_score(y_val_f, preds4))
    
    # PCA k=2
    pca2_f = PCA(n_components=2, random_state=SEED)
    X_tr_pca2 = pca2_f.fit_transform(X_tr_f)
    X_val_pca2 = pca2_f.transform(X_val_f)
    
    # 5. Logistic PCA k=2
    m5 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=SEED)
    m5.fit(X_tr_pca2, y_tr_f)
    cv_scores["Logistic (PCA k=2)"].append(accuracy_score(y_val_f, m5.predict(X_val_pca2)))
    
    # 6. MLP PCA k=2
    m6 = MultiLayerNN(in_features=2)
    tr_ds_p = TensorDataset(torch.FloatTensor(X_tr_pca2), torch.FloatTensor(y_tr_f).unsqueeze(1))
    val_ds_p = TensorDataset(torch.FloatTensor(X_val_pca2), torch.FloatTensor(y_val_f).unsqueeze(1))
    tr_l_p = DataLoader(tr_ds_p, batch_size=32, shuffle=True)
    val_l_p = DataLoader(val_ds_p, batch_size=64, shuffle=False)
    train_neural_net(m6, tr_l_p, val_l_p, epochs=60, lr=0.01)
    m6.eval()
    with torch.no_grad():
        preds6 = (m6(torch.FloatTensor(X_val_pca2)).numpy().ravel() >= 0.5).astype(int)
    cv_scores["MLP (PCA k=2)"].append(accuracy_score(y_val_f, preds6))

cv_df = pd.DataFrame(cv_scores)
print("\n10-Fold Stratified Cross-Validation Summary:")
print(cv_df.describe().T[['mean', 'std', 'min', 'max']])

# ------------------------------------------------------------------------------
# 8.1 ONE-WAY ANOVA
# ------------------------------------------------------------------------------
f_stat, anova_p = stats.f_oneway(
    cv_df["Logistic (Unreg)"],
    cv_df["Logistic (L2)"],
    cv_df["SLP (Single-Layer)"],
    cv_df["MLP (Multi-Layer)"],
    cv_df["Logistic (PCA k=2)"],
    cv_df["MLP (PCA k=2)"]
)

# Test Assumptions: Normality and Homoscedasticity
shapiro_results = {col: stats.shapiro(cv_df[col]) for col in cv_df.columns}
levene_stat, levene_p = stats.levene(*(cv_df[col] for col in cv_df.columns))

# Manual ANOVA Table Computation
K = len(model_names) # 6 groups
N_total = n_splits * K # 60 observations
grand_mean = cv_df.values.mean()

SS_between = n_splits * sum((cv_df[col].mean() - grand_mean)**2 for col in cv_df.columns)
SS_within = sum(sum((cv_df[col] - cv_df[col].mean())**2) for col in cv_df.columns)
SS_total = SS_between + SS_within

df_between = K - 1
df_within = N_total - K
df_total = N_total - 1

MS_between = SS_between / df_between
MS_within = SS_within / df_within

anova_table = {
    "Source": ["Between Groups (Treatments)", "Within Groups (Error)", "Total"],
    "SS": [float(SS_between), float(SS_within), float(SS_total)],
    "df": [int(df_between), int(df_within), int(df_total)],
    "MS": [float(MS_between), float(MS_within), None],
    "F": [float(f_stat), None, None],
    "p_value": [float(anova_p), None, None]
}

# Post-Hoc Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd
melted_cv = cv_df.melt(var_name='model', value_name='score')
tukey = pairwise_tukeyhsd(endog=melted_cv['score'], groups=melted_cv['model'], alpha=0.05)
tukey_summary = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

# ------------------------------------------------------------------------------
# 8.2 FRIEDMAN TEST
# ------------------------------------------------------------------------------
# Each fold is a block. Rank algorithms per fold (1 = best, 6 = worst)
# Note: rankdata by default assigns 1 to lowest value, so we rank -values
ranks = np.array([stats.rankdata(-cv_df.iloc[row].values) for row in range(n_splits)])
mean_ranks = ranks.mean(axis=0)
mean_rank_dict = {name: float(mean_ranks[i]) for i, name in enumerate(model_names)}

# Friedman test via scipy
friedman_stat, friedman_p = stats.friedmanchisquare(
    *(cv_df[col] for col in cv_df.columns)
)

# Iman-Davenport correction: F_F = ( (N - 1) * chi_F^2 ) / ( N * (k - 1) - chi_F^2 )
N_blocks = n_splits
k_treatments = K
F_iman_davenport = ((N_blocks - 1) * friedman_stat) / (N_blocks * (k_treatments - 1) - friedman_stat)
p_iman_davenport = stats.f.sf(F_iman_davenport, k_treatments - 1, (N_blocks - 1) * (k_treatments - 1))

# Nemenyi Critical Difference at alpha=0.05
# q_alpha for k=6, alpha=0.05 is ~2.850 (from Studentized Range distribution table for Nemenyi)
q_alpha_005 = 2.850
CD_nemenyi = q_alpha_005 * np.sqrt((k_treatments * (k_treatments + 1)) / (6.0 * N_blocks))

print("\n=== ANOVA Results ===")
print(f"F-statistic: {f_stat:.4f}, p-value: {anova_p:.4e}")
print(f"Levene Homoscedasticity p-value: {levene_p:.4e}")

print("\n=== Friedman Test Results ===")
print(f"Friedman Chi-Square: {friedman_stat:.4f}, p-value: {friedman_p:.4e}")
print(f"Iman-Davenport F: {F_iman_davenport:.4f}, p-value: {p_iman_davenport:.4e}")
print(f"Critical Difference (Nemenyi alpha=0.05): {CD_nemenyi:.4f}")
print("Mean Ranks (lower is better):", mean_rank_dict)

# Plot Statistical Significance: Boxplots and Friedman Ranks
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot of CV Scores
sns.boxplot(data=cv_df, ax=axes[0], palette="Set2")
sns.stripplot(data=cv_df, ax=axes[0], color='black', alpha=0.5, jitter=0.2, size=5)
axes[0].set_ylabel("10-Fold Stratified CV Accuracy")
axes[0].set_title(f"(a) 10-Fold CV Distribution\nANOVA: F = {f_stat:.2f}, p = {anova_p:.2e}", fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right', fontweight='bold')

# Friedman Mean Ranks Barplot with Critical Difference annotation
rank_series = pd.Series(mean_rank_dict).sort_values()
bars = axes[1].barh(rank_series.index, rank_series.values, color='#2b5c8f', edgecolor='black', height=0.6)
for bar in bars:
    xval = bar.get_width()
    axes[1].text(xval + 0.08, bar.get_y() + bar.get_height()/2.0, f"Rank: {xval:.2f}",
                 va='center', fontweight='bold')
axes[1].set_xlabel("Average Rank across 10 Folds (Lower is Better)")
axes[1].set_xlim(0, 6.5)
axes[1].set_title(f"(b) Friedman Test Model Ranks\n$\\chi_F^2 = {friedman_stat:.2f}$, $p = {friedman_p:.2e}$ | Nemenyi $CD = {CD_nemenyi:.2f}$", fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, '10_statistical_significance_anova_friedman.png'))
plt.close(fig)
print("Saved: 10_statistical_significance_anova_friedman.png")

# ==============================================================================
# 9. EXPORT ALL METRICS TO JSON
# ==============================================================================
final_results = {
    "task1_baseline": metrics_task1,
    "task2_regularization": {
        "best_lambda": float(best_lambda),
        "best_test_accuracy": float(test_accs[best_idx]),
        "weights_at_best_lambda": weights_history[best_idx].tolist(),
        "weight_norm_at_best_lambda": float(weight_norms[best_idx])
    },
    "task3_3d_hyperplane": {
        "features": feat_3d_names,
        "weights": w_3d.tolist(),
        "intercept": float(b_3d),
        "test_accuracy": float(acc_3d)
    },
    "task4_neural_comparison": metrics_models,
    "task5_pca": {
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": evr.tolist(),
        "cumulative_variance_ratio": cum_evr.tolist(),
        "comparison_results": pca_comparison_results
    },
    "task6_statistical_tests": {
        "cv_summary": cv_df.describe().to_dict(),
        "anova": {
            "F_statistic": float(f_stat),
            "p_value": float(anova_p),
            "table": anova_table,
            "levene_p": float(levene_p)
        },
        "friedman": {
            "chi2_statistic": float(friedman_stat),
            "p_value": float(friedman_p),
            "iman_davenport_F": float(F_iman_davenport),
            "iman_davenport_p": float(p_iman_davenport),
            "mean_ranks": mean_rank_dict,
            "nemenyi_CD_005": float(CD_nemenyi)
        },
        "tukey_hsd": tukey_summary.to_dict(orient="records")
    }
}

with open(os.path.join(RESULTS_DIR, 'cat2_results.json'), 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n>>> All experiments completed successfully! Results written to cat2_results.json.")
