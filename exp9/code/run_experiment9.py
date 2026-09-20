#!/usr/bin/env python3
"""
Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment) with
Hyperparameter Tuning
Course: Machine Learning Algorithms Laboratory (ICS1512)
Student Name: Danusu K | Register Number: 3122247001013
Faculty: Dr. Poreddy Ajay Kumar Reddy
Dataset: English Handwritten Characters (3,410 images, 62 classes: 0-9, A-Z, a-z)

This script implements:
1. Dataset loading (grayscale, resize to 28x28, flatten, normalize) and an 80/20
   stratified train-test split.
2. Model A - Single-Layer Perceptron Learning Algorithm (PLA): implemented from
   scratch (One-vs-Rest, step activation, classical weight-update rule) with a
   vectorized per-sample simultaneous multiclass update, tracked epoch-by-epoch.
3. Model B - Multilayer Perceptron (MLP): staged hyperparameter tuning over
   architecture/activation, optimizer/learning-rate, and batch size using
   GridSearchCV with 3-fold Stratified Cross-Validation, followed by a manual
   epoch-tracked retrain of the best configuration for fair A/B comparison with PLA.
4. Evaluation: Accuracy, Precision/Recall/F1 (macro & weighted), confusion matrices,
   micro/macro-average ROC curves, and training convergence curves for both models.
5. Generation of 11 publication-grade visualizations in `output_plots/`.
6. Structured serialization to `experiment9_results.json`.
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(EXP_DIR, 'dataset')
PLOTS_DIR = os.path.join(EXP_DIR, 'output_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

IMG_SIZE = 28
PLA_EPOCHS = 60
PLA_ETA = 0.1


def load_and_preprocess_data():
    """Load the English Handwritten Characters dataset, resize/flatten/normalize."""
    csv_path = os.path.join(DATA_DIR, 'english.csv')
    df = pd.read_csv(csv_path)

    X = np.zeros((len(df), IMG_SIZE * IMG_SIZE), dtype=np.float32)
    for i, rel_path in enumerate(df['image']):
        img_path = os.path.join(DATA_DIR, rel_path)
        img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        X[i] = np.asarray(img, dtype=np.float32).flatten() / 255.0

    le = LabelEncoder()
    y = le.fit_transform(df['label'].astype(str))
    class_names = le.classes_

    print(f"[Data] Loaded {X.shape[0]} images | Feature dim: {X.shape[1]} (={IMG_SIZE}x{IMG_SIZE}) | Classes: {len(class_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[Data] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, class_names, df


def run_perceptron_ovr(X_train, y_train, X_test, y_test, n_classes, epochs=PLA_EPOCHS, eta=PLA_ETA):
    """
    Multiclass One-vs-Rest Perceptron Learning Algorithm implemented from scratch.
    Step activation + classical update rule w_{t+1} = w_t + eta*(y - y_hat)*x,
    applied simultaneously (vectorized) across all 62 one-vs-rest sub-classifiers
    for each training sample, each epoch.
    """
    print("\n==================== Model A: Perceptron Learning Algorithm (PLA) ====================")
    n_train, n_features = X_train.shape

    Xb_train = np.hstack([X_train, np.ones((n_train, 1))])
    Xb_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    Y_train_onehot = np.eye(n_classes)[y_train]

    W = np.zeros((n_classes, n_features + 1))
    rng = np.random.RandomState(RANDOM_STATE)

    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_train)
        t0 = time.time()
        n_updates = 0
        for idx in order:
            x = Xb_train[idx]
            scores = W @ x
            y_hat = (scores >= 0).astype(np.float64)
            y_true_vec = Y_train_onehot[idx]
            error = y_true_vec - y_hat
            if np.any(error != 0):
                W += eta * np.outer(error, x)
                n_updates += 1
        epoch_time = time.time() - t0

        train_scores = Xb_train @ W.T
        train_pred = np.argmax(train_scores, axis=1)
        train_acc = float(np.mean(train_pred == y_train))

        test_scores = Xb_test @ W.T
        test_pred = np.argmax(test_scores, axis=1)
        test_acc = float(np.mean(test_pred == y_test))

        history.append({
            'epoch': epoch, 'train_accuracy': train_acc, 'test_accuracy': test_acc,
            'n_updates': int(n_updates), 'epoch_time_sec': float(epoch_time)
        })
        if epoch % 10 == 0 or epoch == 1:
            print(f"[PLA epoch {epoch:3d}] train_acc={train_acc*100:.2f}% | test_acc={test_acc*100:.2f}% | updates={n_updates}")

    final_test_scores = Xb_test @ W.T
    final_test_pred = np.argmax(final_test_scores, axis=1)

    return W, history, final_test_pred, final_test_scores


def run_pla_lr_sensitivity(X_train, y_train, X_test, y_test, n_classes, etas=(0.001, 0.01, 0.1, 1.0), epochs=20):
    """Quick sensitivity scan: does the learning rate eta affect PLA convergence/accuracy?"""
    print("\n==================== PLA Learning-Rate Sensitivity Scan ====================")
    n_train, n_features = X_train.shape
    Xb_train = np.hstack([X_train, np.ones((n_train, 1))])
    Xb_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    Y_train_onehot = np.eye(n_classes)[y_train]

    results = {}
    for eta in etas:
        rng = np.random.RandomState(RANDOM_STATE)
        W = np.zeros((n_classes, n_features + 1))
        curve = []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n_train)
            for idx in order:
                x = Xb_train[idx]
                y_hat = (W @ x >= 0).astype(np.float64)
                error = Y_train_onehot[idx] - y_hat
                if np.any(error != 0):
                    W += eta * np.outer(error, x)
            test_pred = np.argmax(Xb_test @ W.T, axis=1)
            curve.append(float(np.mean(test_pred == y_test)))
        results[eta] = curve
        print(f"[PLA eta={eta}] final test_acc={curve[-1]*100:.2f}%")
    return results


def run_mlp_tuning(X_train, y_train):
    """Staged hyperparameter tuning: (architecture, activation) -> (optimizer, lr) -> batch_size."""
    print("\n==================== Model B: MLP Hyperparameter Tuning ====================")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Stage 1: Architecture (hidden layers/neurons) x Activation function
    print("\n--- Stage 1: Architecture x Activation ---")
    param_grid_1 = {
        'hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128, 64)],
        'activation': ['relu', 'tanh', 'logistic']
    }
    base_mlp_1 = MLPClassifier(
        solver='adam', learning_rate_init=0.001, batch_size=32,
        max_iter=150, random_state=RANDOM_STATE, early_stopping=False
    )
    gs1 = GridSearchCV(base_mlp_1, param_grid_1, cv=cv, scoring='accuracy', n_jobs=-1)
    gs1.fit(X_train, y_train)
    stage1_records = [
        {'hidden_layer_sizes': str(p['hidden_layer_sizes']), 'activation': p['activation'],
         'mean_cv_accuracy': float(m), 'std_cv_accuracy': float(s)}
        for p, m, s in zip(gs1.cv_results_['params'], gs1.cv_results_['mean_test_score'], gs1.cv_results_['std_test_score'])
    ]
    best_arch = gs1.best_params_['hidden_layer_sizes']
    best_activation = gs1.best_params_['activation']
    print(f"[Stage 1] Best: hidden_layer_sizes={best_arch}, activation={best_activation} | CV Acc={gs1.best_score_*100:.2f}%")

    # Stage 2: Optimizer x Learning Rate (using best architecture/activation)
    print("\n--- Stage 2: Optimizer x Learning Rate ---")
    param_grid_2 = {
        'solver': ['sgd', 'adam'],
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]
    }
    base_mlp_2 = MLPClassifier(
        hidden_layer_sizes=best_arch, activation=best_activation, batch_size=32,
        max_iter=150, random_state=RANDOM_STATE, early_stopping=False
    )
    gs2 = GridSearchCV(base_mlp_2, param_grid_2, cv=cv, scoring='accuracy', n_jobs=-1)
    gs2.fit(X_train, y_train)
    stage2_records = [
        {'solver': p['solver'], 'learning_rate_init': p['learning_rate_init'],
         'mean_cv_accuracy': float(m), 'std_cv_accuracy': float(s)}
        for p, m, s in zip(gs2.cv_results_['params'], gs2.cv_results_['mean_test_score'], gs2.cv_results_['std_test_score'])
    ]
    best_solver = gs2.best_params_['solver']
    best_lr = gs2.best_params_['learning_rate_init']
    print(f"[Stage 2] Best: solver={best_solver}, learning_rate_init={best_lr} | CV Acc={gs2.best_score_*100:.2f}%")

    # Stage 3: Batch Size (using best architecture/activation/optimizer/lr)
    print("\n--- Stage 3: Batch Size ---")
    param_grid_3 = {'batch_size': [16, 32, 64, 128]}
    base_mlp_3 = MLPClassifier(
        hidden_layer_sizes=best_arch, activation=best_activation,
        solver=best_solver, learning_rate_init=best_lr,
        max_iter=150, random_state=RANDOM_STATE, early_stopping=False
    )
    gs3 = GridSearchCV(base_mlp_3, param_grid_3, cv=cv, scoring='accuracy', n_jobs=-1)
    gs3.fit(X_train, y_train)
    stage3_records = [
        {'batch_size': p['batch_size'], 'mean_cv_accuracy': float(m), 'std_cv_accuracy': float(s)}
        for p, m, s in zip(gs3.cv_results_['params'], gs3.cv_results_['mean_test_score'], gs3.cv_results_['std_test_score'])
    ]
    best_batch_size = gs3.best_params_['batch_size']
    print(f"[Stage 3] Best: batch_size={best_batch_size} | CV Acc={gs3.best_score_*100:.2f}%")

    best_config = {
        'hidden_layer_sizes': best_arch, 'activation': best_activation,
        'solver': best_solver, 'learning_rate_init': best_lr, 'batch_size': best_batch_size
    }

    tuning_results = {
        'stage1_architecture_activation': stage1_records,
        'stage2_optimizer_lr': stage2_records,
        'stage3_batch_size': stage3_records,
        'best_config': {k: (str(v) if isinstance(v, tuple) else v) for k, v in best_config.items()}
    }

    return best_config, tuning_results


def train_mlp_epochwise(X_train, y_train, X_test, y_test, n_classes, config, epochs=60, solver_override=None):
    """Manual epoch-tracked training via partial_fit for a fair convergence-curve
    comparison against PLA (and for SGD-vs-Adam convergence visualization)."""
    solver = solver_override if solver_override else config['solver']
    mlp = MLPClassifier(
        hidden_layer_sizes=config['hidden_layer_sizes'], activation=config['activation'],
        solver=solver, learning_rate_init=config['learning_rate_init'],
        batch_size=config['batch_size'], random_state=RANDOM_STATE,
        max_iter=1, warm_start=True
    )
    classes = np.arange(n_classes)
    history = []
    for epoch in range(1, epochs + 1):
        mlp.partial_fit(X_train, y_train, classes=classes)
        train_acc = float(mlp.score(X_train, y_train))
        test_acc = float(mlp.score(X_test, y_test))
        loss = float(mlp.loss_) if hasattr(mlp, 'loss_') else None
        history.append({'epoch': epoch, 'train_accuracy': train_acc, 'test_accuracy': test_acc, 'loss': loss})
    return mlp, history


def compute_metrics(y_true, y_pred, n_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    return {
        'accuracy': float(acc),
        'precision_macro': float(precision), 'recall_macro': float(recall), 'f1_macro': float(f1),
        'precision_weighted': float(precision_w), 'recall_weighted': float(recall_w), 'f1_weighted': float(f1_w),
        'confusion_matrix': cm
    }


def compute_roc_micro_macro(y_test, scores, n_classes):
    """Compute micro- and macro-average ROC curves from decision scores / probabilities."""
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), scores.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    fprs, tprs = [], []
    for c in range(n_classes):
        if y_test_bin[:, c].sum() == 0:
            continue
        fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, c], scores[:, c])
        fprs.append(fpr_c)
        tprs.append(tpr_c)

    all_fpr = np.unique(np.concatenate(fprs))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr_c, tpr_c in zip(fprs, tprs):
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= len(fprs)
    auc_macro = auc(all_fpr, mean_tpr)

    return {
        'fpr_micro': fpr_micro.tolist(), 'tpr_micro': tpr_micro.tolist(), 'auc_micro': float(auc_micro),
        'fpr_macro': all_fpr.tolist(), 'tpr_macro': mean_tpr.tolist(), 'auc_macro': float(auc_macro)
    }


def main():
    t_start = time.time()
    print("=" * 80)
    print("EXPERIMENT 9: Perceptron (PLA) vs Multilayer Perceptron (MLP)")
    print("English Handwritten Characters — 62 classes (0-9, A-Z, a-z)")
    print("=" * 80)

    X_train, X_test, y_train, y_test, class_names, df = load_and_preprocess_data()
    n_classes = len(class_names)

    # ---------------- Model A: PLA ----------------
    W_pla, pla_history, pla_test_pred, pla_test_scores = run_perceptron_ovr(
        X_train, y_train, X_test, y_test, n_classes
    )
    pla_metrics = compute_metrics(y_test, pla_test_pred, n_classes)
    pla_roc = compute_roc_micro_macro(y_test, pla_test_scores, n_classes)
    pla_lr_sensitivity = run_pla_lr_sensitivity(X_train, y_train, X_test, y_test, n_classes)

    # ---------------- Model B: MLP ----------------
    best_config, mlp_tuning_results = run_mlp_tuning(X_train, y_train)

    print(f"\n[MLP] Final chosen configuration: {best_config}")
    mlp_final, mlp_history = train_mlp_epochwise(X_train, y_train, X_test, y_test, n_classes, best_config, epochs=PLA_EPOCHS)
    mlp_test_pred = mlp_final.predict(X_test)
    mlp_test_proba = mlp_final.predict_proba(X_test)
    mlp_metrics = compute_metrics(y_test, mlp_test_pred, n_classes)
    mlp_roc = compute_roc_micro_macro(y_test, mlp_test_proba, n_classes)

    # SGD vs Adam convergence comparison (same architecture, different optimizer)
    print("\n==================== SGD vs Adam Convergence Comparison ====================")
    _, mlp_history_sgd = train_mlp_epochwise(X_train, y_train, X_test, y_test, n_classes, best_config, epochs=PLA_EPOCHS, solver_override='sgd')
    _, mlp_history_adam = train_mlp_epochwise(X_train, y_train, X_test, y_test, n_classes, best_config, epochs=PLA_EPOCHS, solver_override='adam')

    print(f"\n[PLA] Test Accuracy: {pla_metrics['accuracy']*100:.2f}% | Macro-F1: {pla_metrics['f1_macro']*100:.2f}%")
    print(f"[MLP] Test Accuracy: {mlp_metrics['accuracy']*100:.2f}% | Macro-F1: {mlp_metrics['f1_macro']*100:.2f}%")

    # ---------------- Plots ----------------
    generate_plots(
        df, class_names, X_train, y_train, X_test, y_test, n_classes,
        W_pla, pla_history, pla_lr_sensitivity, pla_metrics, pla_roc,
        best_config, mlp_tuning_results, mlp_history, mlp_history_sgd, mlp_history_adam,
        mlp_metrics, mlp_roc
    )

    # ---------------- Serialize results ----------------
    results = {
        'dataset_info': {
            'n_samples': int(len(df)), 'n_features': int(IMG_SIZE * IMG_SIZE),
            'img_size': IMG_SIZE, 'n_classes': int(n_classes),
            'n_train': int(X_train.shape[0]), 'n_test': int(X_test.shape[0])
        },
        'pla': {
            'history': pla_history,
            'lr_sensitivity': {str(k): v for k, v in pla_lr_sensitivity.items()},
            'test_metrics': {k: v for k, v in pla_metrics.items() if k != 'confusion_matrix'},
            'confusion_matrix': pla_metrics['confusion_matrix'].tolist(),
            'roc': {k: v for k, v in pla_roc.items() if 'fpr' not in k and 'tpr' not in k}
        },
        'mlp': {
            'tuning': mlp_tuning_results,
            'best_config': {k: (str(v) if isinstance(v, tuple) else v) for k, v in best_config.items()},
            'history': mlp_history,
            'history_sgd': mlp_history_sgd,
            'history_adam': mlp_history_adam,
            'test_metrics': {k: v for k, v in mlp_metrics.items() if k != 'confusion_matrix'},
            'confusion_matrix': mlp_metrics['confusion_matrix'].tolist(),
            'roc': {k: v for k, v in mlp_roc.items() if 'fpr' not in k and 'tpr' not in k}
        },
        'runtime_sec': time.time() - t_start
    }

    results_path = os.path.join(BASE_DIR, 'experiment9_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] Results JSON: {results_path}")
    print(f"\n[Done] Total runtime: {time.time() - t_start:.1f}s")


def generate_plots(df, class_names, X_train, y_train, X_test, y_test, n_classes,
                    W_pla, pla_history, pla_lr_sensitivity, pla_metrics, pla_roc,
                    best_config, mlp_tuning_results, mlp_history, mlp_history_sgd, mlp_history_adam,
                    mlp_metrics, mlp_roc):
    print("\n==================== Generating Publication Plots ====================")

    # ---------------------------------------------------------
    # Plot 1: Sample Character Gallery (EDA)
    # ---------------------------------------------------------
    rng = np.random.RandomState(RANDOM_STATE)
    sample_classes = rng.choice(class_names, size=20, replace=False)
    fig, axes = plt.subplots(2, 10, figsize=(18, 4.2))
    for ax, cls in zip(axes.flat, sample_classes):
        row = df[df['label'].astype(str) == cls].sample(1, random_state=RANDOM_STATE).iloc[0]
        img = Image.open(os.path.join(DATA_DIR, row['image'])).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        ax.imshow(np.asarray(img), cmap='gray')
        ax.set_title(f"'{cls}'", fontsize=12, fontweight='bold')
        ax.axis('off')
    plt.suptitle('Sample Character Gallery (20 of 62 Classes, 28x28 Preprocessed)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_sample_character_gallery.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 01_sample_character_gallery.png")

    # ---------------------------------------------------------
    # Plot 2: PLA Learning-Rate Sensitivity
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#2980b9', '#27ae60', '#e67e22', '#c0392b']
    for (eta, curve), color in zip(pla_lr_sensitivity.items(), colors):
        ax.plot(range(1, len(curve) + 1), [c * 100 for c in curve], marker='o', markersize=3,
                linewidth=2, color=color, label=f'$\\eta$={eta}')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title('PLA Learning-Rate ($\\eta$) Sensitivity — Test Accuracy vs Epoch', fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_pla_learning_rate_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 02_pla_learning_rate_sensitivity.png")

    # ---------------------------------------------------------
    # Plot 3: PLA Training Convergence (Train vs Test Accuracy)
    # ---------------------------------------------------------
    epochs = [h['epoch'] for h in pla_history]
    train_acc = [h['train_accuracy'] * 100 for h in pla_history]
    test_acc = [h['test_accuracy'] * 100 for h in pla_history]
    n_updates = [h['n_updates'] for h in pla_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(epochs, train_acc, linewidth=2.2, color='#2980b9', label='Train Accuracy')
    ax1.plot(epochs, test_acc, linewidth=2.2, color='#c0392b', label='Test Accuracy')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('PLA: Train vs Test Accuracy Convergence', fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(epochs, n_updates, linewidth=2.2, color='#8e44ad')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Number of Weight Updates (Misclassifications)', fontweight='bold')
    ax2.set_title('PLA: Misclassification Count per Epoch', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_pla_training_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 03_pla_training_convergence.png")

    # ---------------------------------------------------------
    # Plot 4: MLP Architecture x Activation Tuning Heatmap
    # ---------------------------------------------------------
    df_stage1 = pd.DataFrame(mlp_tuning_results['stage1_architecture_activation'])
    pivot1 = df_stage1.pivot(index='hidden_layer_sizes', columns='activation', values='mean_cv_accuracy')
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': '3-Fold CV Accuracy'})
    ax.set_title('MLP Tuning Stage 1: Architecture x Activation Function', fontweight='bold')
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Hidden Layer Sizes')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_mlp_architecture_activation_tuning.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 04_mlp_architecture_activation_tuning.png")

    # ---------------------------------------------------------
    # Plot 5: MLP Optimizer x Learning Rate + SGD vs Adam Convergence
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    df_stage2 = pd.DataFrame(mlp_tuning_results['stage2_optimizer_lr'])
    for solver, color in zip(['sgd', 'adam'], ['#e67e22', '#2980b9']):
        sub = df_stage2[df_stage2['solver'] == solver].sort_values('learning_rate_init')
        axes[0].plot(sub['learning_rate_init'], sub['mean_cv_accuracy'], marker='o', linewidth=2,
                     color=color, label=solver.upper())
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Learning Rate (log scale)', fontweight='bold')
    axes[0].set_ylabel('3-Fold CV Accuracy', fontweight='bold')
    axes[0].set_title('MLP Tuning Stage 2: Optimizer x Learning Rate', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    epochs_h = [h['epoch'] for h in mlp_history_sgd]
    axes[1].plot(epochs_h, [h['test_accuracy'] * 100 for h in mlp_history_sgd], linewidth=2.2,
                 color='#e67e22', label='SGD — Test Accuracy')
    axes[1].plot(epochs_h, [h['test_accuracy'] * 100 for h in mlp_history_adam], linewidth=2.2,
                 color='#2980b9', label='Adam — Test Accuracy')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Test Accuracy (%)', fontweight='bold')
    axes[1].set_title('SGD vs Adam: Convergence Speed (Same Architecture)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_mlp_optimizer_lr_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 05_mlp_optimizer_lr_comparison.png")

    # ---------------------------------------------------------
    # Plot 6: MLP Batch Size Comparison
    # ---------------------------------------------------------
    df_stage3 = pd.DataFrame(mlp_tuning_results['stage3_batch_size']).sort_values('batch_size')
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(df_stage3['batch_size'].astype(str), df_stage3['mean_cv_accuracy'],
                   yerr=df_stage3['std_cv_accuracy'], color='#16a085', edgecolor='black', alpha=0.85, capsize=5)
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('3-Fold CV Accuracy', fontweight='bold')
    ax.set_title('MLP Tuning Stage 3: Batch Size Sensitivity', fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_mlp_batch_size_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 06_mlp_batch_size_comparison.png")

    # ---------------------------------------------------------
    # Plot 7: PLA vs MLP Convergence Curve Comparison (Direct A/B)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(epochs, test_acc, linewidth=2.2, linestyle='--', color='#c0392b', label='PLA — Test Accuracy')
    ax.plot([h['epoch'] for h in mlp_history], [h['test_accuracy'] * 100 for h in mlp_history],
            linewidth=2.5, color='#16a085', label='MLP (Tuned) — Test Accuracy')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title('A/B Comparison: PLA vs Tuned MLP — Convergence Curves', fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_pla_vs_mlp_convergence_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 07_pla_vs_mlp_convergence_curves.png")

    # ---------------------------------------------------------
    # Plot 8: Confusion Matrices (PLA vs MLP)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(pla_metrics['confusion_matrix'], cmap='Blues', ax=axes[0], cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"PLA Confusion Matrix (Test Acc={pla_metrics['accuracy']*100:.1f}%)", fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].tick_params(axis='both', labelsize=5)

    sns.heatmap(mlp_metrics['confusion_matrix'], cmap='Greens', ax=axes[1], cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f"MLP Confusion Matrix (Test Acc={mlp_metrics['accuracy']*100:.1f}%)", fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].tick_params(axis='both', labelsize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_confusion_matrices_pla_vs_mlp.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 08_confusion_matrices_pla_vs_mlp.png")

    # ---------------------------------------------------------
    # Plot 9: ROC Curves (Micro/Macro Average) — PLA vs MLP
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(pla_roc['fpr_micro'], pla_roc['tpr_micro'], linewidth=2, linestyle='--', color='#e74c3c',
            label=f"PLA micro-avg (AUC={pla_roc['auc_micro']:.3f})")
    ax.plot(pla_roc['fpr_macro'], pla_roc['tpr_macro'], linewidth=2, linestyle=':', color='#e67e22',
            label=f"PLA macro-avg (AUC={pla_roc['auc_macro']:.3f})")
    ax.plot(mlp_roc['fpr_micro'], mlp_roc['tpr_micro'], linewidth=2.2, color='#2980b9',
            label=f"MLP micro-avg (AUC={mlp_roc['auc_micro']:.3f})")
    ax.plot(mlp_roc['fpr_macro'], mlp_roc['tpr_macro'], linewidth=2.2, linestyle='-.', color='#16a085',
            label=f"MLP macro-avg (AUC={mlp_roc['auc_macro']:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.2, label='Chance')
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('Micro/Macro-Average ROC Curves: PLA vs MLP (62-Class OvR)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '09_roc_curves_micro_macro.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 09_roc_curves_micro_macro.png")

    # ---------------------------------------------------------
    # Plot 10: Performance Metrics Comparison (PLA vs MLP)
    # ---------------------------------------------------------
    metrics_names = ['Accuracy', 'Precision\n(macro)', 'Recall\n(macro)', 'F1-Score\n(macro)']
    pla_vals = [pla_metrics['accuracy'], pla_metrics['precision_macro'], pla_metrics['recall_macro'], pla_metrics['f1_macro']]
    mlp_vals = [mlp_metrics['accuracy'], mlp_metrics['precision_macro'], mlp_metrics['recall_macro'], mlp_metrics['f1_macro']]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_names))
    width = 0.35
    b1 = ax.bar(x - width/2, pla_vals, width, color='#c0392b', edgecolor='black', label='PLA')
    b2 = ax.bar(x + width/2, mlp_vals, width, color='#16a085', edgecolor='black', label='MLP (Tuned)')
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}',
                    ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Final Performance Comparison: PLA vs Tuned MLP', fontweight='bold')
    ax.set_ylim(0, max(max(pla_vals), max(mlp_vals)) * 1.25)
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '10_performance_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 10_performance_metrics_comparison.png")

    # ---------------------------------------------------------
    # Plot 11: Perceptron Weight Vector Visualization
    # ---------------------------------------------------------
    sample_class_idx = rng.choice(n_classes, size=10, replace=False)
    fig, axes = plt.subplots(1, 10, figsize=(18, 2.2))
    for ax, ci in zip(axes, sample_class_idx):
        w_img = W_pla[ci, :-1].reshape(IMG_SIZE, IMG_SIZE)
        ax.imshow(w_img, cmap='seismic', vmin=-np.max(np.abs(w_img)), vmax=np.max(np.abs(w_img)))
        ax.set_title(f"'{class_names[ci]}'", fontsize=11, fontweight='bold')
        ax.axis('off')
    plt.suptitle('PLA Learned Weight Vectors (Reshaped to 28x28) — 10 Sample Classes', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '11_perceptron_weight_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 11_perceptron_weight_visualization.png")


if __name__ == '__main__':
    main()
