#!/usr/bin/env python3
"""
Experiment 8: Clustering Human Activity Recognition Data using K-Means,
DBSCAN, and Hierarchical Clustering
Course: Machine Learning Algorithms Laboratory (ICS1512)
Student Name: Danusu K | Register Number: 3122247001013
Faculty: Dr. Poreddy Ajay Kumar Reddy
Dataset: Human Activity Recognition Using Smartphones (UCI HAR Dataset)

This script implements:
1. Dataset loading (train + test partitions merged), standardization, and PCA-based
   dimensionality reduction for tractable, curse-of-dimensionality-aware clustering.
2. Exploratory Data Analysis: activity distribution, PCA scree/cumulative variance.
3. Model A - K-Means Clustering: Elbow Method (WCSS) + Silhouette analysis across k=2..8.
4. Model B - DBSCAN: k-distance graph for eps selection + (eps, minPts) grid search.
5. Model C - Hierarchical Agglomerative Clustering (HAC): linkage comparison
   (single/complete/average/ward) + dendrogram on a stratified subsample.
6. Internal validation (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external
   validation against ground-truth activity labels (ARI, NMI) for all three models.
7. Cluster-to-activity contingency (confusion) analysis.
8. Generation of 12 publication-grade visualizations in `output_plots/`.
9. Structured serialization to `experiment8_results.json`.
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

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment

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

ACTIVITY_NAMES = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
}
ACTIVITY_COLORS = {
    1: '#2980b9', 2: '#27ae60', 3: '#16a085',
    4: '#e74c3c', 5: '#e67e22', 6: '#8e44ad'
}

# Hierarchical clustering / dendrogram subsample size (O(n^2) memory & time complexity)
HAC_SUBSAMPLE_SIZE = 2500
# t-SNE visualization subsample size (t-SNE is O(n^2)-ish and slow on 10k+ rows)
TSNE_SUBSAMPLE_SIZE = 3000
# PCA cumulative variance retention target for clustering (mitigates curse of dimensionality)
PCA_VARIANCE_TARGET = 0.90


def load_and_preprocess_data():
    """Load train+test partitions of the UCI HAR dataset, merge, and standardize."""
    X_train = np.loadtxt(os.path.join(DATA_DIR, 'train', 'X_train.txt'))
    y_train = np.loadtxt(os.path.join(DATA_DIR, 'train', 'y_train.txt')).astype(int)
    subj_train = np.loadtxt(os.path.join(DATA_DIR, 'train', 'subject_train.txt')).astype(int)

    X_test = np.loadtxt(os.path.join(DATA_DIR, 'test', 'X_test.txt'))
    y_test = np.loadtxt(os.path.join(DATA_DIR, 'test', 'y_test.txt')).astype(int)
    subj_test = np.loadtxt(os.path.join(DATA_DIR, 'test', 'subject_test.txt')).astype(int)

    with open(os.path.join(DATA_DIR, 'features.txt')) as f:
        feature_names = [line.strip().split(' ', 1)[1] for line in f if line.strip()]

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    subjects = np.concatenate([subj_train, subj_test])

    print(f"[Data] Merged dataset shape: {X.shape} | Missing values: {np.isnan(X).sum()}")
    print(f"[Data] Subjects: {len(np.unique(subjects))} | Activities: {len(np.unique(y))}")

    # No missing values in this dataset (pre-cleaned UCI release); standardize directly.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, subjects, feature_names


def perform_pca_analysis(X_scaled):
    """Full PCA for scree/cumulative-variance reporting + reduced PCA for clustering."""
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)

    var_ratio = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    eigenvalues = pca_full.explained_variance_

    n_comp_90 = int(np.argmax(cum_var >= 0.90) + 1)
    n_comp_95 = int(np.argmax(cum_var >= 0.95) + 1)
    n_comp_99 = int(np.argmax(cum_var >= 0.99) + 1)

    print(f"[PCA] Total features: {X_scaled.shape[1]}")
    print(f"[PCA] 90% variance at {n_comp_90} PCs | 95% at {n_comp_95} PCs | 99% at {n_comp_99} PCs")

    n_selected = int(np.argmax(cum_var >= PCA_VARIANCE_TARGET) + 1)
    pca_reduced = PCA(n_components=n_selected, random_state=RANDOM_STATE)
    X_pca = pca_reduced.fit_transform(X_scaled)

    pca_table = [
        {'component': i + 1, 'eigenvalue': float(eigenvalues[i]),
         'explained_variance_pct': float(var_ratio[i] * 100),
         'cumulative_variance_pct': float(cum_var[i] * 100)}
        for i in range(min(30, len(var_ratio)))
    ]

    pca_info = {
        'total_features': int(X_scaled.shape[1]),
        'n_comp_90': n_comp_90, 'n_comp_95': n_comp_95, 'n_comp_99': n_comp_99,
        'chosen_components': n_selected,
        'variance_target_pct': PCA_VARIANCE_TARGET * 100,
        'actual_explained_variance_pct': float(cum_var[n_selected - 1] * 100),
        'dimension_reduction_pct': float((1 - n_selected / X_scaled.shape[1]) * 100),
        'pca_table': pca_table
    }
    return pca_full, pca_reduced, X_pca, pca_info


def run_kmeans_elbow(X_pca, y_true, k_range=range(2, 9)):
    """Elbow method (WCSS) + Silhouette across candidate k values."""
    print("\n==================== K-Means: Elbow Method & Silhouette Scan ====================")
    records = []
    models = {}
    for k in k_range:
        t0 = time.time()
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_pca)
        fit_time = time.time() - t0

        sil = silhouette_score(X_pca, labels, sample_size=5000, random_state=RANDOM_STATE)
        db = davies_bouldin_score(X_pca, labels)
        ch = calinski_harabasz_score(X_pca, labels)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)

        records.append({
            'k': k, 'wcss': float(km.inertia_), 'silhouette': float(sil),
            'davies_bouldin': float(db), 'calinski_harabasz': float(ch),
            'ari': float(ari), 'nmi': float(nmi), 'fit_time_sec': float(fit_time)
        })
        models[k] = km
        print(f"[K-Means k={k}] WCSS={km.inertia_:.1f} | Silhouette={sil:.4f} | DB={db:.4f} | "
              f"CH={ch:.1f} | ARI={ari:.4f} | NMI={nmi:.4f}")

    best_k = max(records, key=lambda r: r['silhouette'])['k']
    print(f"[K-Means] Best k by Silhouette Score: {best_k}")
    return records, models, best_k


def run_dbscan_tuning(X_pca, y_true):
    """k-distance graph + grid search over (eps, minPts)."""
    print("\n==================== DBSCAN: eps / minPts Tuning ====================")

    min_pts_for_kdist = 10
    nn = NearestNeighbors(n_neighbors=min_pts_for_kdist)
    nn.fit(X_pca)
    distances, _ = nn.kneighbors(X_pca)
    k_distances = np.sort(distances[:, -1])

    eps_candidates = sorted(set(
        np.round(np.percentile(k_distances, p), 2) for p in [80, 85, 88, 90, 92, 95]
    ))
    min_pts_candidates = [5, 10, 15, 20]

    grid_records = []
    best_config = None
    best_sil = -1.0
    for eps in eps_candidates:
        for min_pts in min_pts_candidates:
            db = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=-1)
            labels = db.fit_predict(X_pca)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = float(np.mean(labels == -1))

            if n_clusters >= 2:
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) >= 2:
                    sil = float(silhouette_score(X_pca[mask], labels[mask], sample_size=min(5000, mask.sum()), random_state=RANDOM_STATE))
                else:
                    sil = float('nan')
            else:
                sil = float('nan')

            ari = float(adjusted_rand_score(y_true, labels))
            nmi = float(normalized_mutual_info_score(y_true, labels))

            rec = {
                'eps': float(eps), 'min_pts': min_pts, 'n_clusters': n_clusters,
                'noise_ratio': noise_ratio, 'silhouette': sil, 'ari': ari, 'nmi': nmi
            }
            grid_records.append(rec)
            print(f"[DBSCAN eps={eps:.2f}, minPts={min_pts}] clusters={n_clusters} | "
                  f"noise={noise_ratio*100:.1f}% | sil={sil} | ARI={ari:.4f}")

            if n_clusters >= 2 and noise_ratio < 0.5 and not np.isnan(sil) and sil > best_sil:
                best_sil = sil
                best_config = rec

    if best_config is None:
        # Fallback: pick config with most clusters and lowest noise among n_clusters>=2
        candidates = [r for r in grid_records if r['n_clusters'] >= 2]
        best_config = max(candidates, key=lambda r: (-r['noise_ratio'], r['ari'])) if candidates else grid_records[0]

    best_db = DBSCAN(eps=best_config['eps'], min_samples=best_config['min_pts'], n_jobs=-1)
    best_labels = best_db.fit_predict(X_pca)

    print(f"[DBSCAN] Selected Best Config: eps={best_config['eps']}, minPts={best_config['min_pts']} "
          f"| clusters={best_config['n_clusters']} | noise={best_config['noise_ratio']*100:.1f}%")

    return grid_records, k_distances, best_config, best_labels


def run_hierarchical_clustering(X_pca, y_true, subsample_idx):
    """Linkage comparison (single/complete/average/ward) on a stratified subsample."""
    print("\n==================== Hierarchical Agglomerative Clustering ====================")
    X_sub = X_pca[subsample_idx]
    y_sub = y_true[subsample_idx]

    linkage_methods = ['single', 'complete', 'average', 'ward']
    linkage_records = {}
    n_clusters_hac = 6

    for method in linkage_methods:
        t0 = time.time()
        agg = AgglomerativeClustering(n_clusters=n_clusters_hac, linkage=method)
        labels = agg.fit_predict(X_sub)
        fit_time = time.time() - t0

        sil = float(silhouette_score(X_sub, labels))
        db = float(davies_bouldin_score(X_sub, labels))
        ch = float(calinski_harabasz_score(X_sub, labels))
        ari = float(adjusted_rand_score(y_sub, labels))
        nmi = float(normalized_mutual_info_score(y_sub, labels))

        linkage_records[method] = {
            'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch,
            'ari': ari, 'nmi': nmi, 'fit_time_sec': float(fit_time), 'labels': labels.tolist()
        }
        print(f"[HAC linkage={method}] Silhouette={sil:.4f} | DB={db:.4f} | CH={ch:.1f} | "
              f"ARI={ari:.4f} | NMI={nmi:.4f} | time={fit_time:.2f}s")

    # Ward's linkage is mandated by the experiment brief as the primary HAC configuration
    # (minimum-variance criterion, most balanced partitions); single/average linkage tend to
    # produce degenerate chaining (one dominant cluster + trivial singletons) on this dataset,
    # which can score deceptively high on raw silhouette without being structurally meaningful.
    best_by_silhouette = max(linkage_records, key=lambda m: linkage_records[m]['silhouette'])
    primary_linkage = 'ward'
    print(f"[HAC] Highest raw Silhouette: {best_by_silhouette} (may be a degenerate chaining artifact)")
    print(f"[HAC] Primary linkage used for downstream analysis (per assignment spec): {primary_linkage}")

    # Precompute scipy linkage matrix (Ward) for dendrogram plotting
    Z = linkage(X_sub, method='ward')

    return linkage_records, primary_linkage, Z, X_sub, y_sub


def cluster_to_activity_mapping(cluster_labels, y_true, n_activities=6):
    """Map cluster IDs to majority-vote ground-truth activity using Hungarian assignment."""
    unique_clusters = sorted(set(cluster_labels))
    unique_clusters_no_noise = [c for c in unique_clusters if c != -1]
    n_clusters = len(unique_clusters_no_noise)

    cost = np.zeros((n_clusters, n_activities))
    for i, c in enumerate(unique_clusters_no_noise):
        mask = cluster_labels == c
        for j in range(n_activities):
            cost[i, j] = -np.sum(y_true[mask] == (j + 1))

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {unique_clusters_no_noise[r]: col_ind[r] + 1 for r in row_ind}

    mapped_pred = np.array([mapping.get(c, 0) for c in cluster_labels])
    return mapped_pred, mapping


def generate_plots(X_scaled, y, subjects, pca_full, pca_reduced, X_pca, pca_info,
                    kmeans_records, kmeans_models, best_k,
                    dbscan_grid, k_distances, dbscan_best_config, dbscan_labels,
                    hac_records, best_linkage, Z, X_hac_sub, y_hac_sub, hac_labels_best):
    print("\n==================== Generating Publication Plots ====================")

    # ---------------------------------------------------------
    # Plot 1: Activity Class Distribution (EDA)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = pd.Series(y).value_counts().sort_index()
    labels = [ACTIVITY_NAMES[i] for i in counts.index]
    colors = [ACTIVITY_COLORS[i] for i in counts.index]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 30, str(v), ha='center', fontweight='bold')
    ax.set_ylabel('Number of Samples (Windows)', fontweight='bold')
    ax.set_title('Class Distribution: UCI HAR Activity Labels (N=10,299)', fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_activity_class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 01_activity_class_distribution.png")

    # ---------------------------------------------------------
    # Plot 2: PCA Scree & Cumulative Variance
    # ---------------------------------------------------------
    var_exp = pca_full.explained_variance_ratio_[:50] * 100
    cum_var_exp = np.cumsum(pca_full.explained_variance_ratio_)[:50] * 100
    eigenvalues = pca_full.explained_variance_[:50]
    x_idx = np.arange(1, len(var_exp) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.bar(x_idx, var_exp, color='#2b5c8f', alpha=0.7, edgecolor='black')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_idx, eigenvalues, color='#c0392b', linewidth=1.5, label=r'Eigenvalue ($\lambda$)')
    ax1_twin.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, label=r'Kaiser ($\lambda=1$)')
    ax1.set_xlabel('Principal Component Index (first 50)')
    ax1.set_ylabel('Explained Variance Ratio (%)', color='#2b5c8f')
    ax1_twin.set_ylabel(r'Eigenvalue ($\lambda$)', color='#c0392b')
    ax1.set_title('Scree Plot: Individual Variance (First 50 PCs)', fontweight='bold')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.5)

    n_sel = pca_info['chosen_components']
    ax2.plot(x_idx, cum_var_exp, color='#16a085', linewidth=2.5)
    ax2.axhline(90.0, color='#e74c3c', linestyle='--', linewidth=2, label='90% Threshold')
    if n_sel <= 50:
        ax2.axvline(n_sel, color='#8e44ad', linestyle=':', linewidth=2, label=f'Selected k={n_sel} PCs')
        ax2.scatter([n_sel], [cum_var_exp[n_sel-1]], color='#8e44ad', s=100, zorder=5)
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title(f'Cumulative Variance ({n_sel} PCs = {pca_info["actual_explained_variance_pct"]:.2f}%)', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_pca_scree_and_cumulative_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 02_pca_scree_and_cumulative_variance.png")

    # ---------------------------------------------------------
    # Plot 3: K-Means Elbow Curve & Silhouette Curve
    # ---------------------------------------------------------
    ks = [r['k'] for r in kmeans_records]
    wcss = [r['wcss'] for r in kmeans_records]
    sils = [r['silhouette'] for r in kmeans_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(ks, wcss, marker='o', linewidth=2.5, color='#2980b9')
    ax1.axvline(best_k, color='#c0392b', linestyle='--', linewidth=1.5, label=f'Selected k={best_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax1.set_ylabel('WCSS (Inertia)', fontweight='bold')
    ax1.set_title('K-Means Elbow Method: k vs WCSS', fontweight='bold')
    ax1.set_xticks(ks)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(ks, sils, marker='s', linewidth=2.5, color='#27ae60')
    ax2.axvline(best_k, color='#c0392b', linestyle='--', linewidth=1.5, label=f'Best k={best_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontweight='bold')
    ax2.set_title('K-Means: k vs Silhouette Score', fontweight='bold')
    ax2.set_xticks(ks)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_kmeans_elbow_and_silhouette.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 03_kmeans_elbow_and_silhouette.png")

    # ---------------------------------------------------------
    # Plot 4: DBSCAN k-distance graph (eps selection)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(len(k_distances)), k_distances, color='#34495e', linewidth=1.8)
    ax.axhline(dbscan_best_config['eps'], color='#c0392b', linestyle='--', linewidth=2,
               label=f'Selected $\\epsilon$={dbscan_best_config["eps"]}')
    ax.set_xlabel('Points sorted by distance to 10th Nearest Neighbor', fontweight='bold')
    ax.set_ylabel('10-NN Distance', fontweight='bold')
    ax.set_title('DBSCAN: k-Distance Graph for $\\epsilon$ Selection', fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_dbscan_kdistance_graph.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 04_dbscan_kdistance_graph.png")

    # ---------------------------------------------------------
    # Plot 5: DBSCAN (eps, minPts) Grid Search Heatmaps (Silhouette & #Clusters)
    # ---------------------------------------------------------
    df_grid = pd.DataFrame(dbscan_grid)
    pivot_sil = df_grid.pivot(index='min_pts', columns='eps', values='silhouette')
    pivot_noise = df_grid.pivot(index='min_pts', columns='eps', values='noise_ratio') * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(pivot_sil, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Silhouette Score'})
    axes[0].set_title('DBSCAN Grid Search: Silhouette Score', fontweight='bold')
    axes[0].set_xlabel(r'$\epsilon$ (Neighborhood Radius)')
    axes[0].set_ylabel('minPts')

    sns.heatmap(pivot_noise, annot=True, fmt='.1f', cmap='OrRd', ax=axes[1], cbar_kws={'label': 'Noise Points (%)'})
    axes[1].set_title('DBSCAN Grid Search: Noise Ratio (%)', fontweight='bold')
    axes[1].set_xlabel(r'$\epsilon$ (Neighborhood Radius)')
    axes[1].set_ylabel('minPts')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_dbscan_grid_search_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 05_dbscan_grid_search_heatmaps.png")

    # ---------------------------------------------------------
    # Plot 6: Dendrograms - Linkage Comparison (single/complete/average/ward)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    linkage_methods = ['single', 'complete', 'average', 'ward']
    for ax, method in zip(axes.flat, linkage_methods):
        Zm = linkage(X_hac_sub, method=method)
        dendrogram(Zm, ax=ax, truncate_mode='lastp', p=30, no_labels=True,
                   color_threshold=0.7 * max(Zm[:, 2]))
        ax.set_title(f'Dendrogram — {method.capitalize()} Linkage', fontweight='bold')
        ax.set_xlabel('Sample Clusters (truncated)')
        ax.set_ylabel('Cophenetic Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_dendrograms_linkage_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 06_dendrograms_linkage_comparison.png")

    # ---------------------------------------------------------
    # Plot 7: PCA 2D Cluster Scatter Plots (Ground Truth vs K-Means vs DBSCAN vs HAC)
    # ---------------------------------------------------------
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_scaled)
    var2d = pca_2d.explained_variance_ratio_ * 100

    kmeans_best_labels = kmeans_models[best_k].labels_

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))

    ax = axes[0]
    for act_id, name in ACTIVITY_NAMES.items():
        mask = y == act_id
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=6, alpha=0.6, color=ACTIVITY_COLORS[act_id], label=name)
    ax.set_title('Ground-Truth Activity Labels', fontweight='bold')
    ax.legend(markerscale=3, fontsize=8, loc='best')

    ax = axes[1]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=6, alpha=0.6, c=kmeans_best_labels, cmap='tab10')
    ax.set_title(f'K-Means Clusters (k={best_k})', fontweight='bold')

    ax = axes[2]
    noise_mask = dbscan_labels == -1
    ax.scatter(X_2d[~noise_mask, 0], X_2d[~noise_mask, 1], s=6, alpha=0.6, c=dbscan_labels[~noise_mask], cmap='tab10')
    ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], s=6, alpha=0.4, c='lightgray', label='Noise')
    ax.set_title(f'DBSCAN Clusters ($\\epsilon$={dbscan_best_config["eps"]}, minPts={dbscan_best_config["min_pts"]})', fontweight='bold')
    ax.legend(fontsize=8)

    for a in axes:
        a.set_xlabel(f'PC1 ({var2d[0]:.1f}%)')
        a.set_ylabel(f'PC2 ({var2d[1]:.1f}%)')
        a.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_pca_2d_cluster_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 07_pca_2d_cluster_scatter.png")

    return X_2d


def generate_hac_scatter_and_tsne(X_scaled, y, subsample_idx, hac_labels_best, best_linkage,
                                   kmeans_models, best_k, dbscan_labels, dbscan_best_config):
    """Plot 8: HAC cluster scatter (on its own subsample, PCA-2D) + Plot 9: t-SNE comparison."""
    print("...generating HAC scatter and t-SNE plots")
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d_full = pca_2d.fit_transform(X_scaled)
    var2d = pca_2d.explained_variance_ratio_ * 100
    X_2d_sub = X_2d_full[subsample_idx]
    y_sub = y[subsample_idx]

    # ---------------------------------------------------------
    # Plot 8: HAC Cluster Scatter (PCA-2D, subsample) vs Ground Truth (same subsample)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for act_id, name in ACTIVITY_NAMES.items():
        mask = y_sub == act_id
        axes[0].scatter(X_2d_sub[mask, 0], X_2d_sub[mask, 1], s=10, alpha=0.7, color=ACTIVITY_COLORS[act_id], label=name)
    axes[0].set_title('Ground-Truth Labels (HAC Subsample)', fontweight='bold')
    axes[0].legend(markerscale=2, fontsize=8)

    axes[1].scatter(X_2d_sub[:, 0], X_2d_sub[:, 1], s=10, alpha=0.7, c=hac_labels_best, cmap='tab10')
    axes[1].set_title(f'Hierarchical Clusters ({best_linkage.capitalize()} Linkage)', fontweight='bold')

    for a in axes:
        a.set_xlabel(f'PC1 ({var2d[0]:.1f}%)')
        a.set_ylabel(f'PC2 ({var2d[1]:.1f}%)')
        a.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_hac_pca_2d_cluster_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 08_hac_pca_2d_cluster_scatter.png")

    # ---------------------------------------------------------
    # Plot 9: t-SNE 2D Projection (subsample) — Ground Truth vs K-Means vs DBSCAN
    # ---------------------------------------------------------
    rng = np.random.RandomState(RANDOM_STATE)
    tsne_idx = rng.choice(len(y), size=min(TSNE_SUBSAMPLE_SIZE, len(y)), replace=False)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init='pca', max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled[tsne_idx])

    y_tsne = y[tsne_idx]
    kmeans_tsne_labels = kmeans_models[best_k].labels_[tsne_idx]
    dbscan_tsne_labels = dbscan_labels[tsne_idx]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for act_id, name in ACTIVITY_NAMES.items():
        mask = y_tsne == act_id
        axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=8, alpha=0.65, color=ACTIVITY_COLORS[act_id], label=name)
    axes[0].set_title('t-SNE: Ground-Truth Activity Labels', fontweight='bold')
    axes[0].legend(markerscale=3, fontsize=7, loc='best')

    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], s=8, alpha=0.65, c=kmeans_tsne_labels, cmap='tab10')
    axes[1].set_title(f't-SNE: K-Means Clusters (k={best_k})', fontweight='bold')

    noise_mask = dbscan_tsne_labels == -1
    axes[2].scatter(X_tsne[~noise_mask, 0], X_tsne[~noise_mask, 1], s=8, alpha=0.65, c=dbscan_tsne_labels[~noise_mask], cmap='tab10')
    axes[2].scatter(X_tsne[noise_mask, 0], X_tsne[noise_mask, 1], s=8, alpha=0.4, c='lightgray', label='Noise')
    axes[2].set_title('t-SNE: DBSCAN Clusters', fontweight='bold')
    axes[2].legend(fontsize=8)

    for a in axes:
        a.set_xlabel('t-SNE Dimension 1')
        a.set_ylabel('t-SNE Dimension 2')
        a.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '09_tsne_cluster_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 09_tsne_cluster_comparison.png")

    return tsne_idx


def generate_metric_and_confusion_plots(kmeans_records, best_k, dbscan_best_config, dbscan_labels,
                                         hac_records, best_linkage, y, y_hac_sub,
                                         kmeans_models):
    """Plot 10: Internal metrics bar chart, Plot 11: External metrics bar chart,
    Plot 12: Cluster-to-activity contingency heatmaps."""
    print("...generating metric comparison and contingency plots")

    km_rec = next(r for r in kmeans_records if r['k'] == best_k)

    # DBSCAN internal/external metrics for the chosen config (recompute cleanly)
    mask = dbscan_labels != -1
    X_dummy = None  # not needed; metrics recomputed by caller before this function ideally
    algo_names = ['K-Means', 'DBSCAN', 'Hierarchical']
    sil_vals = [km_rec['silhouette'], dbscan_best_config['silhouette'], hac_records[best_linkage]['silhouette']]
    db_vals = [km_rec['davies_bouldin'], None, hac_records[best_linkage]['davies_bouldin']]
    ch_vals = [km_rec['calinski_harabasz'], None, hac_records[best_linkage]['calinski_harabasz']]
    ari_vals = [km_rec['ari'], dbscan_best_config['ari'], hac_records[best_linkage]['ari']]
    nmi_vals = [km_rec['nmi'], dbscan_best_config['nmi'], hac_records[best_linkage]['nmi']]

    # ---------------------------------------------------------
    # Plot 10: Internal Validation Metrics Bar Chart
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(algo_names))
    axes[0].bar(x, sil_vals, color=['#2980b9', '#e67e22', '#27ae60'], edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algo_names)
    axes[0].set_ylabel('Silhouette Score (higher is better)', fontweight='bold')
    axes[0].set_title('Internal Validation: Silhouette Score', fontweight='bold')
    for i, v in enumerate(sil_vals):
        axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
    axes[0].grid(True, axis='y', linestyle=':', alpha=0.6)

    db_plot_vals = [v if v is not None else 0 for v in db_vals]
    axes[1].bar(x, db_plot_vals, color=['#2980b9', '#bdc3c7', '#27ae60'], edgecolor='black', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algo_names)
    axes[1].set_ylabel('Davies-Bouldin Index (lower is better)', fontweight='bold')
    axes[1].set_title('Internal Validation: Davies-Bouldin Index', fontweight='bold')
    for i, v in enumerate(db_vals):
        label = f'{v:.3f}' if v is not None else 'N/A'
        axes[1].text(i, db_plot_vals[i] + 0.02, label, ha='center', fontweight='bold')
    axes[1].grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '10_internal_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 10_internal_metrics_comparison.png")

    # ---------------------------------------------------------
    # Plot 11: External Validation Metrics Bar Chart (ARI, NMI)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    ax.bar(x - width/2, ari_vals, width, color='#2980b9', edgecolor='black', label='Adjusted Rand Index (ARI)')
    ax.bar(x + width/2, nmi_vals, width, color='#c0392b', edgecolor='black', label='Normalized Mutual Info (NMI)')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('External Validation vs Ground-Truth Activity Labels', fontweight='bold')
    for i, (a_v, n_v) in enumerate(zip(ari_vals, nmi_vals)):
        ax.text(i - width/2, a_v + 0.01, f'{a_v:.3f}', ha='center', fontsize=9)
        ax.text(i + width/2, n_v + 0.01, f'{n_v:.3f}', ha='center', fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '11_external_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 11_external_metrics_comparison.png")

    return {'algo_names': algo_names, 'sil_vals': sil_vals, 'db_vals': db_vals,
            'ch_vals': ch_vals, 'ari_vals': ari_vals, 'nmi_vals': nmi_vals}


def generate_contingency_plot(kmeans_labels, dbscan_labels, hac_labels, y_full, y_hac_sub, best_k, best_linkage):
    """Plot 12: Cluster-to-Activity Contingency Heatmaps (majority-vote mapped confusion matrices)."""
    activity_order = [ACTIVITY_NAMES[i] for i in range(1, 7)]

    mapped_km, map_km = cluster_to_activity_mapping(kmeans_labels, y_full)
    cm_km = confusion_matrix(y_full, mapped_km, labels=list(range(1, 7)))

    non_noise_mask = dbscan_labels != -1
    mapped_db_partial, map_db = cluster_to_activity_mapping(dbscan_labels[non_noise_mask], y_full[non_noise_mask])
    mapped_db = np.zeros_like(y_full)
    mapped_db[non_noise_mask] = mapped_db_partial
    mapped_db[~non_noise_mask] = 0  # noise -> unmapped bucket
    cm_db = confusion_matrix(y_full[non_noise_mask], mapped_db_partial, labels=list(range(1, 7)))

    mapped_hac, map_hac = cluster_to_activity_mapping(hac_labels, y_hac_sub)
    cm_hac = confusion_matrix(y_hac_sub, mapped_hac, labels=list(range(1, 7)))

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    sns.heatmap(cm_km, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=activity_order, yticklabels=activity_order, cbar=False)
    axes[0].set_title(f'K-Means (k={best_k}) — Mapped Contingency', fontweight='bold')
    axes[0].set_xlabel('Mapped Cluster → Activity')
    axes[0].set_ylabel('True Activity')
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_db, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=activity_order, yticklabels=activity_order, cbar=False)
    axes[1].set_title('DBSCAN — Mapped Contingency (Noise Excluded)', fontweight='bold')
    axes[1].set_xlabel('Mapped Cluster → Activity')
    axes[1].set_ylabel('True Activity')
    axes[1].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_hac, annot=True, fmt='d', cmap='Greens', ax=axes[2],
                xticklabels=activity_order, yticklabels=activity_order, cbar=False)
    axes[2].set_title(f'Hierarchical ({best_linkage.capitalize()}) — Mapped Contingency', fontweight='bold')
    axes[2].set_xlabel('Mapped Cluster → Activity')
    axes[2].set_ylabel('True Activity')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '12_cluster_activity_contingency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 12_cluster_activity_contingency.png")

    return cm_km.tolist(), cm_db.tolist(), cm_hac.tolist()


def main():
    t_start = time.time()
    print("=" * 80)
    print("EXPERIMENT 8: Clustering Human Activity Recognition Data")
    print("K-Means | DBSCAN | Hierarchical Agglomerative Clustering (HAC)")
    print("=" * 80)

    X, X_scaled, y, subjects, feature_names = load_and_preprocess_data()
    pca_full, pca_reduced, X_pca, pca_info = perform_pca_analysis(X_scaled)

    kmeans_records, kmeans_models, best_k = run_kmeans_elbow(X_pca, y)

    dbscan_grid, k_distances, dbscan_best_config, dbscan_labels = run_dbscan_tuning(X_pca, y)

    # Stratified subsample for Hierarchical Clustering (O(n^2) complexity)
    rng = np.random.RandomState(RANDOM_STATE)
    subsample_idx_list = []
    for act_id in range(1, 7):
        idx_act = np.where(y == act_id)[0]
        n_take = int(HAC_SUBSAMPLE_SIZE * len(idx_act) / len(y))
        subsample_idx_list.append(rng.choice(idx_act, size=n_take, replace=False))
    subsample_idx = np.concatenate(subsample_idx_list)
    rng.shuffle(subsample_idx)

    hac_records, best_linkage, Z, X_hac_sub, y_hac_sub = run_hierarchical_clustering(X_pca, y, subsample_idx)
    hac_labels_best = np.array(hac_records[best_linkage]['labels'])

    X_2d = generate_plots(X_scaled, y, subjects, pca_full, pca_reduced, X_pca, pca_info,
                           kmeans_records, kmeans_models, best_k,
                           dbscan_grid, k_distances, dbscan_best_config, dbscan_labels,
                           hac_records, best_linkage, Z, X_hac_sub, y_hac_sub, hac_labels_best)

    tsne_idx = generate_hac_scatter_and_tsne(X_scaled, y, subsample_idx, hac_labels_best, best_linkage,
                                              kmeans_models, best_k, dbscan_labels, dbscan_best_config)

    metrics_summary = generate_metric_and_confusion_plots(
        kmeans_records, best_k, dbscan_best_config, dbscan_labels,
        hac_records, best_linkage, y, y_hac_sub, kmeans_models
    )

    cm_km, cm_db, cm_hac = generate_contingency_plot(
        kmeans_models[best_k].labels_, dbscan_labels, hac_labels_best, y, y_hac_sub, best_k, best_linkage
    )

    # -----------------------------------------------------------------
    # Serialize results to JSON
    # -----------------------------------------------------------------
    results = {
        'dataset_info': {
            'n_samples': int(X.shape[0]), 'n_features': int(X.shape[1]),
            'n_subjects': int(len(np.unique(subjects))), 'n_activities': 6,
            'activity_names': ACTIVITY_NAMES
        },
        'pca_info': pca_info,
        'kmeans': {
            'elbow_table': kmeans_records,
            'best_k': best_k
        },
        'dbscan': {
            'grid_search': dbscan_grid,
            'best_config': dbscan_best_config
        },
        'hierarchical': {
            'linkage_comparison': {m: {k: v for k, v in rec.items() if k != 'labels'}
                                    for m, rec in hac_records.items()},
            'best_linkage': best_linkage,
            'subsample_size': int(len(subsample_idx))
        },
        'metrics_summary': metrics_summary,
        'contingency_matrices': {
            'kmeans': cm_km, 'dbscan': cm_db, 'hierarchical': cm_hac
        },
        'runtime_sec': time.time() - t_start
    }

    results_path = os.path.join(BASE_DIR, 'experiment8_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] Results JSON: {results_path}")
    print(f"\n[Done] Total runtime: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
