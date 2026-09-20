# Experiment 8: Observation Sheet

**Experiment Title:** Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering  
**Course:** Machine Learning Algorithms Laboratory (ICS1512)  
**Student Name:** Danusu K  
**Register Number:** 3122247001013  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  
**Academic Year:** 2025-2026 (Odd Semester) | **Batch:** 2024-2029  

---

## 1. Objective
To implement and comparatively analyze three unsupervised clustering paradigms — **K-Means** (centroid-based partitioning), **DBSCAN** (density-based spatial clustering with noise detection), and **Hierarchical Agglomerative Clustering (HAC)** — on the UCI **Human Activity Recognition Using Smartphones (HAR)** dataset. The study selects the optimal number of clusters for K-Means via the Elbow Method and Silhouette Score, tunes DBSCAN's $\epsilon$ and minPts via a k-distance graph and grid search, compares linkage criteria for HAC, and evaluates all three models using both internal (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external (Adjusted Rand Index, Normalized Mutual Information) validation metrics against the ground-truth activity labels.

---

## 2. Dataset Summary & Preprocessing
* **Dataset Name:** Human Activity Recognition Using Smartphones (UCI HAR Dataset)
* **Source:** UCI Machine Learning Repository
* **Total Samples ($N$):** 10,299 sensor windows (7,352 train + 2,947 test, merged for unsupervised analysis)
* **Volunteers:** 30 subjects (aged 19-48), each performing 6 activities with a waist-mounted smartphone
* **Sensors:** 3-axis accelerometer + gyroscope @ 50Hz, windowed into 2.56s segments (128 readings/window, 50% overlap)
* **Number of Features ($D$):** 561 time- and frequency-domain features per window (mean, std, entropy, correlation, FFT-based, etc.)
* **Target Classes:** 6 activities — WALKING (1,722), WALKING_UPSTAIRS (1,544), WALKING_DOWNSTAIRS (1,406), SITTING (1,777), STANDING (1,906), LAYING (1,944)
* **Missing Values:** 0 (pre-cleaned UCI release)
* **Preprocessing:** `StandardScaler` fitted on the full merged corpus ($\mu=0,\sigma=1$); PCA fitted on the standardized data and truncated to the number of components explaining $\ge 90\%$ cumulative variance, used as the working feature space for all three clustering algorithms (mitigates the curse of dimensionality for Euclidean-distance-based methods, and makes DBSCAN/HAC computationally tractable). PCA/t-SNE to 2D is used **only** for visualization, as specified.
* **Design Note — Hierarchical Clustering Subsample:** Agglomerative clustering has $O(n^2)$ memory/time complexity. A stratified random subsample of $n=2{,}497$ points (proportional to class balance) was used for linkage comparison, dendrogram plotting, and HAC metric computation; K-Means and DBSCAN were run on the full $N=10{,}299$ samples.

---

## 3. Observation Tables

### Table 1: PCA Variance Explained & Scree Analysis (First 15 of 561 Components)
*(Eigenvalue Decomposition on Standardized 561-Dimensional Feature Space)*

| Principal Component | Eigenvalue ($\lambda_i$) | Explained Variance (%) | Cumulative Variance (%) |
| :---: | :---: | :---: | :---: |
| **PC1** | 284.67 | 50.74% | 50.74% |
| **PC2** | 35.01 | 6.24% | 56.98% |
| **PC3** | 15.11 | 2.69% | 59.67% |
| **PC4** | 13.76 | 2.45% | 62.12% |
| **PC5** | 10.60 | 1.89% | 64.01% |
| **PC6** | 9.15 | 1.63% | 65.64% |
| **PC7** | 7.94 | 1.41% | 67.06% |
| **PC8** | 6.82 | 1.22% | 68.27% |
| **PC9** | 5.53 | 0.99% | 69.26% |
| **PC10** | 5.33 | 0.95% | 70.21% |
| PC11-PC64 | decreasing | — | 70.21% → 89.9% |
| **PC65** | — | — | **90.05% (Chosen Cutoff — 90% Target)** |
| PC66-PC104 | decreasing | — | 90.05% → 95.02% |
| PC105-PC182 | decreasing | — | 95.02% → 99.01% |

> **Design Choice Justification:**  
> A variance target of **90.0%** was selected for the clustering feature space. The first 65 principal components (of 561) capture **90.05%** of cumulative variance — an **88.4% reduction in dimensionality** — while retaining enough structure for meaningful Euclidean-distance clustering. PC1 alone explains **50.74%** of total variance, dominated by signal-magnitude features that separate dynamic from static activities (confirmed visually in the PCA 2D scatter plots).

---

### Table 2: K-Means Elbow Method & Silhouette Results

| Number of Clusters (k) | WCSS (Inertia) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | ARI (vs. Ground Truth) | NMI (vs. Ground Truth) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | **2,697,926.8** | **0.4366 (Best)** | 0.9611 | **9560.0** | 0.3296 | 0.5455 |
| 3 | 2,346,425.1 | 0.3519 | 1.5974 | 6266.7 | 0.3318 | 0.5184 |
| 4 | 2,207,133.3 | 0.1823 | 2.0573 | 4657.6 | 0.2985 | 0.4693 |
| 5 | 2,081,104.6 | 0.1597 | 2.1379 | 3860.2 | 0.2912 | 0.4604 |
| 6 (= true activity count) | 2,003,581.4 | 0.1400 | 2.0675 | 3287.0 | **0.4201** | **0.5597** |
| 7 | 1,947,312.3 | 0.1184 | 2.2854 | 2867.6 | 0.4155 | 0.5456 |
| 8 | 1,892,509.0 | 0.1017 | 2.2515 | 2571.5 | **0.4307 (Best ARI)** | 0.5542 |

> **Selected k:** The Elbow (WCSS) curve decays smoothly with no sharp "elbow," but the Silhouette Score peaks unambiguously at **k=2** and decreases monotonically thereafter — indicating the strongest internally-valid structure is the coarse **dynamic-vs-static** activity split (visually confirmed along PC1). Note the interesting tension: **k=6** (matching the true number of activities) is *not* the internally optimal k, but achieves the best external agreement (ARI/NMI) among low-k candidates, since it starts to resolve individual activities within the dynamic and static super-clusters.

---

### Table 3: DBSCAN k-Distance Guided (eps, minPts) Grid Search (Selected Rows)

| $\epsilon$ (10-NN percentile) | minPts | Clusters Found | Noise (%) | Silhouette | ARI | NMI |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 13.68 (P80) | 5 | 11 | 9.0% | -0.0137 | 0.0212 | 0.0806 |
| 13.68 (P80) | 20 | 3 | 15.0% | 0.2925 | 0.0458 | 0.1180 |
| 14.50 (P85) | 15 | 2 | 9.1% | 0.3551 | 0.0206 | 0.0720 |
| 15.13 (P88) | 5 | 6 | 5.4% | 0.0411 | 0.0105 | 0.0554 |
| 15.66 (P90) | 10 | 3 | 5.3% | 0.3122 | 0.0102 | 0.0524 |
| 16.35 (P92) | 5 | 2 | 3.3% | 0.3698 | 0.0046 | 0.0259 |
| **18.04 (P95)** | **5** | **2** | **2.2%** | **0.4091 (Best)** | 0.0027 | 0.0191 |
| 18.04 (P95) | 10-20 | 1 | 2.6-3.0% | N/A (single cluster) | 0.0035-0.0041 | 0.0220-0.0243 |

> **Selected Configuration:** $\epsilon=18.04$, minPts=5 (highest Silhouette Score among configurations with $\ge 2$ clusters and noise ratio $< 50\%$). Even at its best internal-validity configuration, DBSCAN converges to the same coarse **2-cluster (dynamic vs. static)** structure as K-Means at k=2 — but with a dramatically lower external agreement (ARI = 0.0027 vs. K-Means' 0.3296), because minor density fluctuations near the cluster boundary get relabeled as noise rather than assigned to a meaningful third group. Lower $\epsilon$ values fragment the dense static-activity region into many spurious micro-clusters (Table 3, row 1: 11 clusters, negative Silhouette).

---

### Table 4: Hierarchical Agglomerative Clustering — Linkage Comparison
*(Subsample $n=2{,}497$, $k=6$ clusters, evaluated on PCA-reduced 65D space)*

| Linkage Criterion | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | ARI | NMI | Structural Character |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Single** | **0.5539 (Highest raw)** | 0.2836 | 12.9 | 0.0003 | 0.0044 | Degenerate chaining — one dominant cluster absorbs >99% of points |
| Complete | 0.3636 | 1.5831 | 630.8 | 0.3253 | 0.5298 | Moderately balanced, activity-aligned |
| Average | 0.3450 | 1.1470 | 57.8 | 0.0031 | 0.0346 | Degenerate chaining (similar to Single) |
| **Ward (Primary — per spec)** | 0.1585 | 2.2087 | **776.6 (Highest)** | **0.3048** | **0.4755** | Most balanced, minimum-variance partitions |

> **Key Finding:** Raw Silhouette Score is **misleading** for linkage selection here — Single and Average linkage score deceptively high because a near-degenerate solution (one giant cluster + a handful of outlier singletons) is trivially "compact." Ward's linkage, mandated by the experiment specification, produces the most structurally meaningful and activity-aligned partition (highest Calinski-Harabasz Index, second-highest ARI/NMI among all linkages), consistent with its minimum-variance merging objective.

---

### Table 5: Cross-Algorithm Internal & External Validation Summary

| Algorithm | Configuration | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means** | k=2 (Silhouette-optimal) | **0.4366 (Best)** | **0.9611 (Best)** | **9560.0 (Best)** | 0.3296 | **0.5455 (Best)** |
| **DBSCAN** | $\epsilon$=18.04, minPts=5 | 0.4091 | N/A (density-based) | N/A | 0.0027 (Worst) | 0.0191 (Worst) |
| **Hierarchical (Ward)** | 6 clusters, subsample | 0.1585 (Worst) | 2.2087 (Worst) | 776.6 | 0.3048 | 0.4755 |

---

## 4. Answers to Observation Questions

### 1. Which algorithm produced the most meaningful clusters? Why?
**K-Means (k=2)** produced the most internally *and* externally valid clustering overall — highest Silhouette (0.4366), lowest Davies-Bouldin (0.9611), highest Calinski-Harabasz (9560.0), and the best external agreement (ARI=0.3296, NMI=0.5455) among the three algorithms' primary configurations. Its centroid-based partitioning naturally captures the dominant, near-linearly-separable **dynamic-vs-static** split visible along PC1 (50.7% variance). **Hierarchical Clustering with Ward linkage** is the best choice when finer-grained activity resolution is needed (6 clusters instead of 2), trading some internal compactness (Silhouette=0.1585) for a partition that still meaningfully separates most activity pairs, particularly the walking-related activities and the SITTING/STANDING/LAYING group boundary (visible in the contingency heatmap). **DBSCAN** was the least meaningful of the three: despite achieving a respectable Silhouette Score (0.4091) at its best configuration, this is an artifact of finding the *same* coarse 2-cluster split as K-Means but assigning ambiguous boundary points to noise rather than to a cluster — its external agreement (ARI=0.0027) is essentially at chance level.

### 2. How sensitive was K-Means to the choice of k?
Highly sensitive, and the sensitivity reveals a genuine methodological tension. Both WCSS and Silhouette Score decrease monotonically as k increases from 2 to 8, with **no elbow** in WCSS and a **sharp Silhouette peak at k=2** — implying the "textbook" unsupervised answer is a 2-cluster solution corresponding to dynamic vs. static activity. However, external validation tells a different story: ARI/NMI against the true 6-activity labels *improve* from k=2 to k=6-8 (ARI: 0.330 → 0.420-0.431), because larger k values progressively resolve the individual activities nested within each dynamic/static super-cluster (e.g., separating WALKING from WALKING_UPSTAIRS, or SITTING from STANDING). This demonstrates that **internal cluster-validity metrics and external label-agreement metrics can disagree substantially**, and the "right" k depends on whether the goal is compact, well-separated groups (favor k=2) or recovering the known activity taxonomy (favor k=6+).

### 3. Did DBSCAN detect noise or small clusters effectively?
Partially. DBSCAN's k-distance-guided $\epsilon$ selection successfully identified a low, stable noise fraction (2.2% at the best configuration, rising to 15% only at aggressively small $\epsilon$), showing the noise mechanism itself works as designed — genuinely sparse/boundary points between the dynamic and static activity manifolds were correctly flagged rather than forced into a cluster. However, DBSCAN was **not effective at detecting small, meaningful activity-level clusters**: at low $\epsilon$ it fragmented the dense static-activity region into many spurious micro-clusters (11 clusters at $\epsilon$=13.68, with a *negative* Silhouette Score, indicating overlapping/incorrect assignments), while at higher $\epsilon$ it collapsed everything back down to the same trivial 2-cluster (or even 1-cluster) split as K-Means. This is a direct consequence of the HAR feature manifold being a continuum of varying local density (SITTING/STANDING/LAYING densely overlap) rather than well-separated, uniform-density blobs — the setting where DBSCAN is theoretically strongest.

### 4. How does linkage choice (single/complete/ward) affect hierarchical clustering?
Dramatically. **Single linkage** (nearest-point distance) is highly susceptible to *chaining*: a sequence of closely-spaced points can bridge otherwise distinct activity groups, resulting in one dominant cluster absorbing nearly all points and several near-empty singleton clusters. **Average linkage** exhibited the same chaining pathology. Both scored deceptively high raw Silhouette (0.55 and 0.35 respectively) purely because a near-single-cluster solution is trivially "compact" by that metric, yet their ARI/NMI against ground truth were essentially zero (0.0003 and 0.0031). **Complete linkage** (farthest-point distance) and **Ward's linkage** (minimum increase in within-cluster variance) both resisted chaining and produced balanced, activity-aligned partitions — Ward achieved the highest Calinski-Harabasz Index (776.6) and the best-balanced trade-off across all metrics, justifying its use as the assignment's mandated linkage criterion for HAC.

### 5. Which internal metric best matched your visual intuition of cluster quality?
The **Calinski-Harabasz Index** best matched visual/PCA-scatter intuition of "good" clustering in this experiment. It correctly rewarded Ward linkage (776.6, highest among HAC linkages) and K-Means k=2 (9560.0) — both of which visibly correspond to compact, well-separated groups in the PCA 2D scatter plots — while correctly *penalizing* neither the fragmented DBSCAN nor rewarding the degenerate single/average-linkage chains as strongly as raw Silhouette did. **Silhouette Score**, by contrast, was actively misleading for Hierarchical Clustering (rewarding degenerate chaining solutions) and required cross-checking against Davies-Bouldin and the external ARI/NMI metrics before drawing conclusions — reinforcing that internal metrics should always be triangulated with several complementary measures and, where available, external ground truth, rather than trusted individually.

---

## 5. Conclusion
This experiment demonstrated that clustering algorithm choice, hyperparameter selection, and validation metric interpretation are deeply interconnected on real-world sensor data. K-Means with a low k (2) delivers the most internally coherent partition (dynamic vs. static activity), while a higher k (6-8) better recovers the true activity taxonomy at the cost of internal compactness. DBSCAN's density-based assumptions are poorly matched to the continuous, overlapping HAR feature manifold, and its apparently strong Silhouette Score at the best configuration masks near-chance-level agreement with ground truth. Hierarchical Clustering is highly sensitive to linkage criterion, with Ward's minimum-variance objective clearly outperforming Single/Average linkage's chaining-prone solutions. Overall, no single internal metric was sufficient in isolation — Silhouette, Davies-Bouldin, and Calinski-Harabasz needed to be interpreted jointly, and cross-referenced with external ARI/NMI, to reach a reliable conclusion about which clustering configuration was genuinely meaningful.
