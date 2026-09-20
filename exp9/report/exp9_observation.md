# Experiment 9: Observation Sheet

**Experiment Title:** Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning  
**Course:** Machine Learning Algorithms Laboratory (ICS1512)  
**Student Name:** Danusu K  
**Register Number:** 3122247001013  
**Faculty:** Dr. Poreddy Ajay Kumar Reddy  
**Academic Year:** 2025-2026 (Odd Semester) | **Batch:** 2024-2029  

---

## 1. Objective
To implement and comparatively analyze **Model A: Single-Layer Perceptron Learning Algorithm (PLA)** and **Model B: Multilayer Perceptron (MLP)** on the English Handwritten Characters dataset (62 classes: digits 0-9, uppercase A-Z, lowercase a-z). The study implements PLA from scratch (step activation, classical weight-update rule), systematically tunes MLP hyperparameters (activation function, cost function, optimizer, learning rate, hidden-layer architecture, batch size) via staged grid search with cross-validation, and rigorously compares both models on accuracy, precision, recall, F1-score, confusion matrices, ROC curves, and training convergence behavior.

---

## 2. Dataset Summary & Preprocessing
* **Dataset Name:** English Handwritten Characters Dataset
* **Source:** Kaggle / UCI-style handwritten character corpus
* **Total Samples:** 3,410 images, exactly 55 samples per class (perfectly balanced)
* **Classes (62):** Digits '0'-'9' (10), Uppercase 'A'-'Z' (26), Lowercase 'a'-'z' (26)
* **Raw Image Format:** 1200×900 RGB, black ink on white background, single character per image
* **Preprocessing Pipeline:**
  1. Convert to grayscale (single channel).
  2. Resize to **28×28** pixels (bicubic/Lanczos resampling) — a compact resolution sufficient to capture stroke topology while keeping the feature dimension tractable for PLA and MLP.
  3. Flatten to a 784-dimensional feature vector per image.
  4. Normalize pixel intensities to **[0, 1]** by dividing by 255.
* **Label Encoding:** `LabelEncoder` mapping each of the 62 class strings to integer indices 0-61.
* **Train-Test Split:** 80% train (2,728 images, ~44/class) / 20% test (682 images, ~11/class), stratified by class, fixed seed (42).

---

## 3. Model A: PLA Implementation & Results

### Table 1: PLA Training Convergence (Selected Epochs)

| Epoch | Train Accuracy | Test Accuracy | Weight Updates (Misclassifications) |
| :---: | :---: | :---: | :---: |
| 1 | 3.70% | 3.81% | 2,689 |
| 10 | 14.81% | 11.29% | 2,544 |
| 20 | 24.71% | 14.96% | 2,427 |
| 30 | 33.54% | 19.50% | 2,369 |
| 40 | 28.37% | 15.40% | 2,325 |
| 50 | 28.37% | 17.60% | 2,283 |
| **60 (Final)** | **37.72%** | **18.62%** | 2,253 |

> **Observation:** PLA never converges to a stable solution — both the accuracy curve and misclassification count oscillate rather than monotonically decreasing, because the 62-class handwritten-character feature space is **not linearly separable**. Each One-vs-Rest sub-perceptron continues flip-flopping its decision boundary across epochs as it chases misclassified points that can never be perfectly separated by a hyperplane.

### Table 2: PLA Learning-Rate ($\eta$) Sensitivity

| Learning Rate ($\eta$) | Final Test Accuracy (20 epochs) |
| :---: | :---: |
| 0.001 | 14.96% |
| 0.01 | 14.96% |
| 0.1 | 14.96% |
| 1.0 | 14.96% |

> **Observation:** Test accuracy is **exactly identical** across all four learning rates tested. This confirms the theoretical property of the classical perceptron: since weights are zero-initialized and the step-activation decision rule depends only on $\text{sign}(\mathbf{w}\cdot\mathbf{x})$, uniformly rescaling $\eta$ by any positive constant rescales the entire weight trajectory proportionally without changing any sign decision at any step — hence identical classification behavior regardless of $\eta$.

### Table 3: PLA Final Test Metrics

| Metric | Value |
| :---: | :---: |
| Test Accuracy | **18.62%** |
| Precision (macro) | 28.75% |
| Recall (macro) | 18.62% |
| F1-Score (macro) | 16.04% |
| ROC-AUC (micro-avg) | 0.8000 |
| ROC-AUC (macro-avg) | 0.8510 |

---

## 4. Model B: MLP Implementation & Hyperparameter Tuning

### Table 4: Stage 1 — Architecture × Activation Function (3-Fold CV Accuracy)

| Hidden Layer Sizes | ReLU | Tanh | Logistic (Sigmoid) |
| :---: | :---: | :---: | :---: |
| (64,) | 1.54% (dead) | 16.97% | 39.04% |
| **(128,)** | 1.54% (dead) | 31.01% | **40.18% (Best)** |
| (128, 64) | 19.24% (unstable, $\sigma$=10.1%) | 26.10% | 39.19% |
| (256, 128, 64) | 1.54% (dead) | 16.42% | 31.27% |

> **Observation:** ReLU catastrophically fails on this task (accuracy $\approx$ 1.5%, essentially random-chance for 62 classes) across almost every architecture — a classic **"dying ReLU"** failure mode: with normalized [0,1] pixel inputs (many exactly 0 over the white background), small default weight initialization, and only 150 training iterations, a large fraction of ReLU units saturate at zero early and never recover a non-zero gradient. **Logistic (sigmoid)** activation was the clear winner at every architecture size, and the shallowest single-hidden-layer network **(128,)** outperformed all deeper architectures.

### Table 5: Stage 2 — Optimizer × Learning Rate (Best Architecture: (128,), Logistic)

| Optimizer | Learning Rate | 3-Fold CV Accuracy |
| :---: | :---: | :---: |
| SGD | 0.0001 | 3.48% |
| Adam | 0.0001 | 28.12% |
| SGD | 0.001 | 16.64% |
| **Adam** | **0.001** | **40.18% (Best)** |
| SGD | 0.01 | 40.14% |
| Adam | 0.01 | 1.58% (diverged) |
| SGD | 0.1 | 22.98% |
| Adam | 0.1 | 1.58% (diverged) |

> **Observation:** **Adam at lr=0.001** and **SGD at lr=0.01** achieve nearly identical peak accuracy (~40%), but through opposite sensitivity profiles: Adam's adaptive per-parameter step sizes let it converge well at a *small* learning rate but **diverge catastrophically** at lr $\ge$ 0.01 (collapsing to near-chance accuracy), whereas plain SGD needs a comparatively *larger* learning rate to make meaningful progress within the fixed 150-iteration budget, but degrades more gracefully at high lr (22.98% at lr=0.1, rather than collapsing entirely). This is a direct, empirical illustration of Adam's optimizer-state sensitivity to learning-rate scale.

### Table 6: Stage 3 — Batch Size (Best Config So Far: (128,), Logistic, Adam, lr=0.001)

| Batch Size | 3-Fold CV Accuracy |
| :---: | :---: |
| 16 | 37.50% |
| 32 | 40.18% |
| **64** | **40.87% (Best)** |
| 128 | 39.63% |

### Table 7: Final Chosen MLP Configuration

| Hyperparameter | Selected Value | Justification |
| :--- | :--- | :--- |
| Hidden Layer Architecture | **(128,)** — single hidden layer, 128 neurons | Outperformed all deeper architectures (Table 4); more parameters overfit the small (~44 samples/class) training set |
| Activation Function | **Logistic (Sigmoid)** | Only activation avoiding the dying-unit failure mode on this normalized, largely-zero-background input |
| Cost Function | Cross-Entropy (multi-class log-loss) | Standard for softmax-output multiclass classification |
| Optimizer | **Adam** | Best CV accuracy at its optimal learning rate; adaptive moment estimation handles the sparse, high-dimensional (784-d) pixel input well |
| Learning Rate | **0.001** | Best Adam configuration (Table 5); avoided divergence seen at lr $\ge$ 0.01 |
| Batch Size | **64** | Best 3-fold CV accuracy (Table 6); balances gradient-noise regularization against convergence speed |

### Table 8: MLP Final Test Metrics

| Metric | Value |
| :---: | :---: |
| Test Accuracy | **38.56%** |
| Precision (macro) | 42.02% |
| Recall (macro) | 38.56% |
| F1-Score (macro) | 37.93% |
| ROC-AUC (micro-avg) | 0.9069 |
| ROC-AUC (macro-avg) | 0.9037 |

---

## 5. A/B Comparison: PLA vs Tuned MLP

| Metric | PLA (Model A) | MLP — Tuned (Model B) | Improvement |
| :--- | :---: | :---: | :---: |
| Test Accuracy | 18.62% | **38.56%** | **+19.94 pp (2.07x)** |
| Precision (macro) | 28.75% | **42.02%** | +13.27 pp |
| Recall (macro) | 18.62% | **38.56%** | +19.94 pp |
| F1-Score (macro) | 16.04% | **37.93%** | +21.89 pp (2.36x) |
| ROC-AUC (micro-avg) | 0.8000 | **0.9069** | +0.1069 |
| ROC-AUC (macro-avg) | 0.8510 | **0.9037** | +0.0527 |
| Training Convergence | Oscillates, never stabilizes | Smooth, monotonic improvement | MLP converges; PLA does not |

---

## 6. Answers to Observation Questions

### 1. Why does PLA underperform compared to MLP?
PLA is a strictly **linear** classifier — each of its 62 One-vs-Rest sub-perceptrons can only carve the 784-dimensional pixel space with a single hyperplane. Handwritten character classes are **not linearly separable**: many character pairs (e.g., '0'/'O', 'l'/'I'/'1', 'S'/'s') differ by subtle, non-linear stroke variations that no single hyperplane can cleanly separate from the other 61 classes simultaneously. This is directly visible in the training convergence curve (Table 1, Figure 3): rather than settling, PLA's misclassification count and accuracy oscillate indefinitely across all 60 epochs, and its confusion matrix (Figure 8) shows predictions collapsing onto a handful of "attractor" columns rather than a clean diagonal. MLP, by contrast, learns a **non-linear** decision boundary through its hidden layer and sigmoid activation, backpropagation, and an adaptive optimizer — enabling it to carve much more flexible, curved decision regions and achieve a clean diagonal-dominant confusion matrix at roughly double PLA's accuracy.

### 2. Which hyperparameters had the most impact on MLP performance?
**Activation function** had by far the largest impact: switching from ReLU to Logistic changed CV accuracy from ~1.5% (chance level) to ~40% at the same architecture — a difference far larger than any other single hyperparameter change. **Learning rate** was the second most impactful, specifically its interaction with optimizer choice: Adam collapsed from 40.18% (lr=0.001) to 1.58% (lr=0.01) with just a 10x increase. **Number of hidden layers/architecture depth** had a clear but smaller effect: deeper networks (2-3 hidden layers) consistently underperformed the single-layer (128,) network by 5-10 percentage points, since the small per-class sample count (~44 training images/class) cannot support the additional parameters of deeper networks. **Batch size** had the smallest impact, varying CV accuracy by only about 3.4 percentage points across the tested range (16-128).

### 3. Did optimizer choice (SGD vs Adam) affect convergence?
Yes, substantially, though the picture is more nuanced than "Adam always wins." At their respective *optimal* learning rates, SGD (lr=0.01, 40.14%) and Adam (lr=0.001, 40.18%) achieved nearly identical peak CV accuracy — but Adam required a learning rate 10x smaller to reach that peak and **diverged catastrophically** (collapsing to ~1.6%, chance level) at learning rates of 0.01 or higher, because its adaptive per-parameter scaling amplifies effective step sizes once gradients shrink near convergence. SGD, lacking this adaptive amplification, degraded much more gracefully as the learning rate increased (22.98% even at lr=0.1). This demonstrates that Adam's fast, low-learning-rate convergence advantage comes with a narrower stable learning-rate range than plain SGD.

### 4. Did adding more hidden layers always improve results? Why or why not?
**No — deeper was consistently worse in this experiment.** With Logistic activation, CV accuracy fell monotonically as depth increased: (128,) → 40.18%, (128,64) → 39.19%, (256,128,64) → 31.27%. The same pattern held for Tanh and (where not dead) ReLU. This is because the training set is very small relative to the 62-way classification problem (~44 images per class after the train/test split). Deeper, wider networks add many more trainable parameters without a corresponding increase in training data, so they overfit the training folds and generalize worse — the classic **bias-variance trade-off** tipping toward variance as capacity grows beyond what the data can support. A single, moderately-sized hidden layer was the sweet spot for this dataset's scale.

### 5. Did MLP show overfitting? How could it be mitigated?
Yes, to a degree. The Stage 1 tuning table shows deeper architectures achieving comparatively strong training-fold fits (not shown directly, but declining CV — not just training — accuracy with depth is itself an overfitting signature) despite worse cross-validated performance. Additionally, the final chosen MLP's train-vs-test convergence curve (Figure 7) shows test accuracy climbing steadily but with a narrowing (though not reversing) gap as epochs increase, consistent with a model that is approaching, but not yet strongly, overfitting given the already-small architecture. Overfitting could be further mitigated by: (a) L2 weight regularization (`alpha` in `MLPClassifier`) to penalize large weights, (b) `early_stopping=True` with a held-out validation split to halt training at the point of best generalization, (c) data augmentation (small rotations, translations, elastic distortions of the character images) to synthetically enlarge the ~44-sample-per-class training set, and (d) dropout-style regularization (available in deep-learning frameworks beyond scikit-learn's MLPClassifier) if migrating to a framework like PyTorch/TensorFlow.

---

## 7. Conclusion
This experiment provides a clear, quantitatively grounded A/B comparison between a linear (PLA) and non-linear (MLP) classifier on a genuinely hard, small-sample, 62-class handwritten-character recognition task. PLA's step-activation, hyperplane-only decision boundaries cannot separate the character classes and its training never converges (accuracy oscillating around 15-20% for the entire 60-epoch run), while systematic staged hyperparameter tuning — first architecture/activation, then optimizer/learning-rate, then batch size — allowed the MLP to more than double PLA's test accuracy (18.62% → 38.56%) and macro-F1 score (16.04% → 37.93%), with a substantially higher and smoothly-converging ROC-AUC (0.80 → 0.91 micro-average). The tuning process itself surfaced two important, non-obvious findings: ReLU activation catastrophically fails ("dies") on this small, sparse, [0,1]-normalized dataset while Logistic activation thrives, and deeper architectures consistently underperform a single hidden layer given the limited ~44-samples-per-class training budget — both valuable lessons in matching model capacity and activation choice to dataset scale and structure.
