import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits, load_iris

# Visualization Configuration
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "../dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 1. Tabular EDA Generator
# ==========================================

def run_tabular_eda(df, target_col, task_type="classification", dataset_name="dataset"):
    print(f"\n=== Running Tabular EDA for {dataset_name.upper()} ===")
    print(f"Data Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target Column: {target_col} | Task: {task_type}")

    # 1. Target Variable Distribution
    plt.figure(figsize=(7, 5))
    if task_type == "classification":
        sns.countplot(data=df, x=target_col, hue=target_col, legend=False, palette="viridis")
        plt.title(f"Class Distribution: {dataset_name.upper()} ({target_col})", fontweight='bold')
    else:
        sns.histplot(data=df, x=target_col, kde=True, color="royalblue", bins=30)
        plt.title(f"Target Distribution: {dataset_name.upper()} ({target_col})", fontweight='bold')
    
    plt.tight_layout()
    target_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_target_dist.png")
    plt.savefig(target_plot_path)
    plt.close()
    print(f"Saved: {target_plot_path}")

    # 2. Correlation Heatmap (numeric columns only)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(f"Feature Correlation Matrix: {dataset_name.upper()}", fontweight='bold')
        plt.tight_layout()
        corr_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_correlation.png")
        plt.savefig(corr_plot_path)
        plt.close()
        print(f"Saved: {corr_plot_path}")

    # 3. Distribution of Numerical Features
    num_cols = [c for c in numeric_df.columns if c != target_col][:6]
    if len(num_cols) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for idx, col in enumerate(num_cols):
            if task_type == "classification" and df[target_col].nunique() <= 10:
                sns.kdeplot(data=df, x=col, hue=target_col, fill=True, ax=axes[idx], palette="Set2")
            else:
                sns.histplot(data=df, x=col, kde=True, ax=axes[idx], color="mediumseagreen")
            axes[idx].set_title(f"Dist of {col}")
        
        for ax in axes[len(num_cols):]:
            ax.set_visible(False)
            
        plt.suptitle(f"Feature Distributions: {dataset_name.upper()}", fontweight='bold')
        plt.tight_layout()
        features_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_features_dist.png")
        plt.savefig(features_plot_path)
        plt.close()
        print(f"Saved: {features_plot_path}")

    # 4. Box Plots for Outlier Analysis
    if len(num_cols) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for idx, col in enumerate(num_cols):
            sns.boxplot(data=df, y=col, ax=axes[idx], color="coral")
            axes[idx].set_title(f"Boxplot of {col}")
        for ax in axes[len(num_cols):]:
            ax.set_visible(False)
        plt.suptitle(f"Feature Boxplots: {dataset_name.upper()}", fontweight='bold')
        plt.tight_layout()
        boxplot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_boxplots.png")
        plt.savefig(boxplot_path)
        plt.close()
        print(f"Saved: {boxplot_path}")


# ==========================================
# 2. Handwritten Character / MNIST EDA Generator
# ==========================================

def run_mnist_digits_eda():
    print("\n=== Running EDA for Handwritten Character / MNIST Digits Dataset ===")
    digits = load_digits()
    X = digits.data
    y = digits.target
    images = digits.images
    
    print(f"Data Shape: {X.shape[0]} samples, {X.shape[1]} features (8x8 pixel grid)")
    print(f"Classes: {np.unique(y)} (Digits 0-9)")
    
    # 1. Target Class Distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(x=y, palette="crest", hue=y, legend=False)
    plt.title("MNIST / Handwritten Digits Class Distribution (0-9)", fontweight='bold')
    plt.xlabel("Digit Class")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mnist_digits_target_dist.png"))
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'mnist_digits_target_dist.png')}")
    
    # 2. Sample Image Grid Plot
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()
    for i in range(10):
        idx = np.where(y == i)[0][0]
        axes[i].imshow(images[idx], cmap=plt.cm.gray_r, interpolation='nearest')
        axes[i].set_title(f"Digit: {i}", fontweight='bold')
        axes[i].axis('off')
    plt.suptitle("Handwritten Character / MNIST Sample Visualizations", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mnist_digits_sample_grid.png"))
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'mnist_digits_sample_grid.png')}")
    
    # 3. Mean Pixel Intensity per Class
    mean_images = [X[y == i].mean(axis=0).reshape(8, 8) for i in range(10)]
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()
    for i in range(10):
        axes[i].imshow(mean_images[i], cmap='hot', interpolation='nearest')
        axes[i].set_title(f"Mean Digit: {i}", fontweight='bold')
        axes[i].axis('off')
    plt.suptitle("MNIST Average Pixel Intensity Heatmaps per Digit Class", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mnist_digits_mean_intensity.png"))
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'mnist_digits_mean_intensity.png')}")


# ==========================================
# 3. Main Execution for All 5 Tasks
# ==========================================

def main():
    print("=== Executing Automated EDA for All 5 Benchmark Tasks ===")
    
    # i.) Loan Amount Prediction
    loan_path = os.path.join(DATASET_DIR, "loan_approval_dataset.csv")
    if os.path.exists(loan_path):
        df_loan = pd.read_csv(loan_path)
        df_loan.columns = df_loan.columns.str.strip()
        target = "loan_status" if "loan_status" in df_loan.columns else df_loan.columns[-1]
        run_tabular_eda(df_loan, target, task_type="classification", dataset_name="loan_amount_prediction")

    # ii.) Handwritten Character Recognition / MNIST Data
    run_mnist_digits_eda()

    # iii.) Classification of Email Spam
    email_path = os.path.join(DATASET_DIR, "email.csv")
    if os.path.exists(email_path):
        df_email = pd.read_csv(email_path)
        df_email.columns = df_email.columns.str.strip()
        target = df_email.columns[-1]
        run_tabular_eda(df_email, target, task_type="classification", dataset_name="email_spam_classification")

    # iv.) Predicting Diabetes
    diabetes_path = os.path.join(DATASET_DIR, "diabetes_prediction_dataset.csv")
    if os.path.exists(diabetes_path):
        df_diab = pd.read_csv(diabetes_path)
        df_diab.columns = df_diab.columns.str.strip()
        target = "diabetes" if "diabetes" in df_diab.columns else df_diab.columns[-1]
        run_tabular_eda(df_diab, target, task_type="classification", dataset_name="predicting_diabetes")

    # v.) Iris Dataset
    iris_path = os.path.join(DATASET_DIR, "Iris.csv")
    if os.path.exists(iris_path):
        df_iris = pd.read_csv(iris_path)
        df_iris.columns = df_iris.columns.str.strip()
        if "Id" in df_iris.columns:
            df_iris = df_iris.drop(columns=["Id"])
        target = "Species" if "Species" in df_iris.columns else df_iris.columns[-1]
        run_tabular_eda(df_iris, target, task_type="classification", dataset_name="iris_dataset")

    print("\n=== Automated EDA Completed for All 5 Tasks! Check 'exp1/output_plots/' ===")


if __name__ == "__main__":
    main()
