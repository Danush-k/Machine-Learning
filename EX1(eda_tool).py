import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# User Configuration
# ==========================================
# Specify the path to your CSV file, image directory, or .npz numpy archive here:
DATASET_PATH = "/Users/danush/Desktop/SEM-5/diabetes_prediction_dataset.csv"

# Optional: Name of target column (defaults to the last column if left None)
TARGET_COLUMN = None

# Visualization Configuration
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 1. EDA Functions for Different Data Types
# ==========================================

def run_tabular_eda(df, target_col, task_type="classification", dataset_name="dataset"):
    """
    Performs EDA on tabular datasets (Classification or Regression).
    """
    print(f"\n=== Running Tabular EDA for {dataset_name.upper()} ===")
    print(f"Data Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n--- Columns and Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    print("\n--- Summary Statistics ---")
    print(df.describe().T)

    # 1. Target Variable Distribution
    plt.figure()
    if task_type == "classification":
        sns.countplot(data=df, x=target_col, hue=target_col, legend=False, palette="viridis")
        plt.title(f"Class Distribution: {target_col}")
    else:
        sns.histplot(data=df, x=target_col, kde=True, color="blue")
        plt.title(f"Target Distribution: {target_col}")
    
    plt.tight_layout()
    target_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_target_dist.png")
    plt.savefig(target_plot_path)
    plt.close()
    print(f"Saved: {target_plot_path}")

    # 2. Correlation Heatmap (numeric columns only)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        corr_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_correlation.png")
        plt.savefig(corr_plot_path)
        plt.close()
        print(f"Saved: {corr_plot_path}")

    # 3. Distribution of Numerical Features
    num_cols = [c for c in numeric_df.columns if c != target_col]
    if len(num_cols) > 0:
        n_features = len(num_cols)
        cols = 3
        rows = (n_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, col in enumerate(num_cols):
            # If classification, show distributions split by class label (maximum of 10 classes)
            if task_type == "classification" and df[target_col].nunique() <= 10:
                sns.kdeplot(data=df, x=col, hue=target_col, fill=True, ax=axes[idx], palette="Set2")
            else:
                sns.histplot(data=df, x=col, kde=True, ax=axes[idx], color="green")
            axes[idx].set_title(f"Distribution of {col}")
        
        for ax in axes[n_features:]:
            ax.set_visible(False)
            
        plt.tight_layout()
        features_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_features_dist.png")
        plt.savefig(features_plot_path)
        plt.close()
        print(f"Saved: {features_plot_path}")

    # 4. Box Plots for Numerical Features (excluding target column)
    if len(num_cols) > 0:
        fig, axes = plt.subplots(1, len(num_cols), figsize=(3 * len(num_cols), 5))
        axes = axes.flatten() if len(num_cols) > 1 else [axes]
        for ax, col in zip(axes, num_cols):
            sns.boxplot(data=df, y=col, ax=ax)
            ax.set_title(f"{col} Box Plot")
        plt.tight_layout()
        box_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_boxplots.png")
        plt.savefig(box_plot_path)
        plt.close()
        print(f"Saved: {box_plot_path}")

    # 5. Scatter Plots between pairs of numerical features
    if len(num_cols) > 1:
        # Sample to 2000 points to speed up plotting and improve clarity
        sample_df = df.sample(min(len(df), 2000), random_state=42)
        hue_val = target_col if task_type == "classification" else None
        sns.pairplot(sample_df, vars=num_cols, hue=hue_val, palette="husl")
        scatter_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_scatterplots.png")
        plt.savefig(scatter_plot_path)
        plt.close()
        print(f"Saved: {scatter_plot_path}")



def run_image_eda(images, labels, image_shape, dataset_name="dataset"):
    """
    Performs EDA on image datasets.
    """
    print(f"\n=== Running Image EDA for {dataset_name.upper()} ===")
    print(f"Dataset Size: {len(images)} images")
    print(f"Image Resolution: {image_shape[0]}x{image_shape[1]} pixels")
    print(f"Pixel Value Range: Min={images.min():.2f}, Max={images.max():.2f}")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n--- Label Distribution ---")
    for lbl, count in zip(unique_labels, counts):
        print(f"Class '{lbl}': {count} samples")

    plt.figure()
    sns.barplot(x=unique_labels, y=counts, hue=unique_labels, legend=False, palette="plasma")
    plt.title(f"Class Label Distribution: {dataset_name}")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    label_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_label_dist.png")
    plt.savefig(label_plot_path)
    plt.close()
    print(f"Saved: {label_plot_path}")

    # Visualize up to 12 sample images
    num_samples = min(len(images), 12)
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2.7))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    indices = np.random.choice(len(images), size=num_samples, replace=False)
    for i, idx in enumerate(indices):
        img = images[idx].reshape(image_shape)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {labels[idx]}")
        axes[i].axis("off")
        
    for ax in axes[num_samples:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    sample_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_samples.png")
    plt.savefig(sample_plot_path)
    plt.close()
    print(f"Saved: {sample_plot_path}")


def run_text_eda(df, text_col, label_col, dataset_name="dataset"):
    """
    Performs EDA on text classification datasets.
    """
    print(f"\n=== Running Text EDA for {dataset_name.upper()} ===")
    print(f"Total Documents: {len(df)}")
    print(f"Classes: {df[label_col].value_counts().to_dict()}")

    df = df.copy()
    df["char_len"] = df[text_col].astype(str).apply(len)
    df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))

    print("\n--- Text Statistics ---")
    print(df[["char_len", "word_count"]].describe().T)

    # 1. Class Distribution Plot
    plt.figure()
    sns.countplot(data=df, x=label_col, hue=label_col, legend=False, palette="viridis")
    plt.title(f"Class Distribution: {label_col}")
    plt.tight_layout()
    class_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_class_dist.png")
    plt.savefig(class_plot_path)
    plt.close()
    print(f"Saved: {class_plot_path}")

    # 2. Character Length Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="char_len", hue=label_col, kde=True, bins=30, multiple="dodge", palette="cool")
    plt.title("Character Length Distribution by Class")
    plt.xlabel("Message Character Length")
    plt.tight_layout()
    length_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_lengths_dist.png")
    plt.savefig(length_plot_path)
    plt.close()
    print(f"Saved: {length_plot_path}")

    # Standard English stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "to", "for", "in", "of", "on", "at", "by", "from", "is", "are", "was", "with", "this", "that", "you", "your", "my", "i", "we", "us"}
    
    classes = df[label_col].unique()
    fig, axes = plt.subplots(1, len(classes), figsize=(12, 5))
    if len(classes) == 1:
        axes = [axes]

    for idx, cls in enumerate(classes):
        texts = df[df[label_col] == cls][text_col].astype(str).str.lower()
        words = []
        for text in texts:
            cleaned = "".join([char if char.isalnum() or char.isspace() else "" for char in text])
            words.extend(cleaned.split())
            
        filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
        word_series = pd.Series(filtered_words)
        top_words = word_series.value_counts().head(10)
        
        sns.barplot(x=top_words.values, y=top_words.index, ax=axes[idx], hue=top_words.index, legend=False, palette="viridis")
        axes[idx].set_title(f"Top 10 Words in '{cls}'")
        axes[idx].set_xlabel("Frequency")

    plt.tight_layout()
    words_plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_top_words.png")
    plt.savefig(words_plot_path)
    plt.close()
    print(f"Saved: {words_plot_path}")


# ==========================================
# 2. Automatic EDA Router
# ==========================================

def run_automatic_eda(file_path, target_col=None):
    """
    Auto-detects the dataset modality (Image directory, NPZ archive, or CSV)
    and executes the appropriate EDA flow.
    """
    if not os.path.exists(file_path):
        print(f"Error: Dataset path '{file_path}' does not exist.")
        return

    # A. Check if the path is a directory (assume folder of subclassified images)
    if os.path.isdir(file_path):
        print(f"\n--- Detected Image Dataset Directory: {file_path} ---")
        images = []
        labels = []
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
        
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.lower().endswith(valid_exts):
                    img_path = os.path.join(root, file)
                    try:
                        img = plt.imread(img_path)
                        images.append(img)
                        labels.append(os.path.basename(root))
                    except Exception:
                        continue
                        
        if not images:
            print("Error: No valid images found in the directory structure.")
            return
            
        images = np.array(images)
        labels = np.array(labels)
        image_shape = images[0].shape[:2]
        
        run_image_eda(images, labels, image_shape, dataset_name=os.path.basename(file_path))
        return

    # B. Check if it's a Numpy archive (.npz or .npy)
    if file_path.endswith(".npz") or file_path.endswith(".npy"):
        print(f"\n--- Detected Numpy Array Archive: {file_path} ---")
        try:
            data = np.load(file_path)
            keys = list(data.keys())
            # Find the arrays representing images (x) and labels (y)
            images_key = next((k for k in keys if "image" in k.lower() or "x" in k.lower()), keys[0])
            labels_key = next((k for k in keys if "label" in k.lower() or "y" in k.lower()), keys[1] if len(keys) > 1 else keys[0])
            
            images = data[images_key]
            labels = data[labels_key]
            image_shape = images[0].shape[:2]
            
            run_image_eda(images, labels, image_shape, dataset_name=os.path.splitext(os.path.basename(file_path))[0])
        except Exception as e:
            print(f"Error loading numpy archive: {e}")
        return

    # C. Check if it's a CSV spreadsheet
    if file_path.endswith(".csv"):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading CSV dataset: {e}")
            return
            
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Clean leading/trailing spaces from column headers
        df.columns = df.columns.str.strip()
        
        # Resolve target column
        target = target_col
        if not target:
            target = df.columns[-1]
            print(f"No TARGET_COLUMN configured. Defaulting to last column: '{target}'")
            
        if target not in df.columns:
            print(f"Error: Target column '{target}' not found in dataset. Columns: {list(df.columns)}")
            return
            
        # Check if the dataset contains text columns (average cell text length > 20 characters)
        text_col = None
        for col in df.columns:
            if col != target and df[col].dtype == object:
                avg_len = df[col].dropna().astype(str).apply(len).mean()
                if avg_len > 20:
                    text_col = col
                    break
                    
        if text_col:
            print("Auto-detected dataset type: Text Classification")
            print(f"Using text feature column: '{text_col}'")
            run_text_eda(df, text_col, target, dataset_name=dataset_name)
            return

        # Check if the target variable is continuous (numeric with >10 distinct values)
        target_series = df[target].dropna()
        n_unique = target_series.nunique()
        
        if pd.api.types.is_numeric_dtype(target_series) and n_unique > 10:
            print("Auto-detected dataset type: Tabular Regression")
            run_tabular_eda(df, target, task_type="regression", dataset_name=dataset_name)
        else:
            print("Auto-detected dataset type: Tabular Classification")
            run_tabular_eda(df, target, task_type="classification", dataset_name=dataset_name)
        return

    print("Error: Unsupported file format. Please provide a CSV file, an image directory, or an .npz archive.")


# ==========================================
# 3. Main execution
# ==========================================

def main():
    run_automatic_eda(DATASET_PATH, TARGET_COLUMN)
    print("\nEDA completed successfully! Check the 'output_plots' directory for output plots.")


if __name__ == "__main__":
    main()
