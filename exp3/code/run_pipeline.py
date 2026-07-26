import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate, validation_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plot styling
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "../dataset/train.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Step 1: Loading Dataset ===")
df = pd.read_csv(DATASET_PATH)
print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

target_col = "Loan Sanction Amount (USD)"

# Drop ID and Name columns
cols_to_drop = [c for c in ["Customer ID", "Name", "Property ID"] if c in df.columns]
df = df.drop(columns=cols_to_drop)

# Drop rows where target variable is missing or negative/invalid if any
df = df.dropna(subset=[target_col])
df = df[df[target_col] >= 0]

X = df.drop(columns=[target_col])
y = df[target_col]

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Split Train/Test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

# Build Preprocessing Transformer
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Fit Preprocessor on Train data
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Feature names after one-hot encoding
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + cat_feature_names

# Define Baseline & Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42, tol=0.01),
    'Elastic Net Regression': ElasticNet(random_state=42, tol=0.01)
}

# Define Hyperparameter Search Spaces as specified in question
param_grids = {
    'Ridge Regression': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
    'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
    'Elastic Net Regression': {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.2, 0.5, 0.8]
    }
}

print("\n=== Step 2: Hyperparameter Tuning (GridSearchCV - K=5) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_estimators = {'Linear Regression': LinearRegression().fit(X_train_proc, y_train)}
tuning_results = []

for name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net Regression']:
    t0 = time.time()
    grid = GridSearchCV(
        estimator=models[name],
        param_grid=param_grids[name],
        cv=kf,
        scoring='r2',
        n_jobs=-1
    )
    grid.fit(X_train_proc, y_train)
    exec_time = time.time() - t0
    
    best_estimators[name] = grid.best_estimator_
    
    params_str = ", ".join([f"{k}={v}" for k, v in grid.best_params_.items()])
    tuning_results.append({
        'Model': name,
        'Search Method': 'GridSearch (5-Fold)',
        'Best Parameters': params_str,
        'Best CV R2': round(grid.best_score_, 4),
        'Execution Time(s)': f"{exec_time:.2f}"
    })

df_table1 = pd.DataFrame(tuning_results)
print("\n--- Table 1: Hyperparameter Tuning Summary ---")
print(df_table1.to_string(index=False))

print("\n=== Step 3: Cross-Validation Performance (K = 5) ===")
cv_results_list = []
scoring = {
    'mae': 'neg_mean_absolute_error',
    'mse': 'neg_mean_squared_error',
    'r2': 'r2'
}

for name, model in best_estimators.items():
    scores = cross_validate(model, X_train_proc, y_train, cv=kf, scoring=scoring, n_jobs=-1)
    mae = -scores['test_mae'].mean()
    mse = -scores['test_mse'].mean()
    rmse = np.sqrt(mse)
    r2 = scores['test_r2'].mean()
    
    cv_results_list.append({
        'Model': name,
        'MAE': round(mae, 2),
        'MSE': f"{mse:.2e}",
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4)
    })

df_table2 = pd.DataFrame(cv_results_list)
print("\n--- Table 2: Cross-Validation Performance (K=5) ---")
print(df_table2.to_string(index=False))

print("\n=== Step 4: Test Set Performance & Training Times ===")
test_results_list = []
exp_train_times = {}

for name, model in best_estimators.items():
    t0 = time.perf_counter()
    model.fit(X_train_proc, y_train)
    tr_time = time.perf_counter() - t0
    exp_train_times[name] = tr_time
    
    y_pred = model.predict(X_test_proc)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    test_results_list.append({
        'Model': name,
        'MAE': round(mae, 2),
        'MSE': f"{mse:.2e}",
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'Training Time(s)': f"{tr_time:.6f}"
    })

df_table3 = pd.DataFrame(test_results_list)
print("\n--- Table 3: Test Set Performance ---")
print(df_table3.to_string(index=False))

print("\n=== Step 5: Effect of Regularization on Coefficients ===")
coeff_dict = {'Feature': all_feature_names}
for name, model in best_estimators.items():
    coeff_dict[name] = model.coef_

df_coeffs = pd.DataFrame(coeff_dict)
df_coeffs['Abs_Sum'] = df_coeffs[['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression']].abs().sum(axis=1)
df_top_coeffs = df_coeffs.sort_values(by='Abs_Sum', ascending=False).head(10).drop(columns=['Abs_Sum'])

# Round coefficients for clean tabular output
for col in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression']:
    df_top_coeffs[col] = df_top_coeffs[col].round(2)

print("\n--- Table 4: Coefficient Comparison (Top 10 Features) ---")
print(df_top_coeffs.to_string(index=False))

print("\n=== Step 6: Generating Required 6 Plots ===")

# Plot 1: Target Variable Distribution
plt.figure(figsize=(8, 5))
sns.histplot(y, kde=True, color="royalblue", bins=30)
plt.title("Plot 1: Target Variable Distribution (Loan Sanction Amount)", fontsize=12, fontweight='bold')
plt.xlabel("Loan Sanction Amount (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_target_distribution.png"))
plt.close()
print("Saved Plot 1: Target distribution")

# Plot 2: Feature vs Target Scatter Plot (Top 4 numerical predictors)
top_num_features = ["Loan Amount Request (USD)", "Property Price", "Credit Score", "Current Loan Expenses (USD)"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, col in enumerate(top_num_features):
    if col in df.columns:
        sns.scatterplot(data=df, x=col, y=target_col, alpha=0.3, ax=axes[idx], color="teal")
        axes[idx].set_title(f"{col} vs. Target", fontweight='bold')
        axes[idx].set_ylabel("Sanction Amount (USD)")
plt.suptitle("Plot 2: Feature vs. Target Scatter Plots", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_feature_vs_target_scatter.png"))
plt.close()
print("Saved Plot 2: Feature vs. target scatter plots")

# Plot 3: Predicted vs Actual Values (using Best Regularized Model)
best_model_name = df_table3.sort_values(by='R2', ascending=False).iloc[0]['Model']
best_model = best_estimators[best_model_name]
y_pred_best = best_model.predict(X_test_proc)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.4, color="crimson", label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Fit (y = x)")
plt.title(f"Plot 3: Predicted vs. Actual Values ({best_model_name})", fontsize=12, fontweight='bold')
plt.xlabel("Actual Loan Sanction Amount (USD)")
plt.ylabel("Predicted Loan Sanction Amount (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_predicted_vs_actual.png"))
plt.close()
print("Saved Plot 3: Predicted vs actual values plot")

# Plot 4: Residual Plot
residuals = y_test - y_pred_best
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_best, residuals, alpha=0.4, color="darkorange")
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.title(f"Plot 4: Residual Plot ({best_model_name})", fontsize=12, fontweight='bold')
plt.xlabel("Predicted Values (USD)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_residual_plot.png"))
plt.close()
print("Saved Plot 4: Residual plot")

# Plot 5: Training Error vs Validation Error Plot (Validation Curve across Ridge Alpha)
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, val_scores = validation_curve(
    Ridge(random_state=42),
    X_train_proc, y_train,
    param_name="alpha",
    param_range=alphas,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
train_rmse = np.sqrt(-train_scores.mean(axis=1))
val_rmse = np.sqrt(-val_scores.mean(axis=1))

plt.figure(figsize=(8, 5))
plt.plot(alphas, train_rmse, marker='o', label="Training Error (RMSE)", color="blue", lw=2)
plt.plot(alphas, val_rmse, marker='s', label="Validation Error (RMSE)", color="red", lw=2)
plt.xscale("log")
plt.title("Plot 5: Training Error vs. Validation Error across Ridge Alpha", fontsize=12, fontweight='bold')
plt.xlabel("Regularization Strength Alpha (log scale)")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "5_learning_curve_errors.png"))
plt.close()
print("Saved Plot 5: Training vs Validation Error plot")

# Plot 6: Coefficient Comparison Bar Plot
df_plot_coeffs = df_top_coeffs.head(6).set_index('Feature')
df_plot_coeffs.plot(kind='barh', figsize=(10, 6), width=0.8)
plt.title("Plot 6: Coefficient Magnitude Comparison (Top Features)", fontsize=12, fontweight='bold')
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.axvline(x=0, color='black', linestyle='--', lw=1)
plt.legend(title="Model", loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "6_coefficient_comparison.png"))
plt.close()
print("Saved Plot 6: Coefficient comparison bar plot")

print("\n=== All Steps Completed Successfully! ===")
