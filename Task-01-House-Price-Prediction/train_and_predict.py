"""
House Price Prediction using Simple and Multiple Linear Regression
Internship Project - Data Preprocessing, Normalization, Modeling and Evaluation
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'housing_data.csv')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_housing_data.csv')
OUTPUT_PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def run_pipeline():
    print("=" * 70)
    print("  HOUSE PRICE PREDICTION - LINEAR REGRESSION PIPELINE  ")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/6] Loading dataset from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"Raw dataset shape: {df.shape}")
    nulls = df.isnull().sum()
    print("\nMissing values detected:")
    print(nulls[nulls > 0])

    # 2. Preprocessing: Imputation and Outlier Removal
    print("\n[2/6] Performing Data Preprocessing...")
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f" - Imputed missing values in '{col}' with median: {median_val}")

    # Outlier removal via 1st and 99th percentiles
    initial_len = len(df_clean)
    for col in ['SquareFootage', 'LotArea', 'Price']:
        q1 = df_clean[col].quantile(0.01)
        q3 = df_clean[col].quantile(0.99)
        df_clean = df_clean[(df_clean[col] >= q1) & (df_clean[col] <= q3)]
    print(f" - Outlier removal: Filtered {initial_len - len(df_clean)} extreme records. Clean shape: {df_clean.shape}")
    df_clean.to_csv(CLEANED_DATA_PATH, index=False)

    # 3. Exploratory Data Analysis & Visualizations
    print("\n[3/6] Generating Visualizations...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
    plt.title('Feature Correlation Matrix - Housing Features vs Price', fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, '01_correlation_heatmap.png'), dpi=200)
    plt.close()

    # Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df_clean['Price'], kde=True, ax=axes[0], color='royalblue', bins=30)
    axes[0].set_title('House Price Distribution', fontsize=12)
    axes[0].set_xlabel('Price ($)')
    sns.histplot(df_clean['SquareFootage'], kde=True, ax=axes[1], color='darkorange', bins=30)
    axes[1].set_title('Square Footage Distribution', fontsize=12)
    axes[1].set_xlabel('Square Footage (sq ft)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, '02_distributions.png'), dpi=200)
    plt.close()

    # 4. Train-Test Split & Normalization
    print("\n[4/6] Feature Selection & Normalization...")
    feature_cols = ['SquareFootage', 'TotalRooms', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'LotArea', 'GarageCars', 'OverallQuality']
    X = df_clean[feature_cols]
    y = df_clean['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Standardization (Z-score normalization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model 1: Simple Linear Regression (Square Footage -> Price)
    print("\n[5/6] Training Model 1: Simple Linear Regression (SquareFootage -> Price)...")
    simple_model = LinearRegression()
    simple_model.fit(X_train[['SquareFootage']], y_train)
    y_pred_simple = simple_model.predict(X_test[['SquareFootage']])

    s_mae = mean_absolute_error(y_test, y_pred_simple)
    s_rmse = np.sqrt(mean_squared_error(y_test, y_pred_simple))
    s_r2 = r2_score(y_test, y_pred_simple)
    print(f" - Simple Regression Slope (Price/sqft): ${simple_model.coef_[0]:.2f}")
    print(f" - Simple Regression Intercept: ${simple_model.intercept_:.2f}")
    print(f" - Simple Linear Regression R²: {s_r2:.4f} | RMSE: ${s_rmse:,.2f} | MAE: ${s_mae:,.2f}")

    # Plot Simple Linear Regression
    plt.figure(figsize=(9, 6))
    plt.scatter(X_test['SquareFootage'], y_test, alpha=0.5, color='teal', label='Actual Test Data')
    sorted_sqft = np.sort(X_test['SquareFootage'].values)
    plt.plot(sorted_sqft, simple_model.predict(pd.DataFrame({'SquareFootage': sorted_sqft})), color='red', linewidth=2.5, label=f'Regression Line (R² = {s_r2:.3f})')
    plt.title('Simple Linear Regression: Square Footage vs Price', fontsize=13)
    plt.xlabel('Square Footage (sq ft)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, '03_simple_regression_fit.png'), dpi=200)
    plt.close()

    # 6. Model 2: Multiple Linear Regression
    print("\n[6/6] Training Model 2: Multiple Linear Regression (All Selected Features)...")
    multi_model = LinearRegression()
    multi_model.fit(X_train_scaled, y_train)
    y_pred_multi = multi_model.predict(X_test_scaled)

    m_mae = mean_absolute_error(y_test, y_pred_multi)
    m_rmse = np.sqrt(mean_squared_error(y_test, y_pred_multi))
    m_r2 = r2_score(y_test, y_pred_multi)
    n, p = len(y_test), X_train.shape[1]
    m_adj_r2 = 1 - (1 - m_r2) * (n - 1) / (n - p - 1)

    print(f" - Multiple Linear Regression R²: {m_r2:.4f} | Adj R²: {m_adj_r2:.4f}")
    print(f" - RMSE: ${m_rmse:,.2f} | MAE: ${m_mae:,.2f}")

    # Plot Actual vs Predicted for Multiple Regression
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_multi, alpha=0.6, color='indigo', label='Predictions')
    min_val = min(y_test.min(), y_pred_multi.min())
    max_val = max(y_test.max(), y_pred_multi.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit Line (y = x)')
    plt.title(f'Multiple Linear Regression: Actual vs Predicted Price (R² = {m_r2:.3f})', fontsize=13)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, '04_actual_vs_predicted_multi.png'), dpi=200)
    plt.close()

    # Residual Analysis
    residuals = y_test - y_pred_multi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred_multi, residuals, alpha=0.6, color='darkgreen')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Residuals vs Fitted Values', fontsize=12)
    axes[0].set_xlabel('Fitted / Predicted Values ($)')
    axes[0].set_ylabel('Residuals ($)')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    sns.histplot(residuals, kde=True, ax=axes[1], color='purple', bins=25)
    axes[1].set_title('Residual Distribution (Normality Check)', fontsize=12)
    axes[1].set_xlabel('Residual ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, '05_residual_diagnostics.png'), dpi=200)
    plt.close()

    # Feature Importance (Standardized Coefficients)
    coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': multi_model.coef_}).sort_values('Coefficient', ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='teal')
    plt.title('Feature Impact on Price (Standardized Coefficients)', fontsize=13)
    plt.xlabel('Impact on Price ($ per 1 Standard Deviation)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, '06_feature_coefficients.png'), dpi=200)
    plt.close()

    # Save models and scalers
    with open(os.path.join(MODELS_DIR, 'simple_linear_regression.pkl'), 'wb') as f:
        pickle.dump(simple_model, f)
    with open(os.path.join(MODELS_DIR, 'multiple_linear_regression.pkl'), 'wb') as f:
        pickle.dump(multi_model, f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    metrics = {
        'Simple_Linear_Regression': {'R2': round(s_r2, 4), 'RMSE': round(s_rmse, 2), 'MAE': round(s_mae, 2)},
        'Multiple_Linear_Regression': {'R2': round(m_r2, 4), 'Adjusted_R2': round(m_adj_r2, 4), 'RMSE': round(m_rmse, 2), 'MAE': round(m_mae, 2)}
    }
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!  ")
    print("  All models, metrics, and plots have been generated and saved.  ")
    print("=" * 70)

if __name__ == '__main__':
    run_pipeline()
