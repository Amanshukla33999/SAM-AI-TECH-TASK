# House Price Prediction - Simple & Multiple Linear Regression
**Data Science Internship Project**

## 1. Project Overview
This repository contains an end-to-end Machine Learning pipeline for predicting residential house prices using **Simple Linear Regression** and **Multiple Linear Regression**. 

The project strictly fulfills all internship requirements:
1. **Housing Prices Dataset**: Contains real estate features including Living Area Square Footage (`SquareFootage`), Total Rooms (`TotalRooms`), Bedrooms (`Bedrooms`), Bathrooms (`Bathrooms`), Construction Year (`YearBuilt`), Lot Area (`LotArea`), Garage Capacity (`GarageCars`), Overall Material/Finish Quality (`OverallQuality`), and Sale Price (`Price`).
2. **Data Preprocessing**:
   - **Missing Value Imputation**: Median imputation for skewed numerical variables.
   - **Outlier Detection & Removal**: Statistical percentile filtering (1st - 99th percentile) to protect Ordinary Least Squares (OLS) estimation.
   - **Feature Selection**: Correlation analysis and multi-collinearity checks.
   - **Normalization & Standardization**: `StandardScaler` (Z-score normalization) applied to scale features without data leakage (fit on training data only).
3. **Model Development**:
   - **Simple Linear Regression**: Modeling the direct relationship between living area square footage and house price ($y = \beta_0 + \beta_1 X$).
   - **Multiple Linear Regression**: Predicting house price using all key property features ($y = \beta_0 + \sum \beta_i X_i$).
4. **Model Evaluation**:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - $R^2$ Score (Coefficient of Determination)
   - Adjusted $R^2$ Score
   - Residual Diagnostics (Homoscedasticity & Normality of Errors)

---

## 2. Project Directory Structure
```text
house prediction project/
│
├── data/
│   ├── housing_data.csv             # Raw input dataset
│   └── cleaned_housing_data.csv     # Preprocessed & cleaned dataset
│
├── notebooks/
│   └── house_price_prediction.ipynb # Step-by-step Jupyter Notebook with EDA, training, and plots
│
├── outputs/
│   └── plots/
│       ├── 01_correlation_heatmap.png      # Feature correlation matrix
│       ├── 02_distributions.png            # Distribution of price & square footage
│       ├── 03_simple_regression_fit.png    # Simple linear regression fit
│       ├── 04_actual_vs_predicted_multi.png# Actual vs predicted prices
│       ├── 05_residual_diagnostics.png     # Residuals vs fitted & error distribution
│       └── 06_feature_coefficients.png     # Standardized regression coefficients
│
├── models/
│   ├── simple_linear_regression.pkl        # Serialized simple regression model
│   ├── multiple_linear_regression.pkl      # Serialized multiple regression model
│   ├── scaler.pkl                          # Fitted StandardScaler object
│   └── model_metrics.json                  # Saved evaluation metrics
│
├── train_and_predict.py             # Executable training and evaluation pipeline
├── predict.py                       # Interactive script for custom house price inference
├── requirements.txt                 # Project dependencies
├── INTERNSHIP_REPORT.md             # Formal Data Science Internship Report
└── README.md                        # Documentation and guide
```

---

## 3. Installation & Setup

1. **Clone or Navigate to the project folder:**
   ```bash
   cd "C:\Users\amans\Desktop\house prediction project"
   ```

2. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. How to Run

### Run the Complete Training Pipeline:
```bash
python train_and_predict.py
```
This script will:
- Clean and preprocess the raw data
- Train both Simple and Multiple Linear Regression models
- Output metrics to the terminal
- Save the trained models into `models/`
- Generate high-resolution plots into `outputs/plots/`

### Run Interactive Predictions for New Houses:
```bash
python predict.py
```

### Run the Interactive Jupyter Notebook:
```bash
jupyter notebook notebooks/house_price_prediction.ipynb
```

---

## 5. Model Evaluation & Results

| Model | $R^2$ Score | Adjusted $R^2$ | RMSE | MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Simple Linear Regression** (SquareFootage) | **0.9018** | 0.9014 | **$34,682.85** | **$25,736.75** |
| **Multiple Linear Regression** (All Key Features) | **0.9633** | **0.9620** | **$21,208.14** | **$15,727.13** |

### Key Findings:
- **Square Footage** is the single strongest predictor of house prices, explaining **~90.2%** of price variance on its own.
- Adding **Overall Quality**, **Year Built**, and **Bathrooms** improves the explanatory power to **96.3%** ($R^2 = 0.9633$) and reduces Root Mean Squared Error from **$34,682** down to **$21,208** (a **38.8% reduction in error**).
- Residual plots verify homoscedasticity (constant variance) and normally distributed error terms, confirming that linear regression assumptions hold well for this dataset.
