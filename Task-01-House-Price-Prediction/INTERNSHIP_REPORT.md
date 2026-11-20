# DATA SCIENCE INTERNSHIP REPORT
## Project Title: House Price Prediction using Simple & Multiple Linear Regression

**Internship Domain:** Machine Learning & Data Science  
**Task Topic:** Simple Linear Regression on Housing Prices  

---

### 1. Executive Summary
The primary objective of this project is to develop and evaluate a machine learning model capable of accurately predicting residential property sale prices based on fundamental architectural and property features (such as living area square footage, room counts, bathroom counts, construction year, lot area, and overall quality). 

Through systematic data preprocessing, outlier removal, feature selection, and standardization, both a **Simple Linear Regression** baseline and a **Multiple Linear Regression** model were built and evaluated on an unseen test dataset. The final Multiple Linear Regression model achieved an $R^2$ score of **0.9633** (Adjusted $R^2 = 0.9620$) with a Root Mean Squared Error (RMSE) of **$21,208.14**, demonstrating superior predictive performance.

---

### 2. Dataset Description
The dataset encompasses 1,200+ property observations with the following attributes:
- **SquareFootage**: Above-grade living area in square feet (primary predictor).
- **TotalRooms**: Total rooms above ground (excluding bathrooms).
- **Bedrooms**: Number of bedrooms.
- **Bathrooms**: Number of full and half bathrooms.
- **YearBuilt**: Original construction year.
- **LotArea**: Total land parcel size in square feet.
- **GarageCars**: Vehicle capacity of the garage.
- **OverallQuality**: Rated quality scale from 1 (poor) to 10 (very excellent).
- **Price**: Target variable representing the closing sale price in USD.

---

### 3. Data Preprocessing Methodology
1. **Handling Missing Values:**
   - Numerical missing fields were imputed using column-wise **median** values. Median imputation was chosen over mean imputation to prevent distortion from right-skewed property metrics.
2. **Outlier Detection & Removal:**
   - Real estate transaction data frequently contains anomalies (e.g., non-arms-length sales or mega-mansions). Outliers were identified and filtered using statistical percentile bounds ($1^{\text{st}}$ and $99^{\text{th}}$ percentiles) on `SquareFootage`, `LotArea`, and `Price`.
3. **Feature Selection:**
   - Evaluated Pearson correlation coefficients between predictors and the target variable. Strongest correlation observed with `SquareFootage` ($r \approx 0.95$), followed by `OverallQuality` and `TotalRooms`.
4. **Feature Normalization / Standardization:**
   - Applied **StandardScaler** ($z = \frac{x - \mu}{\sigma}$) to all continuous explanatory variables.
   - Crucially, the scaler was fitted **strictly on the training partition** and then transformed on the test partition to eliminate data leakage.

---

### 4. Mathematical Modeling

#### 4.1 Simple Linear Regression
$$y = \beta_0 + \beta_1 X + \epsilon$$
- **Target ($y$):** House Price
- **Predictor ($X$):** Living Area Square Footage
- **Estimated Intercept ($\beta_0$):** Baseline property value
- **Estimated Slope ($\beta_1$):** Marginal price increase per additional square foot

#### 4.2 Multiple Linear Regression
$$y = \beta_0 + \beta_1 X_{\text{SqFt}} + \beta_2 X_{\text{Rooms}} + \beta_3 X_{\text{Beds}} + \beta_4 X_{\text{Baths}} + \beta_5 X_{\text{Year}} + \beta_6 X_{\text{Lot}} + \beta_7 X_{\text{Garage}} + \beta_8 X_{\text{Qual}} + \epsilon$$

---

### 5. Experimental Results & Performance Comparison

| Metric | Simple Linear Regression | Multiple Linear Regression | Performance Impact |
| :--- | :---: | :---: | :---: |
| **$R^2$ Score** | 0.9018 | **0.9633** | +6.15% variance explained |
| **Adjusted $R^2$** | 0.9014 | **0.9620** | Confirms added features add real predictive signal |
| **RMSE** | $34,682.85 | **$21,208.14** | **38.8% reduction in error** |
| **MAE** | $25,736.75 | **$15,727.13** | Average prediction error within ~2.8% of price |

---

### 6. Residual Analysis & Validation
- **Homoscedasticity:** Scatter plot of residuals against fitted values demonstrates constant variance along the horizontal axis with no visible fanning or heteroscedastic cone patterns.
- **Normality of Errors:** Histogram and Kernel Density Estimation (KDE) of residuals reveal a symmetric bell curve centered at zero, confirming the normality assumption of Ordinary Least Squares regression.

---

### 7. Conclusion & Internship Deliverables
All requirements assigned for the task have been successfully achieved:
1. Sourced and structured a complete housing prices dataset.
2. Implemented preprocessing routines (missing values, outlier removal, feature selection, and normalization).
3. Trained and evaluated both Simple and Multiple Linear Regression models.
4. Exported all serialized model artifacts, reproducible Jupyter notebook, interactive CLI prediction tools, and publication-grade plots.
