# SAM AI Tech Task - Machine Learning & Data Science Internship

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

This repository contains the complete implementation and technical submissions for the **SAM AI Tech Task**. It features three comprehensive machine learning and data science projects covering regression, classification, exploratory data analysis (EDA), interactive dashboard development, and web-based prediction tools.

---

## 📁 Repository Structure

`	ext
SAM-AI-Tech-Task/
├── README.md                                  # Repository overview and setup guide
├── .gitignore                                 # Standard Git ignore rules
│
├── Task-01-House-Price-Prediction/            # Task 1: Regression Analysis
│   ├── data/                                  # Raw and preprocessed housing datasets
│   ├── models/                                # Trained model weights (.pkl) & metrics
│   ├── notebooks/                             # Jupyter Notebook exploration
│   ├── outputs/plots/                         # Evaluation and residual diagnostic charts
│   ├── gui_app.py                             # Interactive Tkinter Desktop GUI
│   ├── WEB_CALCULATOR.html                    # Standalone Web-based Price Calculator
│   ├── predict.py                             # CLI inference pipeline
│   ├── train_and_predict.py                   # Model training and artifact generation
│   ├── INTERNSHIP_REPORT.md                   # Formal technical internship report
│   └── README.md                              # Task 1 documentation
│
├── Task-02-Unemployment-Analysis/             # Task 2: Time-Series & Socioeconomic Analysis
│   ├── analysis.py                            # Statistical data analysis module
│   ├── app.py                                 # Interactive Streamlit dashboard
│   ├── download_data.py                       # Automated Kaggle data acquisition
│   ├── report.py                              # Analytical summary generator
│   ├── requirements.txt                       # Task dependencies
│   ├── run.bat / run.sh                       # 1-Click launcher scripts
│   └── README.md                              # Task 2 documentation
│
└── Task-03-Titanic-Survival-Prediction/       # Task 3: Classification & Feature Engineering
    ├── results/                               # Trained model (.joblib) & EDA visualizations
    ├── Titanic-Dataset.csv                    # Historical passenger dataset
    ├── titanic_survival_prediction.py         # Full training and evaluation script
    ├── Titanic_Survival_Prediction.ipynb      # Step-by-step Jupyter Notebook
    ├── run_windows.bat / run_mac_linux.command # 1-Click launcher scripts
    ├── requirements.txt                       # Task dependencies
    └── README.md                              # Task 3 documentation
`

---

## 🚀 Projects Overview

### 🏡 Task 1: House Price Prediction (Simple & Multiple Linear Regression)
* **Objective:** Predict residential property values based on key architectural features (Living area square footage, room counts, bathrooms, construction year, overall quality).
* **Key Methodology:**
  * Outlier detection using 1st & 99th percentile truncation.
  * Feature normalization using StandardScaler fitted strictly on training data.
  * Comparison between baseline **Simple Linear Regression** (^2 = 0.898$) and **Multiple Linear Regression** (^2 = 0.9633$, RMSE = ,208.14).
* **Deliverables:** Desktop GUI application (gui_app.py), standalone interactive web calculator (WEB_CALCULATOR.html), Jupyter notebook, diagnostic plots, and formal internship report.

---

### 📊 Task 2: Unemployment Analysis with Python
* **Objective:** Analyze unemployment rate trends across Indian states with special emphasis on the impact of the Covid-19 pandemic.
* **Key Methodology:**
  * Temporal trend tracking and regional socioeconomic impact analysis.
  * Rural vs. Urban comparison across pre-lockdown, strict-lockdown, and post-lockdown phases.
  * Dynamic filtering, choropleth maps, and statistical summary visualizations.
* **Deliverables:** Production-ready interactive Streamlit dashboard (pp.py), automated data downloader (download_data.py), and analytical report script (
eport.py).

---

### 🚢 Task 3: Titanic Survival Prediction
* **Objective:** Build a supervised binary classification model to predict passenger survival on the Titanic based on demographic and voyage factors.
* **Key Methodology:**
  * Feature engineering: Title extraction (Mr, Mrs, Miss, Master, Rare), family size indicators (FamilySize, IsAlone), and ticket fare binning.
  * Handling missing values with median and mode imputation.
  * Model evaluation comparing Logistic Regression and Random Forest classifiers.
* **Deliverables:** Python pipeline script (	itanic_survival_prediction.py), Jupyter notebook, serialized model artifact (	itanic_model.joblib), and visual charts.

---

## 🛠️ Quick Start & Installation

To run any task on your local environment:

1. **Clone the repository:**
   `ash
   git clone https://github.com/<your-username>/SAM-AI-Tech-Task.git
   cd SAM-AI-Tech-Task
   `

2. **Navigate to the desired task directory and install dependencies:**
   * For **Task 1 (House Price Prediction)**:
     `ash
     cd Task-01-House-Price-Prediction
     pip install -r requirements.txt
     python train_and_predict.py
     python gui_app.py
     `
   * For **Task 2 (Unemployment Analysis)**:
     `ash
     cd Task-02-Unemployment-Analysis
     pip install -r requirements.txt
     streamlit run app.py
     `
   * For **Task 3 (Titanic Survival Prediction)**:
     `ash
     cd Task-03-Titanic-Survival-Prediction
     pip install -r requirements.txt
     python titanic_survival_prediction.py
     `

---

## 👨‍💻 Author & Submission Details
* **Repository Name:** SAM-AI-Tech-Task
* **Technologies:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Tkinter, HTML5/CSS3.
* **Submission Purpose:** SAM AI Technical Assessment / Internship Submission.
