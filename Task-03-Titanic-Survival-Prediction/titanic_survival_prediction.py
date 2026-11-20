"""
Titanic Survival Prediction
---------------------------
Usage:
    1. Download the dataset from https://www.kaggle.com/datasets/yasserh/titanic-dataset
       (file: Titanic-Dataset.csv) and place it next to this script.
    2. pip install pandas numpy matplotlib seaborn scikit-learn joblib
    3. python titanic_survival_prediction.py [path/to/Titanic-Dataset.csv]

Outputs (in ./outputs): EDA plots, model comparison, confusion matrix,
ROC curve, feature importances, and the trained model (titanic_model.joblib).
"""

import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # works without a display; remove if using a notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
path = sys.argv[1] if len(sys.argv) > 1 else "Titanic-Dataset.csv"
df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nSurvival rate: {:.1%}".format(df["Survived"].mean()))


# ----------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

sns.countplot(data=df, x="Survived", ax=axes[0, 0])
axes[0, 0].set_title("Survival count (0 = No, 1 = Yes)")

sns.barplot(data=df, x="Sex", y="Survived", ax=axes[0, 1])
axes[0, 1].set_title("Survival rate by Sex")

sns.barplot(data=df, x="Pclass", y="Survived", ax=axes[0, 2])
axes[0, 2].set_title("Survival rate by Passenger Class")

sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30, ax=axes[1, 0])
axes[1, 0].set_title("Age distribution by survival")

sns.boxplot(data=df, x="Survived", y="Fare", ax=axes[1, 1])
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("Fare by survival (log scale)")

sns.barplot(data=df, x="Embarked", y="Survived", ax=axes[1, 2])
axes[1, 2].set_title("Survival rate by Port of Embarkation")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_eda.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix (numeric features)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_correlation.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------------
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()

    # Title extracted from the passenger name (Mr, Mrs, Miss, Master, ...)
    d["Title"] = d["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    d["Title"] = d["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "the Countess": "Rare",
         "Sir": "Rare", "Don": "Rare", "Dona": "Rare", "Jonkheer": "Rare", "Capt": "Rare",
         "Col": "Rare", "Major": "Rare", "Dr": "Rare", "Rev": "Rare"}
    )
    common = {"Mr", "Mrs", "Miss", "Master"}
    d["Title"] = d["Title"].where(d["Title"].isin(common), "Rare")

    # Family features
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)

    # Cabin features (Cabin is ~77% missing, so keep only coarse information)
    d["HasCabin"] = d["Cabin"].notna().astype(int)
    d["Deck"] = d["Cabin"].str[0].fillna("U")

    return d


df = engineer_features(df)

numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
categorical_features = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone", "HasCabin"]
features = numeric_features + categorical_features

# PassengerId, Name, Ticket and Cabin are dropped (identifiers / high cardinality)
X = df[features].copy()
X["Pclass"] = X["Pclass"].astype(str)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")


# ----------------------------------------------------------------------------
# 4. Preprocessing pipeline (imputation + scaling + one-hot encoding)
# ----------------------------------------------------------------------------
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])


# ----------------------------------------------------------------------------
# 5. Compare models with 5-fold stratified cross-validation
# ----------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
print("\n5-fold CV accuracy:")
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores
    print(f"  {name:<20} {scores.mean():.4f} (+/- {scores.std():.4f})")


# ----------------------------------------------------------------------------
# 6. Hyperparameter tuning on the Random Forest
# ----------------------------------------------------------------------------
rf_pipe = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
])
param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [4, 6, 8, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nBest RF params:", grid.best_params_)
print(f"Best RF CV accuracy: {grid.best_score_:.4f}")

best_model = grid.best_estimator_


# ----------------------------------------------------------------------------
# 7. Final evaluation on the held-out test set
# ----------------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n=== Test set performance ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\n", classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Died", "Survived"], cmap="Blues", ax=axes[0]
)
axes[0].set_title("Confusion matrix")
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].set_title("ROC curve")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_evaluation.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------------
# 8. Feature importance
# ----------------------------------------------------------------------------
prep = best_model.named_steps["prep"]
rf = best_model.named_steps["model"]
cat_names = prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
    categorical_features
)
all_names = numeric_features + list(cat_names)
importances = (
    pd.Series(rf.feature_importances_, index=all_names).sort_values(ascending=False).head(15)
)
print("\nTop features:\n", importances.round(4))

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, color="steelblue")
plt.title("Top 15 feature importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_feature_importance.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------------
# 9. Save the model and demo a prediction
# ----------------------------------------------------------------------------
joblib.dump(best_model, f"{OUT_DIR}/titanic_model.joblib")

sample = pd.DataFrame([{
    "Age": 28, "Fare": 70.0, "SibSp": 0, "Parch": 0, "FamilySize": 1,
    "Pclass": "1", "Sex": "female", "Embarked": "S", "Title": "Miss",
    "Deck": "C", "IsAlone": 1, "HasCabin": 1,
}])
prob = best_model.predict_proba(sample)[0, 1]
print(f"\nExample passenger (28F, 1st class): survival probability = {prob:.1%}")
print(f"\nAll outputs saved to ./{OUT_DIR}/")
