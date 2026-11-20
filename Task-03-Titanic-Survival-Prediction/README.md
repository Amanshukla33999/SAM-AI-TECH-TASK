# Titanic Survival Prediction

Task: build a model that predicts whether a Titanic passenger survived.
Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset (Titanic-Dataset.csv, 891 rows)

## If Windows blocks the files (Smart App Control): use Google Colab
1. Go to https://colab.research.google.com and sign in with Google.
2. File > Upload notebook > choose `Titanic_Survival_Prediction.ipynb`.
3. Click the folder icon on the left, then drag `Titanic-Dataset.csv` into the Files panel.
4. Runtime > Run all. Charts and results appear in the notebook.

## How to run on your own PC
1. Install Python from https://www.python.org/downloads/ (tick "Add Python to PATH"). Skip if already installed.
2. Windows: double-click `run_windows.bat`.
   Mac/Linux: double-click `run_mac_linux.command` (or run `bash run_mac_linux.command`).
3. Wait until it says "Done!". Charts and the trained model appear in the new `outputs` folder.

The first run installs the libraries and needs internet. The tuning step can take a minute or two.

## Approach
1. EDA: survival by sex, class, age, fare, port; correlation heatmap
2. Feature engineering: Title (from name), FamilySize, IsAlone, HasCabin, Deck
3. Preprocessing pipeline: median/most-frequent imputation, scaling, one-hot encoding
4. Models compared with 5-fold stratified CV: Logistic Regression, Random Forest, Gradient Boosting
5. Random Forest tuned with GridSearchCV
6. Evaluation on a held-out 20% test set

## Results
| Model | CV accuracy |
|---|---|
| Logistic Regression | 82.3% |
| Random Forest | 81.5% |
| Gradient Boosting | 81.9% |
| Random Forest (tuned) | 83.2% |

Test set (179 passengers), tuned Random Forest: accuracy 79.9%, F1 (survived) 0.72, ROC AUC 0.837.

## Key findings
- Top features: Fare, Age, Title, Sex; passenger class and family size also matter.
- Women and first-class passengers survived at much higher rates.
- Model is better at identifying deaths (recall 87%) than survivors (recall 68%).
