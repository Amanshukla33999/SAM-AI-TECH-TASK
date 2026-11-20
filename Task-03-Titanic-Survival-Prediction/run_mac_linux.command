#!/bin/bash
cd "$(dirname "$0")"
python3 -m pip install pandas numpy matplotlib seaborn scikit-learn joblib
python3 titanic_survival_prediction.py Titanic-Dataset.csv
echo "Done! Open the outputs folder for charts and the saved model."
read -p "Press Enter to close..."
