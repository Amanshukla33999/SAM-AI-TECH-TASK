@echo off
cd /d "%~dp0"
echo ============================================
echo   Titanic Survival Prediction - setup + run
echo ============================================

where python >nul 2>nul
if %errorlevel%==0 (set PY=python) else (
  where py >nul 2>nul
  if %errorlevel%==0 (set PY=py) else (
    echo.
    echo Python is not installed. Install it from https://www.python.org/downloads/
    echo IMPORTANT: tick "Add Python to PATH" during installation, then run this file again.
    pause
    exit /b
  )
)

echo.
echo Installing required libraries (first time only, needs internet)...
%PY% -m pip install pandas numpy matplotlib seaborn scikit-learn joblib

echo.
echo Running the project...
%PY% titanic_survival_prediction.py Titanic-Dataset.csv

echo.
echo ============================================
echo Done! Open the "outputs" folder for charts and the saved model.
echo ============================================
pause
