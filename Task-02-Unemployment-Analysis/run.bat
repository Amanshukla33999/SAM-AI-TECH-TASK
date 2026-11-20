@echo off
title Unemployment Analysis Project
cd /d "%~dp0"
echo ================================================
echo   Unemployment Analysis with Python
echo ================================================
where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Install it from https://www.python.org/downloads/ ^(tick "Add Python to PATH"^)
  pause
  exit /b
)
if not exist ".venv" (
  echo [1/4] Creating virtual environment ...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [2/4] Installing requirements ^(first run takes a few minutes^) ...
python -m pip install -q -r requirements.txt
echo [3/4] Checking dataset ...
python download_data.py
echo [4/4] Building report files and starting dashboard ...  browser opens automatically
python report.py
python -m streamlit run app.py
pause
