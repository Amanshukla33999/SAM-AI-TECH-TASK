#!/bin/bash
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
python download_data.py
python report.py
python -m streamlit run app.py
