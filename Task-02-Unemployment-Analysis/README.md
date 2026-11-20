# Unemployment Analysis with Python

**How to run (Windows):** double-click `run.bat`. First run installs packages and downloads the dataset,
then the dashboard opens in your browser (http://localhost:8501). Mac/Linux: `./run.sh`.

If the dataset doesn't download automatically, get it from
https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india and put the CSV in `data/`
(or upload it in the app's sidebar).

## Files
- `app.py` - dashboard (7 tabs: data/cleaning, trends, COVID-19 impact, seasonality, regions, prediction, insights)
- `analysis.py` - cleaning, COVID-phase analysis, seasonality, Random Forest model, auto-insights
- `download_data.py` - fetches the Kaggle dataset into `data/`
- `report.py` - saves charts, tables and insights into `outputs/`
- `requirements.txt`, `run.bat`, `run.sh`

## Task objectives covered
1. Analyze unemployment rate data -> Trends tab
2. Python cleaning, exploration, visualization -> Data & Trends tabs
3. Impact of Covid-19 -> COVID-19 Impact tab
4. Seasonal patterns -> Seasonality tab
5. Insights for policy -> Insights tab
