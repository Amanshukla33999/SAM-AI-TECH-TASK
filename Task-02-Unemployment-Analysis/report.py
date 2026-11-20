"""Offline report: saves cleaned data, tables, charts (PNG) and insights into the outputs/ folder.
Run:  python report.py   (run.bat does this automatically before opening the dashboard)"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis as a

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

df = a.load()
df.to_csv(OUT / "cleaned_unemployment.csv", index=False)
imp = a.covid_impact(df); imp.round(2).to_csv(OUT / "covid_impact_by_state.csv")
a.seasonality(df).to_csv(OUT / "seasonality.csv", index=False)

n = a.national_monthly(df)
fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(n.Date, n.Unemployment, marker="o")
ax.axvspan("2020-03-25", "2020-05-31", color="red", alpha=.15, label="Lockdown")
ax.set_title("Average unemployment rate (%)"); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(OUT / "trend.png", dpi=150); plt.close(fig)

top = imp.dropna(subset=["Rise (pp)"]).head(15)
fig, ax = plt.subplots(figsize=(10, 5)); ax.barh(top.index[::-1], top["Rise (pp)"][::-1], color="indianred")
ax.set_title("Rise in unemployment, lockdown vs pre-COVID (pp)"); fig.tight_layout()
fig.savefig(OUT / "covid_rise.png", dpi=150); plt.close(fig)

model, states, met, fi = a.train_model(df)
lines = ["# Unemployment Analysis - Findings", ""] + ["- " + i for i in a.insights(df)]
lines += ["", f"Model: Random Forest | MAE {met['MAE']:.2f} pp | R2 {met['R2']:.2f} (n_test={met['n_test']})",
          "", "Note: dataset spans ~17 months, so seasonality/forecast are indicative only."]
(OUT / "insights.md").write_text("\n".join(lines), encoding="utf-8")
print("Report files saved in:", OUT)
