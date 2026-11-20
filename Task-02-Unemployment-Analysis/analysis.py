"""Core logic for the Unemployment Analysis project (cleaning, COVID impact, seasonality, model)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = Path(__file__).parent / "data"

# Standard short names used everywhere in the project
RENAME = {
    "region": "State",
    "date": "Date",
    "frequency": "Frequency",
    "estimated unemployment rate (%)": "Unemployment",
    "estimated employed": "Employed",
    "estimated labour participation rate (%)": "LPR",
    "region.1": "Zone",
    "area": "Area",
    "longitude": "Longitude",
    "latitude": "Latitude",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def find_csv():
    """Return the first CSV found in data/ (prefers the one with 'upto' in its name = has 2020 data)."""
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        return None
    for f in files:
        if "upto" in f.name.lower():
            return f
    return files[0]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Kaggle unemployment dataframe."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: RENAME[c.lower()] for c in df.columns if c.lower() in RENAME})
    needed = {"State", "Date", "Unemployment", "Employed", "LPR"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    df = df.drop_duplicates().dropna(subset=["State", "Date"])
    for c in ["State", "Frequency", "Zone", "Area"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    for c in ["Unemployment", "Employed", "LPR"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Unemployment", "Employed", "LPR"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Month"].map(lambda m: MONTHS[m - 1])
    df["Phase"] = df["Date"].apply(covid_phase)
    return df.sort_values(["State", "Date"]).reset_index(drop=True)


def covid_phase(d: pd.Timestamp) -> str:
    """Pre-COVID / Lockdown (Mar-May 2020) / Recovery (Jun 2020 onwards)."""
    if d < pd.Timestamp("2020-03-01"):
        return "Pre-COVID"
    if d < pd.Timestamp("2020-06-01"):
        return "Lockdown"
    return "Recovery"


def load(path=None, uploaded=None) -> pd.DataFrame:
    if uploaded is not None:
        return clean(pd.read_csv(uploaded))
    path = path or find_csv()
    if path is None:
        raise FileNotFoundError("No CSV found in the data/ folder.")
    return clean(pd.read_csv(path))


def national_monthly(df):
    return df.groupby("Date", as_index=False).agg(Unemployment=("Unemployment", "mean"),
                                                   Employed=("Employed", "sum"),
                                                   LPR=("LPR", "mean"))


def covid_impact(df):
    """Average rate per state before vs. during lockdown + increase in percentage points."""
    t = df.pivot_table(index="State", columns="Phase", values="Unemployment", aggfunc="mean")
    for c in ["Pre-COVID", "Lockdown", "Recovery"]:
        if c not in t.columns:
            t[c] = np.nan
    t["Rise (pp)"] = t["Lockdown"] - t["Pre-COVID"]
    return t[["Pre-COVID", "Lockdown", "Recovery", "Rise (pp)"]].sort_values("Rise (pp)", ascending=False)


def seasonality(df):
    """Average unemployment by calendar month, excluding the 2020 COVID months so they don't distort it."""
    base = df[df["Date"] < pd.Timestamp("2020-03-01")]
    return base.groupby(["Month", "MonthName"], as_index=False)["Unemployment"].mean().sort_values("Month")


# ---------------- prediction model ----------------
FEATURES = ["Month", "Employed", "LPR", "Lockdown", "StateCode"]


def _prep(df):
    d = df.copy()
    d["Lockdown"] = (d["Phase"] == "Lockdown").astype(int)
    states = sorted(d["State"].unique())
    d["StateCode"] = d["State"].map({s: i for i, s in enumerate(states)})
    return d, states


def train_model(df):
    d, states = _prep(df)
    X, y = d[FEATURES], d["Unemployment"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42).fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {"MAE": mean_absolute_error(yte, pred), "R2": r2_score(yte, pred), "n_test": len(yte)}
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return model, states, metrics, imp


def predict(model, states, state, month, employed, lpr, lockdown):
    row = pd.DataFrame([[month, employed, lpr, int(lockdown), states.index(state)]], columns=FEATURES)
    return float(model.predict(row)[0])


def trend_forecast(df, periods=3):
    """Very simple linear-trend forecast of national average rate (short data => indicative only)."""
    n = national_monthly(df)
    n = n[n["Date"] >= pd.Timestamp("2020-06-01")]
    if len(n) < 3:
        return None
    x = np.arange(len(n))
    slope, intercept = np.polyfit(x, n["Unemployment"].values, 1)
    future_dates = pd.date_range(n["Date"].max() + pd.offsets.MonthEnd(1), periods=periods, freq="ME")
    fx = np.arange(len(n), len(n) + periods)
    return pd.DataFrame({"Date": future_dates, "Unemployment": np.clip(intercept + slope * fx, 0, None)})


def insights(df):
    """Auto-generated plain-English findings."""
    out = []
    n = national_monthly(df)
    peak = n.loc[n["Unemployment"].idxmax()]
    out.append(f"National average unemployment peaked at **{peak['Unemployment']:.1f}%** in "
               f"**{peak['Date']:%B %Y}**.")
    imp = covid_impact(df).dropna(subset=["Rise (pp)"])
    if len(imp):
        top = imp.head(3)
        out.append("Largest lockdown-driven jumps: " + ", ".join(
            f"**{s}** (+{r['Rise (pp)']:.1f} pp)" for s, r in top.iterrows()) + ".")
        pre = df[df.Phase == "Pre-COVID"]["Unemployment"].mean()
        lock = df[df.Phase == "Lockdown"]["Unemployment"].mean()
        if not np.isnan(pre) and not np.isnan(lock):
            out.append(f"Average rate rose from **{pre:.1f}%** before COVID to **{lock:.1f}%** during lockdown "
                       f"(x{lock / pre:.1f}).")
        rec = df[df.Phase == "Recovery"]["Unemployment"].mean()
        if not np.isnan(rec):
            out.append(f"In the recovery phase the average eased to **{rec:.1f}%**.")
    if "Zone" in df.columns:
        z = df.groupby("Zone")["Unemployment"].mean().sort_values(ascending=False)
        out.append(f"Highest average unemployment by zone: **{z.index[0]}** ({z.iloc[0]:.1f}%); "
                   f"lowest: **{z.index[-1]}** ({z.iloc[-1]:.1f}%).")
    out.append("Policy pointers: (1) targeted income/job support for the worst-hit states during shocks, "
               "(2) credit and wage support for small businesses and informal workers, who absorb shocks first, "
               "(3) keep labour-participation tracking alongside the headline rate, as the two can diverge.")
    return out
