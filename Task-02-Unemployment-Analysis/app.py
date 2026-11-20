"""Unemployment Analysis with Python - interactive dashboard (Streamlit)."""
import pandas as pd
import plotly.express as px
import streamlit as st
import analysis as a

st.set_page_config(page_title="Unemployment Analysis", page_icon="📉", layout="wide")
st.title("📉 Unemployment Analysis with Python")
st.caption("Data Science Internship Project - India unemployment rate, COVID-19 impact, seasonality & prediction")

# ---------- load data ----------
up = st.sidebar.file_uploader("Upload CSV (optional)", type="csv")
try:
    df = a.load(uploaded=up)
except Exception as e:
    st.error(f"Could not load data: {e}\n\nPut the Kaggle CSV in the `data` folder, or upload it in the sidebar.")
    st.stop()

states = st.sidebar.multiselect("Filter states", sorted(df.State.unique()), default=[])
fdf = df[df.State.isin(states)] if states else df
st.sidebar.markdown(f"**Rows:** {len(fdf):,}  \n**Period:** {fdf.Date.min():%b %Y} - {fdf.Date.max():%b %Y}")

tabs = st.tabs(["1 Data & Cleaning", "2 Trends", "3 COVID-19 Impact", "4 Seasonality",
                "5 Regions", "6 Prediction", "7 Insights"])

# ---------- 1 data ----------
with tabs[0]:
    st.subheader("Cleaned dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(fdf):,}")
    c2.metric("States", fdf.State.nunique())
    c3.metric("Avg unemployment", f"{fdf.Unemployment.mean():.2f}%")
    c4.metric("Max unemployment", f"{fdf.Unemployment.max():.1f}%")
    st.markdown("Cleaning steps: stripped column-name spaces, parsed dates (dd-mm-yyyy), removed duplicates and "
                "missing values, added Year / Month / COVID-phase columns.")
    st.dataframe(fdf.head(100))
    st.write("Summary statistics"); st.dataframe(fdf[["Unemployment", "Employed", "LPR"]].describe())
    st.download_button("Download cleaned CSV", fdf.to_csv(index=False), "cleaned_unemployment.csv")

# ---------- 2 trends ----------
with tabs[1]:
    n = a.national_monthly(fdf)
    fig = px.line(n, x="Date", y="Unemployment", markers=True, title="Average unemployment rate over time (%)")
    fig.add_vrect(x0="2020-03-25", x1="2020-05-31", fillcolor="red", opacity=0.12, annotation_text="Lockdown")
    st.plotly_chart(fig)
    fig = px.line(fdf, x="Date", y="Unemployment", color="State", title="Unemployment rate by state (%)")
    st.plotly_chart(fig)
    st.plotly_chart(px.line(n, x="Date", y="LPR", markers=True, title="Average labour participation rate (%)"),
                    )
    st.plotly_chart(px.imshow(fdf.pivot_table(index="State", columns="Date", values="Unemployment"),
                              aspect="auto", color_continuous_scale="Reds",
                              title="Heatmap: state x month unemployment (%)"))

# ---------- 3 covid ----------
with tabs[2]:
    st.subheader("Impact of COVID-19")
    st.caption("Phases: Pre-COVID (before Mar 2020) | Lockdown (Mar-May 2020) | Recovery (Jun 2020 onwards)")
    imp = a.covid_impact(fdf)
    phase = fdf.groupby("Phase", as_index=False)["Unemployment"].mean()
    phase["Phase"] = pd.Categorical(phase["Phase"], ["Pre-COVID", "Lockdown", "Recovery"], ordered=True)
    st.plotly_chart(px.bar(phase.sort_values("Phase"), x="Phase", y="Unemployment", color="Phase",
                           title="Average unemployment by COVID phase (%)"))
    st.plotly_chart(px.bar(imp.reset_index().dropna(subset=["Rise (pp)"]), x="State", y="Rise (pp)",
                           title="Rise in unemployment: lockdown vs pre-COVID (percentage points)"),
                    )
    st.dataframe(imp.round(2))

# ---------- 4 seasonality ----------
with tabs[3]:
    st.subheader("Seasonal patterns")
    s = a.seasonality(fdf)
    if len(s):
        st.plotly_chart(px.bar(s, x="MonthName", y="Unemployment", title="Avg unemployment by month (pre-COVID data)"),
                        )
    st.plotly_chart(px.box(fdf, x="MonthName", y="Unemployment", category_orders={"MonthName": a.MONTHS},
                           title="Distribution by calendar month (all data)"))
    st.info("Note: the Kaggle dataset covers only ~17 months, so seasonal conclusions are indicative, not definitive.")

# ---------- 5 regions ----------
with tabs[4]:
    st.subheader("Regional comparison")
    avg = fdf.groupby("State", as_index=False)["Unemployment"].mean().sort_values("Unemployment", ascending=False)
    st.plotly_chart(px.bar(avg, x="State", y="Unemployment", title="Average unemployment by state (%)"),
                    )
    if "Zone" in fdf.columns:
        st.plotly_chart(px.box(fdf, x="Zone", y="Unemployment", color="Zone", title="By zone"),
                        )
    if {"Latitude", "Longitude"} <= set(fdf.columns):
        m = fdf.groupby(["State", "Latitude", "Longitude"], as_index=False)["Unemployment"].mean()
        st.plotly_chart(px.scatter_geo(m, lat="Latitude", lon="Longitude", size="Unemployment", color="Unemployment",
                                       hover_name="State", scope="asia", color_continuous_scale="Reds",
                                       title="Map of average unemployment"))

# ---------- 6 prediction ----------
with tabs[5]:
    st.subheader("Predict unemployment rate")
    model, st_list, met, fi = a.train_model(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Random Forest")
    c2.metric("MAE (test)", f"{met['MAE']:.2f} pp")
    c3.metric("R² (test)", f"{met['R2']:.2f}")
    fi_df = pd.DataFrame({"Feature": fi.index, "Importance": fi.values})
    st.plotly_chart(px.bar(fi_df, x="Feature", y="Importance", title="Feature importance"))
    st.markdown("**Try a scenario**")
    c1, c2, c3 = st.columns(3)
    s_ = c1.selectbox("State", st_list)
    mo = c1.selectbox("Month", range(1, 13), format_func=lambda m: a.MONTHS[m - 1])
    emp = c2.number_input("Estimated employed", value=int(df.Employed.median()), step=100000)
    lpr = c2.slider("Labour participation rate (%)", 20.0, 80.0, float(df.LPR.median()))
    lock = c3.checkbox("Lockdown conditions", value=False)
    if st.button("Predict"):
        st.success(f"Predicted unemployment rate: **{a.predict(model, st_list, s_, mo, emp, lpr, lock):.2f}%**")
    fc = a.trend_forecast(df)
    if fc is not None:
        hist = a.national_monthly(df); hist = hist[hist.Date >= "2020-06-01"].assign(Type="Actual")
        fig = px.line(pd.concat([hist[["Date", "Unemployment", "Type"]], fc.assign(Type="Forecast")]),
                      x="Date", y="Unemployment", color="Type", markers=True,
                      title="3-month trend forecast (national average, indicative only)")
        st.plotly_chart(fig)

# ---------- 7 insights ----------
with tabs[6]:
    st.subheader("Key insights & policy suggestions")
    for line in a.insights(fdf):
        st.markdown("- " + line)
