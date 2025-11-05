import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import StringIO

st.set_page_config(page_title="Thailand Inflation – Simple View", layout="centered")
st.title("Thailand Headline Inflation YoY – Simple Result Page")

HISTORY_FILE  = "thai_headline_inflation_yoy_annual_yearly.csv"
FORECAST_FILE = "annual_lstm_forecast.csv"

# --- small helpers ------------------------------------------------------------
def _normalize_year(col):
    # cast to int year if it looks like a date
    try:
        return pd.to_datetime(col).dt.year
    except Exception:
        return pd.to_numeric(col, errors="coerce").astype("Int64")

def load_csv_strict(path, year_col="Year", value_col=None):
    df = pd.read_csv(path)
    # try to find Year
    if year_col not in df.columns:
        # guess first column as Year if not present
        df.rename(columns={df.columns[0]: "Year"}, inplace=True)
    else:
        df.rename(columns={year_col: "Year"}, inplace=True)

    # guess value column if not provided
    if value_col is None:
        candidates = [c for c in df.columns if c.lower() not in ("year",)]
        if len(candidates) != 1:
            st.error(f"Please keep exactly two columns in {path}: Year and Value.")
            st.stop()
        value_col = candidates[0]

    df.rename(columns={value_col: "Value"}, inplace=True)
    df["Year"] = _normalize_year(df["Year"])
    return df.dropna(subset=["Year"]).sort_values("Year")

# --- load data (or allow upload if missing) -----------------------------------
try:
    hist_df = load_csv_strict(HISTORY_FILE, value_col=None)
except Exception:
    st.warning("History CSV not found. Upload it below.")
    up = st.file_uploader("Upload history CSV (Year, Value)", type=["csv"], key="hist")
    if up is None:
        st.stop()
    hist_df = load_csv_strict(StringIO(up.getvalue().decode("utf-8")), value_col=None)

try:
    fc_df = load_csv_strict(FORECAST_FILE, value_col=None)
except Exception:
    st.warning("Forecast CSV not found. Upload it below.")
    up2 = st.file_uploader("Upload forecast CSV (Year, Value)", type=["csv"], key="fc")
    if up2 is None:
        st.stop()
    fc_df = load_csv_strict(StringIO(up2.getvalue().decode("utf-8")), value_col=None)

hist_df.rename(columns={"Value": "Inflation_YoY"}, inplace=True)
fc_df.rename(columns={"Value": "Forecast"}, inplace=True)

# --- show tables --------------------------------------------------------------
st.subheader("Historical data")
st.dataframe(hist_df, use_container_width=True)
st.subheader("Forecast data")
st.dataframe(fc_df, use_container_width=True)

# --- simple combined line chart ----------------------------------------------
st.subheader("History and Forecast")
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(hist_df["Year"], hist_df["Inflation_YoY"], label="Actual (History)", linewidth=2)
ax.plot(fc_df["Year"], fc_df["Forecast"], marker="o", label="Forecast", linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Inflation YoY (%)")
ax.grid(True, alpha=0.25)
ax.legend()
st.pyplot(fig, clear_figure=True)

st.caption("Tip: Keep CSVs with two columns only. History: Year, Inflation_YoY. Forecast: Year, Forecast.")
