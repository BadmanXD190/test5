import os
import io
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Thailand Inflation – History & LSTM Forecast",
                   layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    # Try to standardize column names
    cols = {c.lower().strip(): c for c in df.columns}
    # Find a "year" column
    year_col = next((cols[k] for k in cols if k in ("year", "years", "date", "time", "period")), None)
    if year_col is None:
        raise ValueError(f"Could not find a Year column in {path}. "
                         f"Make sure it has a 'Year' column.")
    # If it's a date, convert to year
    if not np.issubdtype(df[year_col].dtype, np.number):
        try:
            df[year_col] = pd.to_datetime(df[year_col], errors="coerce").dt.year
        except Exception:
            pass
    df.rename(columns={year_col: "Year"}, inplace=True)

    # Try common value column names for inflation
    value_candidates = ["inflation", "inflation_yoy", "headline_inflation",
                        "value", "rate", "yoy", "inflation_yoy_percent", "inflation_yoy_%"]
    for k in list(cols):
        if k in value_candidates:
            df.rename(columns={cols[k]: "Inflation_YoY"}, inplace=True)
    if "Inflation_YoY" not in df.columns:
        # Fallback: if there are exactly two columns, assume the second is the value
        if df.shape[1] == 2:
            second = [c for c in df.columns if c != "Year"][0]
            df.rename(columns={second: "Inflation_YoY"}, inplace=True)
    return df.sort_values("Year").reset_index(drop=True)

@st.cache_data
def load_history_and_forecast(history_csv, forecast_csv):
    hist = load_csv(history_csv)

    fc = pd.read_csv(forecast_csv)
    # Normalize forecast file columns
    fc_cols = {c.lower().strip(): c for c in fc.columns}
    year_fc = next((fc_cols[k] for k in fc_cols if k in ("year", "years", "date", "period")), None)
    if year_fc is None:
        raise ValueError("Forecast CSV must contain a 'Year' column.")
    fc.rename(columns={year_fc: "Year"}, inplace=True)

    # Try common forecast column names
    f_candidates = ["forecast", "pred", "prediction", "yhat", "predicted"]
    for k in list(fc_cols):
        if k in f_candidates:
            fc.rename(columns={fc_cols[k]: "Forecast"}, inplace=True)
    if "Forecast" not in fc.columns:
        # If there's only two columns, assume the non-Year column is forecast
        if fc.shape[1] == 2:
            other = [c for c in fc.columns if c != "Year"][0]
            fc.rename(columns={other: "Forecast"}, inplace=True)

    return hist, fc.sort_values("Year").reset_index(drop=True)

def human_note(txt):
    st.info(textwrap.fill(txt, 100))

# -----------------------------
# Load data
# -----------------------------
HISTORY_CSV  = "thai_headline_inflation_yoy_annual_yearly.csv"
FORECAST_CSV = "annual_lstm_forecast.csv"

with st.sidebar:
    st.header("Files")
    st.caption("These should exist in the repo root.")
    st.code(HISTORY_CSV)
    st.code(FORECAST_CSV)
    show_images = st.checkbox("Also show saved PNG charts (if present)", value=True)

try:
    hist_df, fc_df = load_history_and_forecast(HISTORY_CSV, FORECAST_CSV)
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# Derived frames
min_hist_year = int(hist_df["Year"].min())
max_hist_year = int(hist_df["Year"].max())

merged_df = pd.merge(hist_df, fc_df, on="Year", how="outer").sort_values("Year")
# Tag rows
merged_df["Set"] = np.where(merged_df["Year"] <= max_hist_year, "History", "Forecast")

# -----------------------------
# Header
# -----------------------------
st.title("Thailand Headline Inflation YoY – History, Model, and Forecast")
st.caption(f"History {min_hist_year}–{max_hist_year} with LSTM forecast following years in "
           f"{os.path.basename(FORECAST_CSV)}")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📈 Overview chart", "📊 Data tables", "🖼 Saved PNG charts"])

# -----------------------------
# Tab 1: Main chart
# -----------------------------
with tab1:
    c1, c2 = st.columns([2.2, 1.0], vertical_alignment="top")

    with c1:
        fig, ax = plt.subplots(figsize=(11, 5))

        # Plot history
        ax.plot(hist_df["Year"], hist_df["Inflation_YoY"], label="Actual (History)",
                linewidth=2)

        # Simple in-sample smoothing (optional, dashed)
        # If you trained a model and saved in-sample preds, you can add them here.
        # For now, make a rolling mean to visually resemble a dashed line.
        if len(hist_df) >= 3:
            ax.plot(hist_df["Year"], hist_df["Inflation_YoY"].rolling(3, center=True).mean(),
                    linestyle="--", label="Model (in-sample, proxy)")

        # Plot forecast, if present
        if "Forecast" in fc_df.columns:
            ax.plot(fc_df["Year"], fc_df["Forecast"], marker="o", label="Forecast")

            # Annotate last few forecast points
            for y, v in zip(fc_df["Year"].tolist()[-3:], fc_df["Forecast"].tolist()[-3:]):
                ax.annotate(f"{int(y)}", (y, v), textcoords="offset points", xytext=(6, 6))

        # Shaded regions
        ax.axvspan(hist_df["Year"].min(), max_hist_year, alpha=0.07, color="green",
                   label="Train period")
        if len(fc_df):
            ax.axvspan(max_hist_year + 1, fc_df["Year"].max(), alpha=0.07, color="orange",
                       label="Test/Forecast period")

        ax.set_xlabel("Year")
        ax.set_ylabel("Inflation YoY (%)")
        ax.set_title("History, model proxy, and forecast")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
        st.pyplot(fig, clear_figure=True)

    with c2:
        human_note(
            "Tip: If your forecast CSV contains different column names, the app will try to "
            "auto-detect them. Keep columns as 'Year' and 'Forecast' for best results."
        )
        last_hist = hist_df.dropna(subset=["Inflation_YoY"]).tail(5)
        st.subheader("Recent history")
        st.dataframe(last_hist, use_container_width=True)
        if "Forecast" in fc_df.columns:
            st.subheader("Forecast preview")
            st.dataframe(fc_df.head(5), use_container_width=True)

# -----------------------------
# Tab 2: Data tables
# -----------------------------
with tab2:
    st.subheader("Historical data")
    st.dataframe(hist_df, use_container_width=True)
    st.download_button("Download historical CSV",
                       data=hist_df.to_csv(index=False).encode("utf-8"),
                       file_name="history_clean.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Forecast data")
    st.dataframe(fc_df, use_container_width=True)
    st.download_button("Download forecast CSV",
                       data=fc_df.to_csv(index=False).encode("utf-8"),
                       file_name="forecast_clean.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Merged view")
    st.dataframe(merged_df, use_container_width=True)

# -----------------------------
# Tab 3: Saved PNG charts (optional)
# -----------------------------
with tab3:
    st.caption("If you pushed your generated figures to the repo, they’ll appear here.")
    pngs = [
        ("annual_lstm_forecast_full.png", "Full history + forecast"),
        ("annual_lstm_test_plot.png", "Test window: actual vs predicted"),
        ("annual_lstm_residuals.png", "Residuals histogram (test)"),
        ("annual_lstm_learning_curves.png", "Learning curves (MSE loss)")
    ]
    any_found = False
    for path, title in pngs:
        if show_images and os.path.exists(path):
            any_found = True
            st.subheader(title)
            st.image(path, use_container_width=True)
    if show_images and not any_found:
        st.info("No PNGs found in the repo. Push them if you want this tab to show charts.")
