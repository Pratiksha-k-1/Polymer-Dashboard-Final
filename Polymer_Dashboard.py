# ===========================================================
# Polymer Price Intelligence Dashboard
# Decision-support tool for market monitoring and price outlook
# ===========================================================

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import json
import plotly.graph_objects as go


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "all_data")
SENTIMENT_FILE = os.path.join(DATA_DIR, "sentiment_data_high_quality.csv")

#BASE_MODEL_DIR = "models"
BASE_MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_forecast(polymer: str, horizon: int) -> pd.DataFrame:
    path = os.path.join(
        BASE_MODEL_DIR,
        polymer,
        f"horizon_{horizon}w",
        "forecast.csv"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"No forecast found for {polymer} {horizon}w")
    return pd.read_csv(path)


def load_model_meta(polymer: str, horizon: int) -> dict:
    path = os.path.join(
        BASE_MODEL_DIR,
        polymer,
        f"horizon_{horizon}w",
        "best_model.json"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model meta found for {polymer} {horizon}w")
    with open(path, "r") as f:
        return json.load(f)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===========================================================
# PATH CONFIGURATION (OS-AGNOSTIC)
# ===========================================================
INDICATOR_FILE = os.path.join(DATA_DIR, "merged_final_indicators_weekly.csv")
MONTHLY_FILE   = os.path.join(DATA_DIR, "merged_final_indicators_monthly.csv")
FULL_FILE      = os.path.join(DATA_DIR, "merged_final_indicators.csv")
SPOT_FILE      = os.path.join(DATA_DIR, "polymer_spot_timeseries.csv")
CONTRACT_FILE  = os.path.join(DATA_DIR, "polymer_contract_timeseries.csv")

for name, path in {
    "INDICATOR_FILE": INDICATOR_FILE,
    "MONTHLY_FILE": MONTHLY_FILE,
    "FULL_FILE": FULL_FILE,
    "SPOT_FILE": SPOT_FILE,
    "CONTRACT_FILE": CONTRACT_FILE,
}.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at {path}")

# ===========================================================
# STREAMLIT CONFIGURATION
# ===========================================================
st.set_page_config(
    page_title="Polymer Price Intelligence Dashboard",
    layout="wide"
)

st.title("Polymer Price Intelligence Dashboard")
st.caption(
    "Internal decision-support tool for polymer price monitoring, "
    "driver analysis, and short-term price outlook."
)
st.caption(
    "This dashboard supports commercial and procurement decisions by "
    "combining historical pricing, market indicators, and quantitative models. "
    "Outputs are indicative and should be interpreted alongside market intelligence."
)

# ===========================================================
# PROXY DEFINITIONS (ADDED — DOES NOT CHANGE ORIGINAL OUTPUTS)
# ===========================================================
ENERGY_PROXIES = [
    "Brent Oil",
    "US Crude Oil",
    "Crude Oil",
    "Natural Gas",
    "TTF Gas",
]

# Simple substitution map: selected polymer keyword -> proxy polymer series
# (Only used if those columns exist in your dataset)
CROSS_GRADE_PROXY = {
    "pp copo": "PP Homo Spot",
    "pp copolymer": "PP Homo Spot",
    "lldpe": "HDPE Spot",
    "ldpe": "HDPE Spot",
}

# ===========================================================
# HELPER FUNCTIONS
# ===========================================================
def load_with_date(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next(
        (c for c in df.columns if c.lower() in ["date", "day", "timestamp"]),
        None
    )
    if date_col is None:
        raise ValueError(f"No date column found in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).set_index(date_col)


def detect_polymers(columns):
    detected = []
    for c in columns:
        c_clean = re.sub(r"\s+", "", c.lower())
        if "spot" in c_clean or "contract" in c_clean:
            detected.append(c)
    return detected


def polymer_family(name: str) -> str:
    n = name.lower()
    if "hd" in n: return "HDPE"
    if "lldpe" in n: return "LLDPE"
    if "ldpe" in n: return "LDPE"
    if "pp" in n: return "PP"
    if "pvc" in n: return "PVC"
    if "pet" in n: return "PET"
    if "abs" in n: return "ABS"
    return "Other"


def mase(y_true, y_pred, y_insample):
    naive_scale = np.mean(np.abs(np.diff(y_insample)))
    return np.mean(np.abs(y_true - y_pred)) / naive_scale


# ---------- PROXY HELPERS (ADDED) ----------
def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_cross_grade_proxy(selected_polymer_col: str, df: pd.DataFrame) -> str | None:
    s = selected_polymer_col.lower()
    for key, proxy in CROSS_GRADE_PROXY.items():
        if key in s and proxy in df.columns and proxy != selected_polymer_col:
            return proxy
    return None


def find_counterpart_column(selected: str, all_cols: list[str]) -> str | None:
    # "<Grade> Spot" <-> "<Grade> Contract"
    if selected.endswith(" Spot"):
        c = selected.replace(" Spot", " Contract")
        return c if c in all_cols else None
    if selected.endswith(" Contract"):
        c = selected.replace(" Contract", " Spot")
        return c if c in all_cols else None
    return None


# ===========================================================
# DATA LOADING
# ===========================================================
@st.cache_data
def load_all_data():
    indicators = load_with_date(INDICATOR_FILE)
    spot = load_with_date(SPOT_FILE).resample("W-FRI").last()
    contract = load_with_date(CONTRACT_FILE).resample("W-FRI").last()
    df = indicators.join([spot, contract], how="outer")
    df = df.sort_index()
    return df.select_dtypes(include="number")


df = load_all_data()

# ===========================================================
# COLUMN IDENTIFICATION
# ===========================================================
polymer_cols = detect_polymers(df.columns)
indicator_cols = [c for c in df.columns if c not in polymer_cols]

families = sorted({polymer_family(c) for c in polymer_cols})
family_to_polymers = {
    fam: sorted([c for c in polymer_cols if polymer_family(c) == fam])
    for fam in families
}

# ===========================================================
# SIDEBAR — ANALYSIS PARAMETERS
# ===========================================================
st.sidebar.header("Analysis Parameters")

selected_family = st.sidebar.selectbox("Polymer family", families)
polymer_col = st.sidebar.selectbox(
    "Polymer grade",
    family_to_polymers[selected_family]
)

forecast_horizon = st.sidebar.slider(
    "Forecast horizon (weeks ahead)",
    min_value=1,
    max_value=12,
    value=4
)

lookback_window = st.sidebar.slider(
    "Historical lookback window (weeks)",
    min_value=26,
    max_value=208,
    value=104,
    step=26
)

z_window = st.sidebar.slider(
    "Normalization window (z-score)",
    min_value=4,
    max_value=52,
    value=13
)

top_k_indicators = st.sidebar.slider(
    "Number of key drivers used in model",
    min_value=3,
    max_value=min(15, len(indicator_cols)) if len(indicator_cols) > 0 else 3,
    value=8 if len(indicator_cols) > 0 else 3
)

# ===========================================================
# DATA PREPARATION
# ===========================================================
analysis_df = df.tail(lookback_window).copy()

# Target
analysis_df["future_price"] = analysis_df[polymer_col].shift(-forecast_horizon)

# Lagged price features (keep your original naming style)
for lag in [1, 4, 12]:
    analysis_df[f"{polymer_col}lag{lag}"] = analysis_df[polymer_col].shift(lag)

# Rolling momentum features
analysis_df[f"{polymer_col}_ma_4"] = analysis_df[polymer_col].rolling(4).mean()
analysis_df[f"{polymer_col}_ma_12"] = analysis_df[polymer_col].rolling(12).mean()

# Seasonality
iso_week = analysis_df.index.isocalendar().week.astype(int)
analysis_df["week_sin"] = np.sin(2 * np.pi * iso_week / 52)
analysis_df["week_cos"] = np.cos(2 * np.pi * iso_week / 52)
analysis_df["quarter"] = analysis_df.index.quarter

# ===========================================================
# ✅ PROXIES INTEGRATION (ADDED — NO CHANGE TO ORIGINAL FORECASTING)
# ===========================================================

# A) Energy proxies (only if those columns exist)
energy_cols_in_df = []
for e in ENERGY_PROXIES:
    col = find_first_existing_column(analysis_df, [e])
    if col is not None:
        energy_cols_in_df.append(col)
energy_cols_in_df = sorted(set(energy_cols_in_df))

for e in energy_cols_in_df:
    analysis_df[f"{e}_change_1"] = analysis_df[e].pct_change(1, fill_method=None)
    analysis_df[f"{e}_change_4"] = analysis_df[e].pct_change(4, fill_method=None)

# B) Cross-grade substitution proxy
cross_proxy_col = choose_cross_grade_proxy(polymer_col, df)
if cross_proxy_col is not None:
    analysis_df[f"{cross_proxy_col}_lag_1"] = analysis_df[cross_proxy_col].shift(1)
    analysis_df[f"{cross_proxy_col}_change_4"] = analysis_df[cross_proxy_col].pct_change(4, fill_method=None)

# C) Contract–spot spread proxy (only if both exist)
counterpart_col = find_counterpart_column(polymer_col, polymer_cols)
analysis_df["contract_spot_spread"] = np.nan
if counterpart_col is not None and counterpart_col in analysis_df.columns:
    # Determine which is spot vs contract
    if polymer_col.endswith(" Spot"):
        spot_col = polymer_col
        contract_col = counterpart_col
    elif polymer_col.endswith(" Contract"):
        spot_col = counterpart_col
        contract_col = polymer_col
    else:
        spot_col, contract_col = None, None

    if spot_col is not None and contract_col is not None:
        analysis_df["contract_spot_spread"] = analysis_df[contract_col] - analysis_df[spot_col]
        analysis_df["spread_lag_1"] = analysis_df["contract_spot_spread"].shift(1)
        analysis_df["spread_change_4"] = analysis_df["contract_spot_spread"].diff(4)

# ---- keep your original dropna behavior (but now includes proxies too) ----
#analysis_df = analysis_df.dropna()

# Returns (kept)
returns = analysis_df.pct_change(fill_method=None)
returns["future_return"] = analysis_df["future_price"].pct_change(fill_method=None)

# ===========================================================
# LEADING RELATIONSHIPS
# ===========================================================
corr_rows = []

for col in indicator_cols:
    valid = returns[[col, "future_return"]].dropna()
    corr = valid[col].corr(valid["future_return"]) if len(valid) > 10 else np.nan
    corr_rows.append({"Indicator": col, "Leading relationship": corr})

corr_df = (
    pd.DataFrame(corr_rows)
    .dropna()
    .set_index("Indicator")
    .sort_values("Leading relationship", ascending=False)
)

# ===========================================================
# Z-SCORE POSITIONING
# ===========================================================
z_scores = df[polymer_cols].rolling(z_window).apply(
    lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else np.nan
)
latest_z = z_scores.iloc[-1]

# ===========================================================
# NAVIGATION TABS
# ===========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market Overview",
    "Key Drivers",
    "Price Outlook",
    "Model Validation",
    "Qualitative Signals"
])

# ===========================================================
# TAB 1 — MARKET OVERVIEW
# ===========================================================
with tab1:
    st.subheader("Market Overview")

    current_price = df[polymer_col].iloc[-1]
    price_1m_ago = df[polymer_col].iloc[-5] if len(df) >= 5 else np.nan

    delta_abs = current_price - price_1m_ago
    delta_pct = (delta_abs / price_1m_ago * 100) if price_1m_ago != 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Current price level", f"{current_price:,.0f}")
    c2.metric("Change vs. 1 month", f"{delta_abs:,.0f}", f"{delta_pct:.1f}%")
    c3.metric("Market positioning (z-score)", f"{latest_z[polymer_col]:.2f}")

    with st.expander("What does market positioning (Z-score) mean?"):
        st.markdown(
            """
            *Z-score explains how unusual the current price level is compared to recent history.*

            • *Z ≈ 0* → Price is close to its historical average  
            • *Z > +1* → Price is elevated (tight market conditions)  
            • *Z < –1* → Price is depressed (weak demand or oversupply)
            """
        )

    current_week = int(df.index[-1].isocalendar().week)
    current_quarter = int(df.index[-1].quarter)

    st.markdown(
        f"""
        *Seasonal context*

        The current observation falls in calendar week *{current_week}*
        (Q{current_quarter}).
        """
    )

    short_vol = df[polymer_col].pct_change(fill_method=None).rolling(13).std().iloc[-1]
    long_vol = df[polymer_col].pct_change(fill_method=None).rolling(52).std().iloc[-1]

    regime = "Elevated volatility regime" if short_vol > long_vol else "Normal volatility regime"
    st.info(f"Current market regime: {regime}")

    st.subheader("Polymer portfolio overview")

    portfolio_rows = []
    for p in polymer_cols:
        current = df[p].iloc[-1]
        prev_4w = df[p].iloc[-5] if len(df) >= 5 else np.nan
        change_pct = (current - prev_4w) / prev_4w * 100 if prev_4w != 0 else np.nan
        z = latest_z[p]

        short_vol_p = df[p].pct_change(fill_method=None).rolling(13).std().iloc[-1]
        long_vol_p = df[p].pct_change(fill_method=None).rolling(52).std().iloc[-1]
        vol_regime = "Elevated" if short_vol_p > long_vol_p else "Normal"

        if z >= 1:
            signal = "Tightening"
        elif z <= -1:
            signal = "Weakening"
        else:
            signal = "Balanced"

        portfolio_rows.append({
            "Polymer": p,
            "Current price": round(current, 0),
            "4W change (%)": round(change_pct, 1),
            "Z-score": round(z, 2),
            "Volatility regime": vol_regime,
            "Market signal": signal
        })

    portfolio_df = pd.DataFrame(portfolio_rows)
    st.dataframe(
        portfolio_df.style.background_gradient(
            subset=["4W change (%)", "Z-score"],
            cmap="RdYlGn"
        ),
        use_container_width=True
    )

    st.subheader("Markets requiring attention")

    attention_df = portfolio_df.copy()

    def attention_flag(row):
        if row["Z-score"] > 1.5 and row["Volatility regime"] == "Elevated":
            return "Procurement risk"
        if row["Z-score"] < -1.5:
            return "Buyer advantage"
        if row["Volatility regime"] == "Elevated":
            return "High uncertainty"
        return "Monitor"

    attention_df["Attention flag"] = attention_df.apply(attention_flag, axis=1)

    st.dataframe(
        attention_df[["Polymer", "Attention flag"]],
        use_container_width=True
    )

    price_df = df[[polymer_col]].reset_index()
    fig_price = px.line(
        price_df,
        x=price_df.columns[0],
        y=polymer_col,
        title=f"{polymer_col} – Market Price Development",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Proxy features (explicit)")
    
    st.markdown(
        """
        Because monomer benchmarks are not publicly available, the model includes:
        - *Energy cost proxies* (oil, gas)
        - *Cross-grade substitution* (where applicable)
        - *Contract-spot spread* (when both are available)
        """
    )


# ===========================================================
# TAB 2 — KEY DRIVERS
# ===========================================================
with tab2:
    st.subheader("Key Market Drivers")

    st.markdown(
        """
        The table below highlights indicators that historically lead future
        polymer price movements over the selected horizon.
        """
    )

    st.dataframe(
        corr_df.style.background_gradient(cmap="RdYlGn"),
        use_container_width=True
    )

    st.subheader("Systemic drivers impacting multiple polymers")
    driver_summary = []

    for ind in indicator_cols:
        affected = 0
        strengths = []

        for p in polymer_cols:
            tmp = df[[ind, p]].pct_change(fill_method=None).dropna()
            if len(tmp) > 20:
                corr = tmp[ind].corr(tmp[p])
                if abs(corr) > 0.2:
                    affected += 1
                    strengths.append(abs(corr))

        if affected > 0:
            driver_summary.append({
                "Indicator": ind,
                "Polymers impacted": affected,
                "Avg influence strength": round(np.mean(strengths), 2)
            })

    drivers_df = (
        pd.DataFrame(driver_summary)
        .sort_values(["Polymers impacted", "Avg influence strength"], ascending=False)
        .head(10)
    )

    st.dataframe(drivers_df, use_container_width=True)

    # ✅ This section previously referenced undefined variables — now it works.
    st.subheader("Active proxy signals in this configuration")
    active_energy = [e for e in energy_cols_in_df if f"{e}_change_4" in analysis_df.columns]
    st.write(f"- Energy proxies detected: {', '.join(active_energy) if active_energy else 'None'}")
    st.write(f"- Cross-grade proxy used: *{cross_proxy_col}*" if cross_proxy_col else "- Cross-grade proxy used: None")
    st.write("- Contract–spot spread features: Enabled" if analysis_df["contract_spot_spread"].notna().any()
             else "- Contract–spot spread features: Not available for this series")


# ===========================================================
# TAB 3 — PRICE OUTLOOK
# ===========================================================
# The RandomForest model below is NOT used to generate forecasts.

with tab3:
    st.subheader("Price Outlook (Quantitative Model)")
   # st.subheader("Price Outlook – Driver Attribution Model (RandomForest)")

    top_indicators = corr_df.head(top_k_indicators).index.tolist()

    # ---- FIX: your original lag feature names did not match what you created ----
    lag_features = [
        polymer_col,
        f"{polymer_col}lag1",
        f"{polymer_col}lag4",
        f"{polymer_col}lag12",
        f"{polymer_col}_ma_4",
        f"{polymer_col}_ma_12",
    ]

    seasonal_features = ["week_sin", "week_cos", "quarter"]

    # ===========================================================
    # ✅ ADD PROXY FEATURES INTO THE EXPLANATORY MODEL FEATURE SET
    # (Does NOT affect offline forecast generation)
    # ===========================================================
    proxy_features = []

    for e in energy_cols_in_df:
        for suf in ["_change_1", "_change_4"]:
            c = f"{e}{suf}"
            if c in analysis_df.columns:
                proxy_features.append(c)

    if cross_proxy_col is not None:
        for suf in ["_lag_1", "_change_4"]:
            c = f"{cross_proxy_col}{suf}"
            if c in analysis_df.columns:
                proxy_features.append(c)

    if analysis_df["contract_spot_spread"].notna().any():
        for c in ["contract_spot_spread", "spread_lag_1", "spread_change_4"]:
            if c in analysis_df.columns:
                proxy_features.append(c)

    feature_cols = top_indicators + lag_features + seasonal_features + proxy_features
    feature_cols = [c for c in feature_cols if c in analysis_df.columns]

    #model_df = analysis_df[feature_cols + ["future_price"]].copy()
    #model_df = model_df.rename(columns={"future_price": "target"}).dropna()
    model_df = (
        analysis_df[feature_cols + ["future_price"]]
        .rename(columns={"future_price": "target"})
        .dropna()
    )

    # --- SAFETY GUARD: explanatory model needs enough data ---
    if len(model_df) < 30:
        st.warning(
            "Not enough historical observations to train the explanatory "
            "RandomForest model with the current configuration "
            f"(only {len(model_df)} rows available)."
        )
        st.stop()

    st.caption(f"Explanatory model sample size: {len(model_df)} weekly observations")    

    split_idx = int(len(model_df) * 0.7)
    train, test = model_df.iloc[:split_idx], model_df.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    # NOTE:
    # The RandomForest model below is NOT used to generate forecasts.
    # It is trained only to explain historical driver relationships.

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mase_val = mase(y_test.values, y_pred, y_train.values)

    st.caption(
        "The Random Forest model is used for explanatory analysis only. "
        "Forecast values shown below are generated using the offline "
        "best-performing time-series model."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean absolute error", f"{mae:,.1f}")
    c2.metric("Explained variance (R²)", f"{r2:.3f}")
    c3.metric("MASE", f"{mase_val:.3f}")

    st.markdown(
        """
        *Interpretation*

        - MASE below 1 indicates performance superior to a naïve price persistence benchmark.
        - Results are intended as directional guidance, not exact point forecasts.
        """
    )

    st.subheader("Key drivers in current price outlook")

    importance_df = pd.DataFrame({
        "Driver": feature_cols,
        "Relative importance": model.feature_importances_
    }).sort_values("Relative importance", ascending=False).head(10)

    st.dataframe(importance_df, use_container_width=True)
    #model-table
    

    # ===========================================================
    # SINGLE-STEP FORECAST (Original)
    # ===========================================================
    # ===========================================================
    # AutoGluon canonical multi-horizon forecast dataframe
    # ===========================================================

    available_horizons = [1, 2, 3, 4,5,6,7, 8,9, 10, 11, 12]  # only horizons you actually trained
    def extract_autogluon_return(f_df):
        for col in ["mean", "prediction", "forecast"]:
            if col in f_df.columns:
                vals = pd.to_numeric(f_df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    return float(vals.iloc[-1])
        return None

    forecast_rows = []
    last_date = df.index[-1]
    current_price = df[polymer_col].iloc[-1]

    for h in available_horizons:
        try:
            f_df = load_forecast(polymer_col.replace(" ", "_"), h)
            forecast_return = extract_autogluon_return(f_df)

            if forecast_return is None:
                continue

            forecast_price = current_price * (1 + forecast_return)

            forecast_rows.append({
                "horizon_weeks": h,
                "date": last_date + pd.Timedelta(weeks=h),
                "return": forecast_return,
                "price": forecast_price
            })

        except Exception:
            continue


    if len(forecast_rows) == 0:
        st.error("No AutoGluon forecasts available for this polymer.")
        st.stop()

    forecast_df = (
        pd.DataFrame(forecast_rows)
        .sort_values("horizon_weeks")
        .reset_index(drop=True)
    )
    # ===========================================================
    # Single-Step Price Forecast (AutoGluon, exact table)
    # ===========================================================

    st.subheader("Single-Step Price Forecast")

    # --- pick AutoGluon horizon closest to user selection ---
    selected_row = forecast_df.iloc[
        (forecast_df["horizon_weeks"] - forecast_horizon).abs().argmin()
    ]

    forecast_price = selected_row["price"]
    st.session_state["model_forecast"] = float(forecast_price)


    ret_std = forecast_df["return"].std()

    lower_ci = current_price * (1 + selected_row["return"] - 1.96 * ret_std)
    upper_ci = current_price * (1 + selected_row["return"] + 1.96 * ret_std)


    outlook_rows = [
        {
            "Time Horizon": "Current (Latest Data)",
            "Date": df.index[-1].strftime("%Y-%m-%d"),
            "Price Level": round(current_price, 0),
            "Lower 95% CI": None,
            "Upper 95% CI": None,
            "Market Signal": "Current level"
        },
        {
            "Time Horizon": f"{int(selected_row['horizon_weeks'])} weeks ahead",
            "Date": selected_row["date"].strftime("%Y-%m-%d"),
            "Price Level": round(forecast_price, 0),
            "Lower 95% CI": round(lower_ci, 0),
            "Upper 95% CI": round(upper_ci, 0),
            "Market Signal": (
                "Upward pressure"
                if forecast_price > current_price * 1.02
                else "Downward pressure"
                if forecast_price < current_price * 0.98
                else "Stable outlook"
            )
        }
    ]

    outlook_df = pd.DataFrame(outlook_rows)
    st.dataframe(outlook_df, use_container_width=True)

    # ===========================================================
    # EXTENDED PRICE OUTLOOK (AutoGluon)
    # ===========================================================

    st.subheader("Extended Price Outlook")

    fig = go.Figure()

    # Historical prices (last ~6 months)
    hist_df = df[[polymer_col]].tail(26)
    fig.add_trace(go.Scatter(
        x=hist_df.index,
        y=hist_df[polymer_col],
        mode="lines",
        name="Historical Price"
    ))

    # AutoGluon forecast path
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["price"],
        mode="lines+markers",
        name="AutoGluon Forecast",
        line=dict(dash="dash")
    ))

    forecast_start = df.index[-1]

    # Vertical line
    fig.add_shape(
        type="line",
        x0=forecast_start,
        x1=forecast_start,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="gray", dash="dot")
    )

    # Annotation
    fig.add_annotation(
        x=forecast_start,
        y=1,
        xref="x",
        yref="paper",
        text="Forecast start",
        showarrow=False,
        yshift=10
    )



    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price level",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===========================================================
    # Multi-Month Forecast Summary with Confidence Bands (AutoGluon)
    # ===========================================================

    st.subheader("Multi-Month Forecast Summary")

    # --- estimate return volatility across horizons ---
    ret_std = forecast_df["return"].std()

    monthly_df = (
        forecast_df
        .assign(month=lambda x: x["date"].dt.to_period("M"))
        .groupby("month")
        .agg(
            Avg_Return=("return", "mean"),
            Min_Return=("return", "min"),
            Max_Return=("return", "max")
        )
        .reset_index()
    )

    # Convert returns → prices
    monthly_df["Avg_Price"] = current_price * (1 + monthly_df["Avg_Return"])
    monthly_df["Lower_95_CI"] = current_price * (1 + monthly_df["Avg_Return"] - 1.96 * ret_std)
    monthly_df["Upper_95_CI"] = current_price * (1 + monthly_df["Avg_Return"] + 1.96 * ret_std)

    # Format month
    monthly_df["Month"] = monthly_df["month"].dt.strftime("%B %Y")

    # Rounding
    monthly_df["Avg_Price"] = monthly_df["Avg_Price"].round(0)
    monthly_df["Lower_95_CI"] = monthly_df["Lower_95_CI"].round(0)
    monthly_df["Upper_95_CI"] = monthly_df["Upper_95_CI"].round(0)

    # Change vs current
    monthly_df["Change_vs_Current"] = monthly_df["Avg_Price"] - round(current_price, 0)
    monthly_df["Change_%"] = (
        (monthly_df["Avg_Price"] / current_price - 1) * 100
    ).round(1)

    # Final table
    monthly_df = monthly_df[
        [
            "Month",
            "Avg_Price",
            "Lower_95_CI",
            "Upper_95_CI",
            "Change_vs_Current",
            "Change_%"
        ]
    ]

    st.dataframe(monthly_df, use_container_width=True)

    st.caption(
        "Confidence bands are derived from the dispersion of AutoGluon return forecasts "
        "across horizons and converted to price levels."
    )
    # ===========================================================
    # Key Insights (AutoGluon-based)
    # ===========================================================

    st.subheader("Key Insights")

    # Overall trend based on furthest horizon
    final_price = forecast_df.loc[forecast_df["horizon_weeks"].idxmax(), "price"]

    if final_price > current_price * 1.02:
        price_trend = "Upward"
    elif final_price < current_price * 0.98:
        price_trend = "Downward"
    else:
        price_trend = "Stable"

    # Extremes
    max_price = forecast_df["price"].max()
    min_price = forecast_df["price"].min()

    max_row = forecast_df.loc[forecast_df["price"].idxmax()]
    min_row = forecast_df.loc[forecast_df["price"].idxmin()]

    st.markdown(
        f"""
    - **Overall price trend**: {price_trend}
    - **Highest expected level**: €{max_price:,.0f} around {max_row['date'].strftime('%B %Y')}
    - **Lowest expected level**: €{min_price:,.0f} around {min_row['date'].strftime('%B %Y')}
    - **Expected variation range**: ±{((max_price - min_price) / current_price * 100):.1f}% relative to current levels
    """
    )

# ===========================================================
# TAB 4 — MODEL VALIDATION (AutoGluon-only, RF removed)
# ===========================================================
with tab4:
    st.subheader("Model Validation (Forecast Engine)")

    st.markdown(
        """
        This section validates the **production forecasting engine** using a leakage-safe
        **rolling-origin backtest** and compares it against transparent baselines.
        """
    )

    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    
    # -------------------------------------------------------
    # Baselines
    # -------------------------------------------------------
    def baseline_persistence(current):
        return float(current)

    def baseline_seasonal(series: pd.Series, idx_pos: int, season_lag: int = 52):
        # weekly seasonal naive: use value 52 weeks ago as forecast
        if idx_pos - season_lag >= 0:
            return float(series.iloc[idx_pos - season_lag])
        return np.nan

    def baseline_drift(series: pd.Series, idx_pos: int, h: int, window: int = 26):
        # linear drift using mean weekly change over last N weeks
        start = max(1, idx_pos - window + 1)
        diffs = series.iloc[start:idx_pos+1].diff().dropna()
        if len(diffs) == 0:
            return np.nan
        mu = float(diffs.mean())
        return float(series.iloc[idx_pos] + h * mu)

    def compute_metrics(y_true, y_pred, y_insample_for_mase=None):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5

        # MAPE can be unstable if prices near 0; polymers usually fine, still guard
        denom = np.where(np.array(y_true) == 0, np.nan, np.array(y_true))
        mape = np.nanmean(np.abs((np.array(y_true) - np.array(y_pred)) / denom)) * 100

        # MASE relative to naive one-step differences (scale)
        mase_val = np.nan
        if y_insample_for_mase is not None:
            diffs = np.abs(np.diff(y_insample_for_mase))
            scale = np.mean(diffs) if len(diffs) > 0 else np.nan
            if scale and not np.isnan(scale) and scale != 0:
                mase_val = np.mean(np.abs(np.array(y_true) - np.array(y_pred))) / scale

        return mae, rmse, mape, mase_val

    def direction_accuracy(current_series, y_true, y_pred):
        # compares sign(pred - current) vs sign(actual - current)
        cur = np.array(current_series)
        yt = np.array(y_true)
        yp = np.array(y_pred)
        return (np.sign(yp - cur) == np.sign(yt - cur)).mean() * 100

    
    # -------------------------------------------------------
    # AutoGluon Rolling Backtest
    # -------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def autogluon_rolling_backtest(
        series: pd.Series,
        horizon_weeks: int,
        backtest_points: int = 20,
        min_train_points: int = 80,
        freq: str = "W-FRI",
        presets: str = "medium_quality",
        time_limit: int | None = 60,
    ) -> pd.DataFrame:
        """
        Rolling-origin backtest for a single series.
        Returns a dataframe with per-asof forecasts and errors.

        Requires autogluon.timeseries. If not installed, raises ImportError.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        # ensure datetime index and frequency alignment
        s = series.dropna().copy()
        s = s.asfreq(freq, method="pad")

        # choose as-of dates: last `backtest_points` cutoffs that still have t+h available
        max_i = len(s) - 1 - horizon_weeks
        start_i = max(min_train_points - 1, max_i - backtest_points + 1)
        cut_idxs = list(range(start_i, max_i + 1))

        rows = []
        for i in cut_idxs:
            asof_date = s.index[i]
            target_date = s.index[i + horizon_weeks]
            current = float(s.iloc[i])
            actual = float(s.iloc[i + horizon_weeks])

            train_s = s.iloc[: i + 1]  # up to asof
            # Build AG TSDF
            train_df = pd.DataFrame({
                "item_id": ["series_1"] * len(train_s),
                "timestamp": train_s.index,
                "target": train_s.values
            })
            train_tsdf = TimeSeriesDataFrame.from_data_frame(
                train_df, id_column="item_id", timestamp_column="timestamp"
            )

            predictor = TimeSeriesPredictor(
                target="target",
                prediction_length=horizon_weeks,
                freq=freq,
                eval_metric="MASE",
            )

            predictor.fit(
                train_tsdf,
                presets=presets,
                time_limit=time_limit
            )

            # Forecast next h steps; we take the last step as t+h
            fcst = predictor.predict(train_tsdf)
            # fcst is indexed by (item_id, timestamp)
            # take the prediction at target_date
            try:
                pred = float(fcst.loc[("series_1", target_date)]["mean"])
            except Exception:
                # fallback: grab last row
                pred = float(fcst.xs("series_1").iloc[-1]["mean"])

            rows.append({
                "asof_date": asof_date,
                "target_date": target_date,
                "current_price": current,
                "actual_future_price": actual,
                "pred_autogluon": pred,
            })

        bt = pd.DataFrame(rows).set_index("asof_date").sort_index()
        bt["err_autogluon"] = bt["pred_autogluon"] - bt["actual_future_price"]
        bt["abs_err_autogluon"] = bt["err_autogluon"].abs()
        return bt

    # -------------------------------------------------------
    # Build the series to validate (weekly)
    # -------------------------------------------------------
    s_full = df[polymer_col].copy()
    s_full = s_full.dropna()

    # Controls
    st.sidebar.subheader("Validation Controls")
    bt_points = st.sidebar.slider("Backtest points", 8, 40, 16, 1)
    min_train = st.sidebar.slider("Min training points", 52, 156, 80, 4)

    use_seasonal = st.sidebar.checkbox("Include seasonal naïve (t-52)", value=True)
    use_drift = st.sidebar.checkbox("Include drift baseline", value=True)

    # -------------------------------------------------------
    # Run backtest
    # -------------------------------------------------------
    st.subheader("Rolling-origin backtest (leakage-safe)")

    # Baselines computed without training
    # We'll align them to the same as-of dates as AutoGluon backtest output
    try:
        with st.spinner("Running AutoGluon rolling backtest..."):
            bt_ag = autogluon_rolling_backtest(
                series=s_full,
                horizon_weeks=forecast_horizon,
                backtest_points=bt_points,
                min_train_points=min_train,
                freq="W-FRI",
                presets="medium_quality",
                time_limit=60,   # keep reasonable for dashboard usage; adjust as needed
            )

        # Build baseline predictions for the same asof rows
        s_aligned = s_full.asfreq("W-FRI", method="pad").dropna()
        idx_to_pos = {t: i for i, t in enumerate(s_aligned.index)}

        preds_naive = []
        preds_seasonal = []
        preds_drift = []
        currents = []

        for asof in bt_ag.index:
            pos = idx_to_pos.get(asof, None)
            if pos is None:
                preds_naive.append(np.nan)
                preds_seasonal.append(np.nan)
                preds_drift.append(np.nan)
                currents.append(np.nan)
                continue

            current = float(s_aligned.iloc[pos])
            currents.append(current)

            preds_naive.append(baseline_persistence(current))
            preds_seasonal.append(baseline_seasonal(s_aligned, pos, season_lag=52) if use_seasonal else np.nan)
            preds_drift.append(baseline_drift(s_aligned, pos, forecast_horizon, window=26) if use_drift else np.nan)

        bt = bt_ag.copy()
        bt["pred_naive"] = preds_naive
        if use_seasonal:
            bt["pred_seasonal"] = preds_seasonal
        if use_drift:
            bt["pred_drift"] = preds_drift

        # Metrics table
        y_true = bt["actual_future_price"].values
        y_insample_for_mase = s_aligned.values  # scale reference

        rows = []

        # AutoGluon
        y_pred = bt["pred_autogluon"].values
        mae, rmse, mape, mase_val = compute_metrics(y_true, y_pred, y_insample_for_mase)
        dir_acc = direction_accuracy(bt["current_price"].values, y_true, y_pred)
        rows.append({
            "Model": "AutoGluon (forecast engine)",
            "N forecasts": len(bt),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_%": mape,
            "MASE": mase_val,
            "Direction_Accuracy_%": dir_acc
        })

        # Naive
        y_pred = bt["pred_naive"].values
        mae, rmse, mape, mase_val = compute_metrics(y_true, y_pred, y_insample_for_mase)
        dir_acc = direction_accuracy(bt["current_price"].values, y_true, y_pred)
        rows.append({
            "Model": "Naïve persistence",
            "N forecasts": len(bt),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_%": mape,
            "MASE": mase_val,
            "Direction_Accuracy_%": dir_acc
        })

        
        metrics_df = pd.DataFrame(rows)

        st.dataframe(
            metrics_df.style.format({
                "MAE": "{:,.1f}",
                "RMSE": "{:,.1f}",
                "MAPE_%": "{:.2f}",
                "MASE": "{:.3f}",
                "Direction_Accuracy_%": "{:.1f}",
            }),
            use_container_width=True
        )

        st.caption(
            "MASE < 1 means better than a naïve one-step-change scale baseline. "
            "Comparisons are leakage-safe and computed on the same rolling as-of dates."
        )

        # Plot: actual vs predictions
        st.subheader(f"Backtest: Actual vs Predicted (h={forecast_horizon} weeks)")
        plot_cols = ["actual_future_price", "pred_autogluon", "pred_naive"]
        if use_seasonal and "pred_seasonal" in bt.columns:
            plot_cols.append("pred_seasonal")
        if use_drift and "pred_drift" in bt.columns:
            plot_cols.append("pred_drift")

        plot_df = bt.reset_index()[["asof_date"] + plot_cols]

        fig1 = px.line(
            plot_df,
            x="asof_date",
            y=plot_cols,
            title="Rolling-origin forecast comparison",
            labels={"asof_date": "As-of date", "value": "Future price level", "variable": "Series"}
        )
        st.plotly_chart(fig1, use_container_width=True)


        # Absolute error over time
        st.subheader("Absolute Error Over Time")
        abs_long = []
        for col, name in [
            ("pred_autogluon", "AutoGluon"),
            ("pred_naive", "Naïve"),
            ("pred_seasonal", "Seasonal naïve"),
            ("pred_drift", "Drift"),
        ]:
            if col in bt.columns and bt[col].notna().any():
                abs_err = (bt[col] - bt["actual_future_price"]).abs()
                abs_long.append(pd.DataFrame({
                    "asof_date": bt.index,
                    "Absolute Error": abs_err.values,
                    "Model": name
                }))
        abs_long = pd.concat(abs_long, ignore_index=True)

        fig3 = px.line(
            abs_long,
            x="asof_date",
            y="Absolute Error",
            color="Model",
            title="Absolute error through time"
        )
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Backtest details (per as-of date)"):
            show_cols = ["target_date", "current_price", "actual_future_price", "pred_autogluon", "pred_naive"]
            if use_seasonal and "pred_seasonal" in bt.columns:
                show_cols.append("pred_seasonal")
            if use_drift and "pred_drift" in bt.columns:
                show_cols.append("pred_drift")
            st.dataframe(bt[show_cols], use_container_width=True)

    except ImportError:
        st.warning(
            "AutoGluon is not installed in this environment. "
            "To keep the Streamlit app lightweight, run rolling backtests offline "
            "and load results as a CSV in this tab."
        )
        st.info(
            "If you want, I can give you an offline script that generates "
            "backtest_results.csv for each polymer/horizon and this tab will display it."
        )
    except Exception as e:
        st.error(f"AutoGluon rolling backtest failed: {e}")
    #benchmark
    st.subheader("Multi-model benchmarking (AutoGluon)")

    try:
        ag_leaderboard = pd.read_csv(
            os.path.join(
                BASE_MODEL_DIR,
                polymer_col.replace(" ", "_"),
                f"horizon_{forecast_horizon}w",
                "leaderboard.csv"
            )
        )
    except Exception:
        st.warning("Offline model leaderboard not available.")
        ag_leaderboard = None

    if ag_leaderboard is not None and len(ag_leaderboard) > 0:
        st.markdown("*Model performance leaderboard (lower MASE is better)*")
        st.dataframe(
            ag_leaderboard[
                ["model", "score_val", "fit_time_marginal"]
            ].rename(
                columns={
                    "model": "Model",
                    "score_val": "MASE",
                    "fit_time_marginal": "Training time (s)"
                }
            ),
            use_container_width=True
        )

        best_model = ag_leaderboard.iloc[0]["model"]
        best_mase = ag_leaderboard.iloc[0]["score_val"]

        st.success(
            f"Best-performing model: *{best_model}* "
            f"(MASE = {best_mase:.2f})"
        )    

# ===========================================================
with tab5:
    st.subheader("Qualitative Signals")
#elif selected_tab == "Qualitative Signals":

    st.header("Qualitative Signals from News")
    st.caption(
        "This section analyses uploaded weekly news files to extract sentiment, risk intensity, "
        "and implied price pressure signals relevant for polymer markets."
    )

    uploaded_files = st.file_uploader(
        "Upload weekly news files (.txt)",
        type=["txt"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload one or more weekly news text files.")
    else:
        import re
        import pandas as pd
        import numpy as np

        # -------------------------
        # Simple domain dictionaries
        # -------------------------
        positive_words = {
            "increase", "growth", "recovery", "strong", "tightening",
            "demand", "rebound", "support", "bullish", "improve"
        }

        negative_words = {
            "decline", "drop", "weak", "oversupply", "slowdown",
            "bearish", "fall", "pressure", "reduce", "soft"
        }

        risk_words = {
            "shutdown", "outage", "force majeure", "disruption",
            "strike", "sanction", "conflict", "delay", "shortage"
        }

        price_up_words = {
            "price increase", "price hike", "higher prices", "cost push"
        }

        price_down_words = {
            "price cut", "discount", "lower prices", "price pressure"
        }

        results = []

        for file in uploaded_files:
            text = file.read().decode("utf-8").lower()
            tokens = re.findall(r"\b[a-z]+\b", text)

            pos_count = sum(1 for w in tokens if w in positive_words)
            neg_count = sum(1 for w in tokens if w in negative_words)
            risk_count = sum(1 for w in tokens if w in risk_words)

            sentiment_score = (
                (pos_count - neg_count) / max(pos_count + neg_count, 1)
            )

            if sentiment_score > 0.2:
                tone = "Positive"
            elif sentiment_score < -0.2:
                tone = "Negative"
            else:
                tone = "Neutral"

            price_up_signal = any(p in text for p in price_up_words)
            price_down_signal = any(p in text for p in price_down_words)

            if price_up_signal and not price_down_signal:
                price_signal = "Upward pressure"
            elif price_down_signal and not price_up_signal:
                price_signal = "Downward pressure"
            else:
                price_signal = "Neutral / unclear"

            top_keywords = (
                pd.Series(tokens)
                .value_counts()
                .head(5)
                .index
                .tolist()
            )

            results.append({
                "File": file.name,
                "Sentiment score": round(sentiment_score, 3),
                "Tone": tone,
                "Risk intensity (mentions)": risk_count,
                "Price signal from news": price_signal,
                "Key keywords": ", ".join(top_keywords)
            })

        sentiment_df = pd.DataFrame(results)

     #   st.subheader("Per-file qualitative assessment")
        # --- Reduce qualitative table to business-relevant columns ---

        final_sentiment_table = (
            sentiment_df
            .reset_index(drop=True)
            .reset_index()                     # creates serial number
            .rename(columns={
                "index": "No.",
                "File": "File name",
                "Tone": "Tone",
                "Price signal from news": "Market signal"
            })[
                ["No.", "File name", "Tone", "Market signal"]
            ]
        )

        st.subheader("Weekly Market Signals")
        st.dataframe(final_sentiment_table, use_container_width=True)

      #  st.dataframe(sentiment_df, use_container_width=True)

        # -------------------------
        # Aggregated insights
        # -------------------------
        st.subheader("Aggregated market narrative")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Average sentiment",
                round(sentiment_df["Sentiment score"].mean(), 2)
            )

        with col2:
            dominant_tone = sentiment_df["Tone"].mode()[0]
            st.metric("Dominant tone", dominant_tone)
            


        st.caption(
            "Interpretation: Sentiment captures market mood, risk intensity reflects supply-side stress, "
            "and price signals highlight narrative-driven pricing expectations."
        )