# ===========================================================
# Offline training script (BLACK-BOX AutoGluon)
# Compatible with run_autogluon_forecast(ts_data, horizon)
# ===========================================================

import os
import sys
import json
import pandas as pd

from autogluon_forecasting import (
    prepare_returns_ts,
    run_autogluon_forecast
)

# ===========================================================
# ENVIRONMENT CHECK
# ===========================================================
print("Python version:")
print(sys.version)

print("\nWorking directory:")
print(os.getcwd())

# ===========================================================
# CONFIG
# ===========================================================
TRAINING_MODE = "both"
# options: "spot", "contract", "both"

HORIZONS = list(range(1, 13))   # weeks ahead
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================
# LOAD DATA (MODE-AWARE)
# ===========================================================
print("\nLoading polymer data...")

if TRAINING_MODE == "spot":
    df_raw = pd.read_csv("all_data/polymer_spot_timeseries.csv")

elif TRAINING_MODE == "contract":
    df_raw = pd.read_csv("all_data/polymer_contract_timeseries.csv")

elif TRAINING_MODE == "both":
    spot = pd.read_csv("all_data/polymer_spot_timeseries.csv")
    contract = pd.read_csv("all_data/polymer_contract_timeseries.csv")
    df_raw = spot.merge(contract, on="date", how="outer")

else:
    raise ValueError("Invalid TRAINING_MODE")

df_raw["date"] = pd.to_datetime(df_raw["date"])
df_raw = df_raw.set_index("date").sort_index()

print("Available columns:")
print(df_raw.columns.tolist())

# ===========================================================
# DETECT POLYMER COLUMNS
# ===========================================================
POLYMER_COLUMNS = [
    c for c in df_raw.columns
    if ("spot" in c.lower() or "contract" in c.lower())
    and df_raw[c].dropna().shape[0] >= 52
]

if not POLYMER_COLUMNS:
    raise RuntimeError(
        "No polymers detected for training. "
        "Check TRAINING_MODE and input data."
    )

print("\nDetected polymer columns:")
for c in POLYMER_COLUMNS:
    print("-", c)

# ===========================================================
# TRAINING LOOP
# ===========================================================
for polymer_col in POLYMER_COLUMNS:

    print("\n========================================")
    print(f"Processing polymer: {polymer_col}")
    print("========================================")

    series = df_raw[[polymer_col]].dropna()
    assert len(series) >= 40, f"Not enough data for {polymer_col}"

    polymer_name = polymer_col.replace(" ", "_")
    polymer_dir = os.path.join(OUTPUT_DIR, polymer_name)
    os.makedirs(polymer_dir, exist_ok=True)

    for horizon in HORIZONS:

        print(f"\n--- Forecast horizon: {horizon} weeks ---")

        horizon_dir = os.path.join(polymer_dir, f"horizon_{horizon}w")
        os.makedirs(horizon_dir, exist_ok=True)

        # Prepare time series exactly like dashboard
        ts_data = prepare_returns_ts(
            df_raw,
            polymer_col,
            horizon
        )

        leaderboard, forecast_df = run_autogluon_forecast(
            ts_data,
            horizon
        )

        leaderboard = leaderboard.sort_values("score_val")
        best_row = leaderboard.iloc[0]

        best_model_info = {
            "polymer": polymer_col,
            "horizon_weeks": horizon,
            "best_model": best_row["model"],
            "best_mase": float(best_row["score_val"])
        }

        print(
            f"✔ Best model: {best_row['model']} | "
            f"MASE = {best_row['score_val']:.3f}"
        )

        leaderboard.to_csv(
            os.path.join(horizon_dir, "leaderboard.csv"),
            index=False
        )

        forecast_df.to_csv(
            os.path.join(horizon_dir, "forecast.csv"),
            index=False
        )

        with open(
            os.path.join(horizon_dir, "best_model.json"),
            "w"
        ) as f:
            json.dump(best_model_info, f, indent=2)

print("\n========================================")
print("ALL TRAINING COMPLETED SUCCESSFULLY")
print("========================================")
