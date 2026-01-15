import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def prepare_returns_ts(df, polymer_col, horizon):
    """
    Prepare long-format TimeSeriesDataFrame for AutoGluon.
    Target = forward return over horizon.
    """

    prices = df[[polymer_col]].dropna().copy()

    prices["target"] = (
        prices[polymer_col].shift(-horizon) / prices[polymer_col] - 1
    )

    prices = prices.dropna()

    ts_df = prices.reset_index()
    ts_df["item_id"] = polymer_col
    ts_df = ts_df.rename(columns={ts_df.columns[0]: "timestamp"})

    ts_df = ts_df[["item_id", "timestamp", "target"]]

    # ✅ Correct API usage for your AutoGluon version
    return TimeSeriesDataFrame(ts_df)


def run_autogluon_forecast(ts_data, horizon):
    """
    Train AutoGluon TimeSeries models and return leaderboard + predictions.
    """

    predictor = TimeSeriesPredictor(
    target="target",
    prediction_length=horizon,
    freq="W-FRI",   # ✅ explicit weekly frequency
    eval_metric="MASE",
    verbosity=0
    )

    predictor.fit(
        ts_data,
        presets="medium_quality",
        time_limit=120  # 2 minutes max (adjust if needed)
    )

    leaderboard = predictor.leaderboard(ts_data, silent=True)

    predictions = predictor.predict(ts_data)

    return leaderboard, predictions
