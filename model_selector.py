# ===========================================================
# Model selector for polymer price forecasting
# ===========================================================

import os
import json
from autogluon_forecasting import run_autogluon_forecast


class ModelSelector:
    def __init__(
        self,
        model_store_path="models",
        eval_metric="MASE",
        seasonality=52
    ):
        self.model_store_path = model_store_path
        self.eval_metric = eval_metric
        self.seasonality = seasonality

        os.makedirs(self.model_store_path, exist_ok=True)

    def _model_dir(self, polymer: str, horizon: int):
        return os.path.join(
            self.model_store_path,
            polymer,
            f"horizon_{horizon}w"
        )

    def select_and_train(
        self,
        ts_data,
        polymer,
        horizon,
        target_col,
        time_col,
        id_col
    ):
        """
        Train models and select the best one by MASE
        """

        model_dir = self._model_dir(polymer, horizon)
        os.makedirs(model_dir, exist_ok=True)

        # IMPORTANT:
        # First argument must be positional (ts_df),
        # NOT data=...
        leaderboard = run_autogluon_forecast(
            ts_data,
            target_col,
            time_col,
            id_col,
            horizon,
            eval_metric=self.eval_metric,
            seasonality=self.seasonality,
            save_path=model_dir,
            return_leaderboard=True
        )

        leaderboard = leaderboard.sort_values(self.eval_metric)
        best_row = leaderboard.iloc[0]

        meta = {
            "polymer": polymer,
            "horizon_weeks": horizon,
            "best_model": best_row["model"],
            "best_mase": float(best_row[self.eval_metric]),
            "seasonality": self.seasonality
        }

        with open(os.path.join(model_dir, "model_selection.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return meta, leaderboard

    def forecast(
        self,
        ts_data,
        polymer,
        horizon,
        target_col,
        time_col,
        id_col
    ):
        """
        Forecast using pre-selected best model
        """

        model_dir = self._model_dir(polymer, horizon)
        meta_path = os.path.join(model_dir, "model_selection.json")

        if not os.path.exists(meta_path):
            raise RuntimeError(
                f"No trained model found for {polymer} ({horizon}w)"
            )

        with open(meta_path, "r") as f:
            meta = json.load(f)

        forecast = run_autogluon_forecast(
            ts_data,
            target_col,
            time_col,
            id_col,
            horizon,
            model_names=[meta["best_model"]],
            load_path=model_dir,
            seasonality=meta["seasonality"]
        )

        return forecast, meta
