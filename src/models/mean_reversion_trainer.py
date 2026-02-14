import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


class MeanReversionTrainer:
    """
    Handles the training lifecycle for the Mean Reversion XGBoost model.
    Focuses on learning the 'Snap-Back' potential from market extremes.
    """

    def __init__(self, model_params: dict = None):
        self.model_params = model_params or {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        self.model = xgb.XGBRegressor(**self.model_params)
        self.features = [
            "z_dist_mean", "z_dist_sup", "z_dist_res",
            "macd_momentum_change", "rsi", "bb_bandwidth"
        ]

    def _prepare_training_set(self, df: pl.DataFrame, z_threshold: float = 1.2):
        """
        Calculates features/targets, filters for extremes, and cleans data.
        """
        # 1. Calculate Target and BB Bandwidth locally
        df = df.with_columns([
            (((pl.col("close_1m").shift(-5) - pl.col("close_1m")) /
              pl.col("atr_14").fill_null(0.0001))).alias("y_target_atr"),

            ((pl.col("bb_upper") - pl.col("bb_lower")) /
             pl.col("bb_middle")).alias("bb_bandwidth")
        ])

        # 2. FILTER: Only learn from the 'Bounce Zones'
        df_ml = df.filter(pl.col("z_dist_mean").abs() > z_threshold)

        # 3. Convert and Clean
        pdf = df_ml.select(self.features + ["y_target_atr"]).to_pandas()
        pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()

        return pdf

    def train(self, df: pl.DataFrame, n_splits: int = 5, save_path: str = None):

        # 1. Check if we can skip training
        if save_path and Path(save_path).exists():
            print(f"Loading pre-trained model from {save_path}...")
            self.model = joblib.load(save_path)
            return self.model

        # 2. Existing data prep logic
        data = self._prepare_training_set(df)

        if len(data) < 100:
            raise ValueError(f"Insufficient training data ({len(data)} rows).")

        X = data[self.features]
        y = data["y_target_atr"]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        print(f"Starting Training on {len(data)} 'extreme' samples...")

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

        print("Training complete.")

        # 3. Save the model if a path was provided
        if save_path:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")

        self._log_feature_importance()
        return self.model

    def evaluate_tick_by_tick(self, df_1m_test: pl.DataFrame, df_15m_test: pl.DataFrame, feature_engineer,
                              min_warmup: int = 200):
        """
        Simulates live trading by calculating features and predicting
        one tick at a time using 1m and 15m data.
        """
        print(f"\n--- STARTING TICK-BY-TICK EVALUATION ---")

        # 1. Prepare Ground Truth (Targets)
        # We must explicitly add the target calculation here because get_features doesn't do it.
        df_gt = feature_engineer.get_features(df_1m_test.clone(), df_15m_test.clone())

        # Calculate the future 5-bar ATR-normalized return
        df_gt = df_gt.with_columns([
            (((pl.col("close_1m").shift(-5) - pl.col("close_1m")) /
              pl.col("atr_14").fill_null(0.0001).clip(0.0001, None))).alias("y_target_atr")
        ])

        # Map timestamps to targets for validation
        gt_map = {ts: target for ts, target in zip(df_gt["timestamp"], df_gt["y_target_atr"])}

        all_preds = []
        all_actuals = []

        # 2. Iterate through each 1-minute tick in the test set
        for i in range(min_warmup, len(df_1m_test)):
            current_time = df_1m_test["timestamp"][i]

            # A. Create the 'Live' Slices (No data beyond current_time)
            slice_1m = df_1m_test.slice(0, i + 1)
            slice_15m = df_15m_test.filter(pl.col("timestamp") <= current_time)

            try:
                # B. Calculate Features for the 'Current' Bar
                df_step = feature_engineer.get_features(df_1m=slice_1m, df_15m=slice_15m)
                latest_bar = df_step.tail(1)

                # C. Check if we have a ground truth target for this timestamp
                if current_time in gt_map and gt_map[current_time] is not None:
                    # Ensure bb_bandwidth is present for the model
                    if "bb_bandwidth" not in latest_bar.columns:
                        latest_bar = latest_bar.with_columns([
                            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")).alias("bb_bandwidth")
                        ])

                    # D. Predict
                    X = latest_bar.select(self.features).to_pandas()
                    pred = self.model.predict(X)[0]

                    all_preds.append(pred)
                    all_actuals.append(gt_map[current_time])
            except Exception as e:
                # Silently continue if a specific bar fails (e.g., due to insufficient window)
                continue

        # 3. Calculate Performance Metrics
        if not all_preds:
            print("Error: No predictions were generated. Check your data windows.")
            return {"mae": 0, "dir_acc": 0}

        y_pred = np.array(all_preds)
        y_true = np.array(all_actuals)

        mae = mean_absolute_error(y_true, y_pred)
        # Check for R2 calculation validity
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = 0.0

        correct_dir = np.sign(y_pred) == np.sign(y_true)
        dir_acc = np.mean(correct_dir) * 100

        print("\n--- FINAL TICK-BY-TICK TEST PERFORMANCE ---")
        print(f"Mean Absolute Error: {mae:.6f} ATRs")
        print(f"Directional Accuracy: {dir_acc:.2f}%")
        print(f"R^2 Score: {r2:.4f}")

        return {"mae": mae, "dir_acc": dir_acc}

    def evaluate(self, df_test: pl.DataFrame):
        """
        Evaluates the model on the unseen test set.
        """
        print(f"\n--- STARTING EVALUATION ---")

        # Prepare test data (same logic as training, but on test set)
        test_data = self._prepare_training_set(df_test)
        X_test = test_data[self.features]
        y_test = test_data["y_target_atr"]

        preds = self.model.predict(X_test)

        # Calculate Metrics
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Directional Accuracy (Did we correctly predict a bounce vs a continuation?)
        correct_direction = np.sign(preds) == np.sign(y_test)
        dir_acc = np.mean(correct_direction) * 100

        print("\n--- FINAL TEST PERFORMANCE ---")
        print(f"Mean Absolute Error: {mae:.6f} ATRs")
        print(f"Directional Accuracy: {dir_acc:.2f}%")
        print(f"R^2 Score: {r2:.4f}")

        return {"mae": mae, "dir_acc": dir_acc}

    def _log_feature_importance(self):
        importances = self.model.feature_importances_
        feat_imp = sorted(zip(self.features, importances), key=lambda x: x[1], reverse=True)
        print("\n--- Feature Importance (The Model's Bounce Logic) ---")
        for feat, score in feat_imp:
            print(f"{feat}: {score:.4f}")

    def save_model(self, path: str = "models/mean_reversion_xgb.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"Model saved to {path}")
