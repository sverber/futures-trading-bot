from typing import Dict, Any, Optional
import polars as pl
import numpy as np

from src.core.strategy import Strategy
from src.features.indicators import calculate_bollinger_bands, calculate_rsi, calculate_macd
from src.features.levels import calculate_support_resistance, calculate_swing_levels

class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy implementation.

    This strategy:
    1. Consumes 1m data.
    2. Resamples/Merges 15m data (Trend timeframe).
    3. Calculates Indicators (RSI, MACD, Bollinger Bands).
    4. Calculates Support & Resistance Levels.
    5. Feeds features into a model (optional) or uses rules to generate signals.
    """

    def __init__(self, model=None, threshold_atr: float = 0.15, stop_loss_atr: float = 0.10):
        """
        Args:
            model: Trained XGBoost model (or similar) with a .predict() method.
            threshold_atr: ATR multiple for entry signal threshold.
            stop_loss_atr: ATR multiple for stop loss.
        """
        self.model = model
        self.threshold_atr = threshold_atr
        self.stop_loss_atr = stop_loss_atr

    def generate_signals(self, df_features: pl.DataFrame) -> pl.DataFrame:
        """
        Full pipeline: Processing -> Features -> Signals.

        Args:
            df: Polars DataFrame with 1m data.
                Expects columns: 'timestamp', 'open', 'high', 'low', 'close' (or suffixed with _1m).
        """

        # 4. Generate Predictions & Signals
        if self.model:
            # Prepare X for model
            # Note: We must exclude rows with nulls created by rolling windows before prediction,
            # but we want to return the full dataframe structure.
            # For backtesting, we can predict on everything (handling nulls).
            # For live, we usually care about the last row.

            # We assume the model expects specific column names.
            # Based on previous analysis: z_dist_mean, z_dist_sup, z_dist_res, macd_momentum_change, rsi

            # Predict
            # Convert to pandas for XGBoost compat if needed, or use Polars if supported.
            # Assuming model handles pandas/numpy.

            # Helper to get features
            feature_cols = [
                "z_dist_mean", "z_dist_sup", "z_dist_res",
                "macd_momentum_change", "rsi"
            ]

            # Check availability
            valid_rows = df_features.drop_nulls(subset=feature_cols)

            # If empty (not enough data), return empty signals or default
            if valid_rows.height == 0:
                print("Warning: Not enough data to generate signals.")
                return df_features.with_columns(pl.lit(0).alias("signal"))

            X = valid_rows.select(feature_cols).to_pandas()
            try:
                preds = self.model.predict(X)

                # Assign predictions back to the valid rows
                # This is tricky in Polars without a join on index/timestamp.
                # Easiest is to create a series aligned with valid_rows and join back.

                # Create a DataFrame of predictions with timestamps
                preds_df = pl.DataFrame({
                    "timestamp": valid_rows["timestamp"],
                    "predicted_change_atr": preds
                })

                # Join predictions back to main df
                df_with_preds = df_features.join(preds_df, on="timestamp", how="left")

                # Generate Signal
                # 1 = Long, -1 = Short, 0 = Neutral
                df_signals = df_with_preds.with_columns([
                    pl.when(pl.col("predicted_change_atr") > self.threshold_atr).then(1)
                    .when(pl.col("predicted_change_atr") < -self.threshold_atr).then(-1)
                    .otherwise(0)
                    .alias("signal")
                ])

                return df_signals

            except Exception as e:
                print(f"Prediction error: {e}")
                return df_features.with_columns(pl.lit(0).alias("signal"))
        else:
            # Fallback or Rule-based logic if no model
            return df_features.with_columns(pl.lit(0).alias("signal"))

    def on_bar(self, bar: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Live trading hook.
        This would maintain a buffer of bars, append the new one, run generate_signals,
        and return a trade instruction if signal is found for the LATEST bar.

        For now, this is a placeholder interface implementation.
        """
        pass

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensures columns have _1m suffix and standard names."""
        rename_map = {}
        for col in df.columns:
            if col in ["open", "high", "low", "close", "volume"] and f"{col}_1m" not in df.columns:
                 rename_map[col] = f"{col}_1m"
            # Handle 'vol' -> 'volume'
            if col == "vol" and "volume_1m" not in df.columns:
                 rename_map[col] = "volume_1m"
            if col == "vol_1m":
                 rename_map[col] = "volume_1m"
        return df.rename(rename_map)

    def normalize_15m_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensures columns have _15m suffix."""
        rename_map = {}
        for col in df.columns:
            # If standard names exist, rename them
            if col in ["open", "high", "low", "close", "volume"]:
                 rename_map[col] = f"{col}_15m"
            # Handle 'vol'
            if col == "vol":
                 rename_map[col] = "volume_15m"
        return df.rename(rename_map)

    def process_15m_data(self, df_15m: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates features on 15m data (S&R, Swing Levels).
        Expects columns with _15m suffix.
        """
        # 1. Support & Resistance (Rolling)
        df_15m = calculate_support_resistance(
            df_15m,
            window=5,            # 5 bars of 15m = 75 mins
            high_col="high_15m",
            low_col="low_15m"
        )

        # 2. Swing Levels (Fractals)
        df_15m = calculate_swing_levels(
            df_15m,
            window=3,            # 3 bars of 15m
            high_col="high_15m",
            low_col="low_15m"
        )
        return df_15m

    def merge_15m_data(self, df_1m: pl.DataFrame, df_15m: pl.DataFrame) -> pl.DataFrame:
        """Merges processed 15m data onto 1m data using asof join."""
        # Ensure sorted
        df_1m = df_1m.sort("timestamp")
        df_15m = df_15m.sort("timestamp")

        return df_1m.join_asof(df_15m, on="timestamp", strategy="backward")

    def _add_15m_data(self, df_1m: pl.DataFrame) -> pl.DataFrame:
        """Resamples 1m data to 15m, calculates 15m features, and merges."""
        # Resample
        # Ensure timestamp is sorted
        df_1m = df_1m.sort("timestamp")

        # Check volume col
        vol_col = "volume_1m" if "volume_1m" in df_1m.columns else None

        aggs = [
            pl.col("open_1m").first().alias("open_15m"),
            pl.col("high_1m").max().alias("high_15m"),
            pl.col("low_1m").min().alias("low_15m"),
            pl.col("close_1m").last().alias("close_15m"),
        ]

        if vol_col:
            aggs.append(pl.col(vol_col).sum().alias("volume_15m"))

        df_15m = (df_1m.group_by_dynamic("timestamp", every="15m")
            .agg(aggs)
        )

        # Calculate Features
        df_15m = self.process_15m_data(df_15m)

        # Merge
        return self.merge_15m_data(df_1m, df_15m)

    def _calculate_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Applies 1m indicators and feature engineering."""
        # 1. 1m Indicators
        df = calculate_bollinger_bands(df, source_col="close_1m")
        df = calculate_rsi(df, source_col="close_1m")
        df = calculate_macd(df, source_col="close_1m")

        # 2. Feature Engineering (Normalization)
        df = df.with_columns([
            pl.col("close_1m").rolling_mean(window_size=20).alias("sma_20"),
            (pl.col("high_1m") - pl.col("low_1m")).rolling_mean(window_size=14).alias("atr_14")
        ])

        # 3. Z-Scores and Momentum
        return df.with_columns([
            ((pl.col("close_1m") - pl.col("sma_20")) / pl.col("atr_14").fill_null(0.0001).clip(0.0001, None)).alias("z_dist_mean"),
            ((pl.col("close_1m") - pl.col("swing_sup_level")) / pl.col("atr_14").fill_null(0.0001).clip(0.0001, None)).alias("z_dist_sup"),
            ((pl.col("swing_res_level") - pl.col("close_1m")) / pl.col("atr_14").fill_null(0.0001).clip(0.0001, None)).alias("z_dist_res"),
            (pl.col("macd_hist") - pl.col("macd_hist").shift(1)).alias("macd_momentum_change"),
        ])

    def prepare_training_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepares data for training by calculating features AND targets.
        """
        # 1. Normalize and 15m merge
        df = self._normalize_columns(df)

        if "close_15m" not in df.columns:
            df = self._add_15m_data(df)

        # 2. Features
        df = self._calculate_features(df)

        # 3. Targets (Lookahead)
        # Shift -5 means we look 5 bars into future.
        df = df.with_columns([
            (((pl.col("close_1m").shift(-5) - pl.col("close_1m")) / pl.col("atr_14"))).alias("y_target_atr")
        ])

        return df
