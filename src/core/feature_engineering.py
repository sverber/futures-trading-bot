from abc import ABC

import polars as pl

from src.core.visualization_bot import VisualizationBot
from src.features.indicators import calculate_bollinger_bands, calculate_rsi, calculate_macd
from src.features.levels import calculate_swing_levels


class FeatureEngineering(VisualizationBot, ABC):
    """
    Abstract Base Class for Feature Engineering.
    """

    # @todo: continue building the get_data_pipeline in the morning. It should flow back to the data loader.
    #  Eventually we should be able to register a data loader into at trading bot and it should just flow.
    def get_features(self, df_1m: pl.DataFrame, df_15m: pl.DataFrame) -> pl.DataFrame:
        # Step 1. Calculate swing levels at the 15m timeframe
        df_15m = calculate_swing_levels(
            df_15m,
            window=5,
            high_col='high_15m',
            low_col='low_15m',
        )

        # Step 2. Calculate the indicators at the 1m timeframe
        df_1m = calculate_bollinger_bands(df_1m, source_col="close_1m")
        df_1m = calculate_rsi(df_1m, source_col="close_1m")
        df_1m = calculate_macd(df_1m, source_col="close_1m")

        df_1m = self._calculate_bbands_bounces(df_1m)

        # Step x. Merge the 15m into the 1m timeframe
        df_merged = self.merge_15m_data(df_15m, df_1m)

        # Step x. Get the z-scores and momentum
        df_merged = self._calculate_z_scores(df_merged)

        return df_merged

    @staticmethod
    def _calculate_z_scores(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([
            pl.col("close_1m").rolling_mean(window_size=20).alias("sma_20"),
            (pl.col("high_1m") - pl.col("low_1m")).rolling_mean(window_size=14).alias("atr_14")
        ])

        return df.with_columns([
            ((pl.col("close_1m") - pl.col("sma_20")) / pl.col("atr_14").fill_null(0.0001).clip(0.0001, None))
            .alias("z_dist_mean"),
            ((pl.col("close_1m") - pl.col("swing_sup_level")) / pl.col("atr_14").fill_null(0.0001).clip(0.0001, None))
            .alias("z_dist_sup"),
            ((pl.col("swing_res_level") - pl.col("close_1m")) / pl.col("atr_14").fill_null(0.0001).clip(0.0001, None))
            .alias("z_dist_res"),
            (pl.col("macd_hist") - pl.col("macd_hist").shift(1)).alias("macd_momentum_change"),
        ])

    @staticmethod
    def _calculate_bbands_bounces(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates distance to bands and a touch signal:
        1: Close >= Upper Band
        -1: Close <= Lower Band
        0: Inside Bands
        """
        return df.with_columns([
            # Raw Distances (Positive means price is outside/past the band)
            (pl.col("close_1m") - pl.col("bb_upper")).alias("dist_bb_upper"),
            (pl.col("bb_lower") - pl.col("close_1m")).alias("dist_bb_lower"),

            # Touch Signal (-1, 0, 1)
            pl.when(pl.col("close_1m") >= pl.col("bb_upper")).then(1)
            .when(pl.col("close_1m") <= pl.col("bb_lower")).then(-1)
            .otherwise(0)
            .alias("bb_touch_signal")
        ])

    @staticmethod
    def merge_15m_data(df_1m: pl.DataFrame, df_15m: pl.DataFrame) -> pl.DataFrame:
        """Merges processed 15m data onto 1m data using asof join."""
        # Ensure sorted
        df_1m = df_1m.sort("timestamp")
        df_15m = df_15m.sort("timestamp")

        return df_1m.join_asof(df_15m, on="timestamp", strategy="backward")
