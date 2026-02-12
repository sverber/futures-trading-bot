from abc import ABC, abstractmethod

import polars as pl

from src.core.feature_engineering import FeatureEngineering


class DataLoader(FeatureEngineering, ABC):
    """
    Abstract Base Class for Data Loaders.
    Maintains a single 'df_context' representing the current market state.
    """

    def __init__(self):
        super().__init__()

        self._df_1m_train: pl.DataFrame | None = None
        self._df_1m_test: pl.DataFrame | None = None
        self._df_15m_train: pl.DataFrame | None = None
        self._df_15m_test: pl.DataFrame | None = None
        self._df_context: pl.DataFrame | None = None

    @property
    def df_context(self) -> pl.DataFrame | None:
        return self._df_context

    @property
    def df_1m_train(self) -> pl.DataFrame | None:
        return self._df_1m_train

    @property
    def df_15m_train(self) -> pl.DataFrame | None:
        return self._df_15m_train

    def set_initial_data(self, df_1m: pl.DataFrame, df_15m: pl.DataFrame, split: float = 0.8):
        """Sets the starting historical context and splits into train/test sets."""

        # 1. Ensure data is sorted by time
        df_1m = df_1m.sort("timestamp")
        df_15m = df_15m.sort("timestamp")

        # 2. Calculate the split index for the 1m data
        train_size = int(len(df_1m) * split)

        # Determine the exact timestamp where the split happens
        split_timestamp = df_1m["timestamp"][train_size]

        # 3. Create 1m Train/Test splits
        self._df_1m_train = df_1m.filter(pl.col("timestamp") < split_timestamp)
        self._df_1m_test = df_1m.filter(pl.col("timestamp") >= split_timestamp)

        # 4. Create 15m Train/Test splits using the SAME timestamp
        # This ensures the 15m context matches the 1m timeline
        self._df_15m_train = df_15m.filter(pl.col("timestamp") < split_timestamp)
        self._df_15m_test = df_15m.filter(pl.col("timestamp") >= split_timestamp)

        # 5. Initialize context with the training data
        self._df_context = self.get_features(df_1m=self._df_1m_train, df_15m=self._df_15m_train)

        # Save to csv


    def get_tick_data(self, lookback_ticks: int = 100):
        """Yields the updated df_context one tick at a time."""

        # Construct the features for the last {n} ticks
        self._df_context = self.get_features(self._df_1m_test, self._df_15m_test)
        self._df_context = self.df_context.tail(lookback_ticks)

        return self._df_context
