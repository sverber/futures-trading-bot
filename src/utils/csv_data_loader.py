import os
from datetime import timedelta

import polars as pl

from src.core.data_loader import DataLoader


class CSVDataLoader(DataLoader):
    """
    Loads data from a CSV file.
    Useful for backtesting or simulation using static files.
    """

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Pre-load and sort data
        self.df = pl.read_csv(self.file_path)

        # Ensure timestamp is datetime
        if "timestamp" in self.df.columns:
            self.df = self.df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))  # Adjust format as needed

        self.df = self.df.sort("timestamp")

    def get_historical_data(self, symbol: str, lookback_days: int) -> pl.DataFrame:
        # Determine the end time of the data
        end_time = self.df["timestamp"].max()
        start_time = end_time - timedelta(days=lookback_days)

        return self.df.filter(pl.col("timestamp") >= start_time)

    def get_latest_data(self, symbol: str, lookback_minutes: int) -> pl.DataFrame:
        # Assuming 1m data
        rows = lookback_minutes

        if self.df.height < rows:
            return self.df

        return self.df.tail(rows)

    def get_tick_stream(self, symbol: str, lookback_days: int):
        """
        Simulates a live feed by iterating through historical CSV data.
        """
        # 1. Prepare the full dataset
        full_df = self.get_historical_data(symbol, lookback_days)

        # 2. Iterate through the dataframe starting from a 'warmup' point
        # We start at index 100 so indicators have enough data to calculate
        warmup_period = 100

        for i in range(warmup_period, len(full_df)):
            # Yield the dataframe up to the current 'now' index
            # This prevents the model from 'looking into the future'
            yield full_df.slice(0, i + 1)
