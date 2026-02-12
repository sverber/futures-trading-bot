import os
import time
from datetime import datetime, timedelta

import polars as pl
import requests

from src.core.data_loader import DataLoader


class KrakenDataLoader(DataLoader):
    """
    Loads data from Kraken Futures API.
    """

    def __init__(self, symbol: str, cache_dir: str = "data"):
        super().__init__()
        self.symbol = symbol
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_historical_data(self, symbol: str, lookback_days: int, resolution: str = "1m") -> pl.DataFrame:
        """
        Loads historical data from Kraken Futures API.
        """
        safe_symbol = symbol.replace("/", "_")
        filename = f"../../data/{safe_symbol}_{lookback_days}d_{resolution}.csv"

        if os.path.exists(filename):
            df = pl.read_csv(filename)
            return df.with_columns(pl.col("timestamp").str.to_datetime())

        return self.fetch_historical_candles(symbol, resolution=resolution, lookback_days=lookback_days)

    def get_latest_data(self, symbol: str, lookback_minutes: int) -> pl.DataFrame:
        # For live trading, we need LATEST data.
        # Calling the full fetch_historical_candles might be too heavy if it loops back 30 days.
        # We should optimize fetch_latest.

        # Fetch only what we need or a safe buffer (e.g. 2x lookback or min 1 day)
        # 1 day is minimum for the days parameter in our utils, usually.
        # Let's just fetch 1 day for latest.
        df = self.fetch_historical_candles(symbol, resolution="1m", lookback_days=1)

        if df.height == 0:
            return pl.DataFrame()

        return df.tail(lookback_minutes)

    def get_tick_stream(self, symbol: str, lookback_days: int):
        """
        Provides a real-time stream by polling the Kraken API every minute.
        """
        print(f"Starting live stream for {symbol}...")

        while True:
            # 1. Fetch the latest window of data
            # We fetch a lookback to ensure FeatureEngineering has enough context
            df = self.get_latest_data(symbol, lookback_minutes=200)

            if df.height > 0:
                yield df

            # 2. Wait for the next candle (e.g., 60 seconds for 1m bars)
            # Optimization: Calculate exactly how many seconds until the next minute starts
            time.sleep(60)

    @staticmethod
    def fetch_historical_candles(symbol: str, resolution: str, lookback_days: int) -> pl.DataFrame:
        """
        Fetches historical candles from Kraken Futures Charts API.
        Endpoint: https://futures.kraken.com/api/charts/v1/trade/{symbol}/{resolution}
        Iterates to fetch full history if the API limits response size.
        """

        # Calculate start time
        now = datetime.now()
        start_time = now - timedelta(days=lookback_days)
        current_ts = int(start_time.timestamp())

        # We'll collect all candles here
        all_candles = []

        print(f"Fetching data for {symbol} from {start_time}...")

        while True:
            # Construct URL
            # symbol needs to be uppercase for API usually, check format.
            # Kraken Futures symbols are like 'PI_XBTUSD'. The user might pass 'BTC/USD'.
            # We might need a mapper or assume user passes correct symbol.
            # Attempting direct pass for now.

            url = f"https://futures.kraken.com/api/charts/v1/trade/{symbol}/{resolution}"
            params = {
                "from": current_ts
            }

            try:
                # Note: Requests is blocking.
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                # Response format: { "candles": [ { "time": 168..., "open": ... }, ... ], "more": bool }
                candles = data.get("candles", [])
                more = data.get("more", False)

                if not candles:
                    print("No more candles returned.")
                    break

                all_candles.extend(candles)

                last_candle_time = candles[-1]['time']
                last_candle_dt = datetime.fromtimestamp(last_candle_time / 1000)

                print(
                    f"Fetched {len(candles)} candles. Total: {len(all_candles)}. Last: {last_candle_dt} | More: {more}")

                # Check if we are caught up to now (approx)
                if last_candle_dt >= now - timedelta(minutes=5):
                    break

                # If we got a full batch (e.g. 2000), we assume there might be more,
                # even if 'more' flag is weird (though typically 'more' should be True).
                # If we got significantly less than limit (e.g. < 100), assume end.
                if not more and len(candles) < 10:
                    print("No 'more' flag and low candle count. Stopping.")
                    break

                # Update next 'from'
                # Use last candle time + 1 second
                new_ts = int(last_candle_time / 1000) + 1

                if new_ts <= current_ts:
                    print("Timestamp didn't advance. Stopping to avoid infinite loop.")
                    break

                current_ts = new_ts

                # Sleep to be nice
                time.sleep(0.2)

            except Exception as e:
                print(f"Error fetching candles: {e}")
                break

        if not all_candles:
            return pl.DataFrame()

        print(f"Total candles fetched: {len(all_candles)}")

        # Convert to DataFrame
        df = pl.DataFrame(all_candles)

        # Conversions
        df = df.with_columns([
            pl.from_epoch(pl.col("time"), time_unit="ms").alias("timestamp"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64)
        ])

        # Deduplicate just in case
        df = df.unique(subset=["timestamp"]).sort("timestamp")

        # Rename all columns except 'timestamp' to include the resolution suffix
        # Example: 'close' -> 'close_1m'
        df = df.rename({
            col: f"{col}_{resolution}"
            for col in df.columns if col != "timestamp"
        })

        # Select columns with their new names
        df = df.select([
            "timestamp",
            f"open_{resolution}",
            f"high_{resolution}",
            f"low_{resolution}",
            f"close_{resolution}",
            f"volume_{resolution}"
        ])

        # Ensure data directory exists
        os.makedirs("../../data", exist_ok=True)

        # Save to CSV
        safe_symbol = symbol.replace("/", "_")
        filename = f"../../data/{safe_symbol}_{lookback_days}d_{resolution}.csv"
        df.write_csv(filename)
        print(f"Saved to {filename}")

        return df


if __name__ == "__main__":
    symbol = "PF_XBTUSD"
    resolutions = ["1m", "15m"]
    lookback_days = 30

    data_loader = KrakenDataLoader(symbol=symbol)

    # Get the historical data
    df_1m = data_loader.get_historical_data(symbol, lookback_days, "1m")
    df_15m = data_loader.get_historical_data(symbol, lookback_days, "15m")

    # # Set the initial data
    data_loader.set_initial_data(df_1m=df_1m, df_15m=df_15m)

    df = data_loader.df_context

    # Ensure data directory exists
    os.makedirs("../../data", exist_ok=True)

    # Save to CSV
    safe_symbol = symbol.replace("/", "_")
    filename = f"../../data/{safe_symbol}_{lookback_days}d_debug.csv"
    df.write_csv(filename)
    print(f"Saved to {filename}")

    # Plot the context
    data_loader.plot_features(df=df, symbol=symbol, filename=f"../../plots/{symbol}_features.png")
    data_loader.plot_indicators(df=df, symbol=symbol, filename=f"../../plots/{symbol}_indicators.png")
    data_loader.plot_all(df=df, symbol=symbol, filename=f"../../plots/{symbol}_all.png")

