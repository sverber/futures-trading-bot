from abc import ABC, abstractmethod

import polars as pl


class Strategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    Ensures a unified interface for both Backtesting and Live Trading.
    """

    @abstractmethod
    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Takes a Polars DataFrame (price data) and appends signal columns.
        
        Args:
            df: Polars DataFrame containing OHLCV data.
                Must include 'close', 'high', 'low', etc.
        
        Returns:
            Polars DataFrame with added columns:
            - 'signal': 1 (Buy), -1 (Sell), 0 (Neutral)
            - 'stop_loss': Price level for stop loss
            - 'take_profit': Price level for take profit
            - 'reason': String explaining the signal (optional)
        """
        pass