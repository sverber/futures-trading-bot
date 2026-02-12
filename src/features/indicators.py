import polars as pl


def calculate_bollinger_bands(df: pl.DataFrame, period: int = 20, std_dev: float = 2.0,
                              source_col: str = "close_1m") -> pl.DataFrame:
    """
    Calculates Bollinger Bands.
    """
    return df.with_columns([
        pl.col(source_col).rolling_mean(window_size=period).alias("bb_middle"),
        pl.col(source_col).rolling_std(window_size=period).alias("bb_std")
    ]).with_columns([
        (pl.col("bb_middle") + std_dev * pl.col("bb_std")).alias("bb_upper"),
        (pl.col("bb_middle") - std_dev * pl.col("bb_std")).alias("bb_lower")
    ]).drop("bb_std")


def calculate_rsi(df: pl.DataFrame, period: int = 14, source_col: str = "close_1m") -> pl.DataFrame:
    """
    Calculates RSI using Wilder's Smoothing (approximated by EWM with alpha=1/period).
    """
    delta = df.select(pl.col(source_col).diff().alias("delta"))

    # Needs to be lazy for complex expressions or just series ops
    # Polars has ewm_mean.
    # Wilder's Smoothing is EWM with alpha = 1/N.

    return (
        df.lazy()
        .with_columns([
            pl.col(source_col).diff().alias("delta")
        ])
        .with_columns([
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).alias("loss")
        ])
        .with_columns([
            pl.col("gain").ewm_mean(alpha=1 / period, adjust=False, min_periods=period).alias("avg_gain"),
            pl.col("loss").ewm_mean(alpha=1 / period, adjust=False, min_periods=period).alias("avg_loss")
        ])
        .with_columns([
            (pl.col("avg_gain") / pl.col("avg_loss")).alias("rs")
        ])
        .with_columns([
            (100 - (100 / (1 + pl.col("rs")))).alias("rsi")
        ])
        .drop(["delta", "gain", "loss", "avg_gain", "avg_loss", "rs"])
        .collect()
    )


def calculate_macd(
        df: pl.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        source_col: str = "close_1m"
) -> pl.DataFrame:
    """
    Calculates MACD (Moving Average Convergence Divergence).
    """
    # MACD Line = EMA(12) - EMA(26)
    # Signal Line = EMA(9) of MACD Line
    # Histogram = MACD Line - Signal Line

    # 2/(N+1) is standard alpha for EMA. Polars ewm_mean uses alpha or span.
    # span=N means alpha=2/(N+1).

    return (
        df.lazy()
        .with_columns([
            pl.col(source_col).ewm_mean(span=fast_period, adjust=False).alias("ema_fast"),
            pl.col(source_col).ewm_mean(span=slow_period, adjust=False).alias("ema_slow")
        ])
        .with_columns([
            (pl.col("ema_fast") - pl.col("ema_slow")).alias("macd_line")
        ])
        .with_columns([
            pl.col("macd_line").ewm_mean(span=signal_period, adjust=False).alias("macd_signal")
        ])
        .with_columns([
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
        ])
        .drop(["ema_fast", "ema_slow"])
        .collect()
    )
