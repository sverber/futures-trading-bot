import polars as pl

# def calculate_support_resistance(
#     df: pl.DataFrame,
#     window: int = 5,
#     high_col: str = "high",
#     low_col: str = "low",
#     res_col: str = "res_level",
#     sup_col: str = "sup_level",
#     shift: int = 1
# ) -> pl.DataFrame:
#     """
#     Calculates support and resistance levels based on rolling High/Low (Donchian Channel).
#
#     Args:
#         df: Input DataFrame
#         window: Rolling window size (number of bars)
#         shift: Number of periods to shift the result.
#                shift=1 (default) ensures we use values from COMPLETED bars relative to the current timestamp.
#                This prevents lookahead bias.
#     """
#     # Rolling Max/Min
#     return df.with_columns([
#         pl.col(high_col).rolling_max(window_size=window).shift(shift).alias(res_col),
#         pl.col(low_col).rolling_min(window_size=window).shift(shift).alias(sup_col),
#     ]).fill_null(strategy="forward")

def calculate_swing_levels(
    df: pl.DataFrame, 
    window: int = 5,
    high_col: str = "high_15m",
    low_col: str = "low_15m"
) -> pl.DataFrame:
    """
    Calculates discrete Swing Highs and Swing Lows (Fractals).
    A Swing High is a bar with a higher High than 'window' bars before and after it.
    
    NOTE: Identifying a swing point inherently requires lookahead (waiting for future bars to confirm).
    To use this in real-time, the swing point is only 'confirmed' 'window' bars LATER.
    
    This function adds:
    - is_swing_high: Boolean
    - is_swing_low: Boolean
    - swing_res_level: The Price of the most recent confirmed Swing High (propagated forward)
    - swing_sup_level: The Price of the most recent confirmed Swing Low (propagated forward)
    """
    
    # 1. Identify peaks/valleys
    # We need to look forward 'window' bars.
    # We can use rolling windows with center=True, then check if the center value equals the max/min.
    # BUT for real-time simulation, we must strictly respect time.
    
    # If using rolling_max with center=True, the value at T is max(T-w ... T ... T+w).
    # If High[T] == max, it's a swing high.
    # THIS KNOWS FUTURE DATA at time T.
    
    # So, we calculate the geometric swing points first (with lookahead),
    # THEN we shift the signal forward by 'window' periods to represent when it becomes KNOWN.
    
    # Example: Window=2. Becomes a swing high at T. Confirmed at T+2.
    # We mark T as Swing High. But the "Level" is only available at T+2.
    
    half_window = window # typically small, e.g. 2 for a "5-bar fractal" (2 left, 1 center, 2 right)
    
    # Use larger window for rolling to cover left and right
    full_window = 2 * half_window + 1
    
    # Calculate geometric pivots (Lookahead!)
    pivots = df.select([
        pl.col(high_col).rolling_max(window_size=full_window, center=True).alias("local_max"),
        pl.col(low_col).rolling_min(window_size=full_window, center=True).alias("local_min"),
        pl.col(high_col),
        pl.col(low_col)
    ]).with_columns([
        (pl.col(high_col) == pl.col("local_max")).alias("is_peak"),
        (pl.col(low_col) == pl.col("local_min")).alias("is_valley")
    ])
    
    # Now, project these levels forward from the moment they are CONFIRMED.
    # A peak at Time T is confirmed at Time T + half_window.
    # So we take the 'is_peak' column, shift it forward by 'half_window', 
    # and IF it is true, we update the 'swing_res_level'.
    
    # We want a series that holds the "Last Confirmed Swing High Price".
    
    return df.with_columns([
        # Add the raw detection (with lookahead/nulls at end) if wanted, but better to just add the USABLE levels.
        
        # 1. Detect Peak (True at T, but uses T+window info)
        (pl.col(high_col) == pl.col(high_col).rolling_max(window_size=full_window, center=True)).alias("raw_is_peak"),
        (pl.col(low_col) == pl.col(low_col).rolling_min(window_size=full_window, center=True)).alias("raw_is_valley")
    ]).with_columns([
        # 2. Shift Detection to Confirmation Time (T + half_window)
        pl.col("raw_is_peak").shift(half_window).alias("peak_confirmed"),
        pl.col("raw_is_valley").shift(half_window).alias("valley_confirmed"),
        
        # Capture price at T (shifted to T + half_window)
        pl.col(high_col).shift(half_window).alias("peak_price"),
        pl.col(low_col).shift(half_window).alias("valley_price")
    ]).with_columns([
        # 3. Propagate the last confirmed level forward
        pl.when(pl.col("peak_confirmed")).then(pl.col("peak_price"))
          .otherwise(None)
          .forward_fill()
          .alias("swing_res_level"),
          
        pl.when(pl.col("valley_confirmed")).then(pl.col("valley_price"))
          .otherwise(None)
          .forward_fill()
          .alias("swing_sup_level")
    ]).drop([
        "raw_is_peak", "raw_is_valley", "peak_confirmed", "valley_confirmed", "peak_price", "valley_price"
    ])



# RSI_15m?