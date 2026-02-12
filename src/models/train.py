import polars as pl
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit


def train_mean_reversion(df: pl.DataFrame):
    # 1. Filter for 'Tradeable' setups (near support or resistance)
    # df_ml = df.filter(pl.col("market_zone") != "in_range")
    df_ml = df  # @todo: figure out what we need to provide as input to train our model.

    # 2. ALIGNMENT FIX: Use the features present in your debug output
    features = [
        "z_dist_mean",
        "z_dist_sup",
        "z_dist_res",
        "macd_momentum_change",
        "rsi"  # RSI is available in your valid columns list
    ]

    # 3. Predict the ATR-normalized target
    import numpy as np

    # Select features + target
    df_clean = df_ml.select(features + ["y_target_atr"])

    # Convert to pandas
    df_pd = df_clean.to_pandas()

    # Replace inf with NaN
    df_pd = df_pd.replace([np.inf, -np.inf], np.nan)

    # Drop rows where Target OR Features are NaN
    df_pd = df_pd.dropna(subset=["y_target_atr"] + features)

    if df_pd.empty:
        raise ValueError("No valid training data after cleaning (all rows contained Inf/NaN).")

    X = df_pd[features]
    y = df_pd[["y_target_atr"]]

    # 4. Time-Series Split
    tscv = TimeSeriesSplit(n_splits=5)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.01)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)

    return model, features, X_test, y_test
