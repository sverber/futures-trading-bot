import pandas as pd
import numpy as np


# @todo: continue with adding the maker and taker fees, also consider spread and slippage.
def backtest_model(model, X_test, y_test, df_raw, leverage=1, threshold_atr=0.15, stop_loss_atr=0.10):
    """
    Enhanced backtest with leverage and tighter ATR exits.

    Args:
        threshold_atr: Smaller exit target (e.g. 0.15 ATR)
        leverage: Multiplier (e.g. 1x)
        stop_loss_atr: Maximum loss allowed per trade in ATR units
        df_raw: The full merged dataframe (needed to get 'close_1m' and 'atr_14' for ROE)
    """
    preds = model.predict(X_test)

    # Align the price and volatility data with our test set
    price_data = df_raw.iloc[y_test.index][['close_1m', 'atr_14']]

    results = pd.DataFrame({
        'actual_change_atr': y_test.iloc[:, 0].values,
        'predicted_change_atr': preds,
        'price': price_data['close_1m'].values,
        'atr': price_data['atr_14'].values
    })

    # 1. Generate Signals (Long/Short) based on the smaller threshold
    results['signal'] = 0
    results.loc[results['predicted_change_atr'] > threshold_atr, 'signal'] = 1
    results.loc[results['predicted_change_atr'] < -threshold_atr, 'signal'] = -1

    # 2. Implement Stop Loss (Capping the downside)
    # If the move went against our signal further than stop_loss_atr, we cap the loss
    results['capped_actual_atr'] = results['actual_change_atr']
    # For Longs:
    results.loc[(results['signal'] == 1) & (
            results['actual_change_atr'] < -stop_loss_atr), 'capped_actual_atr'] = -stop_loss_atr
    # For Shorts:
    results.loc[(results['signal'] == -1) & (
            results['actual_change_atr'] > stop_loss_atr), 'capped_actual_atr'] = -stop_loss_atr

    # 3. Calculate PnL in ATR Units
    # Subtracting 0.03 ATR for a high-efficiency exchange/low spread
    cost_atr = 0.03
    results['net_pnl_atr'] = np.where(
        results['signal'] != 0,
        (results['signal'] * results['capped_actual_atr']) - cost_atr,
        0
    )

    # 4. Convert to Percentage ROE (Return on Equity)
    # ROE = (ATR_Move * ATR_Value / Price) * Leverage
    results['roe'] = (results['net_pnl_atr'] * results['atr'] / results['price']) * leverage

    # 5. Metrics
    total_trades = (results['signal'] != 0).sum()
    win_rate = (results['net_pnl_atr'] > 0).sum() / total_trades if total_trades > 0 else 0
    total_roe = results['roe'].sum()

    print(f"\n--- Leveraged Backtest Results ({leverage}x Leverage) ---")
    print(f"Target Threshold: {threshold_atr} ATR | Stop Loss: {stop_loss_atr} ATR")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Return (ROE): {total_roe:.2%}")
    print(f"Avg ROE per Trade: {results.loc[results['signal'] != 0, 'roe'].mean():.4%}")

    # # 6. Plot Equity Curve (ROE %)
    # plt.figure(figsize=(10, 5))
    # (results['roe'].cumsum() * 100).plot()
    # plt.title(f"Leveraged Equity Curve ({leverage}x)")
    # plt.ylabel("Cumulative ROE %")
    # plt.xlabel("Trade Sequence")
    # plt.grid(True)
    # plt.savefig("plots/backtest_leveraged_roe.png")

    # 7. Visualization (Live-style plot for the last portion of backtest)
    try:
        from src.utils.visualization import plot_live_status
        import polars as pl
        
        # We want to visualize the backtest period.
        # df_raw is the full dataset, y_test.index are the indices used for testing.
        # Let's take the test set slice.
        df_test_viz = df_raw.iloc[y_test.index].copy()
        df_test_viz['signal'] = results['signal'].values
        
        # Convert to Polars
        pl_test_viz = pl.from_pandas(df_test_viz)
        
        # Plot the last 200 candles of the backtest
        print("Generating backtest visualization...")
        plot_live_status(pl_test_viz, symbol="Backtest Final", lookback=200, filename="data/plots/backtest_live_status.png")
        print("Saved backtest visualization to data/plots/backtest_live_status.png")
        
    except Exception as e:
        print(f"Failed to generate backtest visualization: {e}")

    return results

