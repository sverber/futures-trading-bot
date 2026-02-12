import os

import matplotlib.pyplot as plt
import polars as pl


class VisualizationBot:
    """
    Base class for visualization capabilities in trading bots.
    """

    @staticmethod
    def plot_features(df: pl.DataFrame, symbol: str, lookback: int = 100,
                      filename: str = "../../plots/features.png"):
        """
        Plots the last N candles with Bollinger Bands and Sup/Res levels.
        Designed for the Paper Trading Bot to update a live image.
        """
        # Filter last N rows
        if df.height > lookback:
            df_plot = df.tail(lookback)
        else:
            df_plot = df

        # Convert to Pandas for plotting compatibility
        pdf = df_plot.to_pandas()

        plt.figure(figsize=(14, 8))

        # Plot Close Price
        plt.plot(pdf["timestamp"], pdf["close_1m"], label="Price", color="black", alpha=0.7)

        # Plot BB
        if "bb_upper" in pdf.columns:
            plt.plot(pdf["timestamp"], pdf["bb_upper"], label="BB Upper", color="green", linestyle="--",
                     alpha=0.5)
        if "bb_middle" in pdf.columns:
            plt.plot(pdf["timestamp"], pdf["bb_middle"], label="BB Middle", color="orange", linestyle="--",
                     alpha=0.5)
        if "bb_lower" in pdf.columns:
            plt.plot(pdf["timestamp"], pdf["bb_lower"], label="BB Lower", color="red", linestyle="--",
                     alpha=0.5)
        if "bb_upper" in pdf.columns and "bb_lower" in pdf.columns:
            plt.fill_between(pdf["timestamp"], pdf["bb_upper"], pdf["bb_lower"], color="gray", alpha=0.1)

        # Plot Support/Resistance
        if "swing_sup_level" in pdf.columns:
            plt.plot(pdf["timestamp"], pdf["swing_sup_level"], label="Support", color="blue", linestyle=":",
                     alpha=0.8)
        if "swing_res_level" in pdf.columns:
            plt.plot(pdf["timestamp"], pdf["swing_res_level"], label="Resistance", color="orange",
                     linestyle=":", alpha=0.8)

        # Plot Signals (if present)
        if "signal" in pdf.columns:
            signal_df = pdf

            buys = signal_df[signal_df["signal"] == 1]
            sells = signal_df[signal_df["signal"] == -1]
            holds = signal_df[signal_df["signal"] == 0]

            if not buys.empty:
                plt.scatter(buys["timestamp"], buys["close_1m"], marker="^", color="green", s=100,
                            label="Buy Signal", zorder=5)
            if not sells.empty:
                plt.scatter(sells["timestamp"], sells["close_1m"], marker="v", color="red", s=100,
                            label="Sell Signal", zorder=5)
            if not holds.empty:
                # Use a small horizontal dash to indicate "flat" or "holding"
                plt.scatter(holds["timestamp"], holds["close_1m"], marker="_", color="blue", s=50, alpha=0.5,
                            label="Hold", zorder=4)

        plt.title(f"Live Bot Status - {symbol} - Last {lookback} Candles")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_indicators(df: pl.DataFrame, symbol: str, lookback: int = 100,
                        filename: str = "../../plots/indicators.png"):
        """
        Plots Z-scores and Momentum using lines and area fills to prevent
        visual crowding and highlight trend reversals.
        """
        if df.height > lookback:
            df_plot = df.tail(lookback)
        else:
            df_plot = df

        pdf = df_plot.to_pandas()

        # Create subplots with a shared x-axis
        fig, (ax_z, ax_mom) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                           gridspec_kw={'height_ratios': [2, 1]})

        # --- Panel 1: Z-Scores (Exhaustion Lines) ---
        ax_z.plot(pdf["timestamp"], pdf["z_dist_mean"], label="Z-Mean (SMA)", color="#34495e", lw=2)

        if "z_dist_sup" in pdf.columns:
            ax_z.plot(pdf["timestamp"], pdf["z_dist_sup"], label="Z-Support", color="#27ae60", linestyle="--",
                      alpha=0.6)
        if "z_dist_res" in pdf.columns:
            ax_z.plot(pdf["timestamp"], pdf["z_dist_res"], label="Z-Resistance", color="#c0392b", linestyle="--",
                      alpha=0.6)

        # Visual 'Danger Zones' for Bounces
        ax_z.axhline(0, color="black", lw=1, alpha=0.4)
        # ax_z.axhline(2, color="#e74c3c", ls=":", alpha=0.6, label="Upper Bound (+2σ)")
        # ax_z.axhline(-2, color="#2ecc71", ls=":", alpha=0.6, label="Lower Bound (-2σ)")

        ax_z.set_title(f"{symbol} - Distance metrics (Z-Scores)")
        ax_z.set_ylabel("Standard Deviations")
        ax_z.legend(loc="upper left", fontsize='small', frameon=True)
        ax_z.grid(True, alpha=0.15)

        # --- Panel 2: Momentum (Smoothed Line + Area) ---
        # We use macd_hist instead of momentum_change as it's cleaner for lines
        momentum_col = "macd_hist" if "macd_hist" in pdf.columns else "macd_momentum_change"

        # Plot the main momentum line
        ax_mom.plot(pdf["timestamp"], pdf[momentum_col], color="#8e44ad", lw=1.5, label=f"Momentum ({momentum_col})")

        # Area fill: Green for positive momentum, Red for negative
        # This solves the 'wide bars' issue by providing a clean, continuous flow
        ax_mom.fill_between(pdf["timestamp"], pdf[momentum_col], 0,
                            where=(pdf[momentum_col] >= 0), color="#2ecc71", alpha=0.3)
        ax_mom.fill_between(pdf["timestamp"], pdf[momentum_col], 0,
                            where=(pdf[momentum_col] < 0), color="#e74c3c", alpha=0.3)

        ax_mom.axhline(0, color="black", lw=1, alpha=0.5)
        ax_mom.set_title(f"Momentum Velocity")
        ax_mom.set_ylabel("Value")
        ax_mom.grid(True, alpha=0.15)

        # Global formatting
        plt.xticks(rotation=45)
        plt.tight_layout()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_all(df: pl.DataFrame, symbol: str, lookback: int = 100,
                 filename: str = "../../plots/bot_status.png"):
        """
        Integrated view: Price/BBands/Levels, Z-Scores, and Momentum.
        This provides a complete 'snapshot' for analyzing potential bounces.
        """
        # 1. Filter and Prepare Data
        if df.height > lookback:
            df_plot = df.tail(lookback)
        else:
            df_plot = df

        pdf = df_plot.to_pandas()

        # 2. Setup Figure (3 Rows)
        # Height ratios: Price plot is the largest, indicators are smaller
        fig, (ax_p, ax_z, ax_m) = plt.subplots(3, 1, figsize=(15, 14), sharex=True,
                                               gridspec_kw={'height_ratios': [3, 1.5, 1]})

        # --- Subplot 1: Price Action & Bands ---
        ax_p.plot(pdf["timestamp"], pdf["close_1m"], label="Price", color="black", alpha=0.8, lw=1.5)

        # Bollinger Bands
        if "bb_upper" in pdf.columns and "bb_lower" in pdf.columns:
            ax_p.plot(pdf["timestamp"], pdf["bb_upper"], color="green", ls="--", alpha=0.4)
            ax_p.plot(pdf["timestamp"], pdf["bb_lower"], color="red", ls="--", alpha=0.4)
            ax_p.fill_between(pdf["timestamp"], pdf["bb_upper"], pdf["bb_lower"], color="gray", alpha=0.05)

        # Support/Resistance
        if "swing_sup_level" in pdf.columns:
            ax_p.plot(pdf["timestamp"], pdf["swing_sup_level"], label="Support", color="blue", ls=":", alpha=0.6)
        if "swing_res_level" in pdf.columns:
            ax_p.plot(pdf["timestamp"], pdf["swing_res_level"], label="Resistance", color="orange", ls=":", alpha=0.6)

        # Signals
        if "signal" in pdf.columns:
            buys = pdf[pdf["signal"] == 1]
            sells = pdf[pdf["signal"] == -1]
            if not buys.empty:
                ax_p.scatter(buys["timestamp"], buys["close_1m"], marker="^", color="green", s=100, label="Buy",
                             zorder=5)
            if not sells.empty:
                ax_p.scatter(sells["timestamp"], sells["close_1m"], marker="v", color="red", s=100, label="Sell",
                             zorder=5)

        ax_p.set_title(f"Bot Status: {symbol} (Last {lookback} Candles)")
        ax_p.set_ylabel("Price")
        ax_p.legend(loc="upper left", fontsize='small')
        ax_p.grid(True, alpha=0.2)

        # --- Subplot 2: Z-Scores (Distance/Exhaustion) ---
        ax_z.plot(pdf["timestamp"], pdf["z_dist_mean"], label="Z-Mean (SMA)", color="#34495e", lw=1.5)
        if "z_dist_sup" in pdf.columns:
            ax_z.plot(pdf["timestamp"], pdf["z_dist_sup"], label="Z-Support", color="#27ae60", ls="--", alpha=0.5)
        if "z_dist_res" in pdf.columns:
            ax_z.plot(pdf["timestamp"], pdf["z_dist_res"], label="Z-Resistance", color="#c0392b", ls="--", alpha=0.5)

        # Thresholds
        ax_z.axhline(0, color="black", lw=1, alpha=0.3)
        ax_z.axhline(2, color="red", ls=":", alpha=0.4)
        ax_z.axhline(-2, color="green", ls=":", alpha=0.4)

        ax_z.set_ylabel("Std Dev (σ)")
        ax_z.legend(loc="upper left", fontsize='small')
        ax_z.grid(True, alpha=0.2)

        # --- Subplot 3: Momentum ---
        mom_col = "macd_hist" if "macd_hist" in pdf.columns else "macd_momentum_change"
        if mom_col in pdf.columns:
            ax_m.plot(pdf["timestamp"], pdf[mom_col], color="#8e44ad", lw=1.2)
            ax_m.fill_between(pdf["timestamp"], pdf[mom_col], 0, where=(pdf[mom_col] >= 0), color="#2ecc71", alpha=0.3)
            ax_m.fill_between(pdf["timestamp"], pdf[mom_col], 0, where=(pdf[mom_col] < 0), color="#e74c3c", alpha=0.3)

        ax_m.axhline(0, color="black", lw=1, alpha=0.3)
        ax_m.set_ylabel("Momentum")
        ax_m.grid(True, alpha=0.2)

        # Formatting
        plt.xticks(rotation=45)
        plt.tight_layout()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()