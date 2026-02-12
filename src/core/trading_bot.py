
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import polars as pl

from src.core.data_loader import DataLoader
from src.core.visualization_bot import VisualizationBot


class TradingBot(VisualizationBot, ABC):
    """
    Abstract Base Class for Trading Bots.
    Standardizes the lifecycle: Warmup -> Train -> Live Execution Loop.
    """
    
    def __init__(self, symbol: str, strategy: Any, data_loader: 'DataLoader', model_trainer_func=None, backtest_only: bool = False):
        self.symbol = symbol
        self.strategy = strategy
        self.data_loader = data_loader
        self.model_trainer_func = model_trainer_func
        self.backtest_only = backtest_only
        
        self.model = None
        self.is_running = False
        self.df_train = None
        
        # State
        self.trading_start_time = None

    def run(self):
        """Main execution entry point."""
        print(f"--- Starting Bot for {self.symbol} ---")
        
        # 1. Warmup & Train
        if self.model_trainer_func:
            self.warmup_and_train()
        else:
            print("No training function provided. Skipping training.")

        # 2. Backtest / Verify (Optional hook)
        if self.model:
           self.on_training_complete()
            
        # 3. Live Loop
        if not self.backtest_only:
            self.start_trading_loop()

    def warmup_and_train(self):
        """Fetches historical data and trains the model."""
        print("Fetching historical data for training...")
        # Configurable lookback? Default to 30 days as per original paper_bot
        df_history = self.data_loader.get_historical_data(self.symbol, lookback_days=5)
        
        if df_history is None or df_history.height == 0:
            print("Error: No historical data found.")
            return

        print(f"Fetched {df_history.height} bars.")
        
        print("Preparing data and training model...")
        # Strategy prepares data (calculates indicators, merges 15m, etc.)
        self.df_train = self.strategy.prepare_training_data(df_history)
        print("Saved df_train to csv.")

        # Save the training data in /data/debug folder
        import os
        os.makedirs("data/debug", exist_ok=True)
        safe_symbol = self.symbol.replace("/", "_")
        self.df_train.write_csv(f"data/debug/{safe_symbol}_train.csv")

        # Train
        if self.model_trainer_func:
             # Expects trainer to return (model, metrics, X_test, y_test) or similar
             # Adjust based on actual train_directional_reversion signature
             self.model, _, _, _ = self.model_trainer_func(self.df_train)
             
             # Update strategy with trained model
             if hasattr(self.strategy, 'model'):
                 self.strategy.model = self.model
                 
             print("Model trained successfully.")

    def start_trading_loop(self):
        """Main Live Trading Loop."""
        self.trading_start_time = datetime.now()
        print(f"--- Live Trading Session Started at {self.trading_start_time} ---")
        self.is_running = True
        
        self.waiting_logic()

        while self.is_running:
            try:
                self._tick()
            except KeyboardInterrupt:
                print("Stopping Bot.")
                self.stop()
            except Exception as e:
                print(f"Error in loop: {e}")
                time.sleep(10)

    def _tick(self):
        """Single iteration of the trading loop."""
        # 1. Wait logic (align to minute)
        self.waiting_logic()
        
        print(f"\nTime: {datetime.now().strftime('%H:%M:%S')}")

        # 2. Fetch recent buffer
        # We need enough data for indicators (lookback_days=1 is safe for 1m indicators)
        # Use get_latest_data from DataLoader
        # For simplicity, convert 'lookback_days=1' logic to minutes or fetch enough bars.
        # Assuming we need ~1440 bars for a day.
        df_recent = self.data_loader.get_latest_data(self.symbol, lookback_minutes=1440)
        
        if df_recent is None or df_recent.height < 50:
            print("Not enough recent data.")
            return

        # 3. Generate Signals
        df_signals = self.strategy.generate_signals(df_recent)
        
        # 4. Visualization
        self.plot_status(df_signals, self.symbol, self.trading_start_time)
        
        # 5. Process Latest Signal
        self.process_signals(df_signals)

    def waiting_logic(self):
        """Sleeps until the start of the next minute."""
        now = datetime.now()
        seconds_to_sleep = 60 - now.second + 2 
        if seconds_to_sleep < 0: 
            seconds_to_sleep = 0
        time.sleep(seconds_to_sleep)

    def stop(self):
        self.is_running = False

    def on_training_complete(self):
        """Hook for backtesting or validation after training."""
        pass

    # --- Abstract Methods ---

    @abstractmethod
    def process_signals(self, df_signals: pl.DataFrame):
        """
        Decide and execute trades based on signals.
        """
        pass
