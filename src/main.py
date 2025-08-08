"""
Main entry point for the Algo Trading System.

This module orchestrates the entire trading system including:
- Data fetching and processing
- Strategy execution
- ML predictions
- Trade logging to Google Sheets
- Telegram notifications
"""

import asyncio
import schedule
import time
from datetime import datetime
import logging
from typing import List, Dict, Any
import sys
import os

# Ensure proper path setup for imports
if __name__ == "__main__":
    # When running directly, add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# Import modules
try:
    from src.utils.config import Config
    from src.utils.logger import setup_logger
    from src.data.data_fetcher import DataFetcher
    from src.strategies.rsi_ma_strategy import RSIMAStrategy
    from src.ml.predictor import MLPredictor
    from src.integrations.google_sheets import GoogleSheetsLogger
    from src.integrations.telegram_bot import TelegramBot
    from src.backtesting.backtest_engine import BacktestEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import method...")
    # Alternative import method
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.utils.config import Config
    from src.utils.logger import setup_logger
    from src.data.data_fetcher import DataFetcher
    from src.strategies.rsi_ma_strategy import RSIMAStrategy
    from src.ml.predictor import MLPredictor
    from src.integrations.google_sheets import GoogleSheetsLogger
    from src.integrations.telegram_bot import TelegramBot
    from src.backtesting.backtest_engine import BacktestEngine


class AlgoTradingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self):
        """Initialize the trading system."""
        print("Initializing trading system...")
        
        print("Loading configuration...")
        self.config = Config()
        
        print("Setting up logger...")
        self.logger = setup_logger(__name__)
        
        print("Initializing data fetcher...")
        self.data_fetcher = DataFetcher()
        
        print("Initializing strategy...")
        self.strategy = RSIMAStrategy()
        
        print("Initializing ML predictor...")
        self.ml_predictor = MLPredictor()
        
        print("Initializing Google Sheets logger...")
        self.sheets_logger = GoogleSheetsLogger()
        
        print("Initializing Telegram bot...")
        self.telegram_bot = TelegramBot()
        
        self.logger.info("Algo Trading System initialized")
        print("✅ System initialization complete!")
    
    async def run_trading_cycle(self) -> None:
        """Execute one complete trading cycle."""
        try:
            self.logger.info("Starting trading cycle")
            
            # Get list of stocks to analyze
            stocks = self.config.get_nifty50_stocks()
            
            # Fetch latest data for all stocks
            self.logger.info(f"Fetching data for {len(stocks)} stocks")
            market_data = await self.data_fetcher.fetch_multiple_stocks(stocks)
            
            # Process each stock
            signals = []
            for symbol, data in market_data.items():
                try:
                    # Generate strategy signals
                    signal = self.strategy.generate_signal(data)
                    
                    # Get ML prediction (await the coroutine)
                    ml_prediction = await self.ml_predictor.predict(data)
                    
                    # Combine signals
                    combined_signal = self._combine_signals(signal, ml_prediction)
                    
                    if combined_signal['action'] != 'HOLD':
                        signals.append({
                            'symbol': symbol,
                            'timestamp': datetime.now(),
                            'action': combined_signal['action'],
                            'price': data['Close'].iloc[-1],
                            'strategy_confidence': signal.get('confidence', 0),
                            'ml_confidence': ml_prediction.get('confidence', 0),
                            'combined_confidence': combined_signal['confidence']
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # Log signals to Google Sheets
            if signals:
                await self.sheets_logger.log_signals(signals)
                
                # Send Telegram notifications
                for signal in signals:
                    await self.telegram_bot.send_signal_alert(signal)
                
                self.logger.info(f"Generated {len(signals)} trading signals")
            else:
                self.logger.info("No trading signals generated")
                
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")
            await self.telegram_bot.send_error_alert(str(e))
    
    def _combine_signals(self, strategy_signal: Dict[str, Any], ml_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Combine strategy and ML signals."""
        # Simple combination logic - can be enhanced
        strategy_action = strategy_signal.get('action', 'HOLD')
        ml_action = ml_prediction.get('action', 'HOLD')
        
        # If both agree, use higher confidence
        if strategy_action == ml_action:
            return {
                'action': strategy_action,
                'confidence': max(
                    strategy_signal.get('confidence', 0),
                    ml_prediction.get('confidence', 0)
                )
            }
        
        # If they disagree, use the one with higher confidence
        strategy_conf = strategy_signal.get('confidence', 0)
        ml_conf = ml_prediction.get('confidence', 0)
        
        if strategy_conf > ml_conf:
            return {'action': strategy_action, 'confidence': strategy_conf}
        elif ml_conf > strategy_conf:
            return {'action': ml_action, 'confidence': ml_conf}
        else:
            return {'action': 'HOLD', 'confidence': 0}
    
    async def run_backtest(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest for a specific symbol."""
        self.logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
        
        backtest_engine = BacktestEngine(
            strategy=self.strategy,
            initial_capital=self.config.get('trading.initial_capital')
        )
        
        results = await backtest_engine.run_backtest(symbol, start_date, end_date)
        
        # Log results to Google Sheets
        await self.sheets_logger.log_backtest_results(results)
        
        # Send summary to Telegram
        await self.telegram_bot.send_backtest_summary(results)
        
        return results
    
    async def train_ml_models(self, symbols: List[str] = None) -> None:
        """Train ML models on historical data."""
        if symbols is None:
            symbols = self.config.get_nifty50_stocks()[:10]  # Train on top 10 for demo
        
        self.logger.info(f"Training ML models on {len(symbols)} stocks")
        
        for symbol in symbols:
            try:
                await self.ml_predictor.train_model(symbol)
                self.logger.info(f"Model trained for {symbol}")
            except Exception as e:
                self.logger.error(f"Error training model for {symbol}: {str(e)}")
    
    async def send_daily_summary(self) -> None:
        """Send daily trading summary."""
        try:
            # Get today's performance from Google Sheets
            summary = await self.sheets_logger.get_daily_summary()
            
            # Send to Telegram
            await self.telegram_bot.send_daily_summary(summary)
            
            self.logger.info("Daily summary sent")
            
        except Exception as e:
            self.logger.error(f"Error sending daily summary: {str(e)}")
    
    def schedule_tasks(self) -> None:
        """Schedule recurring tasks."""
        # Run trading cycle every 5 minutes during market hours
        schedule.every(5).minutes.do(lambda: asyncio.create_task(self.run_trading_cycle()))
        
        # Send daily summary at 6 PM
        schedule.every().day.at("18:00").do(lambda: asyncio.create_task(self.send_daily_summary()))
        
        # Train ML models weekly on Sunday at 2 AM
        schedule.every().sunday.at("02:00").do(lambda: asyncio.create_task(self.train_ml_models()))
        
        self.logger.info("Tasks scheduled successfully")
    
    async def run(self) -> None:
        """Run the main trading system."""
        self.logger.info("Starting Algo Trading System")
        
        # Send startup notification
        await self.telegram_bot.send_startup_notification()
        
        # Schedule tasks
        self.schedule_tasks()
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Shutting down trading system")
                await self.telegram_bot.send_shutdown_notification()
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                await self.telegram_bot.send_error_alert(str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retrying


async def main():
    """Main function."""
    system = AlgoTradingSystem()
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())
