"""
Logging utilities for the algo trading system.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
import sys


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        format_string: Custom format string
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TradingLogger:
    """Specialized logger for trading operations."""
    
    def __init__(self, name: str = "trading", log_dir: str = "logs"):
        """Initialize trading logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main trading logger
        self.logger = setup_logger(
            name,
            log_file=str(self.log_dir / "trading.log")
        )
        
        # Separate loggers for different activities
        self.trade_logger = setup_logger(
            f"{name}.trades",
            log_file=str(self.log_dir / "trades.log")
        )
        
        self.error_logger = setup_logger(
            f"{name}.errors",
            log_file=str(self.log_dir / "errors.log"),
            level="ERROR"
        )
        
        self.performance_logger = setup_logger(
            f"{name}.performance",
            log_file=str(self.log_dir / "performance.log")
        )
    
    def log_trade(self, symbol: str, action: str, price: float, quantity: int, 
                  timestamp: str, strategy: str, confidence: float, **kwargs) -> None:
        """Log a trade execution."""
        trade_info = {
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity,
            'timestamp': timestamp,
            'strategy': strategy,
            'confidence': confidence,
            **kwargs
        }
        
        trade_msg = f"TRADE: {action} {quantity} {symbol} @ {price} " \
                   f"[{strategy}] confidence: {confidence:.2f}"
        
        if kwargs:
            trade_msg += f" extras: {kwargs}"
        
        self.trade_logger.info(trade_msg)
        self.logger.info(trade_msg)
    
    def log_signal(self, symbol: str, signal_type: str, confidence: float,
                   strategy: str, **kwargs) -> None:
        """Log a trading signal."""
        signal_msg = f"SIGNAL: {signal_type} {symbol} " \
                    f"[{strategy}] confidence: {confidence:.2f}"
        
        if kwargs:
            signal_msg += f" extras: {kwargs}"
        
        self.logger.info(signal_msg)
    
    def log_performance(self, portfolio_value: float, daily_return: float,
                       total_return: float, trades_count: int, **kwargs) -> None:
        """Log performance metrics."""
        perf_msg = f"PERFORMANCE: Portfolio: ${portfolio_value:.2f} " \
                  f"Daily: {daily_return:.2%} Total: {total_return:.2%} " \
                  f"Trades: {trades_count}"
        
        if kwargs:
            perf_msg += f" extras: {kwargs}"
        
        self.performance_logger.info(perf_msg)
        self.logger.info(perf_msg)
    
    def log_error(self, error_msg: str, exception: Exception = None, **kwargs) -> None:
        """Log an error with optional exception details."""
        error_info = f"ERROR: {error_msg}"
        
        if exception:
            error_info += f" Exception: {str(exception)}"
        
        if kwargs:
            error_info += f" extras: {kwargs}"
        
        self.error_logger.error(error_info)
        self.logger.error(error_info)
    
    def log_backtest_result(self, symbol: str, start_date: str, end_date: str,
                           total_return: float, sharpe_ratio: float,
                           max_drawdown: float, **kwargs) -> None:
        """Log backtest results."""
        backtest_msg = f"BACKTEST: {symbol} ({start_date} to {end_date}) " \
                      f"Return: {total_return:.2%} Sharpe: {sharpe_ratio:.2f} " \
                      f"MaxDD: {max_drawdown:.2%}"
        
        if kwargs:
            backtest_msg += f" extras: {kwargs}"
        
        self.performance_logger.info(backtest_msg)
        self.logger.info(backtest_msg)


# Global trading logger instance
trading_logger = TradingLogger()
