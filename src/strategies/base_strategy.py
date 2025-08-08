"""
Base strategy class for all trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
import logging

from ..utils.logger import setup_logger


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.logger = setup_logger(f"strategy.{name}")
        self.parameters = {}
        self.performance_metrics = {}
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on data.
        
        Args:
            data: Stock data with technical indicators
        
        Returns:
            Dictionary containing signal information
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict[str, Any], portfolio_value: float) -> int:
        """
        Calculate position size for a given signal.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
        
        Returns:
            Number of shares to trade
        """
        pass
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters: Dictionary of parameters
        """
        self.parameters.update(parameters)
        self.logger.info(f"Updated parameters for {self.name}: {parameters}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.parameters.copy()
    
    def validate_data(self, data: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate input data for strategy.
        
        Args:
            data: Input DataFrame
            required_columns: List of required columns
        
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.error("Data is None or empty")
            return False
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        if len(data) < 50:  # Minimum data points
            self.logger.warning(f"Insufficient data points: {len(data)}")
            return False
        
        return True
    
    def calculate_stop_loss(self, entry_price: float, action: str, atr: float = None) -> float:
        """
        Calculate stop-loss price.
        
        Args:
            entry_price: Entry price
            action: 'BUY' or 'SELL'
            atr: Average True Range for dynamic stop-loss
        
        Returns:
            Stop-loss price
        """
        stop_loss_pct = self.parameters.get('stop_loss_percent', 0.05)
        
        if atr is not None:
            # Use ATR-based stop-loss
            atr_multiplier = self.parameters.get('atr_multiplier', 2.0)
            stop_distance = atr * atr_multiplier
        else:
            # Use percentage-based stop-loss
            stop_distance = entry_price * stop_loss_pct
        
        if action == 'BUY':
            return entry_price - stop_distance
        else:  # SELL
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, action: str) -> float:
        """
        Calculate take-profit price.
        
        Args:
            entry_price: Entry price
            action: 'BUY' or 'SELL'
        
        Returns:
            Take-profit price
        """
        take_profit_pct = self.parameters.get('take_profit_percent', 0.10)
        take_profit_distance = entry_price * take_profit_pct
        
        if action == 'BUY':
            return entry_price + take_profit_distance
        else:  # SELL
            return entry_price - take_profit_distance
    
    def update_performance(self, metric: str, value: float) -> None:
        """
        Update performance metrics.
        
        Args:
            metric: Metric name
            value: Metric value
        """
        self.performance_metrics[metric] = value
    
    def get_performance(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def reset_performance(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics.clear()
    
    def log_signal(self, signal: Dict[str, Any], data: pd.DataFrame) -> None:
        """
        Log signal details.
        
        Args:
            signal: Generated signal
            data: Input data used for signal generation
        """
        if signal['action'] != 'HOLD':
            latest_price = data['Close'].iloc[-1]
            timestamp = data.index[-1] if hasattr(data.index[-1], 'strftime') else 'N/A'
            
            self.logger.info(
                f"Signal generated - Action: {signal['action']}, "
                f"Price: {latest_price:.2f}, "
                f"Confidence: {signal.get('confidence', 0):.2f}, "
                f"Timestamp: {timestamp}"
            )
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name} Strategy"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"
