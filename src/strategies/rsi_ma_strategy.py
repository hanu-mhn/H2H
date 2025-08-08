"""
RSI and Moving Average crossover trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from .base_strategy import BaseStrategy
from .technical_indicators import TechnicalIndicators
from ..utils.config import Config


class RSIMAStrategy(BaseStrategy):
    """
    RSI and Moving Average crossover trading strategy.
    
    Strategy Rules:
    - Buy Signal: RSI < 30 AND 20-day MA crosses above 50-day MA
    - Sell Signal: RSI > 70 OR 20-day MA crosses below 50-day MA
    - Additional filters: Volume confirmation, trend alignment
    """
    
    def __init__(self):
        """Initialize RSI MA strategy."""
        super().__init__("RSI_MA_Crossover")
        
        # Load configuration
        config = Config()
        strategy_config = config.get_strategy_config()
        
        # Default parameters
        self.parameters = {
            'rsi_period': strategy_config.get('rsi', {}).get('period', 14),
            'rsi_buy_threshold': strategy_config.get('rsi', {}).get('buy_threshold', 30),
            'rsi_sell_threshold': strategy_config.get('rsi', {}).get('sell_threshold', 70),
            'short_ma_period': strategy_config.get('moving_average', {}).get('short_period', 20),
            'long_ma_period': strategy_config.get('moving_average', {}).get('long_period', 50),
            'volume_threshold': 1.2,  # Volume should be 20% above average
            'min_crossover_strength': 0.01,  # Minimum % difference for crossover
            'trend_confirmation': True,  # Use trend confirmation
            'stop_loss_percent': 0.05,  # 5% stop loss
            'take_profit_percent': 0.10,  # 10% take profit
            'max_position_size': 0.1  # 10% of portfolio per trade
        }
        
        self.tech_indicators = TechnicalIndicators()
        
        # Track previous values for crossover detection
        self.previous_short_ma = None
        self.previous_long_ma = None
        self.previous_rsi = None
        
        self.logger.info(f"Initialized {self.name} with parameters: {self.parameters}")
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on RSI and MA crossover.
        
        Args:
            data: Stock data DataFrame with OHLCV columns
        
        Returns:
            Dictionary with signal information
        """
        # Validate input data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not self.validate_data(data, required_columns):
            return {'action': 'HOLD', 'confidence': 0, 'reason': 'Invalid data'}
        
        try:
            # Add technical indicators if not present
            if 'RSI' not in data.columns:
                data = self.tech_indicators.calculate_all_indicators(data)
            
            # Get latest values
            latest_idx = -1
            current_price = data['Close'].iloc[latest_idx]
            current_rsi = data['RSI'].iloc[latest_idx]
            current_short_ma = data[f'SMA_{self.parameters["short_ma_period"]}'].iloc[latest_idx]
            current_long_ma = data[f'SMA_{self.parameters["long_ma_period"]}'].iloc[latest_idx]
            current_volume = data['Volume'].iloc[latest_idx]
            volume_ma = data['Volume_SMA'].iloc[latest_idx] if 'Volume_SMA' in data.columns else data['Volume'].rolling(20).mean().iloc[latest_idx]
            
            # Previous values for crossover detection
            prev_short_ma = data[f'SMA_{self.parameters["short_ma_period"]}'].iloc[-2]
            prev_long_ma = data[f'SMA_{self.parameters["long_ma_period"]}'].iloc[-2]
            
            # Check for NaN values
            if pd.isna(current_rsi) or pd.isna(current_short_ma) or pd.isna(current_long_ma):
                return {'action': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data for indicators'}
            
            # Initialize signal
            signal = {
                'action': 'HOLD',
                'confidence': 0,
                'price': current_price,
                'timestamp': data.index[latest_idx] if hasattr(data.index[latest_idx], 'strftime') else datetime.now(),
                'indicators': {
                    'rsi': current_rsi,
                    'short_ma': current_short_ma,
                    'long_ma': current_long_ma,
                    'volume_ratio': current_volume / volume_ma
                },
                'reason': '',
                'stop_loss': None,
                'take_profit': None
            }
            
            # Calculate signal components
            buy_signal, buy_confidence, buy_reasons = self._calculate_buy_signal(
                current_rsi, current_short_ma, current_long_ma, prev_short_ma, prev_long_ma,
                current_volume, volume_ma, data
            )
            
            sell_signal, sell_confidence, sell_reasons = self._calculate_sell_signal(
                current_rsi, current_short_ma, current_long_ma, prev_short_ma, prev_long_ma,
                current_volume, volume_ma, data
            )
            
            # Determine final signal
            if buy_signal and buy_confidence > sell_confidence:
                signal['action'] = 'BUY'
                signal['confidence'] = buy_confidence
                signal['reason'] = '; '.join(buy_reasons)
                signal['stop_loss'] = self.calculate_stop_loss(current_price, 'BUY', 
                                                             data.get('ATR', {}).iloc[latest_idx] if 'ATR' in data.columns else None)
                signal['take_profit'] = self.calculate_take_profit(current_price, 'BUY')
                
            elif sell_signal and sell_confidence > buy_confidence:
                signal['action'] = 'SELL'
                signal['confidence'] = sell_confidence
                signal['reason'] = '; '.join(sell_reasons)
                signal['stop_loss'] = self.calculate_stop_loss(current_price, 'SELL',
                                                             data.get('ATR', {}).iloc[latest_idx] if 'ATR' in data.columns else None)
                signal['take_profit'] = self.calculate_take_profit(current_price, 'SELL')
            
            # Log signal if not HOLD
            if signal['action'] != 'HOLD':
                self.log_signal(signal, data)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0, 'reason': f'Error: {str(e)}'}
    
    def _calculate_buy_signal(
        self,
        rsi: float,
        short_ma: float,
        long_ma: float,
        prev_short_ma: float,
        prev_long_ma: float,
        volume: float,
        volume_ma: float,
        data: pd.DataFrame
    ) -> tuple:
        """Calculate buy signal components."""
        confidence = 0
        reasons = []
        
        # RSI oversold condition
        rsi_oversold = rsi < self.parameters['rsi_buy_threshold']
        if rsi_oversold:
            confidence += 0.3
            reasons.append(f"RSI oversold ({rsi:.1f})")
        
        # Moving average crossover (bullish)
        ma_crossover = (short_ma > long_ma and 
                       prev_short_ma <= prev_long_ma and
                       abs(short_ma - long_ma) / long_ma > self.parameters['min_crossover_strength'])
        
        if ma_crossover:
            confidence += 0.4
            reasons.append("Bullish MA crossover")
        elif short_ma > long_ma:
            confidence += 0.2
            reasons.append("Short MA above Long MA")
        
        # Volume confirmation
        volume_confirmation = volume / volume_ma > self.parameters['volume_threshold']
        if volume_confirmation:
            confidence += 0.2
            reasons.append(f"High volume ({volume/volume_ma:.1f}x)")
        
        # Trend confirmation
        if self.parameters['trend_confirmation']:
            trend_bullish = self._is_trend_bullish(data)
            if trend_bullish:
                confidence += 0.1
                reasons.append("Bullish trend")
        
        # Additional technical confirmations
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_Signal'].iloc[-1]
            if macd > macd_signal:
                confidence += 0.1
                reasons.append("MACD bullish")
        
        if 'ADX' in data.columns:
            adx = data['ADX'].iloc[-1]
            if adx > 25:  # Strong trend
                confidence += 0.1
                reasons.append(f"Strong trend (ADX: {adx:.1f})")
        
        # Normalize confidence to 0-1 range
        confidence = min(confidence, 1.0)
        
        # Buy signal requires RSI oversold OR MA crossover with minimum confidence
        buy_signal = (rsi_oversold or ma_crossover) and confidence >= 0.5
        
        return buy_signal, confidence, reasons
    
    def _calculate_sell_signal(
        self,
        rsi: float,
        short_ma: float,
        long_ma: float,
        prev_short_ma: float,
        prev_long_ma: float,
        volume: float,
        volume_ma: float,
        data: pd.DataFrame
    ) -> tuple:
        """Calculate sell signal components."""
        confidence = 0
        reasons = []
        
        # RSI overbought condition
        rsi_overbought = rsi > self.parameters['rsi_sell_threshold']
        if rsi_overbought:
            confidence += 0.4
            reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # Moving average crossover (bearish)
        ma_crossover = (short_ma < long_ma and 
                       prev_short_ma >= prev_long_ma and
                       abs(short_ma - long_ma) / long_ma > self.parameters['min_crossover_strength'])
        
        if ma_crossover:
            confidence += 0.5
            reasons.append("Bearish MA crossover")
        elif short_ma < long_ma:
            confidence += 0.2
            reasons.append("Short MA below Long MA")
        
        # Volume confirmation
        volume_confirmation = volume / volume_ma > self.parameters['volume_threshold']
        if volume_confirmation:
            confidence += 0.1
            reasons.append(f"High volume ({volume/volume_ma:.1f}x)")
        
        # Trend confirmation
        if self.parameters['trend_confirmation']:
            trend_bearish = self._is_trend_bearish(data)
            if trend_bearish:
                confidence += 0.1
                reasons.append("Bearish trend")
        
        # Additional technical confirmations
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_Signal'].iloc[-1]
            if macd < macd_signal:
                confidence += 0.1
                reasons.append("MACD bearish")
        
        # Normalize confidence to 0-1 range
        confidence = min(confidence, 1.0)
        
        # Sell signal requires RSI overbought OR MA crossover with minimum confidence
        sell_signal = (rsi_overbought or ma_crossover) and confidence >= 0.4
        
        return sell_signal, confidence, reasons
    
    def _is_trend_bullish(self, data: pd.DataFrame) -> bool:
        """Check if overall trend is bullish."""
        try:
            # Use multiple timeframe analysis
            close = data['Close']
            
            # Short-term trend (last 10 days)
            short_trend = close.iloc[-1] > close.iloc[-10]
            
            # Medium-term trend (last 20 days)
            medium_trend = close.rolling(5).mean().iloc[-1] > close.rolling(5).mean().iloc[-20]
            
            # Long-term trend (price above 200 MA if available)
            if 'SMA_200' in data.columns:
                long_trend = close.iloc[-1] > data['SMA_200'].iloc[-1]
                return short_trend and medium_trend and long_trend
            
            return short_trend and medium_trend
            
        except Exception:
            return False
    
    def _is_trend_bearish(self, data: pd.DataFrame) -> bool:
        """Check if overall trend is bearish."""
        try:
            # Use multiple timeframe analysis
            close = data['Close']
            
            # Short-term trend (last 10 days)
            short_trend = close.iloc[-1] < close.iloc[-10]
            
            # Medium-term trend (last 20 days)
            medium_trend = close.rolling(5).mean().iloc[-1] < close.rolling(5).mean().iloc[-20]
            
            # Long-term trend (price below 200 MA if available)
            if 'SMA_200' in data.columns:
                long_trend = close.iloc[-1] < data['SMA_200'].iloc[-1]
                return short_trend and medium_trend and long_trend
            
            return short_trend and medium_trend
            
        except Exception:
            return False
    
    def calculate_position_size(self, signal: Dict[str, Any], portfolio_value: float) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
        
        Returns:
            Number of shares to trade
        """
        if signal['action'] == 'HOLD':
            return 0
        
        # Maximum position size as percentage of portfolio
        max_position_value = portfolio_value * self.parameters['max_position_size']
        
        # Adjust based on confidence
        confidence_multiplier = signal.get('confidence', 0.5)
        position_value = max_position_value * confidence_multiplier
        
        # Calculate number of shares
        price = signal['price']
        shares = int(position_value / price)
        
        # Ensure minimum position size
        if shares > 0 and shares * price < 1000:  # Minimum ₹1000 position
            shares = max(1, int(1000 / price))
        
        self.logger.info(f"Position size calculated: {shares} shares for {signal['action']} signal")
        return shares
    
    def update_strategy_performance(self, trades: list) -> None:
        """
        Update strategy performance metrics.
        
        Args:
            trades: List of completed trades
        """
        if not trades:
            return
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(trade.get('pnl', 0) for trade in trades)
        avg_win = np.mean([trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([trade['pnl'] for trade in trades if trade.get('pnl', 0) < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf')
        
        # Update performance metrics
        self.performance_metrics.update({
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'last_updated': datetime.now()
        })
        
        self.logger.info(f"Strategy performance updated: Win rate: {win_rate:.2%}, Total PnL: ₹{total_pnl:.2f}")
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy summary including parameters and performance."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'performance': self.performance_metrics,
            'description': 'RSI oversold + Moving Average crossover strategy with volume and trend confirmation'
        }
