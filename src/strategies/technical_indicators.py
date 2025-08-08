"""
Technical indicators for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import ta
import logging

from ..utils.logger import setup_logger


class TechnicalIndicators:
    """Technical analysis indicators calculator."""
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        self.logger = setup_logger(__name__)
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series (usually Close)
            window: RSI period
        
        Returns:
            RSI series
        """
        try:
            return ta.momentum.RSIIndicator(close=prices, window=window).rsi()
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            # Return a series with NaN values if calculation fails
            return pd.Series([np.nan] * len(prices), index=prices.index)
    
    def calculate_moving_averages(self, prices: pd.Series, windows: list = [20, 50]) -> Dict[str, pd.Series]:
        """
        Calculate multiple moving averages.
        
        Args:
            prices: Price series
            windows: List of periods for moving averages
        
        Returns:
            Dictionary of moving average series
        """
        mas = {}
        for window in windows:
            mas[f'SMA_{window}'] = ta.trend.SMAIndicator(close=prices, window=window).sma_indicator()
            mas[f'EMA_{window}'] = ta.trend.EMAIndicator(close=prices, window=window).ema_indicator()
        
        return mas
    
    def calculate_macd(
        self,
        prices: pd.Series,
        window_fast: int = 12,
        window_slow: int = 26,
        window_sign: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD indicators.
        
        Args:
            prices: Price series
            window_fast: Fast EMA period
            window_slow: Slow EMA period
            window_sign: Signal line period
        
        Returns:
            Dictionary with MACD, Signal, and Histogram
        """
        macd_indicator = ta.trend.MACD(
            close=prices,
            window_fast=window_fast,
            window_slow=window_slow,
            window_sign=window_sign
        )
        
        return {
            'MACD': macd_indicator.macd(),
            'MACD_Signal': macd_indicator.macd_signal(),
            'MACD_Histogram': macd_indicator.macd_diff()
        }
    
    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        window: int = 20,
        window_dev: int = 2
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            window: Moving average period
            window_dev: Standard deviation multiplier
        
        Returns:
            Dictionary with Upper, Middle, Lower bands and additional metrics
        """
        bb_indicator = ta.volatility.BollingerBands(
            close=prices,
            window=window,
            window_dev=window_dev
        )
        
        upper = bb_indicator.bollinger_hband()
        middle = bb_indicator.bollinger_mavg()
        lower = bb_indicator.bollinger_lband()
        
        # Additional calculations
        width = (upper - lower) / middle
        position = (prices - lower) / (upper - lower)
        
        return {
            'BB_Upper': upper,
            'BB_Middle': middle,
            'BB_Lower': lower,
            'BB_Width': width,
            'BB_Position': position,
            'BB_Squeeze': bb_indicator.bollinger_wband() < 0.1  # Width < 10%
        }
    
    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        smooth_window: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Stochastic period
            smooth_window: Smoothing period
        
        Returns:
            Dictionary with %K and %D lines
        """
        stoch_indicator = ta.momentum.StochasticOscillator(
            high=high,
            low=low,
            close=close,
            window=window,
            smooth_window=smooth_window
        )
        
        return {
            'Stoch_K': stoch_indicator.stoch(),
            'Stoch_D': stoch_indicator.stoch_signal()
        }
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: ATR period
        
        Returns:
            ATR series
        """
        return ta.volatility.AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=window
        ).average_true_range()
    
    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index and related indicators.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: ADX period
        
        Returns:
            Dictionary with ADX, +DI, -DI
        """
        adx_indicator = ta.trend.ADXIndicator(
            high=high,
            low=low,
            close=close,
            window=window
        )
        
        return {
            'ADX': adx_indicator.adx(),
            'ADX_Plus': adx_indicator.adx_pos(),
            'ADX_Minus': adx_indicator.adx_neg()
        }
    
    def calculate_volume_indicators(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators.
        
        Args:
            close: Close price series
            volume: Volume series
        
        Returns:
            Dictionary with volume indicators
        """
        try:
            # On-Balance Volume
            obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
            
            # Volume Weighted Average Price (using simple calculation)
            vwap = (close * volume).cumsum() / volume.cumsum()
            
            # Money Flow Index
            mfi = ta.volume.MFIIndicator(
                high=close,  # Simplified - ideally use actual high/low
                low=close,
                close=close,
                volume=volume,
                window=14
            ).money_flow_index()
            
            return {
                'OBV': obv,
                'VWAP': vwap,
                'MFI': mfi,
                'Volume_SMA': volume.rolling(window=20).mean(),
                'Volume_Ratio': volume / volume.rolling(window=20).mean()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {str(e)}")
            # Return default values if calculation fails
            return {
                'OBV': pd.Series([np.nan] * len(close), index=close.index),
                'VWAP': pd.Series([np.nan] * len(close), index=close.index),
                'MFI': pd.Series([np.nan] * len(close), index=close.index),
                'Volume_SMA': volume.rolling(window=20).mean(),
                'Volume_Ratio': pd.Series([1.0] * len(close), index=close.index)
            }
    
    def calculate_momentum_indicators(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate momentum indicators.
        
        Args:
            close: Close price series
        
        Returns:
            Dictionary with momentum indicators
        """
        # Rate of Change
        roc_5 = ta.momentum.ROCIndicator(close=close, window=5).roc()
        roc_10 = ta.momentum.ROCIndicator(close=close, window=10).roc()
        
        # Williams %R
        williams_r = ta.momentum.WilliamsRIndicator(
            high=close,  # Simplified
            low=close,
            close=close,
            lbp=14
        ).williams_r()
        
        # Commodity Channel Index
        cci = ta.trend.CCIIndicator(
            high=close,
            low=close,
            close=close,
            window=20
        ).cci()
        
        return {
            'ROC_5': roc_5,
            'ROC_10': roc_10,
            'Williams_R': williams_r,
            'CCI': cci
        }
    
    def calculate_support_resistance(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 20
    ) -> Dict[str, pd.Series]:
        """
        Calculate dynamic support and resistance levels.
        
        Args:
            high: High price series
            low: Low price series
            window: Lookback period
        
        Returns:
            Dictionary with support and resistance levels
        """
        resistance = high.rolling(window=window).max()
        support = low.rolling(window=window).min()
        
        # Pivot points (simplified daily calculation)
        pivot = (high + low + high.shift(1)) / 3  # Using close as high for simplification
        
        return {
            'Resistance': resistance,
            'Support': support,
            'Pivot': pivot
        }
    
    def calculate_trend_indicators(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate trend-following indicators.
        
        Args:
            close: Close price series
        
        Returns:
            Dictionary with trend indicators
        """
        # Parabolic SAR
        psar = ta.trend.PSARIndicator(
            high=close,  # Simplified
            low=close,
            close=close
        ).psar()
        
        # Aroon Indicator
        aroon = ta.trend.AroonIndicator(
            high=close,
            low=close,
            window=25
        )
        
        return {
            'PSAR': psar,
            'Aroon_Up': aroon.aroon_up(),
            'Aroon_Down': aroon.aroon_down(),
            'Aroon_Indicator': aroon.aroon_indicator()
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            DataFrame with all indicators added
        """
        try:
            result = df.copy()
            
            # Basic indicators
            result['RSI'] = self.calculate_rsi(df['Close'])
            
            # Moving averages
            try:
                ma_dict = self.calculate_moving_averages(df['Close'])
                for key, value in ma_dict.items():
                    result[key] = value
            except Exception as e:
                self.logger.error(f"Error calculating moving averages: {str(e)}")
            
            # MACD
            try:
                macd_dict = self.calculate_macd(df['Close'])
                for key, value in macd_dict.items():
                    result[key] = value
            except Exception as e:
                self.logger.error(f"Error calculating MACD: {str(e)}")
            
            # Bollinger Bands
            try:
                bb_dict = self.calculate_bollinger_bands(df['Close'])
                for key, value in bb_dict.items():
                    result[key] = value
            except Exception as e:
                self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            
            # Stochastic
            try:
                stoch_dict = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
                for key, value in stoch_dict.items():
                    result[key] = value
            except Exception as e:
                self.logger.error(f"Error calculating Stochastic: {str(e)}")
            
            # ATR
            try:
                result['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
            except Exception as e:
                self.logger.error(f"Error calculating ATR: {str(e)}")
            
            # ADX
            try:
                adx_dict = self.calculate_adx(df['High'], df['Low'], df['Close'])
                for key, value in adx_dict.items():
                    result[key] = value
            except Exception as e:
                self.logger.error(f"Error calculating ADX: {str(e)}")
            
            # Volume indicators
            if 'Volume' in df.columns:
                vol_dict = self.calculate_volume_indicators(df['Close'], df['Volume'])
                for key, value in vol_dict.items():
                    result[key] = value
            
            # Momentum indicators
            try:
                momentum_dict = self.calculate_momentum_indicators(df['Close'])
                for key, value in momentum_dict.items():
                    result[key] = value
            except Exception as e:
                self.logger.error(f"Error calculating momentum indicators: {str(e)}")
            
            # Support/Resistance
            try:
                sr_dict = self.calculate_support_resistance(df['High'], df['Low'])
                for key, value in sr_dict.items():
                    result[key] = value
            except Exception as e:
                self.logger.error(f"Error calculating support/resistance: {str(e)}")
            
            # Trend indicators
            try:
                trend_dict = self.calculate_trend_indicators(df['Close'])
                for key, value in trend_dict.items():
                    result[key] = value
            except Exception as e:
                self.logger.error(f"Error calculating trend indicators: {str(e)}")
            
            self.logger.info(f"Calculated technical indicators. Shape: {result.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df
