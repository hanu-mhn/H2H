"""
Data processing and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

from ..utils.helpers import validate_dataframe, calculate_returns
from ..utils.logger import setup_logger


class DataProcessor:
    """Data processing and preprocessing for trading strategies."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = setup_logger(__name__)
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: Stock data DataFrame with OHLCV columns
        
        Returns:
            DataFrame with additional technical indicators
        """
        if not validate_dataframe(df, ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Invalid DataFrame for technical analysis")
        
        data = df.copy()
        
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # MACD
        macd_data = self._calculate_macd(data['Close'])
        data['MACD'] = macd_data['MACD']
        data['MACD_Signal'] = macd_data['Signal']
        data['MACD_Histogram'] = macd_data['Histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(data['Close'])
        data['BB_Upper'] = bb_data['Upper']
        data['BB_Middle'] = bb_data['Middle']
        data['BB_Lower'] = bb_data['Lower']
        data['BB_Width'] = bb_data['Width']
        data['BB_Position'] = bb_data['Position']
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Price change indicators
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
        data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
        
        # Volatility
        data['Volatility'] = data['Price_Change'].rolling(window=20).std()
        
        # Support and Resistance levels
        data['Support'] = data['Low'].rolling(window=20).min()
        data['Resistance'] = data['High'].rolling(window=20).max()
        
        # Average True Range (ATR)
        data['ATR'] = self._calculate_atr(data)
        
        self.logger.debug(f"Added technical indicators. DataFrame shape: {data.shape}")
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Calculate MACD indicators."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'MACD': macd,
            'Signal': macd_signal,
            'Histogram': macd_histogram
        }
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Additional metrics
        width = (upper_band - lower_band) / rolling_mean
        position = (prices - lower_band) / (upper_band - lower_band)
        
        return {
            'Upper': upper_band,
            'Middle': rolling_mean,
            'Lower': lower_band,
            'Width': width,
            'Position': position
        }
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def prepare_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning models.
        
        Args:
            df: DataFrame with technical indicators
        
        Returns:
            DataFrame with ML-ready features
        """
        # Ensure we have technical indicators
        if 'RSI' not in df.columns:
            df = self.add_technical_indicators(df)
        
        features = df.copy()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'Close_lag_{lag}'] = features['Close'].shift(lag)
            features[f'Volume_lag_{lag}'] = features['Volume'].shift(lag)
            features[f'RSI_lag_{lag}'] = features['RSI'].shift(lag)
        
        # Moving average ratios
        features['Close_SMA20_ratio'] = features['Close'] / features['SMA_20']
        features['Close_SMA50_ratio'] = features['Close'] / features['SMA_50']
        features['SMA20_SMA50_ratio'] = features['SMA_20'] / features['SMA_50']
        
        # Momentum features
        features['ROC_5'] = ((features['Close'] - features['Close'].shift(5)) / 
                           features['Close'].shift(5)) * 100
        features['ROC_10'] = ((features['Close'] - features['Close'].shift(10)) / 
                            features['Close'].shift(10)) * 100
        
        # Volatility features
        features['Price_Range'] = (features['High'] - features['Low']) / features['Close']
        features['Gap'] = (features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1)
        
        # Market cap proxy (Volume * Price)
        features['Market_Cap_Proxy'] = features['Volume'] * features['Close']
        
        # Time-based features
        features['DayOfWeek'] = features.index.dayofweek
        features['Month'] = features.index.month
        features['Quarter'] = features.index.quarter
        
        # Target variable for classification (next day direction)
        features['Next_Day_Return'] = features['Close'].shift(-1) / features['Close'] - 1
        features['Target'] = (features['Next_Day_Return'] > 0).astype(int)
        
        # Drop rows with NaN values
        features = features.dropna()
        
        self.logger.debug(f"Prepared {len(features)} samples with {features.shape[1]} features")
        return features
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Fraction of data for testing
            shuffle: Whether to shuffle the data (not recommended for time series)
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        self.logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
        return train_df, test_df
    
    def normalize_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Normalize features using training set statistics.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            feature_columns: List of columns to normalize
        
        Returns:
            Tuple of (normalized_train_df, normalized_test_df, normalization_params)
        """
        train_normalized = train_df.copy()
        test_normalized = test_df.copy()
        normalization_params = {}
        
        for col in feature_columns:
            if col in train_df.columns and col in test_df.columns:
                mean = train_df[col].mean()
                std = train_df[col].std()
                
                if std > 0:  # Avoid division by zero
                    train_normalized[col] = (train_df[col] - mean) / std
                    test_normalized[col] = (test_df[col] - mean) / std
                    
                    normalization_params[col] = {'mean': mean, 'std': std}
        
        return train_normalized, test_normalized, normalization_params
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect outliers in specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Method to use ('iqr', 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outlier flags
        """
        outlier_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_df[f'{col}_outlier'] = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                )
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_df[f'{col}_outlier'] = z_scores > threshold
        
        return outlier_df
    
    def resample_data(
        self,
        df: pd.DataFrame,
        frequency: str = 'D',
        agg_method: str = 'last'
    ) -> pd.DataFrame:
        """
        Resample data to different frequency.
        
        Args:
            df: Input DataFrame with datetime index
            frequency: Target frequency ('D', 'W', 'M', etc.)
            agg_method: Aggregation method ('last', 'first', 'mean', 'ohlc')
        
        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for resampling")
        
        if agg_method == 'ohlc':
            # For OHLC data
            agg_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
            # Add other columns with 'last' method
            for col in df.columns:
                if col not in agg_dict:
                    agg_dict[col] = 'last'
            
            resampled = df.resample(frequency).agg(agg_dict)
        else:
            resampled = df.resample(frequency).agg(agg_method)
        
        return resampled.dropna()
    
    def calculate_market_regime(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Calculate market regime indicators.
        
        Args:
            df: Input DataFrame with price data
            window: Window for regime calculation
        
        Returns:
            DataFrame with regime indicators
        """
        regime_df = df.copy()
        
        # Trend regime based on moving averages
        regime_df['Trend_Regime'] = np.where(
            regime_df['Close'] > regime_df['SMA_50'], 'Uptrend', 'Downtrend'
        )
        
        # Volatility regime
        volatility_ma = regime_df['Volatility'].rolling(window=window).mean()
        regime_df['Volatility_Regime'] = np.where(
            regime_df['Volatility'] > volatility_ma, 'High_Vol', 'Low_Vol'
        )
        
        # Volume regime
        volume_ma = regime_df['Volume'].rolling(window=window).mean()
        regime_df['Volume_Regime'] = np.where(
            regime_df['Volume'] > volume_ma, 'High_Volume', 'Low_Volume'
        )
        
        return regime_df
