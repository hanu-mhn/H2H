"""
Feature engineering for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime, timedelta

from ..utils.logger import setup_logger
from ..strategies.technical_indicators import TechnicalIndicators


class FeatureEngineer:
    """Feature engineering for ML-based trading models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.logger = setup_logger(__name__)
        self.tech_indicators = TechnicalIndicators()
        
        # Feature categories
        self.price_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Price_Change', 'Log_Return', 'High_Low_Ratio'
        ]
        
        self.technical_features = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'BB_Width', 'Stoch_K', 'Stoch_D',
            'ADX', 'ATR', 'OBV', 'MFI'
        ]
        
        self.moving_average_features = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
            'Close_SMA20_Ratio', 'Close_SMA50_Ratio', 'SMA20_SMA50_Ratio'
        ]
        
        self.volatility_features = [
            'Volatility_5', 'Volatility_10', 'Volatility_20',
            'ATR_Ratio', 'BB_Width'
        ]
        
        self.volume_features = [
            'Volume_SMA', 'Volume_Ratio', 'OBV', 'VWAP', 'MFI'
        ]
    
    def create_features(self, df: pd.DataFrame, target_days: int = 1) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models.
        
        Args:
            df: Stock data DataFrame
            target_days: Number of days ahead to predict
        
        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info(f"Creating features for {len(df)} data points")
            
            # Start with original data
            features = df.copy()
            
            # Add technical indicators if not present
            if 'RSI' not in features.columns:
                features = self.tech_indicators.calculate_all_indicators(features)
            
            # Price-based features
            features = self._add_price_features(features)
            
            # Lag features
            features = self._add_lag_features(features)
            
            # Rolling statistics
            features = self._add_rolling_features(features)
            
            # Momentum features
            features = self._add_momentum_features(features)
            
            # Volatility features
            features = self._add_volatility_features(features)
            
            # Volume features
            features = self._add_volume_features(features)
            
            # Market structure features
            features = self._add_market_structure_features(features)
            
            # Time-based features
            features = self._add_time_features(features)
            
            # Cross-sectional features
            features = self._add_cross_sectional_features(features)
            
            # Target variables
            features = self._add_target_variables(features, target_days)
            
            # Remove rows with NaN values
            initial_length = len(features)
            features = features.dropna()
            final_length = len(features)
            
            self.logger.info(f"Feature engineering complete: {initial_length} -> {final_length} samples, "
                           f"{features.shape[1]} features")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        features = df.copy()
        
        # Price changes
        features['Price_Change'] = features['Close'].pct_change()
        features['Log_Return'] = np.log(features['Close'] / features['Close'].shift(1))
        
        # Price ratios
        features['High_Low_Ratio'] = (features['High'] - features['Low']) / features['Close']
        features['Open_Close_Ratio'] = features['Open'] / features['Close']
        features['High_Close_Ratio'] = features['High'] / features['Close']
        features['Low_Close_Ratio'] = features['Low'] / features['Close']
        
        # Gap analysis
        features['Gap'] = (features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1)
        features['Gap_Up'] = (features['Gap'] > 0.02).astype(int)  # 2% gap up
        features['Gap_Down'] = (features['Gap'] < -0.02).astype(int)  # 2% gap down
        
        return features
    
    def _add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged features."""
        features = df.copy()
        
        key_columns = ['Close', 'Volume', 'RSI', 'MACD', 'Price_Change']
        
        for lag in lags:
            for col in key_columns:
                if col in features.columns:
                    features[f'{col}_lag_{lag}'] = features[col].shift(lag)
        
        return features
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features."""
        features = df.copy()
        
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Rolling means
            features[f'Close_SMA_{window}'] = features['Close'].rolling(window).mean()
            features[f'Volume_SMA_{window}'] = features['Volume'].rolling(window).mean()
            
            # Rolling standard deviations
            features[f'Close_STD_{window}'] = features['Close'].rolling(window).std()
            features[f'Return_STD_{window}'] = features['Price_Change'].rolling(window).std()
            
            # Rolling min/max
            features[f'Close_Min_{window}'] = features['Close'].rolling(window).min()
            features[f'Close_Max_{window}'] = features['Close'].rolling(window).max()
            
            # Position within range
            features[f'Close_Position_{window}'] = (
                (features['Close'] - features[f'Close_Min_{window}']) / 
                (features[f'Close_Max_{window}'] - features[f'Close_Min_{window}'])
            )
        
        # Moving average ratios
        features['Close_SMA5_Ratio'] = features['Close'] / features['Close_SMA_5']
        features['Close_SMA20_Ratio'] = features['Close'] / features['Close_SMA_20']
        features['Close_SMA50_Ratio'] = features['Close'] / features['Close_SMA_50']
        features['SMA5_SMA20_Ratio'] = features['Close_SMA_5'] / features['Close_SMA_20']
        features['SMA20_SMA50_Ratio'] = features['Close_SMA_20'] / features['Close_SMA_50']
        
        return features
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        features = df.copy()
        
        # Rate of change
        for period in [3, 5, 10, 20]:
            features[f'ROC_{period}'] = (
                (features['Close'] - features['Close'].shift(period)) / 
                features['Close'].shift(period) * 100
            )
        
        # Momentum oscillators
        features['RSI_Momentum'] = features['RSI'].diff()
        features['MACD_Momentum'] = features['MACD'].diff() if 'MACD' in features.columns else 0
        
        # Price momentum
        features['Price_Momentum_5'] = features['Close'] / features['Close'].shift(5) - 1
        features['Price_Momentum_10'] = features['Close'] / features['Close'].shift(10) - 1
        
        # Acceleration (second derivative)
        features['Price_Acceleration'] = features['Price_Change'].diff()
        
        return features
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        features = df.copy()
        
        # Historical volatility
        for window in [5, 10, 20]:
            features[f'Volatility_{window}'] = features['Price_Change'].rolling(window).std() * np.sqrt(252)
        
        # ATR-based features
        if 'ATR' in features.columns:
            features['ATR_Ratio'] = features['ATR'] / features['Close']
            features['ATR_Percentile'] = features['ATR'].rolling(50).rank(pct=True)
        
        # Bollinger Band features
        if 'BB_Width' in features.columns:
            features['BB_Squeeze'] = (features['BB_Width'] < features['BB_Width'].rolling(20).quantile(0.2)).astype(int)
        
        # Volatility regime
        vol_20 = features['Volatility_20'] if 'Volatility_20' in features.columns else features['Price_Change'].rolling(20).std()
        vol_ma = vol_20.rolling(50).mean()
        features['High_Vol_Regime'] = (vol_20 > vol_ma * 1.5).astype(int)
        
        return features
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        features = df.copy()
        
        # Volume ratios
        features['Volume_Ratio_5'] = features['Volume'] / features['Volume'].rolling(5).mean()
        features['Volume_Ratio_20'] = features['Volume'] / features['Volume'].rolling(20).mean()
        
        # Volume trend
        features['Volume_Trend'] = features['Volume'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
        )
        
        # Price-volume relationship
        features['PV_Correlation'] = features['Close'].rolling(20).corr(features['Volume'])
        
        # Volume momentum
        features['Volume_Momentum'] = features['Volume'].pct_change()
        
        # On-balance volume momentum
        if 'OBV' in features.columns:
            features['OBV_Momentum'] = features['OBV'].pct_change()
        
        return features
    
    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features."""
        features = df.copy()
        
        # Higher highs and lower lows
        features['Higher_High'] = (features['High'] > features['High'].shift(1)).astype(int)
        features['Lower_Low'] = (features['Low'] < features['Low'].shift(1)).astype(int)
        
        # Support and resistance levels
        features['Near_Resistance'] = (
            features['Close'] > features['Close'].rolling(20).max() * 0.98
        ).astype(int)
        
        features['Near_Support'] = (
            features['Close'] < features['Close'].rolling(20).min() * 1.02
        ).astype(int)
        
        # Trend strength
        if 'ADX' in features.columns:
            features['Strong_Trend'] = (features['ADX'] > 25).astype(int)
            features['Weak_Trend'] = (features['ADX'] < 20).astype(int)
        
        # Market phases
        features['Bull_Phase'] = (
            (features['Close'] > features['Close_SMA_20']) & 
            (features['Close_SMA_20'] > features['Close_SMA_50'])
        ).astype(int)
        
        features['Bear_Phase'] = (
            (features['Close'] < features['Close_SMA_20']) & 
            (features['Close_SMA_20'] < features['Close_SMA_50'])
        ).astype(int)
        
        return features
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        features = df.copy()
        
        # Day of week effect
        features['Day_of_Week'] = features.index.dayofweek
        features['Monday'] = (features['Day_of_Week'] == 0).astype(int)
        features['Friday'] = (features['Day_of_Week'] == 4).astype(int)
        
        # Month effect
        features['Month'] = features.index.month
        features['Quarter'] = features.index.quarter
        
        # Beginning/End of month
        features['Month_Start'] = (features.index.day <= 5).astype(int)
        features['Month_End'] = (features.index.day >= 25).astype(int)
        
        # Holiday effects (simplified)
        features['Year_End'] = (features.index.month == 12).astype(int)
        features['Year_Start'] = (features.index.month == 1).astype(int)
        
        return features
    
    def _add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional features (market relative)."""
        features = df.copy()
        
        # These would typically require market index data
        # For now, we'll create placeholders that can be enhanced
        
        # Relative strength (vs own history)
        features['Relative_Strength'] = features['Close'] / features['Close'].rolling(252).mean()
        
        # Performance ranking (vs own historical performance)
        features['Performance_Rank'] = features['Price_Change'].rolling(50).rank(pct=True)
        
        return features
    
    def _add_target_variables(self, df: pd.DataFrame, target_days: int = 1) -> pd.DataFrame:
        """Add target variables for ML models."""
        features = df.copy()
        
        # Future returns
        features['Future_Return'] = features['Close'].shift(-target_days) / features['Close'] - 1
        
        # Binary classification targets
        features['Future_Up'] = (features['Future_Return'] > 0).astype(int)
        features['Future_Strong_Up'] = (features['Future_Return'] > 0.02).astype(int)  # >2%
        features['Future_Down'] = (features['Future_Return'] < 0).astype(int)
        features['Future_Strong_Down'] = (features['Future_Return'] < -0.02).astype(int)  # <-2%
        
        # Multi-class target
        features['Future_Direction'] = np.where(
            features['Future_Return'] > 0.01, 2,  # Strong up
            np.where(features['Future_Return'] > 0, 1,  # Weak up
                    np.where(features['Future_Return'] > -0.01, 0,  # Sideways
                            -1))  # Down
        )
        
        # Volatility target
        features['Future_Volatility'] = features['Price_Change'].shift(-target_days).rolling(5).std()
        
        return features
    
    def select_features(self, df: pd.DataFrame, method: str = 'correlation', 
                       target_col: str = 'Future_Up', max_features: int = 50) -> List[str]:
        """
        Select most important features for modeling.
        
        Args:
            df: DataFrame with features
            method: Feature selection method
            target_col: Target column name
            max_features: Maximum number of features to select
        
        Returns:
            List of selected feature names
        """
        try:
            # Exclude non-feature columns
            exclude_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Future_Return', 'Future_Up', 'Future_Strong_Up',
                'Future_Down', 'Future_Strong_Down', 'Future_Direction',
                'Future_Volatility'
            ]
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if method == 'correlation':
                # Select features based on correlation with target
                correlations = df[feature_cols].corrwith(df[target_col]).abs()
                selected_features = correlations.nlargest(max_features).index.tolist()
                
            elif method == 'variance':
                # Select features with highest variance
                variances = df[feature_cols].var()
                selected_features = variances.nlargest(max_features).index.tolist()
                
            else:
                # Default: use predefined important features
                important_features = [
                    'RSI', 'MACD', 'MACD_Signal', 'BB_Position', 'ATR_Ratio',
                    'Volume_Ratio_20', 'Close_SMA20_Ratio', 'Close_SMA50_Ratio',
                    'ROC_5', 'ROC_10', 'Volatility_20', 'Price_Momentum_5',
                    'Volume_Momentum', 'Higher_High', 'Lower_Low'
                ]
                selected_features = [f for f in important_features if f in feature_cols]
            
            self.logger.info(f"Selected {len(selected_features)} features using {method} method")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            return feature_cols[:max_features]  # Fallback
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained ML model
            feature_names: List of feature names
        
        Returns:
            Dictionary of feature importances
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            else:
                return {}
            
            importance_dict = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {}
