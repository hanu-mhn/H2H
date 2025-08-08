"""
ML prediction engine for trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import logging

from ..utils.logger import setup_logger
from ..utils.config import Config
from ..data.data_fetcher import DataFetcher
from .models import MLModels
from .features import FeatureEngineer


class MLPredictor:
    """ML-based prediction engine for trading signals."""
    
    def __init__(self):
        """Initialize ML predictor."""
        self.logger = setup_logger(__name__)
        self.config = Config()
        self.ml_config = self.config.get_ml_config()
        
        self.data_fetcher = DataFetcher()
        self.ml_models = MLModels()
        self.feature_engineer = FeatureEngineer()
        
        # Prediction thresholds
        self.confidence_threshold = 0.6  # Minimum confidence for signals
        self.probability_thresholds = {
            'strong_buy': 0.8,
            'buy': 0.6,
            'hold': 0.4,
            'sell': 0.4,
            'strong_sell': 0.2
        }
        
        self.logger.info("ML Predictor initialized")
    
    async def train_model(self, symbol: str, period: str = "2y", 
                         retrain: bool = False) -> Dict[str, Any]:
        """
        Train ML models for a specific symbol.
        
        Args:
            symbol: Stock symbol
            period: Data period for training
            retrain: Force retraining even if models exist
        
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info(f"Starting ML model training for {symbol}")
            
            # Check if models already exist and are recent
            if not retrain and self.ml_models.load_models(symbol):
                self.logger.info(f"Using existing models for {symbol}")
                return {'status': 'loaded_existing', 'models': self.ml_models.get_model_summary()}
            
            # Fetch historical data
            self.logger.info(f"Fetching training data for {symbol}")
            data = await self.data_fetcher.fetch_stock_data(symbol, period=period)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Create features
            self.logger.info("Engineering features for ML training")
            features_df = self.feature_engineer.create_features(data)
            
            if len(features_df) < 100:  # Minimum data for training
                raise ValueError(f"Insufficient data for training: {len(features_df)} samples")
            
            # Prepare data for training
            X_train, X_test, y_train, y_test, feature_names = self.ml_models.prepare_data(
                features_df,
                target_col='Future_Up',
                test_size=self.ml_config.get('training', {}).get('test_size', 0.2)
            )
            
            # Train models
            model_names = self.ml_config.get('models', ['decision_tree', 'logistic_regression'])
            training_results = self.ml_models.train_models(X_train, y_train, model_names)
            
            # Evaluate models
            evaluation_results = self.ml_models.evaluate_models(X_test, y_test, feature_names)
            
            # Hyperparameter tuning for best models
            best_models = sorted(training_results.items(), 
                               key=lambda x: x[1]['cv_mean'], reverse=True)[:2]
            
            tuning_results = {}
            for model_name, _ in best_models:
                self.logger.info(f"Tuning hyperparameters for {model_name}")
                tuning_result = self.ml_models.hyperparameter_tuning(model_name, X_train, y_train)
                if tuning_result:
                    tuning_results[model_name] = tuning_result
            
            # Re-evaluate after tuning
            final_evaluation = self.ml_models.evaluate_models(X_test, y_test, feature_names)
            
            # Save models
            self.ml_models.save_models(symbol)
            
            results = {
                'status': 'training_complete',
                'symbol': symbol,
                'data_points': len(features_df),
                'features_count': len(feature_names),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'training_results': training_results,
                'evaluation_results': final_evaluation,
                'tuning_results': tuning_results,
                'feature_names': feature_names,
                'model_summary': self.ml_models.get_model_summary()
            }
            
            self.logger.info(f"ML model training completed for {symbol}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training ML models for {symbol}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def predict(self, data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """
        Generate ML-based predictions for trading.
        
        Args:
            data: Stock data DataFrame
            symbol: Stock symbol (for loading specific models)
        
        Returns:
            Prediction results dictionary
        """
        try:
            # Load models if symbol provided and not already loaded
            if symbol and not self.ml_models.models:
                if not self.ml_models.load_models(symbol):
                    self.logger.warning(f"No trained models found for {symbol}")
                    return self._default_prediction()
            
            # Check if we have any trained models
            if not self.ml_models.models:
                return self._default_prediction()
            
            # Create features
            features_df = self.feature_engineer.create_features(data)
            
            if features_df.empty:
                return self._default_prediction()
            
            # Get the latest features (most recent data point)
            latest_features = features_df.iloc[-1:].copy()
            
            # Select same features used in training
            available_features = [col for col in latest_features.columns 
                                if col not in ['Open', 'High', 'Low', 'Close', 'Volume',
                                             'Future_Return', 'Future_Up', 'Future_Strong_Up',
                                             'Future_Down', 'Future_Strong_Down', 'Future_Direction']]
            
            # Ensure we have enough features
            if len(available_features) < 10:
                self.logger.warning("Insufficient features for prediction")
                return self._default_prediction()
            
            # Select top features (limit to avoid overfitting)
            selected_features = available_features[:30]
            X = latest_features[selected_features].values
            
            # Check for NaN values
            if np.isnan(X).any():
                # Fill NaN with median values or use interpolation
                X = np.nan_to_num(X, nan=0.0)
            
            # Scale features if scaler is available
            if hasattr(self.ml_models.scaler, 'transform'):
                try:
                    X_scaled = self.ml_models.scaler.transform(X)
                except Exception:
                    X_scaled = X  # Use unscaled if scaling fails
            else:
                X_scaled = X
            
            # Get ensemble prediction
            predictions, probabilities = self.ml_models.get_ensemble_prediction(
                X_scaled, method='weighted'
            )
            
            if len(predictions) == 0:
                return self._default_prediction()
            
            # Individual model predictions for analysis
            individual_predictions = {}
            for model_name, model in self.ml_models.models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    prob = None
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[0, 1]
                    
                    individual_predictions[model_name] = {
                        'prediction': int(pred),
                        'probability': float(prob) if prob is not None else None,
                        'confidence': float(prob) if prob is not None else float(pred)
                    }
                except Exception as e:
                    self.logger.warning(f"Error getting prediction from {model_name}: {str(e)}")
                    continue
            
            # Generate final signal
            ensemble_prediction = int(predictions[0])
            ensemble_probability = float(probabilities[0]) if probabilities is not None else 0.5
            
            signal = self._generate_signal_from_prediction(
                ensemble_prediction, 
                ensemble_probability,
                individual_predictions
            )
            
            # Add metadata
            signal.update({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'individual_predictions': individual_predictions,
                'ensemble_prediction': ensemble_prediction,
                'ensemble_probability': ensemble_probability,
                'feature_count': len(selected_features),
                'model_count': len(self.ml_models.models)
            })
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {str(e)}")
            return self._default_prediction()
    
    def _generate_signal_from_prediction(self, prediction: int, probability: float,
                                       individual_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal from ML prediction."""
        
        # Calculate consensus among models
        model_agreement = len([p for p in individual_predictions.values() 
                             if p['prediction'] == prediction]) / len(individual_predictions)
        
        # Adjust confidence based on model agreement
        base_confidence = probability if prediction == 1 else (1 - probability)
        adjusted_confidence = base_confidence * model_agreement
        
        # Determine action based on probability and confidence
        if prediction == 1 and adjusted_confidence >= self.confidence_threshold:
            if probability >= self.probability_thresholds['strong_buy']:
                action = 'BUY'
                strength = 'STRONG'
            elif probability >= self.probability_thresholds['buy']:
                action = 'BUY'
                strength = 'MODERATE'
            else:
                action = 'BUY'
                strength = 'WEAK'
        elif prediction == 0 and (1 - probability) >= self.confidence_threshold:
            if probability <= self.probability_thresholds['strong_sell']:
                action = 'SELL'
                strength = 'STRONG'
            elif probability <= self.probability_thresholds['sell']:
                action = 'SELL'
                strength = 'MODERATE'
            else:
                action = 'SELL'
                strength = 'WEAK'
        else:
            action = 'HOLD'
            strength = 'NEUTRAL'
        
        return {
            'action': action,
            'confidence': adjusted_confidence,
            'strength': strength,
            'probability_up': probability,
            'model_agreement': model_agreement,
            'reason': f"ML prediction: {action} ({strength}) - "
                     f"Probability: {probability:.2f}, Agreement: {model_agreement:.2f}"
        }
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when ML models are not available."""
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'strength': 'NEUTRAL',
            'probability_up': 0.5,
            'model_agreement': 0.0,
            'reason': 'No ML models available or insufficient data',
            'timestamp': datetime.now(),
            'model_count': 0
        }
    
    async def batch_predict(self, symbols: List[str], period: str = "1y") -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            period: Data period for prediction
        
        Returns:
            Dictionary mapping symbols to predictions
        """
        predictions = {}
        
        # Fetch data for all symbols
        market_data = await self.data_fetcher.fetch_multiple_stocks(symbols, period=period)
        
        for symbol, data in market_data.items():
            try:
                prediction = await self.predict(data, symbol)
                predictions[symbol] = prediction
                
            except Exception as e:
                self.logger.error(f"Error predicting for {symbol}: {str(e)}")
                predictions[symbol] = self._default_prediction()
        
        return predictions
    
    def get_model_performance(self, symbol: str) -> Dict[str, Any]:
        """
        Get performance metrics for trained models.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Performance metrics dictionary
        """
        try:
            if not self.ml_models.load_models(symbol):
                return {}
            
            return self.ml_models.model_performance.copy()
            
        except Exception as e:
            self.logger.error(f"Error getting model performance for {symbol}: {str(e)}")
            return {}
    
    def update_model_performance(self, symbol: str, actual_returns: List[float],
                               predicted_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update model performance based on actual results.
        
        Args:
            symbol: Stock symbol
            actual_returns: List of actual returns
            predicted_signals: List of predicted signals
        
        Returns:
            Updated performance metrics
        """
        try:
            if len(actual_returns) != len(predicted_signals):
                raise ValueError("Mismatch between actual returns and predictions")
            
            # Calculate performance metrics
            correct_predictions = 0
            total_predictions = len(actual_returns)
            
            for actual, predicted in zip(actual_returns, predicted_signals):
                actual_direction = 1 if actual > 0 else 0
                predicted_direction = 1 if predicted['action'] == 'BUY' else 0
                
                if actual_direction == predicted_direction:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Update performance tracking
            performance_update = {
                'symbol': symbol,
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'last_updated': datetime.now().isoformat()
            }
            
            self.logger.info(f"Updated performance for {symbol}: {accuracy:.2%} accuracy "
                           f"({correct_predictions}/{total_predictions})")
            
            return performance_update
            
        except Exception as e:
            self.logger.error(f"Error updating model performance: {str(e)}")
            return {}
    
    async def retrain_all_models(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Retrain models for all provided symbols.
        
        Args:
            symbols: List of symbols to retrain
        
        Returns:
            Dictionary with retraining results
        """
        results = {}
        
        for symbol in symbols:
            try:
                self.logger.info(f"Retraining models for {symbol}")
                result = await self.train_model(symbol, retrain=True)
                results[symbol] = result
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error retraining models for {symbol}: {str(e)}")
                results[symbol] = {'status': 'error', 'error': str(e)}
        
        return results
