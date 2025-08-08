"""
Machine learning models for stock price prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Model selection and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ..utils.logger import setup_logger
from ..utils.config import Config
from .features import FeatureEngineer


class MLModels:
    """Machine learning models for stock prediction."""
    
    def __init__(self):
        """Initialize ML models."""
        self.logger = setup_logger(__name__)
        self.config = Config()
        self.ml_config = self.config.get_ml_config()
        
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.models = {}
        self.model_performance = {}
        
        # Model configurations
        self.model_configs = {
            'decision_tree': {
                'class': DecisionTreeClassifier,
                'params': {
                    'random_state': 42,
                    'max_depth': 10,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5
                }
            },
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'n_jobs': -1
                }
            },
            'logistic_regression': {
                'class': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'C': 1.0
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'learning_rate': 0.1,
                    'max_depth': 6
                }
            },
            'svm': {
                'class': SVC,
                'params': {
                    'random_state': 42,
                    'kernel': 'rbf',
                    'C': 1.0,
                    'probability': True
                }
            },
            'naive_bayes': {
                'class': GaussianNB,
                'params': {}
            },
            'neural_network': {
                'class': MLPClassifier,
                'params': {
                    'random_state': 42,
                    'hidden_layer_sizes': (100, 50),
                    'max_iter': 500,
                    'alpha': 0.01
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'class': xgb.XGBClassifier,
                'params': {
                    'random_state': 42,
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
            }
        
        # Paths for saving models
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized ML models: {list(self.model_configs.keys())}")
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Future_Up', 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for ML training.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            test_size: Test set proportion
        
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        try:
            # Create features if not present
            if 'Future_Up' not in df.columns:
                df = self.feature_engineer.create_features(df)
            
            # Select features
            feature_names = self.feature_engineer.select_features(
                df, 
                target_col=target_col,
                max_features=self.ml_config.get('max_features', 30)
            )
            
            # Prepare feature matrix and target
            X = df[feature_names].values
            y = df[target_col].values
            
            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size,
                random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.logger.info(f"Data prepared: {len(X_train)} train samples, {len(X_test)} test samples, "
                           f"{len(feature_names)} features")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, feature_names
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    model_names: List[str] = None) -> Dict[str, Any]:
        """
        Train multiple ML models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_names: List of model names to train (None for all)
        
        Returns:
            Dictionary with training results
        """
        if model_names is None:
            model_names = list(self.model_configs.keys())
        
        results = {}
        
        for model_name in model_names:
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Get model configuration
                config = self.model_configs[model_name]
                model_class = config['class']
                params = config['params']
                
                # Create and train model
                model = model_class(**params)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Train on full training set
                model.fit(X_train, y_train)
                
                # Store model and results
                self.models[model_name] = model
                results[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist(),
                    'training_accuracy': model.score(X_train, y_train)
                }
                
                self.logger.info(f"{model_name} trained - CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                       feature_names: List[str]) -> Dict[str, Any]:
        """
        Evaluate trained models on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            feature_names: Feature names
        
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Classification report
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # Confusion matrix
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # ROC AUC if probabilities available
                roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                # Feature importance
                feature_importance = self.feature_engineer.get_feature_importance(model, feature_names)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'roc_auc': roc_auc,
                    'feature_importance': feature_importance,
                    'predictions': y_pred.tolist(),
                    'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
                }
                
                # Store performance for later use
                self.model_performance[model_name] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'precision': class_report['1']['precision'],
                    'recall': class_report['1']['recall'],
                    'f1_score': class_report['1']['f1-score']
                }
                
                self.logger.info(f"{model_name} evaluation - Accuracy: {accuracy:.4f}, "
                               f"AUC: {roc_auc:.4f if roc_auc else 'N/A'}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        return results
    
    def hyperparameter_tuning(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                            param_grid: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name: Name of model to tune
            X_train: Training features
            y_train: Training targets
            param_grid: Parameter grid for tuning
        
        Returns:
            Dictionary with tuning results
        """
        try:
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Default parameter grids
            default_param_grids = {
                'decision_tree': {
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [5, 10, 20],
                    'min_samples_leaf': [2, 5, 10]
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [5, 10, 20]
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                },
                'svm': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
            
            if param_grid is None:
                param_grid = default_param_grids.get(model_name, {})
            
            if not param_grid:
                self.logger.warning(f"No parameter grid defined for {model_name}")
                return {}
            
            self.logger.info(f"Starting hyperparameter tuning for {model_name}...")
            
            # Get base model
            config = self.model_configs[model_name]
            model_class = config['class']
            base_params = config['params']
            
            # Create base model
            base_model = model_class(**base_params)
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            best_params = {**base_params, **grid_search.best_params_}
            self.models[model_name] = model_class(**best_params)
            self.models[model_name].fit(X_train, y_train)
            
            # Update model config
            self.model_configs[model_name]['params'] = best_params
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            self.logger.info(f"Hyperparameter tuning complete for {model_name}. "
                           f"Best score: {grid_search.best_score_:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning for {model_name}: {str(e)}")
            return {}
    
    def get_ensemble_prediction(self, X: np.ndarray, method: str = 'voting') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ensemble prediction from multiple models.
        
        Args:
            X: Features for prediction
            method: Ensemble method ('voting', 'weighted', 'stacking')
        
        Returns:
            Predictions and probabilities
        """
        try:
            if not self.models:
                raise ValueError("No trained models available")
            
            predictions = []
            probabilities = []
            weights = []
            
            for model_name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[:, 1]
                    probabilities.append(prob)
                
                # Use accuracy as weight
                weight = self.model_performance.get(model_name, {}).get('accuracy', 1.0)
                weights.append(weight)
            
            predictions = np.array(predictions)
            probabilities = np.array(probabilities) if probabilities else None
            weights = np.array(weights)
            
            if method == 'voting':
                # Simple majority voting
                ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
                ensemble_prob = np.mean(probabilities, axis=0) if probabilities is not None else None
                
            elif method == 'weighted':
                # Weighted voting based on model performance
                weights_norm = weights / weights.sum()
                ensemble_pred = np.round(np.average(predictions, axis=0, weights=weights_norm)).astype(int)
                ensemble_prob = np.average(probabilities, axis=0, weights=weights_norm) if probabilities is not None else None
                
            else:  # Default to voting
                ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
                ensemble_prob = np.mean(probabilities, axis=0) if probabilities is not None else None
            
            return ensemble_pred, ensemble_prob
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {str(e)}")
            return np.array([]), np.array([])
    
    def save_models(self, symbol: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            symbol: Stock symbol for naming
        """
        try:
            symbol_dir = self.model_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Save models
            for model_name, model in self.models.items():
                model_path = symbol_dir / f"{model_name}.joblib"
                joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = symbol_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            
            # Save performance metrics
            performance_path = symbol_dir / "performance.json"
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)
            
            # Save model configurations
            config_path = symbol_dir / "model_configs.json"
            
            # Convert model configs to serializable format
            serializable_configs = {}
            for model_name, config in self.model_configs.items():
                serializable_configs[model_name] = {
                    'class_name': config['class'].__name__,
                    'params': config['params']
                }
            
            with open(config_path, 'w') as f:
                json.dump(serializable_configs, f, indent=2)
            
            self.logger.info(f"Models saved for {symbol} in {symbol_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, symbol: str) -> bool:
        """
        Load trained models from disk.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            True if successful, False otherwise
        """
        try:
            symbol_dir = self.model_dir / symbol
            
            if not symbol_dir.exists():
                self.logger.warning(f"No saved models found for {symbol}")
                return False
            
            # Load models
            self.models.clear()
            for model_file in symbol_dir.glob("*.joblib"):
                if model_file.name == "scaler.joblib":
                    continue
                
                model_name = model_file.stem
                self.models[model_name] = joblib.load(model_file)
            
            # Load scaler
            scaler_path = symbol_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            # Load performance metrics
            performance_path = symbol_dir / "performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    self.model_performance = json.load(f)
            
            self.logger.info(f"Loaded {len(self.models)} models for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models and their performance."""
        summary = {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'performance': self.model_performance.copy(),
            'best_model': None,
            'ensemble_available': len(self.models) > 1
        }
        
        # Find best model based on accuracy
        if self.model_performance:
            best_model = max(self.model_performance.items(), 
                           key=lambda x: x[1].get('accuracy', 0))
            summary['best_model'] = {
                'name': best_model[0],
                'performance': best_model[1]
            }
        
        return summary
