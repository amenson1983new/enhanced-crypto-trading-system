"""
Model Trainer for Enhanced Crypto Trading System
Handles model training, hyperparameter optimization, and prediction.
"""
import logging
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


class ModelTrainer:
    def __init__(self, config):
        """Initialize ModelTrainer with configuration."""
        self.config = config
        self.model = None
        self.feature_columns = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Tune hyperparameters using Optuna if available."""
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available. Using default parameters.")
            return self.config['model']['default_params']
            
        def objective(trial):
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 200)
            rf_max_depth = trial.suggest_int('rf_max_depth', 5, 20)
            gb_n_estimators = trial.suggest_int('gb_n_estimators', 50, 200)
            gb_learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.3)
            gb_max_depth = trial.suggest_int('gb_max_depth', 3, 10)
            
            rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate, max_depth=gb_max_depth, random_state=42)
            
            model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            return f1_score(y_val, y_pred, average='weighted')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['model'].get('optimization_trials', 3))
        
        best_params = {
            'rf_n_estimators': study.best_params['rf_n_estimators'],
            'rf_max_depth': study.best_params['rf_max_depth'],
            'gb_n_estimators': study.best_params['gb_n_estimators'],
            'gb_learning_rate': study.best_params['gb_learning_rate'],
            'gb_max_depth': study.best_params['gb_max_depth']
        }
        
        self.logger.info(f"Best hyperparameters: {best_params}")
        return best_params
    
    def train(self, X_train, y_train, params=None, incremental=False, X_val=None, y_val=None):
        """Train the model with given data."""
        self.logger.info("Starting model training...")
        
        if params is None:
            params = self.config['model']['default_params']
            
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Check if we have enough classes
        if len(y_train.unique()) < 2:
            self.logger.error("Not enough classes to train model. Need at least 2 different signal types.")
            raise ValueError("Training requires data with multiple signal types.")
        
        # Extract parameters for each model
        rf_params = {k.replace('rf_', ''): v for k, v in params.items() if 'rf_' in k}
        gb_params = {k.replace('gb_', ''): v for k, v in params.items() if 'gb_' in k}
        
        # Create models
        rf = RandomForestClassifier(**rf_params, random_state=42)
        gb = GradientBoostingClassifier(**gb_params, random_state=42)
        
        self.model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft', n_jobs=-1)
        
        # Handle class imbalance with SMOTE if available
        if SMOTE_AVAILABLE and len(y_train.unique()) > 1:
            try:
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                self.logger.info("Applied SMOTE for class balance")
            except Exception as e:
                self.logger.warning(f"SMOTE failed, using original data: {e}")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train the model
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Save model
        symbol = self.config.get('symbol', 'UNKNOWN')
        interval = self.config.get('interval', '5m')
        model_path = f"models/{symbol}_{interval}_trading_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump({
            'model': self.model, 
            'feature_columns': self.feature_columns, 
            'params': params
        }, model_path)
        
        self.logger.info(f"Model saved to {model_path}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val)
            self.logger.info(f"Validation accuracy: {f1_score(y_val, y_pred, average='weighted'):.4f}")
        
        return self.model
        
    def load_model(self):
        """Load a trained model."""
        symbol = self.config.get('symbol', 'UNKNOWN')
        interval = self.config.get('interval', '5m')
        model_path = f"models/{symbol}_{interval}_trading_model.pkl"
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return False
            
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            if not self.load_model():
                raise ValueError("No model available for prediction")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities