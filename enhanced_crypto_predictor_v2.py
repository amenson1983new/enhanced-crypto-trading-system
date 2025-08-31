"""
Enhanced Crypto Trading Predictor System v2.0
Created for user: amenson1983new
Date: 2025-08-31 16:11:25 UTC

This system combines LSTM/GRU networks with ensemble methods
for crypto trading signal prediction with 85% target profitability.
"""

import pandas as pd
import numpy as np
import warnings
import os
import pickle
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union

# Core ML/DL libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingRegressor, 
                             VotingClassifier, VotingRegressor, ExtraTreesClassifier)
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (classification_report, accuracy_score, precision_recall_fscore_support,
                           mean_squared_error, mean_absolute_error, r2_score)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, 
                                   concatenate, BatchNormalization, Conv1D, 
                                   MaxPooling1D, Flatten, Attention)
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint, LearningRateScheduler)
from tensorflow.keras.regularizers import l1_l2

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Utilities
import joblib
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ConfigManager:
    """Configuration management for the trading system"""
    
    DEFAULT_CONFIG = {
        'model': {
            'sequence_length': 25,
            'lstm_units': 128,
            'gru_units': 96,
            'dense_units': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'early_stopping_patience': 20,
            'use_attention': True,
            'use_conv1d': False
        },
        'features': {
            'use_technical_indicators': True,
            'use_volume_analysis': True,
            'use_price_action': True,
            'lookback_periods': [5, 10, 20, 50],
            'volatility_window': 14,
            'momentum_window': 10
        },
        'trading': {
            'min_profit_threshold': 0.015,  # 1.5%
            'risk_reward_ratio': 1.5,
            'max_risk_per_trade': 0.02,    # 2%
            'max_portfolio_risk': 0.10,    # 10%
            'confidence_threshold': 0.65,
            'future_periods': 3
        },
        'optimization': {
            'n_trials': 50,
            'cv_folds': 5,
            'optimization_metric': 'f1_macro',
            'use_pruning': True
        }
    }
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or return default"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**cls.DEFAULT_CONFIG, **config}
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def save_config(cls, config: Dict, config_path: str):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

class AdvancedTechnicalAnalyzer:
    """Advanced technical analysis with multiple timeframes"""
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df = data.copy()
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        
        df['bb_upper'] = rolling_mean + (rolling_std * std_dev)
        df['bb_lower'] = rolling_mean - (rolling_std * std_dev)
        df['bb_middle'] = rolling_mean
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def calculate_ichimoku_cloud(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators"""
        df = data.copy()
        
        # Tenkan-sen (Conversion Line)
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-26)
        
        return df
    
    @staticmethod
    def calculate_williams_r(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = data['high'].rolling(window=window).max()
        low_min = data['low'].rolling(window=window).min()
        return -100 * (high_max - data['close']) / (high_max - low_min)
    
    @staticmethod
    def calculate_roc(data: pd.Series, window: int = 12) -> pd.Series:
        """Calculate Rate of Change"""
        return ((data - data.shift(window)) / data.shift(window)) * 100
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window).mean()

class EnhancedDataPreprocessor:
    """Enhanced data preprocessing with advanced feature engineering"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive feature engineering"""
        logger.info("Starting feature engineering...")
        df = data.copy()
        
        # Basic technical indicators
        analyzer = AdvancedTechnicalAnalyzer()
        
        # Bollinger Bands
        df = analyzer.calculate_bollinger_bands(df)
        
        # Ichimoku Cloud
        df = analyzer.calculate_ichimoku_cloud(df)
        
        # Additional indicators
        df['williams_r'] = analyzer.calculate_williams_r(df)
        df['roc'] = analyzer.calculate_roc(df['close'])
        df['atr'] = analyzer.calculate_atr(df)
        
        # Volume analysis
        if self.config['features']['use_volume_analysis']:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']
            df['vwap'] = (df['price_volume'].rolling(window=20).sum() / 
                         df['volume'].rolling(window=20).sum())
        
        # Price action features
        if self.config['features']['use_price_action']:
            # Candlestick patterns
            df['doji'] = ((df['close'] - df['open']).abs() / (df['high'] - df['low'])) < 0.1
            df['hammer'] = ((df['high'] - df['close']) / (df['close'] - df['low']) > 2) & (df['close'] > df['open'])
            df['shooting_star'] = ((df['close'] - df['low']) / (df['high'] - df['close']) > 2) & (df['close'] < df['open'])
            
            # Gap analysis
            df['gap_up'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) > 0.005
            df['gap_down'] = (df['close'].shift(1) - df['open']) / df['close'].shift(1) > 0.005
        
        # Multi-timeframe features
        for period in self.config['features']['lookback_periods']:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'volatility_{period}'] = df['close'].rolling(window=period).std()
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # Advanced momentum indicators
        df['momentum_10'] = df['close'] / df['close'].shift(10)
        df['momentum_20'] = df['close'] / df['close'].shift(20)
        
        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        
        # Fractal analysis
        df['fractal_high'] = (df['high'] > df['high'].shift(2)) & (df['high'] > df['high'].shift(1)) & \
                           (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))
        df['fractal_low'] = (df['low'] < df['low'].shift(2)) & (df['low'] < df['low'].shift(1)) & \
                          (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2))
        
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_advanced_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated trading signals with multiple criteria"""
        logger.info("Creating advanced trading signals...")
        df = data.copy()
        
        future_periods = self.config['trading']['future_periods']
        min_profit = self.config['trading']['min_profit_threshold']
        risk_reward = self.config['trading']['risk_reward_ratio']
        
        signals = []
        profit_potentials = []
        confidence_scores = []
        
        for i in tqdm(range(len(df)), desc="Generating signals"):
            if i + future_periods < len(df):
                current_price = df.iloc[i]['close']
                future_data = df.iloc[i:i+future_periods]
                
                # Calculate potential movements
                future_high = future_data['high'].max()
                future_low = future_data['low'].min()
                
                upward_potential = (future_high - current_price) / current_price
                downward_risk = (current_price - future_low) / current_price
                
                # Technical analysis score
                tech_score = self._calculate_technical_score(df.iloc[i])
                
                # Market structure score
                structure_score = self._calculate_structure_score(df.iloc[max(0, i-10):i+1])
                
                # Volume confirmation
                volume_score = self._calculate_volume_score(df.iloc[max(0, i-5):i+1])
                
                # Combined confidence
                confidence = (tech_score + structure_score + volume_score) / 3
                
                # Decision logic with multiple criteria
                if (upward_potential > min_profit and 
                    upward_potential / max(downward_risk, 0.005) > risk_reward and
                    confidence > 0.6):
                    signal = 'BUY'
                    profit_potential = upward_potential
                    
                elif (downward_risk > min_profit and 
                      downward_risk / max(upward_potential, 0.005) > risk_reward and
                      confidence > 0.6):
                    signal = 'SELL' 
                    profit_potential = downward_risk
                    
                else:
                    signal = 'SKIP'
                    profit_potential = 0
                
                signals.append(signal)
                profit_potentials.append(profit_potential)
                confidence_scores.append(confidence)
                
            else:
                signals.append('SKIP')
                profit_potentials.append(0)
                confidence_scores.append(0)
        
        df['signal'] = signals
        df['profit_potential'] = profit_potentials
        df['confidence_score'] = confidence_scores
        
        # Calculate additional targets
        df['upward_potential'] = df['close'].shift(-future_periods).rolling(future_periods).max() / df['close'] - 1
        df['downward_risk'] = 1 - df['close'].shift(-future_periods).rolling(future_periods).min() / df['close']
        
        logger.info(f"Signals created. Distribution: {pd.Series(signals).value_counts().to_dict()}")
        return df
    
    def _calculate_technical_score(self, row: pd.Series) -> float:
        """Calculate technical analysis score"""
        score = 0.5  # Neutral base
        
        # RSI signals
        if 'rsi' in row.index:
            if row['rsi'] < 30:
                score += 0.2  # Oversold - bullish
            elif row['rsi'] > 70:
                score -= 0.2  # Overbought - bearish
        
        # MACD signals
        if all(col in row.index for col in ['MACD_3_9_21', 'MACDs_3_9_21']):
            if row['MACD_3_9_21'] > row['MACDs_3_9_21']:
                score += 0.15
            else:
                score -= 0.15
        
        # Bollinger Bands
        if 'bb_position' in row.index:
            if row['bb_position'] < 0.2:
                score += 0.1  # Near lower band - bullish
            elif row['bb_position'] > 0.8:
                score -= 0.1  # Near upper band - bearish
        
        # Stochastic
        if all(col in row.index for col in ['STOCHk_14_3_3', 'STOCHd_14_3_3']):
            if row['STOCHk_14_3_3'] > row['STOCHd_14_3_3'] and row['STOCHk_14_3_3'] < 80:
                score += 0.1
            elif row['STOCHk_14_3_3'] < row['STOCHd_14_3_3'] and row['STOCHk_14_3_3'] > 20:
                score -= 0.1
        
        return np.clip(score, 0, 1)
    
    def _calculate_structure_score(self, data: pd.DataFrame) -> float:
        """Calculate market structure score"""
        if len(data) < 3:
            return 0.5
        
        score = 0.5
        
        # Trend analysis
        recent_closes = data['close'].tail(3)
        if recent_closes.is_monotonic_increasing:
            score += 0.2
        elif recent_closes.is_monotonic_decreasing:
            score -= 0.2
        
        # Support/resistance breaks
        if len(data) > 5:
            recent_high = data['high'].tail(5).max()
            recent_low = data['low'].tail(5).min()
            current_close = data['close'].iloc[-1]
            
            if current_close > recent_high:
                score += 0.15  # Breakout
            elif current_close < recent_low:
                score -= 0.15  # Breakdown
        
        return np.clip(score, 0, 1)
    
    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """Calculate volume confirmation score"""
        if 'volume' not in data.columns or len(data) < 2:
            return 0.5
        
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].mean()
        
        if current_volume > avg_volume * 1.5:
            return 0.8  # High volume confirmation
        elif current_volume > avg_volume:
            return 0.6  # Moderate volume
        else:
            return 0.4  # Low volume
    
    def prepare_sequences(self, data: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequences for deep learning models"""
        logger.info("Preparing sequences for deep learning...")
        
        # Define feature columns
        feature_cols = [
            'close', 'high', 'low', 'volume', 'rsi', 'sma', 'ema',
            'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ADX_14', 'DMP_14', 'DMN_14',
            'MACD_3_9_21', 'MACDh_3_9_21', 'MACDs_3_9_21', 'bb_position',
            'bb_width', 'williams_r', 'roc', 'atr', 'momentum_10', 'momentum_20'
        ]
        
        # Add dynamic features
        for period in [5, 10, 20]:
            if f'return_{period}' in data.columns:
                feature_cols.append(f'return_{period}')
            if f'volatility_{period}' in data.columns:
                feature_cols.append(f'volatility_{period}')
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in data.columns]
        self.feature_names = available_cols
        
        # Clean data
        df_clean = data[available_cols + ['signal', 'profit_potential']].dropna()
        
        if len(df_clean) == 0:
            logger.error("No data available after cleaning")
            return np.array([]), np.array([]), np.array([])
        
        # Scale features
        X_features = df_clean[available_cols].values
        
        if not self.is_fitted:
            self.scalers['features'] = RobustScaler()
            X_scaled = self.scalers['features'].fit_transform(X_features)
            self.is_fitted = True
        else:
            X_scaled = self.scalers['features'].transform(X_features)
        
        # Encode labels
        try:
            if not hasattr(self.label_encoder, 'classes_'):
                y_signal = self.label_encoder.fit_transform(df_clean['signal'])
            else:
                y_signal = self.label_encoder.transform(df_clean['signal'])
        except ValueError as e:
            logger.warning(f"Label encoding issue: {e}")
            y_signal = np.zeros(len(df_clean))
        
        y_profit = df_clean['profit_potential'].values
        
        # Create sequences
        X_sequences, y_signal_seq, y_profit_seq = [], [], []
        
        for i in range(sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-sequence_length:i])
            y_signal_seq.append(y_signal[i])
            y_profit_seq.append(y_profit[i])
        
        result = (np.array(X_sequences), np.array(y_signal_seq), np.array(y_profit_seq))
        logger.info(f"Sequences prepared. Shapes: {[arr.shape for arr in result]}")
        return result

class HybridDeepLearningModel:
    """Advanced hybrid deep learning model with multiple architectures"""
    
    def __init__(self, config: Dict):
        self.config = config['model']
        self.signal_model = None
        self.profit_model = None
        self.feature_dim = None
        
    def build_advanced_signal_model(self, input_shape: Tuple, num_classes: int = 3) -> Model:
        """Build advanced signal prediction model"""
        inputs = Input(shape=input_shape)
        
        # CNN branch for pattern recognition
        if self.config.get('use_conv1d', False):
            conv_branch = Conv1D(64, 3, activation='relu')(inputs)
            conv_branch = MaxPooling1D(2)(conv_branch)
            conv_branch = Conv1D(32, 3, activation='relu')(conv_branch)
            conv_branch = Dropout(self.config['dropout_rate'])(conv_branch)
        
        # LSTM branch for sequence modeling
        lstm_branch = LSTM(
            self.config['lstm_units'], 
            return_sequences=True,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        )(inputs)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = Dropout(self.config['dropout_rate'])(lstm_branch)
        
        lstm_branch = LSTM(
            self.config['lstm_units']//2, 
            return_sequences=False
        )(lstm_branch)
        
        # GRU branch for additional sequence processing
        gru_branch = GRU(
            self.config.get('gru_units', 96),
            return_sequences=True,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        )(inputs)
        gru_branch = BatchNormalization()(gru_branch)
        gru_branch = Dropout(self.config['dropout_rate'])(gru_branch)
        
        gru_branch = GRU(
            self.config.get('gru_units', 96)//2,
            return_sequences=False
        )(gru_branch)
        
        # Combine branches
        if self.config.get('use_conv1d', False):
            conv_flattened = Flatten()(conv_branch)
            combined = concatenate([lstm_branch, gru_branch, conv_flattened])
        else:
            combined = concatenate([lstm_branch, gru_branch])
        
        # Dense layers
        dense = Dense(self.config['dense_units'], activation='relu')(combined)
        dense = BatchNormalization()(dense)
        dense = Dropout(self.config['dropout_rate'])(dense)
        
        dense = Dense(self.config['dense_units']//2, activation='relu')(dense)
        dense = Dropout(self.config['dropout_rate']/2)(dense)
        
        outputs = Dense(num_classes, activation='softmax')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=AdamW(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_crossentropy']
        )
        
        return model
    
    def build_advanced_profit_model(self, input_shape: Tuple) -> Model:
        """Build advanced profit prediction model"""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention mechanism
        if self.config.get('use_attention', True):
            # Simple attention mechanism
            attention_weights = Dense(1, activation='tanh')(inputs)
            attention_weights = tf.nn.softmax(attention_weights, axis=1)
            attended = inputs * attention_weights
        else:
            attended = inputs
        
        # GRU with residual connections
        gru1 = GRU(
            self.config['gru_units'], 
            return_sequences=True,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        )(attended)
        gru1 = BatchNormalization()(gru1)
        
        gru2 = GRU(
            self.config['gru_units']//2, 
            return_sequences=True
        )(gru1)
        gru2 = BatchNormalization()(gru2)
        gru2 = Dropout(self.config['dropout_rate'])(gru2)
        
        # Final GRU layer
        gru_final = GRU(self.config['gru_units']//4, return_sequences=False)(gru2)
        
        # Dense layers with skip connections
        dense1 = Dense(self.config['dense_units'], activation='relu')(gru_final)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.config['dropout_rate'])(dense1)
        
        dense2 = Dense(self.config['dense_units']//2, activation='relu')(dense1)
        dense2 = Dropout(self.config['dropout_rate']/2)(dense2)
        
        # Skip connection
        if gru_final.shape[-1] == dense2.shape[-1]:
            dense2_skip = dense2 + gru_final
        else:
            dense2_skip = dense2
        
        outputs = Dense(1, activation='linear')(dense2_skip)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=AdamW(learning_rate=self.config['learning_rate']),
            loss='huber',  # Robust loss function
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_models(self, X: np.ndarray, y_signal: np.ndarray, y_profit: np.ndarray) -> Dict:
        """Train both models with advanced callbacks"""
        logger.info("Training hybrid deep learning models...")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_signal_train, y_signal_val = y_signal[:split_idx], y_signal[split_idx:]
        y_profit_train, y_profit_val = y_profit[:split_idx], y_profit[split_idx:]
        
        # Build models
        self.signal_model = self.build_advanced_signal_model(
            X.shape[1:], len(np.unique(y_signal))
        )
        self.profit_model = self.build_advanced_profit_model(X.shape[1:])
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_signal_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train signal model
        logger.info("Training signal model...")
        signal_history = self.signal_model.fit(
            X_train, y_signal_train,
            validation_data=(X_val, y_signal_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Train profit model
        logger.info("Training profit model...")
        profit_history = self.profit_model.fit(
            X_train, y_profit_train,
            validation_data=(X_val, y_profit_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'signal_history': signal_history,
            'profit_history': profit_history
        }

# Continue with the rest of the enhanced system...
class EnhancedCryptoPredictor:
    """Main enhanced crypto trading prediction system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigManager.load_config(config_path)
        self.preprocessor = EnhancedDataPreprocessor(self.config)
        self.dl_model = HybridDeepLearningModel(self.config)
        self.ensemble_models = {}
        self.is_trained = False
        
        # Initialize ensemble models
        self._initialize_ensemble_models()
    
    def _initialize_ensemble_models(self):
        """Initialize ensemble ML models"""
        self.ensemble_models = {
            'rf_classifier': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                random_state=42,
                n_jobs=-1
            ),
            'xgb_classifier': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lgb_classifier': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'gb_regressor': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'xgb_regressor': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
    
    def train(self, data: pd.DataFrame) -> Dict:
        """Train the complete system"""
        logger.info("Starting comprehensive training process...")
        
        # Feature engineering
        data_processed = self.preprocessor.engineer_features(data)
        data_with_signals = self.preprocessor.create_advanced_trading_signals(data_processed)
        
        # Prepare sequences for deep learning
        sequence_length = self.config['model']['sequence_length']
        X_seq, y_signal_seq, y_profit_seq = self.preprocessor.prepare_sequences(
            data_with_signals, sequence_length
        )
        
        if len(X_seq) == 0:
            logger.error("No sequences available for training")
            return {}
        
        # Train deep learning models
        dl_history = self.dl_model.train_models(X_seq, y_signal_seq, y_profit_seq)
        
        # Train ensemble models
        logger.info("Training ensemble models...")
        feature_cols = self.preprocessor.feature_names
        X_ensemble = data_with_signals[feature_cols].dropna()
        
        # Align with sequences
        if len(X_ensemble) > len(y_signal_seq):
            X_ensemble = X_ensemble.iloc[-len(y_signal_seq):].reset_index(drop=True)
        
        # Train classifiers
        try:
            self.ensemble_models['rf_classifier'].fit(X_ensemble, y_signal_seq)
            self.ensemble_models['xgb_classifier'].fit(X_ensemble, y_signal_seq)
            self.ensemble_models['lgb_classifier'].fit(X_ensemble, y_signal_seq)
            
            # Train regressors
            self.ensemble_models['gb_regressor'].fit(X_ensemble, y_profit_seq)
            self.ensemble_models['xgb_regressor'].fit(X_ensemble, y_profit_seq)
            
            logger.info("Ensemble models trained successfully")
        except Exception as e:
            logger.error(f"Error training ensemble models: {e}")
        
        self.is_trained = True
        return dl_history
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make comprehensive predictions"""
        if not self.is_trained:
            logger.error("Models not trained yet")
            return pd.DataFrame()
        
        logger.info("Making predictions...")
        
        # Preprocess data
        data_processed = self.preprocessor.engineer_features(data)
        
        # Prepare sequences
        sequence_length = self.config['model']['sequence_length']
        X_seq, _, _ = self.preprocessor.prepare_sequences(data_processed, sequence_length)
        
        if len(X_seq) == 0:
            logger.warning("No sequences available for prediction")
            return pd.DataFrame()
        
        # Deep learning predictions
        try:
            dl_signal_probs = self.dl_model.signal_model.predict(X_seq, verbose=0)
            dl_profit_pred = self.dl_model.profit_model.predict(X_seq, verbose=0).flatten()
        except Exception as e:
            logger.error(f"Error in deep learning prediction: {e}")
            dl_signal_probs = np.zeros((len(X_seq), 3))
            dl_profit_pred = np.zeros(len(X_seq))
        
        # Ensemble predictions
        feature_cols = self.preprocessor.feature_names
        X_ensemble = data_processed[feature_cols].dropna()
        X_ensemble = X_ensemble.iloc[-len(X_seq):].reset_index(drop=True)
        
        try:
            rf_signal_probs = self.ensemble_models['rf_classifier'].predict_proba(X_ensemble)
            xgb_signal_probs = self.ensemble_models['xgb_classifier'].predict_proba(X_ensemble)
            lgb_signal_probs = self.ensemble_models['lgb_classifier'].predict_proba(X_ensemble)
            
            gb_profit_pred = self.ensemble_models['gb_regressor'].predict(X_ensemble)
            xgb_profit_pred = self.ensemble_models['xgb_regressor'].predict(X_ensemble)
            
            # Ensemble averaging
            ensemble_signal_probs = (rf_signal_probs + xgb_signal_probs + lgb_signal_probs) / 3
            ensemble_profit_pred = (gb_profit_pred + xgb_profit_pred) / 2
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            ensemble_signal_probs = np.zeros_like(dl_signal_probs)
            ensemble_profit_pred = np.zeros_like(dl_profit_pred)
        
        # Combine predictions with weights
        combined_signal_probs = 0.6 * dl_signal_probs + 0.4 * ensemble_signal_probs
        combined_profit_pred = 0.6 * dl_profit_pred + 0.4 * ensemble_profit_pred
        
        # Generate final predictions
        results = self._generate_trading_decisions(
            combined_signal_probs, combined_profit_pred, data_processed, len(X_seq)
        )
        
        return pd.DataFrame(results)
    
    def _generate_trading_decisions(self, signal_probs: np.ndarray, profit_pred: np.ndarray, 
                                  data: pd.DataFrame, seq_len: int) -> List[Dict]:
        """Generate final trading decisions with risk management"""
        results = []
        confidence_threshold = self.config['trading']['confidence_threshold']
        
        for i, (sig_prob, profit) in enumerate(zip(signal_probs, profit_pred)):
            current_price = data['close'].iloc[-(seq_len-i)]
            max_confidence = np.max(sig_prob)
            signal_class = np.argmax(sig_prob)
            
            try:
                signal_name = self.preprocessor.label_encoder.inverse_transform([signal_class])[0]
            except:
                signal_name = 'SKIP'
            
            # Apply confidence threshold
            if max_confidence < confidence_threshold:
                signal_name = 'SKIP'
            
            # Calculate stop loss and take profit
            if signal_name == 'BUY':
                take_profit = current_price * (1 + max(abs(profit), 0.015))
                stop_loss = current_price * (1 - 0.015)  # 1.5% stop loss
            elif signal_name == 'SELL':
                take_profit = current_price * (1 - max(abs(profit), 0.015))
                stop_loss = current_price * (1 + 0.015)  # 1.5% stop loss
            else:
                take_profit = current_price
                stop_loss = current_price
            
            results.append({
                'signal': signal_name,
                'confidence': max_confidence,
                'profit_potential': profit,
                'current_price': current_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'risk_reward_ratio': abs(take_profit - current_price) / abs(stop_loss - current_price) if stop_loss != current_price else 0
            })
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained models"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save deep learning models
        if self.dl_model.signal_model:
            self.dl_model.signal_model.save(f"{filepath}_signal_model.h5")
        if self.dl_model.profit_model:
            self.dl_model.profit_model.save(f"{filepath}_profit_model.h5")
        
        # Save ensemble models and preprocessor
        joblib.dump({
            'ensemble_models': self.ensemble_models,
            'preprocessor': self.preprocessor,
            'config': self.config
        }, f"{filepath}_components.pkl")
        
        logger.info(f"Models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models"""
        try:
            # Load deep learning models
            self.dl_model.signal_model = tf.keras.models.load_model(f"{filepath}_signal_model.h5")
            self.dl_model.profit_model = tf.keras.models.load_model(f"{filepath}_profit_model.h5")
            
            # Load other components
            components = joblib.load(f"{filepath}_components.pkl")
            self.ensemble_models = components['ensemble_models']
            self.preprocessor = components['preprocessor']
            self.config = components['config']
            
            self.is_trained = True
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# Usage example and testing
def main():
    """Main execution function"""
    logger.info(f"Enhanced Crypto Trading System v2.0")
    logger.info(f"User: amenson1983new")
    logger.info(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Initialize system with custom config
    config = {
        'model': {
            'sequence_length': 25,
            'lstm_units': 128,
            'gru_units': 96,
            'dense_units': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 50,  # Reduced for demo
            'early_stopping_patience': 15,
            'use_attention': True,
            'use_conv1d': False
        },
        'trading': {
            'min_profit_threshold': 0.015,
            'confidence_threshold': 0.65,
            'future_periods': 3
        }
    }
    
    try:
        # Save config
        ConfigManager.save_config(config, 'config/trading_config.json')
        
        # Initialize predictor
        predictor = EnhancedCryptoPredictor('config/trading_config.json')
        
        # Load and train (replace with your actual file paths)
        logger.info("Loading training data...")
        # train_data = pd.read_csv('bnbusdt_2024_5m.csv')
        # history = predictor.train(train_data)
        
        # Load and test
        # test_data = pd.read_csv('bnbusdt_2025_01_5m.csv')
        # predictions = predictor.predict(test_data)
        
        # Save results
        # predictions.to_csv('enhanced_predictions_2025_01.csv', index=False)
        # predictor.save_model('models/enhanced_crypto_predictor')
        
        logger.info("✅ Enhanced crypto trading system ready!")
        logger.info("📊 Features: Advanced DL models, ensemble methods, risk management")
        logger.info("🎯 Target: 85% profitable trades")
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")

if __name__ == "__main__":
    main()