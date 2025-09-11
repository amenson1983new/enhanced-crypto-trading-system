"""
Data Processor for Enhanced Crypto Trading System
Handles data loading, technical indicator calculation (including MACD and ATR), and feature preparation.
"""
import pandas as pd
import numpy as np
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Handle pandas-ta import with compatibility fix
try:
    import numpy as np
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    if not hasattr(np, 'NAN'): np.NAN = np.nan
    if not hasattr(np, 'Inf'): np.Inf = np.inf
    if not hasattr(np, 'NINF'): np.NINF = np.NINF if hasattr(np, 'NINF') else -np.inf
    if not hasattr(np, 'PINF'): np.PINF = np.PINF if hasattr(np, 'PINF') else np.inf
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pandas-ta not available: {e}")
    PANDAS_TA_AVAILABLE = False
    ta = None


class DataProcessor:
    def __init__(self, config, strategies):
        """Initialize DataProcessor with configuration and strategies."""
        self.config = config
        self.strategies = strategies
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_data(self, file_path):
        """Load data from CSV file."""
        self.logger.info(f"Loading data from {file_path}")
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df.columns = [col.lower() for col in df.columns]
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
        self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def calculate_indicators(self, df):
        """Calculate technical indicators including MACD and ATR using pandas-ta."""
        self.logger.info("Calculating technical indicators...")
        if not PANDAS_TA_AVAILABLE:
            self.logger.warning("pandas-ta not available. Some features may be missing.")
            return df
        
        data = df.copy()
        try:
            # Set datetime index for pandas-ta
            if 'open_time' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                data.set_index('open_time', inplace=True)

            indicators_config = self.config['indicators']
            regime_filter_config = self.config.get('regime_filter', {})
            
            # Define comprehensive technical analysis strategy including MACD and ATR
            strategy_ta = [
                {"kind": "rsi", "length": indicators_config.get('rsi_period', 14)},
                {"kind": "stoch", "k": indicators_config.get('stoch_k', 14), "d": indicators_config.get('stoch_d', 3)},
                {"kind": "willr"}, {"kind": "roc"}, {"kind": "cci"},
                {"kind": "sma", "length": 20}, {"kind": "sma", "length": 50},
                {"kind": "ema", "length": 12}, {"kind": "ema", "length": 26},
                # MACD indicator with configurable parameters
                {"kind": "macd", "fast": indicators_config.get('macd_fast', 12), 
                 "slow": indicators_config.get('macd_slow', 26), 
                 "signal": indicators_config.get('macd_signal', 9)},
                # ATR indicator with configurable period
                {"kind": "atr", "length": indicators_config.get('atr_period', 14)},
                {"kind": "bbands", "length": indicators_config.get('bb_period', 20), "std": indicators_config.get('bb_std', 2.0)},
                {"kind": "obv"}, {"kind": "vwap"}, {"kind": "mfi"}
            ]

            # Add regime filter SMA if enabled
            if regime_filter_config.get('enabled', False):
                period = regime_filter_config.get('period', 200)
                strategy_ta.append({"kind": "sma", "length": period})

            # Calculate all indicators using pandas-ta strategy
            data.ta.strategy(ta.Strategy(name="Comprehensive", ta=strategy_ta))
            data.columns = [col.lower() for col in data.columns]
            
            # Rename columns for consistency
            rename_map = {
                f"atrr_{indicators_config.get('atr_period', 14)}": "atr",
                f"rsi_{indicators_config.get('rsi_period', 14)}": "rsi",
            }
            if regime_filter_config.get('enabled', False):
                period = regime_filter_config.get('period', 200)
                rename_map[f"sma_{period}"] = "regime_sma"

            data.rename(columns=rename_map, inplace=True)

            # Calculate Bollinger Bands width
            bb_period = indicators_config.get('bb_period', 20)
            bb_std = float(indicators_config.get('bb_std', 2.0))
            bbl_col = f'bbl_{bb_period}_{bb_std}'.lower()
            bbm_col = f'bbm_{bb_period}_{bb_std}'.lower()
            bbu_col = f'bbu_{bb_period}_{bb_std}'.lower()

            if bbu_col in data.columns and bbl_col in data.columns and not data[bbm_col].isnull().all():
                 data['bb_width'] = (data[bbu_col] - data[bbl_col]) / data[bbm_col]
            else:
                 data['bb_width'] = np.nan
            
            data.reset_index(inplace=True)
            return data.loc[:, ~data.columns.duplicated()]

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            if isinstance(data.index, pd.DatetimeIndex):
                data.reset_index(inplace=True)
            return df

    def generate_signals(self, df):
        """Generate trading signals based on technical indicators."""
        self.logger.info("Generating trading signals...")
        data = df.copy()
        signals = []
        sg_config = self.config['signal_generation']
        future_periods = sg_config['future_periods']
        profit_threshold = sg_config['profit_threshold']
        buy_rr_ratio = sg_config.get('buy_risk_reward_ratio', 2.0)
        sell_rr_ratio = sg_config.get('sell_risk_reward_ratio', 2.0)
        
        # Calculate volatility threshold using Bollinger Bands width
        if 'bb_width' not in data.columns or data['bb_width'].isnull().all():
            volatility_threshold = np.inf
        else:
            volatility_threshold = data['bb_width'].quantile(0.75)
        self.logger.info(f"Volatility threshold (max bb_width): {volatility_threshold:.4f}")

        for i in range(len(data) - future_periods):
            row = data.iloc[i]
            future_slice = data.iloc[i+1:i+future_periods+1]
            upward_potential = (future_slice['high'].max() - row['close']) / row['close']
            downward_potential = (row['close'] - future_slice['low'].min()) / row['close']
            
            is_low_volatility = row.get('bb_width', 1.0) < volatility_threshold
            
            buy_conditions = is_low_volatility and upward_potential > profit_threshold and upward_potential / max(downward_potential, 1e-5) > buy_rr_ratio
            sell_conditions = is_low_volatility and downward_potential > profit_threshold and downward_potential / max(upward_potential, 1e-5) > sell_rr_ratio
            
            if buy_conditions: 
                signals.append('BUY')
            elif sell_conditions: 
                signals.append('SELL')
            else: 
                signals.append('SKIP')
        
        # Pad remaining signals
        signals.extend(['SKIP'] * future_periods)
        data['signal'] = signals[:len(data)]
        return data

    def prepare_features(self, df):
        """Prepare feature matrix and target labels for machine learning."""
        self.logger.info("Preparing features for ML model...")
        
        # Select feature columns (excluding target and metadata)
        feature_cols = [col for col in df.columns if col not in ['signal', 'open_time', 'close_time']]
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.ffill().fillna(0)
        
        # Create target labels if signal column exists
        if 'signal' in df.columns:
            # Map signals to numeric labels
            signal_map = {'BUY': 0, 'SELL': 1, 'SKIP': 2}
            y = df['signal'].map(signal_map)
            return X, y
        else:
            return X, None

    def run_for_training(self):
        """Run complete data processing pipeline for training."""
        try:
            # Load training data
            train_path = self.config['data']['train_path']
            df = self.load_data(train_path)
            
            # Calculate technical indicators including MACD and ATR
            df = self.calculate_indicators(df)
            
            # Generate trading signals
            df = self.generate_signals(df)
            
            # Prepare features and labels
            X, y = self.prepare_features(df)
            
            self.logger.info(f"Training data prepared. Features: {X.shape}, Labels: {y.shape if y is not None else 'None'}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in training pipeline: {e}", exc_info=True)
            return None, None

    def run_for_prediction(self, predict_df):
        """Run data processing pipeline for prediction."""
        try:
            # Calculate technical indicators including MACD and ATR
            processed_df = self.calculate_indicators(predict_df)
            
            # Prepare features (no signal generation needed for prediction)
            X_predict, _ = self.prepare_features(processed_df)
            
            self.logger.info(f"Prediction data prepared. Features: {X_predict.shape}")
            return processed_df, X_predict
            
        except Exception as e:
            self.logger.error(f"Error in prediction pipeline: {e}", exc_info=True)
            return predict_df, None