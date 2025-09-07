import pandas as pd
import numpy as np
import json
import logging
import os
import sys
from datetime import datetime, timedelta
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
    print("pandas-ta imported successfully")
except ImportError as e:
    print(f"Warning: pandas-ta not available: {e}")
    PANDAS_TA_AVAILABLE = False
    ta = None

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available for hyperparameter optimization. Set 'optimize_hyperparameters' to false in config.")


try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from multi_timeframe_analyzer import MultiTimeframeAnalyzer

class CryptoTradingSystem:
    def __init__(self, symbol, interval="5m"):
        print("Initializing Enhanced Crypto Trading System...")
        self.create_directories()
        
        self.symbol = symbol
        self.interval = interval
        config_path = f"config/{self.symbol}_config.json"
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                print(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Please ensure it exists.")
            raise
        
        self.model = None
        self.feature_columns = None
        
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Enhanced Crypto Trading System initialized successfully")
        self.logger.info(f"Symbol: {self.symbol}, Interval: {self.interval}")

    def create_directories(self):
        dirs = ['logs', 'models', 'output', 'data', 'config', 'temp']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
        
    def setup_logging(self):
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_filename = f'logs/{self.symbol}_{self.interval}_trading.log'
        
        logger = logging.getLogger()
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                
        logging.basicConfig(level=logging.INFO, format=log_format,
                            handlers=[logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
                                      logging.StreamHandler()])
        
        print(f"Logging setup completed: {log_filename}")

    def load_data(self, file_path):
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
        self.logger.info("Calculating technical indicators...")
        if not PANDAS_TA_AVAILABLE:
            self.logger.warning("pandas-ta not available. Some features may be missing.")
            return df
        
        data = df.copy()
        try:
            if 'open_time' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                data.set_index('open_time', inplace=True)

            indicators_config = self.config['indicators']
            regime_filter_config = self.config.get('regime_filter', {})
            
            strategy_ta = [
                {"kind": "rsi", "length": indicators_config.get('rsi_period', 14)},
                {"kind": "stoch", "k": indicators_config.get('stoch_k', 14), "d": indicators_config.get('stoch_d', 3)},
                {"kind": "willr"}, {"kind": "roc"}, {"kind": "cci"},
                {"kind": "sma", "length": 20}, {"kind": "sma", "length": 50},
                {"kind": "ema", "length": 12}, {"kind": "ema", "length": 26},
                {"kind": "macd", "fast": indicators_config.get('macd_fast', 12), "slow": indicators_config.get('macd_slow', 26), "signal": indicators_config.get('macd_signal', 9)},
                {"kind": "atr", "length": indicators_config.get('atr_period', 14)},
                {"kind": "bbands", "length": indicators_config.get('bb_period', 20), "std": indicators_config.get('bb_std', 2.0)},
                {"kind": "obv"}, {"kind": "vwap"}, {"kind": "mfi"}
            ]

            if regime_filter_config.get('enabled', False):
                period = regime_filter_config.get('period', 200)
                strategy_ta.append({"kind": "sma", "length": period})

            data.ta.strategy(ta.Strategy(name="Comprehensive", ta=strategy_ta))
            data.columns = [col.lower() for col in data.columns]
            
            rename_map = {
                f"atrr_{indicators_config.get('atr_period', 14)}": "atr",
                f"rsi_{indicators_config.get('rsi_period', 14)}": "rsi",
            }
            if regime_filter_config.get('enabled', False):
                period = regime_filter_config.get('period', 200)
                rename_map[f"sma_{period}"] = "regime_sma"

            data.rename(columns=rename_map, inplace=True)

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
        self.logger.info("Generating trading signals...")
        data = df.copy()
        signals = []
        sg_config = self.config['signal_generation']
        future_periods = sg_config['future_periods']
        profit_threshold = sg_config['profit_threshold']
        buy_rr_ratio = sg_config.get('buy_risk_reward_ratio', 2.0)
        sell_rr_ratio = sg_config.get('sell_risk_reward_ratio', 2.0)
        
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
            
            if buy_conditions: signals.append('BUY')
            elif sell_conditions: signals.append('SELL')
            else: signals.append('SKIP')
        
        data['signal'] = signals + ['SKIP'] * future_periods
        self.logger.info(f"Signal distribution: {dict(data['signal'].value_counts())}")
        return data

    def prepare_features(self, df, for_training=True):
        self.logger.info("Preparing features...")
        exclude_cols = ['open_time', 'close_time', 'dt', 'signal', 'predicted_signal', 'confidence', 'trade_outcome', 'profit_pct', 'exit_price', 'stop_loss_price', 'take_profit_price']
        
        if for_training:
            features_df = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')
            self.feature_columns = features_df.select_dtypes(include=np.number).columns.tolist()
            X = features_df[self.feature_columns].copy()
            y = df['signal']
        else:
            if not self.feature_columns:
                raise ValueError("Feature columns not set. Train or load a model first.")
            X = pd.DataFrame(columns=self.feature_columns)
            for col in self.feature_columns:
                if col in df.columns:
                    X[col] = df[col]
                else:
                    X[col] = 0
            y = None
        
        X.fillna(method='ffill', inplace=True)
        X.fillna(method='bfill', inplace=True)
        X.replace([np.inf, -np.inf], 0, inplace=True)
        X.fillna(0, inplace=True)
        
        return X, y

    def optimize_hyperparameters(self, X, y):
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, skipping optimization.")
            return self.config['model']['default_params']

        if len(np.unique(y)) < 2:
            self.logger.warning("Only one class present; skipping optimization.")
            return self.config['model']['default_params']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        def objective(trial):
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
            rf_max_depth = trial.suggest_int('rf_max_depth', 5, 30)
            gb_n_estimators = trial.suggest_int('gb_n_estimators', 50, 300)
            gb_learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True)
            gb_max_depth = trial.suggest_int('gb_max_depth', 3, 10)

            rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate, max_depth=gb_max_depth, random_state=42)
            
            model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            return f1_score(y_val, y_pred, average='weighted')

        self.logger.info(f"Starting hyperparameter optimization with Optuna...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['model'].get('optimization_trials', 50))

        self.logger.info(f"Optimization finished. Best trial: {study.best_trial.params}")
        
        best_params = self.config['model']['default_params'].copy()
        best_params.update(study.best_trial.params)
        return best_params

    def train_model(self, df):
        self.logger.info("Starting model training...")
        X, y = self.prepare_features(df, for_training=True)
        trainable_mask = y.isin(['BUY', 'SELL'])
        X_trainable, y_trainable = X[trainable_mask], y[trainable_mask]

        if len(y_trainable.unique()) < 2:
            self.logger.error("Not enough classes to train model. Need both BUY and SELL signals in data.")
            raise ValueError("Training requires data with both BUY and SELL signals.")

        if self.config['model'].get('optimize_hyperparameters', False):
            params = self.optimize_hyperparameters(X_trainable, y_trainable)
        else:
            self.logger.info("Using default hyperparameters from config.")
            params = self.config['model']['default_params']

        rf_params = {k.replace('rf_', ''): v for k, v in params.items() if 'rf_' in k}
        gb_params = {k.replace('gb_', ''): v for k, v in params.items() if 'gb_' in k}
        
        rf = RandomForestClassifier(**rf_params, random_state=42)
        gb = GradientBoostingClassifier(**gb_params, random_state=42)
        
        self.model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft', n_jobs=-1)
        self.model.fit(X_trainable, y_trainable)
        
        model_name = f"models/{self.symbol}_{self.interval}_trading_model.pkl"
        joblib.dump({'model': self.model, 'feature_columns': self.feature_columns, 'params': params}, model_name)
        self.logger.info(f"Model and parameters saved to {model_name}")
        return self.model

    def load_model(self):
        model_path = f"models/{self.symbol}_{self.interval}_trading_model.pkl"
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return False
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.logger.info(f"Model loaded from {model_path}")
        return True

    def predict_signals(self, df):
        self.logger.info("Predicting signals...")
        if not self.model:
            if not self.load_model():
                raise FileNotFoundError("Could not load model for prediction.")
        
        X, _ = self.prepare_features(df, for_training=False)
        
        result_df = df.copy()
        
        result_df['predicted_signal'] = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        result_df['confidence'] = np.max(probabilities, axis=1)
        
        confidence_threshold = self.config['signal_generation']['confidence_threshold']
        result_df.loc[result_df['confidence'] < confidence_threshold, 'predicted_signal'] = 'SKIP'

        regime_filter_config = self.config.get('regime_filter', {})
        if regime_filter_config.get('enabled', False) and 'regime_sma' in result_df.columns:
            self.logger.info("Applying regime filter...")
            result_df.loc[(result_df['predicted_signal'] == 'BUY') & (result_df['close'] < result_df['regime_sma']), 'predicted_signal'] = 'SKIP'
            self.logger.info("Regime filter applied.")

        self.logger.info(f"Prediction distribution after filters: {dict(result_df['predicted_signal'].value_counts())}")
        return result_df

    def _apply_multi_timeframe_features(self, df):
        if self.config.get('multi_timeframe', {}).get('enabled', False):
            self.logger.info("Adding multi-timeframe features...")
        return df

    def calculate_stop_loss_take_profit(self, df):
        self.logger.info("Calculating stop loss and take profit levels...")
        result_df = df.copy()
        rm_config = self.config['risk_management']
        sl_multiplier = rm_config['stop_loss_atr_multiplier']
        tp_multiplier = rm_config['take_profit_atr_multiplier']
        
        result_df['stop_loss_price'] = np.nan
        result_df['take_profit_price'] = np.nan
        
        buy_mask = result_df['predicted_signal'] == 'BUY'
        sell_mask = result_df['predicted_signal'] == 'SELL'
        
        if 'atr' not in result_df.columns:
            self.logger.error("'atr' column not found for stop loss calculation. Using fallback.")
            atr_values = result_df['close'] * 0.02 
        else:
            atr_values = result_df['atr'].fillna(result_df['close'] * 0.02)
        
        result_df.loc[buy_mask, 'stop_loss_price'] = result_df['close'] - (atr_values * sl_multiplier)
        result_df.loc[buy_mask, 'take_profit_price'] = result_df['close'] + (atr_values * tp_multiplier)
        
        result_df.loc[sell_mask, 'stop_loss_price'] = result_df['close'] + (atr_values * sl_multiplier)
        result_df.loc[sell_mask, 'take_profit_price'] = result_df['close'] - (atr_values * tp_multiplier)
        
        self.logger.info("Stop loss and take profit levels calculated.")
        return result_df

    def analyze_trades(self, df):
        self.logger.info("Analyzing trade outcomes with trailing stop-loss...")
        result_df = df.copy()
        rm_config = self.config['risk_management']
        analysis_config = self.config['analysis']
        
        trailing_enabled = rm_config.get('trailing_stop_enabled', False)
        trailing_atr_multiplier = rm_config.get('trailing_atr_multiplier', 1.5)
        activation_profit_pct = rm_config.get('activation_profit_pct', 0.01)
        
        future_periods = analysis_config['future_periods_for_analysis']

        result_df['profit_pct'] = 0.0
        result_df['exit_price'] = np.nan
        result_df['trade_outcome'] = 'NO_TRADE'
        
        for i in result_df.index:
            signal = result_df.at[i, 'predicted_signal']
            if signal not in ['BUY', 'SELL']:
                continue

            entry_price = result_df.at[i, 'close']
            initial_sl = result_df.at[i, 'stop_loss_price']
            tp_price = result_df.at[i, 'take_profit_price']
            
            if pd.isna(initial_sl) or pd.isna(tp_price):
                continue
            
            future_slice = result_df.loc[i+1 : i + future_periods]
            if future_slice.empty:
                continue

            exit_price = None
            trade_outcome = 'TIMEOUT'
            current_sl = initial_sl
            trailing_activated = False
            
            peak_price = entry_price

            for index, future_row in future_slice.iterrows():
                atr_val = future_row.get('atr', entry_price * 0.02)
                
                if signal == 'BUY':
                    peak_price = max(peak_price, future_row['high'])
                    
                    if not trailing_activated and trailing_enabled:
                        if (future_row['high'] - entry_price) / entry_price >= activation_profit_pct:
                            trailing_activated = True
                            self.logger.debug(f"Trade at index {i}: Trailing stop activated.")
                    
                    if trailing_activated:
                        new_sl = peak_price - (atr_val * trailing_atr_multiplier)
                        current_sl = max(current_sl, new_sl)

                    if future_row['low'] <= current_sl:
                        exit_price, trade_outcome = current_sl, 'STOP_LOSS' if not trailing_activated else 'TRAILING_STOP'
                        break
                    elif future_row['high'] >= tp_price:
                        exit_price, trade_outcome = tp_price, 'TAKE_PROFIT'
                        break
                
                elif signal == 'SELL':
                    peak_price = min(peak_price, future_row['low'])
                    
                    if not trailing_activated and trailing_enabled:
                        if (entry_price - future_row['low']) / entry_price >= activation_profit_pct:
                            trailing_activated = True
                            self.logger.debug(f"Trade at index {i}: Trailing stop activated.")

                    if trailing_activated:
                        new_sl = peak_price + (atr_val * trailing_atr_multiplier)
                        current_sl = min(current_sl, new_sl)

                    if future_row['high'] >= current_sl:
                        exit_price, trade_outcome = current_sl, 'STOP_LOSS' if not trailing_activated else 'TRAILING_STOP'
                        break
                    elif future_row['low'] <= tp_price:
                        exit_price, trade_outcome = tp_price, 'TAKE_PROFIT'
                        break
            
            if exit_price is None:
                exit_price = future_slice.iloc[-1]['close']
            
            profit_pct = ((exit_price - entry_price) / entry_price * 100) if signal == 'BUY' else ((entry_price - exit_price) / entry_price * 100)
            
            result_df.at[i, 'profit_pct'] = profit_pct
            result_df.at[i, 'exit_price'] = exit_price
            result_df.at[i, 'trade_outcome'] = trade_outcome

        self.logger.info("Trade analysis completed.")
        return result_df

def main(symbol, interval="5m", train_model=False):
    """
    Main function to run the full training and backtesting pipeline for a given symbol.
    :param symbol: The crypto symbol to process (e.g., 'BNBUSDT').
    :param interval: The data interval (e.g., '5m').
    :param train_model: Boolean flag to control model training. If False, loads an existing model for prediction.
    """
    print(f"\n{'='*20} Running Pipeline for: {symbol.upper()} {'='*20}")
    
    try:
        trading_system = CryptoTradingSystem(symbol=symbol, interval=interval)
        
        training_file = f"data/{symbol}_{interval}_assembley.csv"
        current_file = f"data/{symbol}_{interval}_current.csv"
        output_file = f"output/{symbol}_{interval}_trading_results.csv"
        
        if train_model:
            print("\n--- Training Phase ---")
            if not os.path.exists(training_file):
                print(f"Training file not found at {training_file}. Skipping training.")
            else:
                df_train = trading_system.load_data(training_file)
                df_train_features = trading_system._apply_multi_timeframe_features(df_train)
                df_train_features = trading_system.calculate_indicators(df_train_features)
                df_with_signals = trading_system.generate_signals(df_train_features)
                trading_system.train_model(df_with_signals)
                print("Training complete. Model saved.")
        else:
            print("\n--- Prediction Phase (loading existing model) ---")
            # This block is for prediction only. The model must already exist.
            if not os.path.exists(f"models/{symbol}_{interval}_trading_model.pkl"):
                 print("Model not found. Cannot run prediction. Please train a model first.")
                 return

        # Always run the analysis phase
        print("\n--- Analysis Phase ---")
        if not os.path.exists(current_file):
            print(f"Current data file not found at {current_file}. Skipping analysis.")
            return
            
        df_current = trading_system.load_data(current_file)
        df_current_features = trading_system._apply_multi_timeframe_features(df_current)
        df_current_features = trading_system.calculate_indicators(df_current_features)
        
        df_with_predictions = trading_system.predict_signals(df_current_features)
        df_with_risk = trading_system.calculate_stop_loss_take_profit(df_with_predictions)
        final_results = trading_system.analyze_trades(df_with_risk)
        
        if 'open_time' in final_results.columns:
            dt_series = pd.to_datetime(final_results['open_time'])
            final_results['year'] = dt_series.dt.year
            final_results['month'] = dt_series.dt.month
            final_results['day'] = dt_series.dt.day
            final_results['hour'] = dt_series.dt.hour
            final_results['minute'] = dt_series.dt.minute
        
        final_results.to_csv(output_file, index=False)
        print(f"\nProcessing completed successfully for {symbol}!")
        print(f"Results saved to: {output_file}")
        
        trade_outcomes = final_results['trade_outcome'].value_counts()
        print("\nTrade Outcome Summary:")
        print(trade_outcomes)

    except Exception as e:
        print(f"\nAn error occurred during pipeline execution for {symbol}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crypto Trading System")
    parser.add_argument("--symbol", type=str, default="BNBUSDT", help="The crypto symbol to process.")
    parser.add_argument("--interval", type=str, default="5m", help="The data interval to use.")
    parser.add_argument("--train", action='store_true', help="Flag to run the training phase.")
    parser.add_argument("--no-train", dest='train', action='store_false', help="Flag to run the prediction phase (default).")
    parser.set_defaults(train=False)

    args = parser.parse_args()

    main(symbol=args.symbol, interval=args.interval, train_model=args.train)