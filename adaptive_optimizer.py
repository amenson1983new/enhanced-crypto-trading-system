import pandas as pd
import numpy as np
import json
import os
import subprocess
import logging
import argparse

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_market_regime(symbol, interval="5m", data_file=None, lookback_days=14):
    """
    Analyzes the most recent market data to determine the current regime.
    
    Returns:
        str: 'trending' or 'ranging'
    """
    logging.info(f"Detecting market regime for {symbol}...")
    
    if data_file is None:
        data_file = f"data/{symbol}_{interval}_current.csv"
        
    if not os.path.exists(data_file):
        logging.error(f"Data file not found at {data_file}. Cannot detect regime.")
        raise FileNotFoundError(f"Data file not found: {data_file}")
        
    df = pd.read_csv(data_file)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    
    recent_data = df[df['open_time'] >= (df['open_time'].max() - pd.Timedelta(days=lookback_days))]
    if len(recent_data) < 50:
        logging.warning("Not enough recent data to determine regime. Defaulting to 'ranging'.")
        return 'ranging'

    try:
        import pandas_ta as ta
        atr = ta.atr(recent_data['high'], recent_data['low'], recent_data['close'], length=14)
        if atr is None or atr.isnull().all():
            raise ImportError("pandas_ta ATR calculation failed.")
        
        normalized_atr = (atr / recent_data['close']).dropna()
        avg_volatility = normalized_atr.mean()
        volatility_threshold = 0.002

        logging.info(f"Average normalized ATR over last {lookback_days} days: {avg_volatility:.5f}")

        if avg_volatility > volatility_threshold:
            logging.info("Regime detected: Trending (due to high volatility)")
            return 'trending'
        else:
            logging.info("Regime detected: Ranging (due to low volatility)")
            return 'ranging'

    except (ImportError, Exception) as e:
        logging.error(f"Could not use pandas_ta for regime detection: {e}. Defaulting to 'ranging'.")
        recent_data['price_change'] = recent_data['close'].pct_change().abs()
        if recent_data['price_change'].mean() > 0.0015:
             return 'trending'
        return 'ranging'


def main(symbol, interval="5m", incremental_train=True):
    """
    Main function to run the full, end-to-end adaptive optimization and prediction pipeline.
    :param incremental_train: If True, combines historical and new data. If False, trains from scratch.
    """
    logging.info(f"--- Starting Full Pipeline for {symbol} (Incremental Train: {incremental_train}) ---")
    
    # 1. Detect Regime
    try:
        current_regime = detect_market_regime(symbol, interval)
    except FileNotFoundError as e:
        logging.error(f"Stopping execution. {e}")
        return

    # 2. Select Strategy
    strategies_file = 'strategies.json'
    if not os.path.exists(strategies_file):
        logging.error(f"Strategies portfolio file not found at '{strategies_file}'. Aborting.")
        return
        
    with open(strategies_file, 'r') as f:
        strategies = json.load(f)
        
    if current_regime not in strategies:
        logging.error(f"No strategy found for the detected regime '{current_regime}'. Aborting.")
        return
        
    selected_strategy = strategies[current_regime]
    logging.info(f"Selected strategy for '{current_regime}' regime: {selected_strategy['description']}")

    # 3. Update Main Config
    main_config_path = f"config/{symbol}_config.json"
    if not os.path.exists(main_config_path):
        logging.error(f"Main config file not found at '{main_config_path}'. Aborting.")
        return

    with open(main_config_path, 'r') as f:
        main_config = json.load(f)

    logging.info(f"Updating '{main_config_path}' with the selected strategy...")
    
    for key, value in selected_strategy['config'].items():
        if key in main_config:
            main_config[key].update(value)
        else:
            main_config[key] = value
            
    with open(main_config_path, 'w') as f:
        json.dump(main_config, f, indent=4)
    
    logging.info("Main configuration has been automatically updated.")

    # 4. Trigger Model Retraining (Conditional)
    logging.info("\n--- Phase 1: Triggering Model Retraining ---")
    
    try:
        if incremental_train:
            logging.info("Running INCREMENTAL training...")
            train_script = 'incremental_trainer.py'
            train_args = ['python', train_script, '--symbol', symbol, '--interval', interval]
        else:
            logging.info("Running training FROM SCRATCH...")
            train_script = 'crypto_trading_app.py'
            train_args = ['python', train_script, '--train', '--symbol', symbol, '--interval', interval]

        result_train = subprocess.run(
            train_args,
            check=True, capture_output=True, text=True, encoding='utf-8'
        )
        logging.info(f"--- {train_script} Log (stdout) ---")
        if result_train.stdout: logging.info(result_train.stdout)
        logging.info(f"--- {train_script} Log (stderr) ---")
        if result_train.stderr: logging.info(result_train.stderr)
        logging.info("Model retraining process completed successfully.")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"--- FAILED to execute {train_script} ---")
        logging.error(e.stderr)
        return # Stop if training fails

    # 5. Automatically Trigger Prediction and Analysis
    logging.info("\n--- Phase 2: Triggering Prediction & Analysis ---")
    try:
        result_predict = subprocess.run(
            ['python', 'crypto_trading_app.py', '--no-train', '--symbol', symbol, '--interval', interval],
            check=True, capture_output=True, text=True, encoding='utf-8'
        )
        logging.info("--- Prediction & Analysis Log (stdout) ---")
        if result_predict.stdout: logging.info(result_predict.stdout)
        logging.info("--- Prediction & Analysis Log (stderr) ---")
        if result_predict.stderr: logging.info(result_predict.stderr)
        
        logging.info("Prediction and analysis completed successfully.")
        logging.info(f"Final results are available in: output/{symbol}_{interval}_trading_results.csv")

    except subprocess.CalledProcessError as e:
        logging.error("--- FAILED to execute prediction and analysis ---")
        logging.error(e.stderr)

    logging.info(f"\n--- Full Pipeline for {symbol} Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Pipeline: Adapt, Train, and Predict")
    parser.add_argument("--symbol", type=str, default="BNBUSDT", help="The crypto symbol to process.")
    parser.add_argument("--interval", type=str, default="5m", help="The data interval to use.")
    
    # Add mutually exclusive flags for controlling the training type
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument("--incremental-train", dest='incremental_train', action='store_true', help="Run incremental training (default).")
    train_group.add_argument("--no-incremental-train", dest='incremental_train', action='store_false', help="Run training from scratch.")

    parser.set_defaults(incremental_train=False) # Default behavior is NOT incremental training

    args = parser.parse_args()
    
    main(symbol=args.symbol, interval=args.interval, incremental_train=args.incremental_train)

    # First step: parser.set_defaults(incremental_train=False)
    # Second step: parser.set_defaults(incremental_train=True)