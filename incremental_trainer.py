import os
import json
import logging
import argparse
import pandas as pd
import glob
from pathlib import Path

from crypto_trading_app import CryptoTradingSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def find_latest_file(directory_path):
    """Finds the most recently modified CSV file in a directory."""
    if not os.path.isdir(directory_path):
        return None
    
    list_of_files = glob.glob(os.path.join(directory_path, '*.csv'))
    if not list_of_files:
        return None
        
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def main(symbol, interval="5m"):
    """
    Main function to run the incremental training for a given symbol.
    This now combines old and new data to prevent catastrophic forgetting.
    It automatically finds the latest data file in the incremental learning directory.
    """
    print(f"\n{'='*20} Running Smart Incremental Training for: {symbol.upper()} {'='*20}")
    
    config_path = f"config/{symbol}_config.json"
    if not os.path.exists(config_path):
        print(f"Error: Config file for {symbol} not found at {config_path}. Aborting.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    inc_learn_config = config.get('incremental_learning', {})
    new_data_dir_template = inc_learn_config.get('new_data_directory')
    
    if not new_data_dir_template:
        print("Error: `new_data_directory` not specified in the incremental_learning config. Aborting.")
        return
    
    new_data_dir = new_data_dir_template.format(symbol=symbol, interval=interval)
    
    print(f"Searching for latest data file in: {new_data_dir}")
    new_data_file = find_latest_file(new_data_dir)

    if not new_data_file:
        print(f"Error: No new data files found in '{new_data_dir}'. Aborting.")
        return
    
    print(f"Found latest data file to be: {new_data_file}")

    original_data_file = f"data/{symbol}_{interval}_assembley.csv"
    if not os.path.exists(original_data_file):
        print(f"Error: Original assembly data file not found at '{original_data_file}'. Aborting.")
        return

    try:
        trading_system = CryptoTradingSystem(symbol=symbol, interval=interval)

        print(f"Loading original data from {original_data_file}...")
        df_original = trading_system.load_data(original_data_file)
        
        print(f"Loading new data from {new_data_file}...")
        df_new = trading_system.load_data(new_data_file)
        
        print("Combining original and new data to retain model memory...")
        df_combined = pd.concat([df_original, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=['open_time'], keep='last', inplace=True)
        df_combined.sort_values('open_time', inplace=True)
        
        print(f"Combined data shape: {df_combined.shape}")

        print("Generating signals and features for combined data...")
        df_features = trading_system.calculate_indicators(df_combined)
        df_with_signals = trading_system.generate_signals(df_features)

        print("Retraining model on the full, updated dataset...")
        trading_system.train_model(df_with_signals)
        
        print(f"\nSmart incremental training for {symbol} completed successfully!")
        print(f"Model at models/{symbol}_{interval}_trading_model.pkl is now updated with new data while retaining old knowledge.")

    except Exception as e:
        print(f"\nAn error occurred during incremental training for {symbol}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Incremental Model Trainer")
    parser.add_argument("--symbol", type=str, default="BNBUSDT", help="The crypto symbol to train (e.g., BNBUSDT).")
    parser.add_argument("--interval", type=str, default="5m", help="The interval to use (e.g., 5m).")
    args = parser.parse_args()
    
    main(symbol=args.symbol, interval=args.interval)