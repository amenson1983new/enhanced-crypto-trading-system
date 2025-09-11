import json
import logging
import argparse
import os
from sklearn.model_selection import train_test_split
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from trading_simulator import TradingSimulator

def setup_logging(config):
    log_path = config['logging']['path']
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(level=config['logging']['level'], format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

def load_config(symbol):
    try:
        with open(f'config/{symbol}/config.json', 'r') as f: config = json.load(f)
        with open(f'config/{symbol}/strategies.json', 'r') as f: strategies = json.load(f)
        return config, strategies
    except FileNotFoundError:
        logging.error(f"Config files for symbol '{symbol}' not found.")
        return None, None

def main(symbol, interval, run_training, optimize, incremental_train):
    """Main function to run the trading pipeline."""
    config, strategies = load_config(symbol)
    if not config:
        return

    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting process for symbol: {symbol} on {interval} interval.")
    
    data_processor = DataProcessor(config, strategies)
    model_trainer = ModelTrainer(config)

    if run_training:
        logger.info("--- Starting Training Pipeline ---")
        X, y = data_processor.run_for_training()
        
        if X is None or y is None:
            logger.error("Failed to process data for training. Exiting.")
            return
        
        # Always split data to get a validation set
        logger.info("Splitting data for training and validation...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logger.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

        best_params = None
        if optimize:
            logger.info("--- Running Hyperparameter Optimization ---")
            best_params = model_trainer.tune_hyperparameters(X_train, y_train, X_val, y_val)
            
            logger.info("Retraining model with best parameters...")
            model_trainer.train(X_train, y_train, params=best_params, X_val=X_val, y_val=y_val)
        else:
            logger.info("--- Starting Model Training ---")
            model_trainer.train(X_train, y_train, incremental=incremental_train, X_val=X_val, y_val=y_val)
    else:
        logger.info("--- Starting Prediction Pipeline ---")
        
        predict_df = data_processor.load_data(config['data']['predict_path'])
        if predict_df.empty:
            logger.error("Prediction data not found or empty. Exiting.")
            return

        processed_df, X_predict = data_processor.run_for_prediction(predict_df)
        
        model_trainer.load_model()
        if model_trainer.model:
            booster = model_trainer.model.get_booster()
            if booster:
                model_cols = booster.feature_names
                X_predict = X_predict.reindex(columns=model_cols, fill_value=0)
            else:
                logger.error("Model booster not found. Cannot align features.")
                return
        else:
            logger.error("Model is not loaded. Cannot align features. Prediction might be inaccurate.")
            return

        predictions, probabilities = model_trainer.predict(X_predict)
        
        if predictions is not None:
            prediction_start_index = len(processed_df) - len(predict_df)

            final_df = processed_df.iloc[prediction_start_index:].copy()
            final_df['prediction'] = predictions[prediction_start_index:]
            final_df['prob_buy'] = probabilities[prediction_start_index:, 0]
            final_df['prob_sell'] = probabilities[prediction_start_index:, 1]
            final_df['prob_skip'] = probabilities[prediction_start_index:, 2]

            simulator = TradingSimulator(config)
            results_df = simulator.run_simulation(final_df)

            output_path = config['data']['output_path']
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Prediction results saved to {output_path}")
        else:
            logger.error("Prediction failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Trading Pipeline")
    parser.add_argument("--symbol", type=str, default="BNBUSDT", help="The crypto symbol to process.")
    parser.add_argument("--interval", type=str, default="5m", help="The data interval to use.")
    
    parser.add_argument("--train", dest='run_training', action='store_true', help="Run the training phase.")
    parser.add_argument("--predict", dest='run_training', action='store_false', help="Run the prediction phase (default).")
    
    parser.add_argument("--optimize", action='store_true', help="Run hyperparameter optimization before training.")
    
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument("--incremental", dest='incremental_train', action='store_true', help="Run incremental training.")
    train_group.add_argument("--from-scratch", dest='incremental_train', action='store_false', help="Run training from scratch (default).")
    
    parser.set_defaults(run_training=False, incremental_train=False, optimize=False)

    args = parser.parse_args()
    
    if args.optimize:
        args.run_training = True

    main(
        symbol=args.symbol, 
        interval=args.interval, 
        run_training=args.run_training, 
        optimize=args.optimize,
        incremental_train=args.incremental_train
    )