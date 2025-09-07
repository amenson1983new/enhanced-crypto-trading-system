import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class MultiTimeframeAnalyzer:
    """Analyze data across multiple timeframes and downscale predictions"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def aggregate_to_higher_timeframe(self, df, target_interval):
        """Aggregate 5m data to higher timeframes"""
        self.logger.info(f"Aggregating data to {target_interval}")
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'number_of_trades': 'sum',
            'quote_asset_volume': 'sum',
            'taker_buy_base_asset_volume': 'sum',
            'taker_buy_quote_asset_volume': 'sum'
        }
        
        # Convert interval to pandas frequency
        freq_map = {
            '15m': '15T',
            '1h': '1H', 
            '4h': '4H',
            '1d': '1D'
        }
        
        if target_interval not in freq_map:
            raise ValueError(f"Unsupported interval: {target_interval}")
            
        # Resample data
        df_copy = df.copy()
        df_copy.set_index('open_time', inplace=True)
        
        resampled = df_copy.resample(freq_map[target_interval]).agg(agg_rules)
        resampled = resampled.dropna()
        
        # Reset index and add interval column
        resampled.reset_index(inplace=True)
        resampled['interval'] = target_interval
        
        # --- FIX: Ensure the aggregated data is sorted by time before returning ---
        resampled = resampled.sort_values('open_time').reset_index(drop=True)
        
        self.logger.info(f"Aggregated to {target_interval}: {resampled.shape}")
        return resampled
    
    def downscale_predictions(self, higher_tf_predictions, target_tf_data):
        """Downscale predictions from higher timeframe to lower timeframe"""
        self.logger.info("Downscaling predictions to lower timeframe")
        
        # Merge predictions with target timeframe data
        # This is a simplified approach - you might want more sophisticated alignment
        
        result = target_tf_data.copy()
        
        # Forward fill higher timeframe predictions
        higher_tf_predictions['prediction_start'] = higher_tf_predictions['open_time']
        
        # Create prediction intervals based on higher timeframe
        for i, pred_row in higher_tf_predictions.iterrows():
            start_time = pred_row['open_time']
            
            # Find corresponding rows in target timeframe
            mask = (result['open_time'] >= start_time) & \
                   (result['open_time'] < start_time + pd.Timedelta(minutes=15))  # Assuming 15m higher TF
            
            if mask.any():
                result.loc[mask, 'higher_tf_signal'] = pred_row['predicted_signal']
                result.loc[mask, 'higher_tf_confidence'] = pred_row['confidence']
        
        self.logger.info("Downscaling completed")
        return result
    
    def combine_multi_timeframe_signals(self, signals_dict):
        """Combine signals from multiple timeframes"""
        self.logger.info("Combining multi-timeframe signals")
        
        # Weight different timeframes
        weights = {
            '5m': 0.4,
            '15m': 0.3, 
            '1h': 0.2,
            '4h': 0.1
        }
        
        # This is a placeholder - implement your combination logic
        # You might want to use voting, weighted averaging, or more sophisticated methods
        
        combined_signals = signals_dict['5m'].copy()  # Start with base timeframe
        
        # Add higher timeframe signals as features
        for tf, signals in signals_dict.items():
            if tf != '5m':
                combined_signals[f'{tf}_signal'] = signals.get('predicted_signal', 'SKIP')
                combined_signals[f'{tf}_confidence'] = signals.get('confidence', 0.5)
        
        return combined_signals