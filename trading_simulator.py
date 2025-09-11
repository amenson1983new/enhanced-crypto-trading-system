"""
Trading Simulator for Enhanced Crypto Trading System
Handles trade simulation and backtesting with risk management.
"""
import logging
import pandas as pd
import numpy as np


class TradingSimulator:
    def __init__(self, config):
        """Initialize TradingSimulator with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def run_simulation(self, df):
        """Run trading simulation on processed data with predictions."""
        self.logger.info("Starting trading simulation...")
        
        results = df.copy()
        results['trade_outcome'] = 'NONE'
        results['profit_pct'] = 0.0
        results['exit_price'] = np.nan
        results['stop_loss_price'] = np.nan
        results['take_profit_price'] = np.nan
        
        # Get risk management configuration
        risk_config = self.config.get('risk_management', {})
        stop_loss_atr_mult = risk_config.get('stop_loss_atr_multiplier', 1.2)
        take_profit_atr_mult = risk_config.get('take_profit_atr_multiplier', 5.0)
        trailing_enabled = risk_config.get('trailing_stop_enabled', True)
        trailing_atr_mult = risk_config.get('trailing_atr_multiplier', 1.2)
        
        # Simulate trades based on predictions
        position = None
        entry_price = None
        entry_idx = None
        stop_loss = None
        take_profit = None
        trailing_stop = None
        
        for i, row in results.iterrows():
            current_price = row['close']
            atr_value = row.get('atr', 0.01)  # Default ATR if not available
            
            # Close existing position if needed
            if position is not None:
                exit_trade = False
                exit_reason = 'NONE'
                exit_price = current_price
                
                # Check stop loss
                if position == 'LONG' and current_price <= stop_loss:
                    exit_trade = True
                    exit_reason = 'STOP_LOSS'
                    exit_price = stop_loss
                elif position == 'SHORT' and current_price >= stop_loss:
                    exit_trade = True
                    exit_reason = 'STOP_LOSS'
                    exit_price = stop_loss
                
                # Check take profit
                if not exit_trade:
                    if position == 'LONG' and current_price >= take_profit:
                        exit_trade = True
                        exit_reason = 'TAKE_PROFIT'
                        exit_price = take_profit
                    elif position == 'SHORT' and current_price <= take_profit:
                        exit_trade = True
                        exit_reason = 'TAKE_PROFIT'
                        exit_price = take_profit
                
                # Update trailing stop
                if trailing_enabled and not exit_trade:
                    if position == 'LONG':
                        new_trailing = current_price - (trailing_atr_mult * atr_value)
                        if trailing_stop is None or new_trailing > trailing_stop:
                            trailing_stop = new_trailing
                        if current_price <= trailing_stop:
                            exit_trade = True
                            exit_reason = 'TRAILING_STOP'
                            exit_price = trailing_stop
                    elif position == 'SHORT':
                        new_trailing = current_price + (trailing_atr_mult * atr_value)
                        if trailing_stop is None or new_trailing < trailing_stop:
                            trailing_stop = new_trailing
                        if current_price >= trailing_stop:
                            exit_trade = True
                            exit_reason = 'TRAILING_STOP'
                            exit_price = trailing_stop
                
                # Exit trade if conditions met
                if exit_trade:
                    if position == 'LONG':
                        profit_pct = (exit_price - entry_price) / entry_price
                    else:  # SHORT
                        profit_pct = (entry_price - exit_price) / entry_price
                    
                    # Update results
                    results.loc[entry_idx, 'trade_outcome'] = exit_reason
                    results.loc[entry_idx, 'profit_pct'] = profit_pct
                    results.loc[entry_idx, 'exit_price'] = exit_price
                    
                    # Reset position
                    position = None
                    entry_price = None
                    entry_idx = None
                    stop_loss = None
                    take_profit = None
                    trailing_stop = None
            
            # Open new position based on prediction
            if position is None and 'prediction' in row:
                signal = row.get('prediction', 'SKIP')
                
                if signal == 0:  # BUY signal (from numeric mapping)
                    position = 'LONG'
                    entry_price = current_price
                    entry_idx = i
                    stop_loss = entry_price - (stop_loss_atr_mult * atr_value)
                    take_profit = entry_price + (take_profit_atr_mult * atr_value)
                    trailing_stop = None
                    
                    results.loc[i, 'stop_loss_price'] = stop_loss
                    results.loc[i, 'take_profit_price'] = take_profit
                    
                elif signal == 1:  # SELL signal (from numeric mapping)
                    position = 'SHORT'
                    entry_price = current_price
                    entry_idx = i
                    stop_loss = entry_price + (stop_loss_atr_mult * atr_value)
                    take_profit = entry_price - (take_profit_atr_mult * atr_value)
                    trailing_stop = None
                    
                    results.loc[i, 'stop_loss_price'] = stop_loss
                    results.loc[i, 'take_profit_price'] = take_profit
        
        # Close any remaining open position at the end
        if position is not None:
            final_price = results.iloc[-1]['close']
            if position == 'LONG':
                profit_pct = (final_price - entry_price) / entry_price
            else:  # SHORT
                profit_pct = (entry_price - final_price) / entry_price
            
            results.loc[entry_idx, 'trade_outcome'] = 'END_OF_DATA'
            results.loc[entry_idx, 'profit_pct'] = profit_pct
            results.loc[entry_idx, 'exit_price'] = final_price
        
        # Calculate summary statistics
        trades = results[results['trade_outcome'] != 'NONE']
        if len(trades) > 0:
            total_return = trades['profit_pct'].sum()
            win_rate = len(trades[trades['profit_pct'] > 0]) / len(trades)
            avg_profit = trades['profit_pct'].mean()
            
            self.logger.info(f"Simulation completed:")
            self.logger.info(f"Total trades: {len(trades)}")
            self.logger.info(f"Win rate: {win_rate:.2%}")
            self.logger.info(f"Total return: {total_return:.2%}")
            self.logger.info(f"Average profit per trade: {avg_profit:.2%}")
        else:
            self.logger.info("No trades executed in simulation")
        
        return results