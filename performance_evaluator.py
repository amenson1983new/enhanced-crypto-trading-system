import pandas as pd
import numpy as np
from typing import Dict, Any

class PerformanceEvaluator:
    """
    Evaluates the performance of trading signals against historical data.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PerformanceEvaluator.

        Args:
            config (Dict[str, Any]): A dictionary containing trading parameters, 
                                     including risk management settings.
        """
        self.rm_config = config.get('risk_management', {})
        self.sig_config = config.get('signal_generation', {})
        
        # Risk management parameters
        self.stop_loss_mult = self.rm_config.get('stop_loss_atr_multiplier', 1.5)
        self.take_profit_mult = self.rm_config.get('take_profit_atr_multiplier', 3.0)
        self.trailing_stop_enabled = self.rm_config.get('trailing_stop_enabled', False)
        self.trailing_atr_mult = self.rm_config.get('trailing_atr_multiplier', 2.0)
        self.activation_profit_pct = self.rm_config.get('activation_profit_pct', 0.01)
        self.breakeven_period = self.rm_config.get('breakeven_period', None)

        # Signal generation parameters
        self.future_periods = self.sig_config.get('future_periods', 10)

    def evaluate_signals(self, signals_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates trading signals to determine trade outcomes and profit/loss.

        Args:
            signals_df (pd.DataFrame): DataFrame with trading signals ('signal', 'confidence', etc.).
            data_df (pd.DataFrame): DataFrame with historical market data (OHLCV, ATR).

        Returns:
            pd.DataFrame: A DataFrame containing the results of each trade.
        """
        results = []
        active_trade = None

        # Merge data to align signals with market data
        merged_df = pd.merge_asof(signals_df.sort_index(), data_df.sort_index(), 
                                  left_index=True, right_index=True, direction='forward')
        
        for i in range(len(merged_df)):
            row = merged_df.iloc[i]
            
            if active_trade:
                # Check for exit conditions
                exit_reason, exit_price, exit_time = self._check_exit_conditions(i, merged_df, active_trade)
                
                if exit_reason:
                    profit_pct = (exit_price - active_trade['entry_price']) / active_trade['entry_price']
                    if active_trade['trade_type'] == 'SELL':
                        profit_pct = -profit_pct
                    
                    active_trade['exit_price'] = exit_price
                    active_trade['exit_time'] = exit_time
                    active_trade['profit_pct'] = profit_pct
                    active_trade['trade_outcome'] = exit_reason
                    results.append(active_trade)
                    active_trade = None

            # Check for new trade entry
            if not active_trade and row['signal'] in ['BUY', 'SELL']:
                active_trade = self._enter_trade(row)

        return pd.DataFrame(results)

    def _enter_trade(self, row: pd.Series) -> Dict[str, Any]:
        """Initializes a new trade."""
        entry_price = row['close']
        atr_at_entry = row['ATR']
        
        if row['signal'] == 'BUY':
            stop_loss_price = entry_price - (self.stop_loss_mult * atr_at_entry)
            take_profit_price = entry_price + (self.take_profit_mult * atr_at_entry)
        else: # SELL
            stop_loss_price = entry_price + (self.stop_loss_mult * atr_at_entry)
            take_profit_price = entry_price - (self.take_profit_mult * atr_at_entry)

        return {
            'entry_time': row.name,
            'entry_price': entry_price,
            'trade_type': row['signal'],
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'trailing_stop_activated': False,
            'trailing_stop_price': None,
            'peak_price': entry_price
        }

    def _check_exit_conditions(self, current_index: int, df: pd.DataFrame, trade: Dict[str, Any]) -> tuple[str, float, pd.Timestamp]:
        """Checks all exit conditions for an active trade."""
        current_row = df.iloc[current_index]
        low_price, high_price = current_row['low'], current_row['high']
        
        # --- Update Trailing Stop ---
        if self.trailing_stop_enabled:
            trade = self._update_trailing_stop(high_price, low_price, trade)

        # --- Check Exit Triggers (Order of checks matters) ---
        # 1. Stop Loss
        if trade['trade_type'] == 'BUY' and low_price <= trade['stop_loss']:
            return 'STOP_LOSS', trade['stop_loss'], current_row.name
        if trade['trade_type'] == 'SELL' and high_price >= trade['stop_loss']:
            return 'STOP_LOSS', trade['stop_loss'], current_row.name

        # 2. Trailing Stop
        if trade['trailing_stop_activated']:
            if trade['trade_type'] == 'BUY' and low_price <= trade['trailing_stop_price']:
                return 'TRAILING_STOP', trade['trailing_stop_price'], current_row.name
            if trade['trade_type'] == 'SELL' and high_price >= trade['trailing_stop_price']:
                return 'TRAILING_STOP', trade['trailing_stop_price'], current_row.name
        
        # 3. Take Profit
        if trade['trade_type'] == 'BUY' and high_price >= trade['take_profit']:
            return 'TAKE_PROFIT', trade['take_profit'], current_row.name
        if trade['trade_type'] == 'SELL' and low_price <= trade['take_profit']:
            return 'TAKE_PROFIT', trade['take_profit'], current_row.name

        # 4. Breakeven Stop (NEW LOGIC)
        if self.breakeven_period is not None:
            periods_in_trade = current_index - df.index.get_loc(trade['entry_time'])
            if periods_in_trade >= self.breakeven_period:
                is_profitable = (trade['trade_type'] == 'BUY' and current_row['close'] > trade['entry_price']) or \
                                (trade['trade_type'] == 'SELL' and current_row['close'] < trade['entry_price'])
                if not is_profitable:
                    return 'BREAKEVEN_STOP', current_row['close'], current_row.name

        # 5. Timeout
        time_in_trade = (current_row.name - trade['entry_time']).total_seconds() / 60
        # This logic assumes 1 period = 1 minute, adjust if data frequency is different
        # A more robust way is to count periods directly.
        periods_in_trade = current_index - df.index.get_loc(trade['entry_time'])
        if periods_in_trade >= self.future_periods:
            return 'TIMEOUT', current_row['close'], current_row.name

        return None, None, None

    def _update_trailing_stop(self, high_price: float, low_price: float, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Updates the trailing stop price if conditions are met."""
        if trade['trade_type'] == 'BUY':
            current_profit_pct = (high_price - trade['entry_price']) / trade['entry_price']
            if not trade['trailing_stop_activated'] and current_profit_pct >= self.activation_profit_pct:
                trade['trailing_stop_activated'] = True

            if trade['trailing_stop_activated']:
                # Update peak price
                trade['peak_price'] = max(trade['peak_price'], high_price)
                # Update trailing stop price
                atr_val = 0 # Placeholder: Need ATR at current row
                try:
                    atr_val = self.rm_config.get('latest_atr', 0)
                except: # Fallback if not available
                    atr_val = (trade['peak_price'] - trade['stop_loss']) / self.stop_loss_mult
                
                new_trailing_stop = trade['peak_price'] - (self.trailing_atr_mult * atr_val)
                # Ensure trailing stop only moves up
                if trade['trailing_stop_price'] is None or new_trailing_stop > trade['trailing_stop_price']:
                    trade['trailing_stop_price'] = new_trailing_stop
                # Also, ensure the trailing stop doesn't move below the initial stop loss
                trade['trailing_stop_price'] = max(trade['trailing_stop_price'], trade['stop_loss'])

        else: # SELL
            current_profit_pct = (trade['entry_price'] - low_price) / trade['entry_price']
            if not trade['trailing_stop_activated'] and current_profit_pct >= self.activation_profit_pct:
                trade['trailing_stop_activated'] = True

            if trade['trailing_stop_activated']:
                trade['peak_price'] = min(trade['peak_price'], low_price)
                atr_val = self.rm_config.get('latest_atr', 0)
                try:
                    atr_val = self.rm_config.get('latest_atr', 0)
                except:
                    atr_val = (trade['stop_loss'] - trade['peak_price']) / self.stop_loss_mult
                
                new_trailing_stop = trade['peak_price'] + (self.trailing_atr_mult * atr_val)
                if trade['trailing_stop_price'] is None or new_trailing_stop < trade['trailing_stop_price']:
                    trade['trailing_stop_price'] = new_trailing_stop
                trade['trailing_stop_price'] = min(trade['trailing_stop_price'], trade['stop_loss'])
                
        return trade
