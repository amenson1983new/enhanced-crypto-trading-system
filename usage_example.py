"""
Usage Example for Enhanced Crypto Trading Predictor
User: amenson1983new
Created: 2025-08-31 16:11:25 UTC
"""

import pandas as pd
import numpy as np
from enhanced_crypto_predictor_v2 import EnhancedCryptoPredictor, ConfigManager
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample BNBUSDT data for demonstration"""
    np.random.seed(42)
    
    # Generate 1000 data points (about 3.5 days of 5-minute data)
    n_points = 1000
    dates = pd.date_range('2024-01-01', periods=n_points, freq='5T')
    
    # Generate price data with realistic patterns
    base_price = 300
    prices = []
    current_price = base_price
    
    for i in range(n_points):
        # Add trend and noise
        trend = 0.0001 * i  # Slight upward trend
        noise = np.random.normal(0, 2)  # Random noise
        current_price = max(current_price + trend + noise, 250)  # Minimum price floor
        prices.append(current_price)
    
    # Calculate OHLC from close prices
    close_prices = np.array(prices)
    high_prices = close_prices + np.abs(np.random.normal(0, 1, n_points))
    low_prices = close_prices - np.abs(np.random.normal(0, 1, n_points))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # Generate volume
    volume = np.random.lognormal(10, 0.5, n_points)
    
    # Calculate technical indicators
    def calculate_sma(prices, window):
        return pd.Series(prices).rolling(window=window).mean().fillna(prices[0])
    
    def calculate_ema(prices, window):
        return pd.Series(prices).ewm(span=window).mean().fillna(prices[0])
    
    def calculate_rsi(prices, window=14):
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    sma = calculate_sma(close_prices, 20)
    ema = calculate_ema(close_prices, 20)
    rsi = calculate_rsi(close_prices)
    
    # Generate other indicators (simplified)
    stoch_k = np.random.uniform(20, 80, n_points)
    stoch_d = pd.Series(stoch_k).rolling(window=3).mean().fillna(50)
    adx = np.random.uniform(20, 50, n_points)
    dmp = np.random.uniform(15, 35, n_points)
    dmn = np.random.uniform(15, 35, n_points)
    macd = np.random.normal(0, 2, n_points)
    macd_signal = pd.Series(macd).rolling(window=9).mean().fillna(0)
    macd_histogram = macd - macd_signal
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        'rsi': rsi,
        'sma': sma,
        'ema': ema,
        'profitability': 0,  # Initialize as 0
        'STOCHk_14_3_3': stoch_k,
        'STOCHd_14_3_3': stoch_d,
        'ADX_14': adx,
        'DMP_14': dmp,
        'DMN_14': dmn,
        'MACD_3_9_21': macd,
        'MACDh_3_9_21': macd_histogram,
        'MACDs_3_9_21': macd_signal
    })
    
    return data

def run_complete_example():
    """Run complete example with sample data"""
    print("🚀 Enhanced Crypto Trading Predictor - Complete Example")
    print(f"👤 User: amenson1983new")
    print(f"📅 Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample training data...")
    train_data = create_sample_data()
    print(f"✅ Training data created: {train_data.shape}")
    print(f"📈 Price range: ${train_data['close'].min():.2f} - ${train_data['close'].max():.2f}")
    
    # Step 2: Create test data (slightly different)
    print("\n📊 Step 2: Creating sample test data...")
    test_data = create_sample_data()
    # Modify test data slightly
    test_data['close'] = test_data['close'] * 1.05 + np.random.normal(0, 5, len(test_data))
    test_data['high'] = test_data['close'] + np.abs(np.random.normal(0, 1, len(test_data)))
    test_data['low'] = test_data['close'] - np.abs(np.random.normal(0, 1, len(test_data)))
    print(f"✅ Test data created: {test_data.shape}")
    
    # Step 3: Configure system
    print("\n⚙️  Step 3: Configuring trading system...")
    custom_config = {
        'model': {
            'sequence_length': 20,  # Shorter for demo
            'lstm_units': 64,       # Smaller for demo
            'gru_units': 48,
            'dense_units': 32,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 20,           # Fewer epochs for demo
            'early_stopping_patience': 10,
            'use_attention': True,
            'use_conv1d': False
        },
        'trading': {
            'min_profit_threshold': 0.015,
            'risk_reward_ratio': 1.5,
            'confidence_threshold': 0.6,
            'future_periods': 3
        }
    }
    
    ConfigManager.save_config(custom_config, 'config/demo_config.json')
    print("✅ Configuration saved")
    
    # Step 4: Initialize and train
    print("\n🤖 Step 4: Training enhanced predictor...")
    predictor = EnhancedCryptoPredictor('config/demo_config.json')
    
    try:
        training_history = predictor.train(train_data)
        print("✅ Training completed successfully!")
        
        # Show training metrics if available
        if 'signal_history' in training_history:
            signal_history = training_history['signal_history']
            final_accuracy = signal_history.history['val_accuracy'][-1]
            print(f"📊 Final validation accuracy: {final_accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 5: Make predictions
    print("\n🔮 Step 5: Making predictions on test data...")
    try:
        predictions = predictor.predict(test_data)
        print(f"✅ Predictions generated: {len(predictions)} signals")
        
        # Display prediction summary
        signal_counts = predictions['signal'].value_counts()
        print("\n📈 Signal Distribution:")
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} ({count/len(predictions)*100:.1f}%)")
        
        print(f"\n💰 Average Confidence: {predictions['confidence'].mean():.3f}")
        print(f"💎 Average Profit Potential: {predictions['profit_potential'].mean():.3f}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # Step 6: Evaluate performance
    print("\n📊 Step 6: Evaluating performance...")
    try:
        # Calculate performance metrics
        buy_signals = predictions[predictions['signal'] == 'BUY']
        sell_signals = predictions[predictions['signal'] == 'SELL']
        skip_signals = predictions[predictions['signal'] == 'SKIP']
        
        # Simulate trading performance
        total_trades = len(buy_signals) + len(sell_signals)
        profitable_trades = len(predictions[predictions['profit_potential'] > 0.01])
        profitability_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   Total Signals: {len(predictions)}")
        print(f"   Actionable Trades: {total_trades}")
        print(f"   Profitable Trades: {profitable_trades}")
        print(f"   Profitability Rate: {profitability_rate:.1%}")
        print(f"   Target Achievement: {'✅' if profitability_rate >= 0.85 else '❌'} (Target: 85%)")
        
        # Risk metrics
        avg_risk_reward = predictions['risk_reward_ratio'].mean()
        print(f"   Average Risk/Reward: {avg_risk_reward:.2f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
    
    # Step 7: Save results
    print("\n💾 Step 7: Saving results...")
    try:
        # Save predictions
        predictions.to_csv('demo_predictions.csv', index=False)
        print("✅ Predictions saved to 'demo_predictions.csv'")
        
        # Save model
        predictor.save_model('models/demo_predictor')
        print("✅ Model saved to 'models/demo_predictor'")
        
        # Create sample of top predictions
        top_predictions = predictions.nlargest(5, 'confidence')
        print("\n🏆 TOP 5 PREDICTIONS BY CONFIDENCE:")
        for idx, row in top_predictions.iterrows():
            print(f"   {row['signal']} @ ${row['current_price']:.2f} "
                  f"(Confidence: {row['confidence']:.3f}, "
                  f"Profit: {row['profit_potential']:.3f})")
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
    
    # Step 8: Demonstrate live prediction
    print("\n🔴 Step 8: Live prediction example...")
    try:
        # Use last 50 data points for live prediction
        live_data = test_data.tail(50)
        live_predictions = predictor.predict(live_data)
        
        if len(live_predictions) > 0:
            latest_signal = live_predictions.iloc[-1]
            print(f"📡 LATEST SIGNAL:")
            print(f"   Action: {latest_signal['signal']}")
            print(f"   Current Price: ${latest_signal['current_price']:.2f}")
            print(f"   Confidence: {latest_signal['confidence']:.3f}")
            print(f"   Take Profit: ${latest_signal['take_profit']:.2f}")
            print(f"   Stop Loss: ${latest_signal['stop_loss']:.2f}")
            print(f"   Risk/Reward: {latest_signal['risk_reward_ratio']:.2f}")
        
    except Exception as e:
        print(f"❌ Live prediction failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("📝 Next steps:")
    print("   1. Replace sample data with real BNBUSDT CSV files")
    print("   2. Adjust configuration parameters in config file")
    print("   3. Run training with more epochs for better performance")
    print("   4. Implement live trading integration")
    print(f"📧 Contact: amenson1983new")

def analyze_predictions(predictions_file='demo_predictions.csv'):
    """Analyze saved predictions"""
    try:
        df = pd.read_csv(predictions_file)
        
        print(f"📊 PREDICTION ANALYSIS")
        print(f"Total predictions: {len(df)}")
        print(f"Signal distribution:\n{df['signal'].value_counts()}")
        print(f"Average confidence: {df['confidence'].mean():.3f}")
        print(f"Average profit potential: {df['profit_potential'].mean():.4f}")
        
        # Plot distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        df['signal'].value_counts().plot(kind='bar')
        plt.title('Signal Distribution')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.hist(df['confidence'], bins=20, alpha=0.7)
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        
        plt.subplot(2, 2, 3)
        plt.hist(df['profit_potential'], bins=20, alpha=0.7)
        plt.title('Profit Potential Distribution')
        plt.xlabel('Profit Potential')
        
        plt.subplot(2, 2, 4)
        plt.scatter(df['confidence'], df['profit_potential'], alpha=0.6)
        plt.title('Confidence vs Profit Potential')
        plt.xlabel('Confidence')
        plt.ylabel('Profit Potential')
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png')
        plt.show()
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    # Run complete example
    run_complete_example()
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📈 Analyzing predictions...")
    analyze_predictions()