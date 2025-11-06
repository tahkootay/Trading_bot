#!/usr/bin/env python3
"""
Fixed Real LSTM Backtesting for July, August, September 2025
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set JAX backend for Keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

class FixedLSTMBacktester:
    def __init__(self):
        self.window_size = 15  # Quick LSTM uses 15
        self.commission = 0.001
        self.initial_capital = 10000
        self.position_size = 0.1
        
        # Features used in quick LSTM
        self.features = ['rsi_14', 'ema_20', 'ema_50', 'macd_line', 'atr_14']
        
        # Load artifacts
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load quick LSTM model"""
        print("Loading Quick LSTM artifacts...")
        
        try:
            self.model = keras.models.load_model("models/quick_real_lstm.keras")
            print("✅ Quick Real LSTM model loaded")
            
            # Create new scaler for this backtest
            self.scaler = StandardScaler()
            print("✅ New scaler created")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def create_sequences(self, data):
        """Create sequences for LSTM prediction"""
        if len(data) < self.window_size:
            return np.array([])
        
        X = []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
        
        return np.array(X)
    
    def filter_monthly_data(self, df, year, month):
        """Filter data for specific month"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        mask = (df['timestamp'].dt.year == year) & (df['timestamp'].dt.month == month)
        return df[mask].copy()
    
    def prepare_data(self, file_path, year, month, month_name):
        """Prepare data for specific month"""
        print(f"\nPreparing {month_name} {year} data...")
        
        df = pd.read_csv(file_path)
        print(f"Total data loaded: {len(df)} records")
        
        # Filter for specific month
        df_month = self.filter_monthly_data(df, year, month)
        print(f"{month_name} data: {len(df_month)} records")
        
        if len(df_month) < 200:
            print(f"❌ Insufficient data for {month_name} ({len(df_month)} records)")
            return None
        
        # Create target for evaluation (3 bars ahead)
        df_month["target"] = (df_month["close"].shift(-3) > df_month["close"]).astype(int)
        
        # Check available features
        available_features = [f for f in self.features if f in df_month.columns]
        missing_features = [f for f in self.features if f not in df_month.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        if len(available_features) < 3:
            print(f"❌ Too few features available: {available_features}")
            return None
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Clean data
        required_cols = ['timestamp', 'close', 'target'] + available_features
        df_clean = df_month[required_cols].dropna()
        
        print(f"Clean data: {len(df_clean)} rows")
        
        if len(df_clean) < 100:
            print(f"❌ Insufficient clean data: {len(df_clean)} rows")
            return None
        
        return df_clean
    
    def generate_signals(self, df):
        """Generate trading signals using LSTM"""
        print("Generating LSTM signals...")
        
        if len(df) < self.window_size + 10:
            print(f"❌ Not enough data for sequences (need >{self.window_size + 10})")
            return df
        
        # Get available features for this dataset
        available_features = [f for f in self.features if f in df.columns]
        
        # Prepare feature data
        feature_data = df[available_features].values
        
        # Scale features
        feature_scaled = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X_sequences = self.create_sequences(feature_scaled)
        
        if len(X_sequences) == 0:
            print("❌ Could not create sequences")
            return df
        
        print(f"Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
        
        # Get predictions
        try:
            predictions_prob = self.model.predict(X_sequences, verbose=0)
            predictions = (predictions_prob > 0.5).astype(int).flatten()
            probabilities = predictions_prob.flatten()
            
            print(f"Generated {len(predictions)} predictions")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return df
        
        # Add signals to dataframe
        df_signals = df.copy()
        df_signals['signal'] = 0
        df_signals['prob_up'] = 0.5
        df_signals['confidence'] = 0.0
        
        # Map predictions to dataframe rows
        start_idx = self.window_size
        end_idx = start_idx + len(predictions)
        
        if end_idx <= len(df_signals):
            df_signals.iloc[start_idx:end_idx, df_signals.columns.get_loc('signal')] = predictions
            df_signals.iloc[start_idx:end_idx, df_signals.columns.get_loc('prob_up')] = probabilities
            df_signals.iloc[start_idx:end_idx, df_signals.columns.get_loc('confidence')] = np.abs(probabilities - 0.5) * 2
        
        return df_signals
    
    def simulate_trading(self, df_signals, month_name):
        """Simulate trading with LSTM signals"""
        print(f"Simulating trading for {month_name}...")
        
        total_pnl = 0
        trades = 0
        winning_trades = 0
        
        # Use confidence threshold
        confidences = df_signals['confidence'].values
        valid_confidences = confidences[confidences > 0]
        
        if len(valid_confidences) == 0:
            print("❌ No valid predictions for trading")
            return {
                'period': month_name,
                'total_return': 0,
                'total_pnl': 0,
                'trades': 0,
                'winning_trades': 0,
                'win_rate': 0
            }
        
        threshold = np.percentile(valid_confidences, 60)  # Top 40% confidence
        print(f"Trading threshold: {threshold:.3f}")
        
        for i in range(len(df_signals) - 3):
            row = df_signals.iloc[i]
            
            confidence = row['confidence']
            if confidence < threshold:
                continue
            
            entry_price = row['close']
            exit_price = df_signals.iloc[i + 3]['close']
            prob_up = row['prob_up']
            
            # Calculate P&L
            if prob_up > 0.5:  # Long
                pnl = (exit_price - entry_price) / entry_price * self.initial_capital * self.position_size
            else:  # Short
                pnl = (entry_price - exit_price) / entry_price * self.initial_capital * self.position_size
            
            # Subtract commission
            pnl -= self.initial_capital * self.position_size * self.commission * 2
            
            total_pnl += pnl
            trades += 1
            
            if pnl > 0:
                winning_trades += 1
        
        win_rate = winning_trades / trades if trades > 0 else 0
        total_return = total_pnl / self.initial_capital if self.initial_capital > 0 else 0
        
        return {
            'period': month_name,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'trades': trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate
        }
    
    def evaluate_predictions(self, df_signals, month_name):
        """Evaluate prediction accuracy"""
        print(f"Evaluating predictions for {month_name}...")
        
        # Get predictions that can be evaluated
        eval_df = df_signals[:-3].copy()
        has_predictions = eval_df['confidence'] > 0
        eval_subset = eval_df[has_predictions]
        
        if len(eval_subset) == 0:
            print("⚠️ No predictions to evaluate")
            return None
        
        y_true = eval_subset['target'].values
        y_pred = eval_subset['signal'].values
        
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'period': month_name,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'total_predictions': len(y_true)
        }
    
    def run_monthly_backtest(self, file_path, year, month, month_name):
        """Run backtest for specific month"""
        print(f"\n{'='*60}")
        print(f"Quick LSTM Backtest: {month_name} {year}")
        print(f"{'='*60}")
        
        # Prepare data
        df_clean = self.prepare_data(file_path, year, month, month_name)
        if df_clean is None:
            return None
        
        # Generate signals
        df_signals = self.generate_signals(df_clean)
        
        # Evaluate predictions
        pred_results = self.evaluate_predictions(df_signals, month_name)
        
        # Simulate trading
        trading_results = self.simulate_trading(df_signals, month_name)
        
        return {
            'prediction_results': pred_results,
            'trading_results': trading_results
        }

def main():
    print("=== Fixed LSTM Backtesting: July, August, September 2025 ===")
    print(f"Backend: {keras.config.backend()}")
    
    # Initialize backtester
    backtester = FixedLSTMBacktester()
    
    # Data file path
    data_path = "/Users/alexey/Documents/Development/Python/Trading_bot/data/raw/SOLUSDT_5m_20250101_20251031_advanced_indicators.csv"
    
    # Test periods
    periods = [
        (2025, 7, "July"),
        (2025, 8, "August"), 
        (2025, 9, "September")
    ]
    
    all_results = {}
    
    # Run backtests
    for year, month, month_name in periods:
        try:
            results = backtester.run_monthly_backtest(data_path, year, month, month_name)
            if results:
                all_results[month_name] = results
                
                # Print quick summary
                trading = results['trading_results']
                pred = results['prediction_results']
                
                print(f"\n📊 {month_name} {year} Results:")
                print(f"   Return: {trading['total_return']*100:+.2f}%")
                print(f"   P&L: ${trading['total_pnl']:+.2f}")
                print(f"   Trades: {trading['trades']}")
                print(f"   Win Rate: {trading['win_rate']*100:.1f}%")
                if pred:
                    print(f"   Accuracy: {pred['accuracy']*100:.1f}%")
                
        except Exception as e:
            print(f"❌ Error in {month_name}: {e}")
    
    # Generate summary
    if all_results:
        print(f"\n{'='*80}")
        print("QUICK LSTM BACKTEST SUMMARY - JULY, AUGUST, SEPTEMBER 2025")
        print(f"{'='*80}")
        print(f"{'Period':<12} {'Return %':<10} {'P&L $':<12} {'Trades':<8} {'Win Rate %':<12} {'Accuracy %':<12}")
        print("-" * 80)
        
        total_return = 0
        total_pnl = 0
        total_trades = 0
        total_accuracy = 0
        valid_periods = 0
        
        for period, results in all_results.items():
            trading = results['trading_results']
            pred = results['prediction_results']
            
            accuracy = pred['accuracy'] * 100 if pred else 0
            
            print(f"{period:<12} {trading['total_return']*100:>8.1f}% "
                  f"${trading['total_pnl']:>10.2f} {trading['trades']:>6} "
                  f"{trading['win_rate']*100:>10.1f}% {accuracy:>10.1f}%")
            
            total_return += trading['total_return']
            total_pnl += trading['total_pnl']
            total_trades += trading['trades']
            if pred:
                total_accuracy += pred['accuracy']
                valid_periods += 1
        
        avg_return = total_return / len(all_results) * 100 if all_results else 0
        avg_accuracy = total_accuracy / valid_periods * 100 if valid_periods > 0 else 0
        
        print("-" * 80)
        print(f"{'AVERAGE':<12} {avg_return:>8.1f}% "
              f"${total_pnl:>10.2f} {total_trades:>6} {'':>10} {avg_accuracy:>10.1f}%")
        
        # Compare with previous results
        print(f"\n{'='*60}")
        print("COMPARISON WITH PREVIOUS BACKTESTS")
        print(f"{'='*60}")
        print(f"Previous Quick LSTM: -10.20% return, 51.8% accuracy")
        print(f"Current (Jul-Sep): {avg_return:+.2f}% return, {avg_accuracy:.1f}% accuracy")
        
        improvement_return = avg_return + 10.20
        improvement_accuracy = avg_accuracy - 51.8
        
        if improvement_return > 0:
            print(f"✅ Return improvement: {improvement_return:+.2f}%")
        else:
            print(f"⚠️ Return decline: {improvement_return:+.2f}%")
            
        if improvement_accuracy > 0:
            print(f"✅ Accuracy improvement: {improvement_accuracy:+.1f}%")
        else:
            print(f"⚠️ Accuracy decline: {improvement_accuracy:+.1f}%")
        
        # Save results
        os.makedirs('reports', exist_ok=True)
        
        report_content = f"""Quick Real LSTM Backtesting Results - July, August, September 2025
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Backend: {keras.config.backend()}
Model: Quick Real LSTM (window=15, features=5)

=== SUMMARY ===
Average Return: {avg_return:.2f}%
Total P&L: ${total_pnl:.2f}
Total Trades: {total_trades}
Average Accuracy: {avg_accuracy:.1f}%

=== COMPARISON ===
Previous Backtest (Reference): -10.20% return, 51.8% accuracy
Current Backtest (Jul-Sep): {avg_return:+.2f}% return, {avg_accuracy:.1f}% accuracy
Performance Change: {improvement_return:+.2f}% return, {improvement_accuracy:+.1f}% accuracy

=== DETAILED RESULTS ===
"""
        
        for period, results in all_results.items():
            trading = results['trading_results']
            pred = results['prediction_results']
            
            report_content += f"""
{period} 2025:
  Return: {trading['total_return']*100:.2f}%
  P&L: ${trading['total_pnl']:.2f}
  Trades: {trading['trades']}
  Win Rate: {trading['win_rate']*100:.1f}%"""
            
            if pred:
                report_content += f"""
  Accuracy: {pred['accuracy']*100:.1f}%
  Total Predictions: {pred['total_predictions']}
  Confusion Matrix:
{pred['confusion_matrix']}"""
        
        # Assessment
        is_improved = improvement_return > 0 and improvement_accuracy > 0
        is_mixed = improvement_return > 0 or improvement_accuracy > 0
        
        report_content += f"""

=== ASSESSMENT ===
Model Performance: {'✅ IMPROVED' if is_improved else '⚠️ MIXED RESULTS' if is_mixed else '❌ DECLINED'}
Profitability: {'❌ STILL UNPROFITABLE' if avg_return < 0 else '✅ PROFITABLE'}
Accuracy Level: {'✅ ABOVE RANDOM' if avg_accuracy > 52 else '⚠️ NEAR RANDOM' if avg_accuracy > 48 else '❌ BELOW RANDOM'}

Key Insights:
1. {'Model shows promise with better temporal patterns' if avg_accuracy > 52 else 'Model needs architecture improvements'}
2. {'Risk management needed to achieve profitability' if avg_return > -5 else 'Fundamental strategy revision required'}
3. {'Consistent across periods' if len(all_results) == 3 else 'Inconsistent performance across periods'}

Next Steps:
- {'Test full LSTM architecture' if avg_accuracy > 52 else 'Improve feature engineering'}
- {'Implement dynamic position sizing' if avg_return > -10 else 'Reconsider market selection'}
- {'Add portfolio management layer' if total_trades > 100 else 'Increase signal frequency'}
"""
        
        report_path = 'reports/lstm_jul_aug_sep_2025_backtest.txt'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"\n✅ Backtesting completed!")
        print(f"Report saved to: {report_path}")
        
    else:
        print("❌ No successful backtests completed")

if __name__ == "__main__":
    main()