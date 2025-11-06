#!/usr/bin/env python3
"""
Real LSTM Backtesting for July, August, September 2025
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class RealLSTMBacktester:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.commission = 0.001
        self.initial_capital = 10000
        self.position_size = 0.1
        
        # Load artifacts
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load trained model, scaler, and features"""
        print("Loading Real LSTM artifacts...")
        
        try:
            # Try to load the full model first
            self.model = keras.models.load_model("models/lstm_best_weights.keras")
            print("✅ Best LSTM weights loaded")
            
            # Load scaler
            with open("models/scaler_real_lstm.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler loaded")
            
            # Load features
            with open("models/features_real_lstm.txt", 'r') as f:
                self.features = [line.strip() for line in f.readlines()]
            print(f"✅ Features loaded: {len(self.features)} features")
            
        except Exception as e:
            print(f"⚠️ Error loading main artifacts: {e}")
            print("Trying quick LSTM fallback...")
            
            try:
                self.model = keras.models.load_model("models/quick_real_lstm.keras")
                with open("models/features_lstm.txt", 'r') as f:
                    self.features = [line.strip() for line in f.readlines()]
                self.scaler = StandardScaler()
                self.window_size = 15
                print("✅ Quick LSTM fallback loaded")
                
            except Exception as e2:
                print(f"❌ Error loading fallback: {e2}")
                raise e2
    
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
        
        if len(df_month) < 100:
            print(f"❌ Insufficient data for {month_name} ({len(df_month)} records)")
            return None, None
        
        # Create target for evaluation (3 bars ahead)
        df_month["target"] = (df_month["close"].shift(-3) > df_month["close"]).astype(int)
        
        # Filter available features
        available_features = [f for f in self.features if f in df_month.columns]
        missing_features = [f for f in self.features if f not in df_month.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features[:5]}...")
        
        if not available_features:
            print("❌ No features available!")
            return None, None
        
        print(f"Using {len(available_features)} features")
        
        # Clean data
        required_cols = ['timestamp', 'close', 'target'] + available_features
        df_clean = df_month[required_cols].dropna()
        
        print(f"Clean data: {len(df_clean)} rows")
        
        return df_clean, available_features
    
    def generate_signals(self, df, features):
        """Generate trading signals using Real LSTM"""
        print("Generating LSTM signals...")
        
        if len(df) < self.window_size + 10:
            print(f"❌ Not enough data for sequences (need >{self.window_size + 10})")
            return df
        
        # Prepare feature data
        feature_data = df[features].values
        
        # Scale features
        feature_scaled = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X_sequences = self.create_sequences(feature_scaled)
        
        if len(X_sequences) == 0:
            print("❌ Could not create sequences")
            return df
        
        print(f"Created {len(X_sequences)} sequences")
        
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
        
        # Dynamic threshold
        confidences = df_signals['confidence'].values
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences)
        threshold = conf_mean + 0.3 * conf_std
        
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
            'total_predictions': len(y_true),
            'positive_predictions': sum(y_pred),
            'actual_positive': sum(y_true)
        }
    
    def run_monthly_backtest(self, file_path, year, month, month_name):
        """Run backtest for specific month"""
        print(f"\n{'='*60}")
        print(f"LSTM Backtest: {month_name} {year}")
        print(f"{'='*60}")
        
        # Prepare data
        df_clean, features = self.prepare_data(file_path, year, month, month_name)
        if df_clean is None:
            return None
        
        # Generate signals
        df_signals = self.generate_signals(df_clean, features)
        
        # Evaluate predictions
        pred_results = self.evaluate_predictions(df_signals, month_name)
        
        # Simulate trading
        trading_results = self.simulate_trading(df_signals, month_name)
        
        return {
            'prediction_results': pred_results,
            'trading_results': trading_results
        }

def main():
    print("=== Real LSTM Backtesting: July, August, September 2025 ===")
    print(f"Backend: {keras.config.backend()}")
    
    # Initialize backtester
    backtester = RealLSTMBacktester()
    
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
        print("REAL LSTM BACKTEST SUMMARY - JULY, AUGUST, SEPTEMBER 2025")
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
        
        avg_return = total_return / len(all_results) * 100
        avg_accuracy = total_accuracy / valid_periods * 100 if valid_periods > 0 else 0
        
        print("-" * 80)
        print(f"{'AVERAGE':<12} {avg_return:>8.1f}% "
              f"${total_pnl:>10.2f} {total_trades:>6} {'':>10} {avg_accuracy:>10.1f}%")
        
        # Compare with previous results
        print(f"\n{'='*60}")
        print("COMPARISON WITH PREVIOUS BACKTESTS")
        print(f"{'='*60}")
        print(f"Previous (Aug-Oct): -10.20% return, 51.8% accuracy")
        print(f"Current (Jul-Sep): {avg_return:+.2f}% return, {avg_accuracy:.1f}% accuracy")
        
        if avg_return > -10.20:
            print(f"✅ Return improvement: {avg_return + 10.20:+.2f}%")
        else:
            print(f"⚠️ Return decline: {avg_return + 10.20:+.2f}%")
            
        if avg_accuracy > 51.8:
            print(f"✅ Accuracy improvement: {avg_accuracy - 51.8:+.1f}%")
        else:
            print(f"⚠️ Accuracy decline: {avg_accuracy - 51.8:+.1f}%")
        
        # Save results
        os.makedirs('reports', exist_ok=True)
        
        report_content = f"""Real LSTM Backtesting Results - July, August, September 2025
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Backend: {keras.config.backend()}

=== SUMMARY ===
Average Return: {avg_return:.2f}%
Total P&L: ${total_pnl:.2f}
Total Trades: {total_trades}
Average Accuracy: {avg_accuracy:.1f}%

=== COMPARISON ===
Previous Backtest (Aug-Oct): -10.20% return, 51.8% accuracy
Current Backtest (Jul-Sep): {avg_return:+.2f}% return, {avg_accuracy:.1f}% accuracy
Performance Change: {avg_return + 10.20:+.2f}% return, {avg_accuracy - 51.8:+.1f}% accuracy

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
  Confusion Matrix:
{pred['confusion_matrix']}"""
        
        # Assessment
        report_content += f"""

=== ASSESSMENT ===
Model Performance: {'✅ IMPROVED' if avg_return > -10.20 and avg_accuracy > 51.8 else '⚠️ MIXED RESULTS' if avg_return > -10.20 or avg_accuracy > 51.8 else '❌ DECLINED'}
Consistency: {'✅ STABLE' if all(r['trading_results']['total_return'] > -0.15 for r in all_results.values()) else '⚠️ VOLATILE'}
Profitability: {'❌ STILL UNPROFITABLE' if avg_return < 0 else '✅ PROFITABLE'}

Recommendations:
1. {'Continue model optimization' if avg_return < 0 else 'Test with live trading'}
2. {'Improve feature engineering' if avg_accuracy < 55 else 'Features performing well'}
3. {'Add risk management' if any(r['trading_results']['total_return'] < -0.2 for r in all_results.values()) else 'Risk levels acceptable'}
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