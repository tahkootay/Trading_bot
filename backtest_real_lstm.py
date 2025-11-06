#!/usr/bin/env python3
"""
Real LSTM Backtesting Script
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
    def __init__(self, model_path, scaler_path, features_path, window_size=30):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        self.window_size = window_size
        
        # Load artifacts
        self.load_artifacts()
        
        # Trading parameters
        self.commission = 0.001
        self.initial_capital = 10000
        self.position_size = 0.1
        
    def load_artifacts(self):
        """Load trained model, scaler, and features"""
        print("Loading Real LSTM artifacts...")
        
        try:
            # Load model
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ LSTM Model loaded from {self.model_path}")
            
            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler loaded from {self.scaler_path}")
            
            # Load features
            with open(self.features_path, 'r') as f:
                self.features = [line.strip() for line in f.readlines()]
            print(f"✅ Features loaded: {len(self.features)} features")
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            print("Using fallback quick LSTM model...")
            
            # Fallback to quick model if available
            try:
                self.model = keras.models.load_model("models/quick_real_lstm.keras")
                self.features = ['rsi_14', 'ema_20', 'ema_50', 'macd_line', 'atr_14']
                self.window_size = 15
                
                # Create a basic scaler
                self.scaler = StandardScaler()
                print("✅ Using quick LSTM fallback")
                
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
    
    def prepare_data(self, file_path, period_name):
        """Prepare data for backtesting"""
        print(f"\\nPreparing {period_name} data...")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
        
        # Use sample for speed
        if len(df) > 2000:
            df = df.tail(2000).copy()
            print(f"Using last 2000 records for speed")
        
        # Create target for evaluation
        df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
        
        # Filter available features
        available_features = [f for f in self.features if f in df.columns]
        missing_features = [f for f in self.features if f not in df.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        if not available_features:
            print("❌ No features available!")
            return None, None
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Clean data
        required_cols = ['timestamp', 'close', 'target'] + available_features
        df_clean = df[required_cols].dropna()
        
        print(f"Clean data: {len(df_clean)} rows")
        
        return df_clean, available_features
    
    def generate_signals(self, df, features):
        """Generate trading signals using Real LSTM"""
        print("Generating Real LSTM signals...")
        
        if len(df) < self.window_size + 10:
            print(f"❌ Not enough data for sequences (need >{self.window_size + 10})")
            return df
        
        # Prepare feature data
        feature_data = df[features].values
        
        # Scale features (fit on all data for simplicity in backtest)
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
        
        # Add signals to dataframe (align with sequences)
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
    
    def simulate_trading(self, df_signals, period_name):
        """Simulate trading with Real LSTM signals"""
        print(f"Simulating Real LSTM trading for {period_name}...")
        
        capital = self.initial_capital
        trades = 0
        winning_trades = 0
        total_pnl = 0
        
        # Dynamic threshold based on confidence distribution
        confidences = df_signals['confidence'].values
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences)
        threshold = conf_mean + 0.5 * conf_std
        
        print(f"Confidence stats: mean={conf_mean:.3f}, std={conf_std:.3f}")
        print(f"Trading threshold: {threshold:.3f}")
        
        trades_list = []
        
        for i in range(len(df_signals) - 3):
            row = df_signals.iloc[i]
            
            confidence = row['confidence']
            signal = row['signal']
            prob_up = row['prob_up']
            
            # Only trade with high confidence
            if confidence < threshold:
                continue
            
            entry_price = row['close']
            exit_price = df_signals.iloc[i + 3]['close']
            
            # Determine position type
            if prob_up > 0.5:  # Long position
                pnl = (exit_price - entry_price) / entry_price * capital * self.position_size
            else:  # Short position
                pnl = (entry_price - exit_price) / entry_price * capital * self.position_size
            
            # Subtract commission
            pnl -= capital * self.position_size * self.commission * 2
            
            total_pnl += pnl
            trades += 1
            
            if pnl > 0:
                winning_trades += 1
            
            trades_list.append({
                'timestamp': row['timestamp'],
                'signal': 'LONG' if prob_up > 0.5 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'confidence': confidence
            })
        
        win_rate = winning_trades / trades if trades > 0 else 0
        total_return = total_pnl / capital if capital > 0 else 0
        
        results = {
            'period': period_name,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'trades': trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'trades_list': trades_list
        }
        
        return results
    
    def evaluate_predictions(self, df_signals, period_name):
        """Evaluate Real LSTM prediction accuracy"""
        print(f"Evaluating Real LSTM predictions for {period_name}...")
        
        # Get predictions that can be evaluated (have future targets)
        eval_df = df_signals[:-3].copy()
        
        # Only evaluate rows with predictions
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
            'period': period_name,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'total_predictions': len(y_true),
            'positive_predictions': sum(y_pred),
            'actual_positive': sum(y_true)
        }
    
    def run_backtest(self, file_path, period_name):
        """Run complete Real LSTM backtest"""
        print(f"\\n{'='*60}")
        print(f"Real LSTM Backtest: {period_name}")
        print(f"{'='*60}")
        
        # Prepare data
        df_clean, features = self.prepare_data(file_path, period_name)
        if df_clean is None:
            return None
        
        # Generate signals
        df_signals = self.generate_signals(df_clean, features)
        
        # Evaluate predictions
        pred_results = self.evaluate_predictions(df_signals, period_name)
        
        # Simulate trading
        trading_results = self.simulate_trading(df_signals, period_name)
        
        return {
            'prediction_results': pred_results,
            'trading_results': trading_results,
            'signals_data': df_signals
        }

def main():
    print("=== Real LSTM Backtesting ===")
    print(f"Backend: {keras.config.backend()}")
    
    # Initialize backtester
    try:
        backtester = RealLSTMBacktester(
            model_path="models/real_lstm_horizon3.keras",
            scaler_path="models/scaler_real_lstm.pkl",
            features_path="models/features_real_lstm.txt"
        )
    except:
        print("Using fallback configuration...")
        backtester = RealLSTMBacktester(
            model_path="models/quick_real_lstm.keras",
            scaler_path="models/scaler_horizon3.pkl",  # Use existing scaler
            features_path="models/features_lstm.txt"
        )
    
    # Test periods
    periods = {
        'August 2025': 'data/processed/test_aug_2025.csv',
        'September 2025': 'data/processed/test_sep_2025.csv',
        'October 2025': 'data/processed/test_oct_2025.csv'
    }
    
    all_results = {}
    
    # Run backtests
    for period_name, file_path in periods.items():
        try:
            results = backtester.run_backtest(file_path, period_name)
            if results:
                all_results[period_name] = results
                
                # Print quick summary
                trading = results['trading_results']
                pred = results['prediction_results']
                
                print(f"\\n📊 {period_name} Results:")
                print(f"   Return: {trading['total_return']*100:+.2f}%")
                print(f"   Trades: {trading['trades']}")
                print(f"   Win Rate: {trading['win_rate']*100:.1f}%")
                if pred:
                    print(f"   Accuracy: {pred['accuracy']*100:.1f}%")
                
        except Exception as e:
            print(f"❌ Error in {period_name}: {e}")
    
    # Generate summary
    if all_results:
        print(f"\\n{'='*70}")
        print("REAL LSTM SUMMARY RESULTS")
        print(f"{'='*70}")
        print(f"{'Period':<15} {'Return %':<10} {'Trades':<8} {'Win Rate %':<12} {'Accuracy %':<12}")
        print("-" * 70)
        
        total_return = 0
        total_trades = 0
        total_accuracy = 0
        
        for period, results in all_results.items():
            trading = results['trading_results']
            pred = results['prediction_results']
            
            accuracy = pred['accuracy'] * 100 if pred else 0
            
            print(f"{period:<15} {trading['total_return']*100:>8.1f}% {trading['trades']:>6} "
                  f"{trading['win_rate']*100:>10.1f}% {accuracy:>10.1f}%")
            
            total_return += trading['total_return']
            total_trades += trading['trades']
            if pred:
                total_accuracy += pred['accuracy']
        
        avg_return = total_return / len(all_results) * 100
        avg_accuracy = total_accuracy / len(all_results) * 100
        
        print("-" * 70)
        print(f"{'AVERAGE':<15} {avg_return:>8.1f}% {total_trades:>6} {'':>10} {avg_accuracy:>10.1f}%")
        
        # Compare with previous model
        print(f"\\n{'='*50}")
        print("COMPARISON WITH PREVIOUS MODEL")
        print(f"{'='*50}")
        print(f"Previous (Logistic): -13.46% return, 48.48% accuracy")
        print(f"Real LSTM: {avg_return:+.2f}% return, {avg_accuracy:.1f}% accuracy")
        print(f"Improvement: {avg_return + 13.46:+.2f}% return, {avg_accuracy - 48.48:+.1f}% accuracy")
        
        # Save results
        os.makedirs('reports', exist_ok=True)
        
        report_content = f"""Real LSTM Backtesting Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Backend: {keras.config.backend()}

=== SUMMARY ===
Average Return: {avg_return:.2f}%
Total Trades: {total_trades}
Average Accuracy: {avg_accuracy:.1f}%

=== COMPARISON ===
Previous Model: -13.46% return, 48.48% accuracy
Real LSTM: {avg_return:+.2f}% return, {avg_accuracy:.1f}% accuracy
Improvement: {avg_return + 13.46:+.2f}% return, {avg_accuracy - 48.48:+.1f}% accuracy

=== DETAILED RESULTS ===
"""
        
        for period, results in all_results.items():
            trading = results['trading_results']
            pred = results['prediction_results']
            
            report_content += f"""
{period}:
  Return: {trading['total_return']*100:.2f}%
  P&L: ${trading['total_pnl']:.2f}
  Trades: {trading['trades']}
  Win Rate: {trading['win_rate']*100:.1f}%
"""
            
            if pred:
                report_content += f"""  Accuracy: {pred['accuracy']*100:.1f}%
  Confusion Matrix:
{pred['confusion_matrix']}
"""
        
        with open('reports/real_lstm_backtest_results.txt', 'w') as f:
            f.write(report_content)
        
        print(f"\\n✅ Real LSTM backtesting completed!")
        print("Results saved to: reports/real_lstm_backtest_results.txt")
        
    else:
        print("❌ No successful backtests completed")

if __name__ == "__main__":
    main()