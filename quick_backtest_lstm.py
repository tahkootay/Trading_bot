#!/usr/bin/env python3
"""
Quick LSTM Backtesting Script - Simplified version
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_model_artifacts():
    """Load trained model artifacts"""
    print("Loading model artifacts...")
    
    with open("models/lstm_horizon3.pkl", 'rb') as f:
        model = pickle.load(f)
    
    with open("models/scaler_horizon3.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    with open("models/features_lstm.txt", 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    print(f"✅ Loaded model with features: {features}")
    return model, scaler, features

def quick_backtest(file_path, period_name, model, scaler, features):
    """Run quick backtest for a period"""
    print(f"\n=== {period_name} Quick Backtest ===")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    
    # Take sample for speed
    if len(df) > 1000:
        df = df.tail(1000).copy()
        print(f"Using last 1000 records for speed")
    
    # Create target
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    
    # Prepare features
    df_clean = df[features + ['timestamp', 'close', 'target']].dropna()
    print(f"Clean data: {len(df_clean)} rows")
    
    if len(df_clean) < 50:
        print("⚠️ Not enough clean data")
        return None
    
    # Generate predictions
    X = df_clean[features].values
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Evaluate predictions (remove last 3 rows for target alignment)
    eval_data = df_clean[:-3].copy()
    eval_predictions = predictions[:-3]
    eval_targets = eval_data['target'].values
    
    accuracy = accuracy_score(eval_targets, eval_predictions)
    cm = confusion_matrix(eval_targets, eval_predictions)
    
    print(f"Prediction Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\\n{cm}")
    
    # Simple trading simulation
    capital = 10000
    position_size = 0.1
    commission = 0.001
    
    trades = 0
    winning_trades = 0
    total_pnl = 0
    
    # Check probability distribution
    prob_stats = {
        'min': np.min(probabilities),
        'max': np.max(probabilities),
        'mean': np.mean(probabilities),
        'std': np.std(probabilities)
    }
    print(f"Probability stats: min={prob_stats['min']:.3f}, max={prob_stats['max']:.3f}, mean={prob_stats['mean']:.3f}")
    
    # Use more flexible thresholds
    high_threshold = prob_stats['mean'] + 0.5 * prob_stats['std']
    low_threshold = prob_stats['mean'] - 0.5 * prob_stats['std']
    
    print(f"Trading thresholds: buy>{high_threshold:.3f}, sell<{low_threshold:.3f}")
    
    for i in range(len(df_clean) - 3):
        if probabilities[i] > high_threshold:  # Buy signal
            entry_price = df_clean.iloc[i]['close']
            exit_price = df_clean.iloc[i + 3]['close']
            
            # Long position
            pnl = (exit_price - entry_price) / entry_price * capital * position_size
            pnl -= capital * position_size * commission * 2  # Commission
            
            trades += 1
            total_pnl += pnl
            if pnl > 0:
                winning_trades += 1
                
        elif probabilities[i] < low_threshold:  # Sell signal
            entry_price = df_clean.iloc[i]['close']
            exit_price = df_clean.iloc[i + 3]['close']
            
            # Short position
            pnl = (entry_price - exit_price) / entry_price * capital * position_size
            pnl -= capital * position_size * commission * 2
            
            trades += 1
            total_pnl += pnl
            if pnl > 0:
                winning_trades += 1
    
    win_rate = winning_trades / trades if trades > 0 else 0
    total_return = total_pnl / capital
    
    results = {
        'period': period_name,
        'accuracy': accuracy,
        'total_return': total_return,
        'total_pnl': total_pnl,
        'trades': trades,
        'win_rate': win_rate,
        'confusion_matrix': cm
    }
    
    print(f"Trading Results:")
    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Trades: {trades}")
    print(f"  Win Rate: {win_rate*100:.2f}%")
    
    return results

def main():
    print("=== Quick LSTM Backtesting ===\\n")
    
    # Load model
    model, scaler, features = load_model_artifacts()
    
    # Test periods
    periods = {
        'August 2025': 'data/processed/test_aug_2025.csv',
        'September 2025': 'data/processed/test_sep_2025.csv', 
        'October 2025': 'data/processed/test_oct_2025.csv'
    }
    
    results = {}
    
    # Run backtests
    for period_name, file_path in periods.items():
        try:
            result = quick_backtest(file_path, period_name, model, scaler, features)
            if result:
                results[period_name] = result
        except Exception as e:
            print(f"❌ Error in {period_name}: {e}")
    
    # Summary
    if results:
        print(f"\\n{'='*60}")
        print("SUMMARY RESULTS")
        print(f"{'='*60}")
        print(f"{'Period':<15} {'Return %':<10} {'Trades':<8} {'Win Rate %':<12} {'Accuracy %':<12}")
        print("-" * 60)
        
        total_return = 0
        total_trades = 0
        total_accuracy = 0
        
        for period, result in results.items():
            print(f"{period:<15} {result['total_return']*100:>8.2f}% {result['trades']:>6} "
                  f"{result['win_rate']*100:>10.2f}% {result['accuracy']*100:>10.2f}%")
            
            total_return += result['total_return']
            total_trades += result['trades']
            total_accuracy += result['accuracy']
        
        avg_return = total_return / len(results) * 100
        avg_accuracy = total_accuracy / len(results) * 100
        
        print("-" * 60)
        print(f"{'AVERAGE':<15} {avg_return:>8.2f}% {total_trades:>6} {'':>10} {avg_accuracy:>10.2f}%")
        
        # Save results
        os.makedirs('reports', exist_ok=True)
        
        report_content = f"""LSTM Horizon 3 Quick Backtest Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== SUMMARY ===
Average Return: {avg_return:.2f}%
Total Trades: {total_trades}
Average Accuracy: {avg_accuracy:.2f}%

=== DETAILED RESULTS ===
"""
        
        for period, result in results.items():
            report_content += f"""
{period}:
  Return: {result['total_return']*100:.2f}%
  P&L: ${result['total_pnl']:.2f}
  Trades: {result['trades']}
  Win Rate: {result['win_rate']*100:.2f}%
  Accuracy: {result['accuracy']*100:.2f}%
  Confusion Matrix:
{result['confusion_matrix']}
"""
        
        with open('reports/lstm_quick_backtest.txt', 'w') as f:
            f.write(report_content)
        
        print(f"\\n✅ Results saved to: reports/lstm_quick_backtest.txt")
    else:
        print("❌ No successful backtests")

if __name__ == "__main__":
    main()