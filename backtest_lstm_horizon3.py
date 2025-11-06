#!/usr/bin/env python3
"""
LSTM Horizon 3 Backtesting Script
Run backtests on August, September, and October 2025 data
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class LSTMBacktester:
    def __init__(self, model_path, scaler_path, features_path):
        """Initialize backtester with trained model artifacts"""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        
        # Load artifacts
        self.load_artifacts()
        
        # Trading parameters
        self.commission = 0.001  # 0.1% commission
        self.initial_capital = 10000
        self.position_size = 0.1  # 10% of capital per trade
        
    def load_artifacts(self):
        """Load trained model, scaler, and features"""
        print("Loading model artifacts...")
        
        # Load model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Model loaded from {self.model_path}")
        
        # Load scaler
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✅ Scaler loaded from {self.scaler_path}")
        
        # Load features
        with open(self.features_path, 'r') as f:
            self.features = [line.strip() for line in f.readlines()]
        print(f"✅ Features loaded: {self.features}")
    
    def prepare_data(self, file_path, period_name):
        """Prepare data for backtesting"""
        print(f"\nPreparing {period_name} data...")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
        
        # Create target for evaluation
        df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
        
        # Filter available features
        available_features = [f for f in self.features if f in df.columns]
        missing_features = [f for f in self.features if f not in df.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Clean data
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + available_features + ['target']
        df_clean = df[required_cols].dropna()
        
        print(f"Clean data: {len(df_clean)} rows")
        
        return df_clean, available_features
    
    def generate_signals(self, df, features):
        """Generate trading signals using the trained model"""
        print("Generating trading signals...")
        
        # Prepare features for prediction
        X = df[features].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of UP
        
        # Add signals to dataframe
        df_signals = df.copy()
        df_signals['signal'] = predictions  # 1 = BUY, 0 = SELL/HOLD
        df_signals['prob_up'] = probabilities
        df_signals['confidence'] = np.abs(probabilities - 0.5) * 2  # Confidence [0, 1]
        
        return df_signals
    
    def simulate_trading(self, df_signals, period_name):
        """Simulate trading based on signals"""
        print(f"Simulating trading for {period_name}...")
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Track performance
        total_trades = 0
        winning_trades = 0
        total_pnl = 0
        
        for i in range(len(df_signals) - 3):  # -3 because we need future price
            row = df_signals.iloc[i]
            future_price = df_signals.iloc[i + 3]['close']  # Price after 3 bars
            
            timestamp = row['timestamp']
            current_price = row['close']
            signal = row['signal']
            confidence = row['confidence']
            
            # Only trade with high confidence (> 0.6)
            if confidence < 0.6:
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': capital,
                    'position': position,
                    'price': current_price
                })
                continue
            
            # Close existing position and open new one
            if position != 0:
                # Close existing position
                if position == 1:  # Close long
                    pnl = (current_price - entry_price) * (capital * self.position_size / entry_price)
                    pnl -= capital * self.position_size * self.commission * 2  # Entry + exit commission
                else:  # Close short
                    pnl = (entry_price - current_price) * (capital * self.position_size / entry_price)
                    pnl -= capital * self.position_size * self.commission * 2
                
                capital += pnl
                total_pnl += pnl
                total_trades += 1
                
                if pnl > 0:
                    winning_trades += 1
                
                trades.append({
                    'entry_time': entry_timestamp,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position_type': 'LONG' if position == 1 else 'SHORT',
                    'pnl': pnl,
                    'capital_after': capital
                })
            
            # Open new position based on signal
            if signal == 1:  # BUY signal
                position = 1
                entry_price = current_price
                entry_timestamp = timestamp
            else:  # SELL signal
                position = -1
                entry_price = current_price
                entry_timestamp = timestamp
            
            equity_curve.append({
                'timestamp': timestamp,
                'equity': capital,
                'position': position,
                'price': current_price
            })
        
        # Close final position if exists
        if position != 0:
            final_price = df_signals.iloc[-1]['close']
            if position == 1:
                pnl = (final_price - entry_price) * (capital * self.position_size / entry_price)
            else:
                pnl = (entry_price - final_price) * (capital * self.position_size / entry_price)
            
            pnl -= capital * self.position_size * self.commission * 2
            capital += pnl
            total_pnl += pnl
            total_trades += 1
            
            if pnl > 0:
                winning_trades += 1
        
        # Calculate metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        # Calculate additional metrics
        if trades:
            profits = [t['pnl'] for t in trades if t['pnl'] > 0]
            losses = [t['pnl'] for t in trades if t['pnl'] < 0]
            
            avg_win = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = abs(sum(profits) / sum(losses)) if losses else float('inf')
            
            # Sharpe-like ratio (simplified)
            returns = [t['pnl'] / self.initial_capital for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        else:
            avg_win = avg_loss = profit_factor = sharpe_ratio = 0
        
        results = {
            'period': period_name,
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        return results
    
    def evaluate_predictions(self, df_signals, period_name):
        """Evaluate prediction accuracy"""
        print(f"Evaluating predictions for {period_name}...")
        
        # Remove rows without future target
        df_eval = df_signals[:-3].copy()
        
        if 'target' not in df_eval.columns:
            print("⚠️ Target column not available for evaluation")
            return None
        
        y_true = df_eval['target'].values
        y_pred = df_eval['signal'].values
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'period': period_name,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'total_predictions': len(y_true),
            'positive_predictions': sum(y_pred),
            'actual_positive': sum(y_true)
        }
    
    def run_backtest(self, file_path, period_name):
        """Run complete backtest for a period"""
        print(f"\n{'='*50}")
        print(f"Running backtest for {period_name}")
        print(f"{'='*50}")
        
        # Prepare data
        df_clean, features = self.prepare_data(file_path, period_name)
        
        # Generate signals
        df_signals = self.generate_signals(df_clean, features)
        
        # Evaluate predictions
        pred_results = self.evaluate_predictions(df_signals, period_name)
        
        # Simulate trading
        trading_results = self.simulate_trading(df_signals, period_name)
        
        # Combine results
        results = {
            'prediction_results': pred_results,
            'trading_results': trading_results,
            'signals_data': df_signals
        }
        
        return results
    
    def plot_results(self, results, period_name):
        """Plot backtest results"""
        equity_curve = results['trading_results']['equity_curve']
        
        if not equity_curve:
            print(f"No equity curve data for {period_name}")
            return
        
        eq_df = pd.DataFrame(equity_curve)
        eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        ax1.plot(eq_df['timestamp'], eq_df['equity'], label='Portfolio Value', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title(f'{period_name} - Portfolio Performance')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Price with positions
        ax2.plot(eq_df['timestamp'], eq_df['price'], label='Price', alpha=0.7)
        
        # Mark positions
        long_positions = eq_df[eq_df['position'] == 1]
        short_positions = eq_df[eq_df['position'] == -1]
        
        if not long_positions.empty:
            ax2.scatter(long_positions['timestamp'], long_positions['price'], 
                       color='green', marker='^', s=30, alpha=0.7, label='Long Position')
        
        if not short_positions.empty:
            ax2.scatter(short_positions['timestamp'], short_positions['price'], 
                       color='red', marker='v', s=30, alpha=0.7, label='Short Position')
        
        ax2.set_title(f'{period_name} - Price & Positions')
        ax2.set_ylabel('Price')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(f'results/lstm_backtests', exist_ok=True)
        plt.savefig(f'results/lstm_backtests/{period_name.lower()}_backtest.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, all_results):
        """Generate comprehensive backtest report"""
        print("\nGenerating comprehensive report...")
        
        report_lines = []
        report_lines.append("LSTM Horizon 3 Backtesting Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        
        # Summary table
        report_lines.append("\n=== PERFORMANCE SUMMARY ===")
        report_lines.append(f"{'Period':<12} {'Return %':<10} {'Trades':<8} {'Win Rate':<10} {'Accuracy':<10}")
        report_lines.append("-" * 60)
        
        total_return_sum = 0
        total_trades_sum = 0
        total_wins_sum = 0
        
        for period_name, results in all_results.items():
            trading = results['trading_results']
            pred = results['prediction_results']
            
            return_pct = trading['total_return'] * 100
            win_rate = trading['win_rate'] * 100
            accuracy = pred['accuracy'] * 100 if pred else 0
            
            total_return_sum += trading['total_return']
            total_trades_sum += trading['total_trades']
            total_wins_sum += trading['winning_trades']
            
            report_lines.append(f"{period_name:<12} {return_pct:>8.2f}% {trading['total_trades']:>6} {win_rate:>8.2f}% {accuracy:>8.2f}%")
        
        # Overall summary
        avg_return = total_return_sum / len(all_results) * 100
        overall_win_rate = total_wins_sum / total_trades_sum * 100 if total_trades_sum > 0 else 0
        
        report_lines.append("-" * 60)
        report_lines.append(f"{'AVERAGE':<12} {avg_return:>8.2f}% {total_trades_sum:>6} {overall_win_rate:>8.2f}%")
        
        # Detailed results for each period
        for period_name, results in all_results.items():
            trading = results['trading_results']
            pred = results['prediction_results']
            
            report_lines.append(f"\n=== {period_name.upper()} DETAILED RESULTS ===")
            
            # Trading metrics
            report_lines.append("\nTrading Performance:")
            report_lines.append(f"  Initial Capital: ${trading['initial_capital']:,.2f}")
            report_lines.append(f"  Final Capital: ${trading['final_capital']:,.2f}")
            report_lines.append(f"  Total Return: {trading['total_return']*100:.2f}%")
            report_lines.append(f"  Total P&L: ${trading['total_pnl']:,.2f}")
            report_lines.append(f"  Total Trades: {trading['total_trades']}")
            report_lines.append(f"  Winning Trades: {trading['winning_trades']}")
            report_lines.append(f"  Win Rate: {trading['win_rate']*100:.2f}%")
            report_lines.append(f"  Average Win: ${trading['avg_win']:,.2f}")
            report_lines.append(f"  Average Loss: ${trading['avg_loss']:,.2f}")
            report_lines.append(f"  Profit Factor: {trading['profit_factor']:.2f}")
            report_lines.append(f"  Sharpe Ratio: {trading['sharpe_ratio']:.2f}")
            
            # Prediction metrics
            if pred:
                report_lines.append("\nPrediction Accuracy:")
                report_lines.append(f"  Overall Accuracy: {pred['accuracy']*100:.2f}%")
                report_lines.append(f"  Total Predictions: {pred['total_predictions']}")
                report_lines.append(f"  Positive Predictions: {pred['positive_predictions']}")
                report_lines.append(f"  Actual Positive: {pred['actual_positive']}")
                
                # Confusion matrix
                cm = pred['confusion_matrix']
                report_lines.append(f"\nConfusion Matrix:")
                report_lines.append(f"  True Neg: {cm[0,0]:>4} | False Pos: {cm[0,1]:>4}")
                report_lines.append(f"  False Neg: {cm[1,0]:>4} | True Pos: {cm[1,1]:>4}")
                
                # Classification report
                clf_report = pred['classification_report']
                report_lines.append(f"\nPrecision/Recall:")
                report_lines.append(f"  Down - Precision: {clf_report['0']['precision']:.3f}, Recall: {clf_report['0']['recall']:.3f}")
                report_lines.append(f"  Up - Precision: {clf_report['1']['precision']:.3f}, Recall: {clf_report['1']['recall']:.3f}")
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        report_path = 'reports/lstm_backtest_results.txt'
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Report saved to: {report_path}")
        
        # Print summary to console
        print("\n" + "\n".join(report_lines[:20]))  # Print first 20 lines
        print("...")
        print(f"Full report saved to: {report_path}")

def main():
    print("=== LSTM Horizon 3 Backtesting ===\n")
    
    # Initialize backtester
    backtester = LSTMBacktester(
        model_path="models/lstm_horizon3.pkl",
        scaler_path="models/scaler_horizon3.pkl", 
        features_path="models/features_lstm.txt"
    )
    
    # Define test periods
    test_periods = {
        'August 2025': 'data/processed/test_aug_2025.csv',
        'September 2025': 'data/processed/test_sep_2025.csv',
        'October 2025': 'data/processed/test_oct_2025.csv'
    }
    
    # Run backtests
    all_results = {}
    
    for period_name, file_path in test_periods.items():
        try:
            results = backtester.run_backtest(file_path, period_name)
            all_results[period_name] = results
            
            # Plot results
            backtester.plot_results(results, period_name)
            
            # Print quick summary
            trading = results['trading_results']
            pred = results['prediction_results']
            
            print(f"\n📊 {period_name} Quick Summary:")
            print(f"   Return: {trading['total_return']*100:.2f}%")
            print(f"   Trades: {trading['total_trades']}")
            print(f"   Win Rate: {trading['win_rate']*100:.2f}%")
            if pred:
                print(f"   Accuracy: {pred['accuracy']*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error processing {period_name}: {e}")
            continue
    
    # Generate comprehensive report
    if all_results:
        backtester.generate_report(all_results)
        
        print(f"\n✅ Backtesting completed for {len(all_results)} periods!")
        print("Check 'reports/lstm_backtest_results.txt' for detailed results")
        print("Check 'results/lstm_backtests/' for charts")
    else:
        print("❌ No successful backtests completed")

if __name__ == "__main__":
    main()