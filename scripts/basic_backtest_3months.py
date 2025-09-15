#!/usr/bin/env python3
"""
Basic backtest for MACD+RSI strategy on 3-month data (June-August 2025).
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('/Users/alexey/Documents/Development/Python/Trading_bot')

def load_data():
    """Load the corrected detailed CSV data."""
    
    data_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/detailed_backtest_corrected_timestamps.csv'
    
    print(f"📊 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    print(f"📅 Data period: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"📈 Total bars: {len(df):,}")
    print(f"🔧 Indicators ready: {df['indicators_ready'].sum():,}")
    
    return df

def run_basic_backtest(df):
    """Run basic backtest simulation."""
    
    print(f"\n🚀 Starting basic backtest simulation...")
    
    # Initial parameters
    initial_capital = 10000.0
    current_capital = initial_capital
    position_size_pct = 0.05  # 5% per trade
    
    # Trade tracking
    trades = []
    current_position = None
    position_entry_bar = 0
    position_entry_price = 0.0
    equity_curve = []
    
    # Performance tracking
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_profit = 0.0
    total_loss = 0.0
    max_capital = initial_capital
    max_drawdown = 0.0
    
    # Process each bar
    for idx, row in df.iterrows():
        if not row['indicators_ready']:
            continue
            
        current_price = row['close']
        signal_type = row['signal_type']
        
        # Record equity curve point (every 100 bars to reduce data)
        if idx % 100 == 0:
            equity_curve.append({
                'bar': row['bar'],
                'timestamp': row['timestamp'],
                'capital': current_capital,
                'price': current_price
            })
        
        # Process trading signals
        if signal_type in ['BUY', 'SELL'] and current_position is None:
            # Open new position
            current_position = 'LONG' if signal_type == 'BUY' else 'SHORT'
            position_entry_bar = row['bar']
            position_entry_price = current_price
            
            print(f"🟢 {signal_type} at bar {row['bar']}: ${current_price:.2f} (RSI: {row['rsi']:.1f}, MACD: {row['macd_line']:.4f})")
            
        elif signal_type in ['CLOSE_LONG', 'CLOSE_SHORT'] and current_position is not None:
            # Close position
            if ((current_position == 'LONG' and signal_type == 'CLOSE_LONG') or 
                (current_position == 'SHORT' and signal_type == 'CLOSE_SHORT')):
                
                # Calculate P&L
                position_size = current_capital * position_size_pct
                
                if current_position == 'LONG':
                    pnl_pct = ((current_price - position_entry_price) / position_entry_price) * 100
                else:  # SHORT
                    pnl_pct = ((position_entry_price - current_price) / position_entry_price) * 100
                
                pnl_dollars = (pnl_pct / 100) * position_size
                current_capital += pnl_dollars
                
                # Track performance
                total_trades += 1
                if pnl_dollars > 0:
                    winning_trades += 1
                    total_profit += pnl_dollars
                else:
                    losing_trades += 1
                    total_loss += abs(pnl_dollars)
                
                # Track drawdown
                if current_capital > max_capital:
                    max_capital = current_capital
                current_drawdown = (max_capital - current_capital) / max_capital * 100
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                
                # Record trade
                duration_bars = row['bar'] - position_entry_bar
                trade = {
                    'entry_bar': position_entry_bar,
                    'exit_bar': row['bar'],
                    'entry_time': df[df['bar'] == position_entry_bar]['timestamp'].iloc[0],
                    'exit_time': row['timestamp'],
                    'direction': current_position,
                    'entry_price': position_entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'duration_bars': duration_bars,
                    'duration_hours': duration_bars * 0.25,  # 15min bars
                    'capital_after': current_capital
                }
                trades.append(trade)
                
                print(f"🔴 {signal_type} at bar {row['bar']}: ${current_price:.2f} | P&L: {pnl_pct:+.2f}% (${pnl_dollars:+.2f}) | Capital: ${current_capital:.2f}")
                
                # Reset position
                current_position = None
                position_entry_bar = 0
                position_entry_price = 0.0
    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'final_capital': current_capital,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'max_drawdown': max_drawdown
    }

def calculate_metrics(results, initial_capital):
    """Calculate performance metrics."""
    
    trades = results['trades']
    final_capital = results['final_capital']
    
    if len(trades) == 0:
        return {
            'total_return_pct': 0,
            'win_rate_pct': 0,
            'profit_factor': 0,
            'avg_trade_pnl': 0,
            'best_trade_pct': 0,
            'worst_trade_pct': 0,
            'avg_trade_duration_hours': 0
        }
    
    # Basic metrics
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    win_rate_pct = (results['winning_trades'] / results['total_trades']) * 100
    profit_factor = results['total_profit'] / results['total_loss'] if results['total_loss'] > 0 else float('inf')
    
    # Trade analysis
    trade_pnls = [t['pnl_pct'] for t in trades]
    avg_trade_pnl = np.mean(trade_pnls)
    best_trade_pct = max(trade_pnls)
    worst_trade_pct = min(trade_pnls)
    avg_trade_duration_hours = np.mean([t['duration_hours'] for t in trades])
    
    return {
        'total_return_pct': total_return_pct,
        'win_rate_pct': win_rate_pct,
        'profit_factor': profit_factor,
        'avg_trade_pnl': avg_trade_pnl,
        'best_trade_pct': best_trade_pct,
        'worst_trade_pct': worst_trade_pct,
        'avg_trade_duration_hours': avg_trade_duration_hours
    }

def display_results(results, metrics, initial_capital):
    """Display backtest results."""
    
    print(f"\n" + "="*60)
    print(f"📊 BACKTEST RESULTS - MACD+RSI Strategy")
    print(f"="*60)
    
    print(f"💰 PERFORMANCE:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Capital: ${results['final_capital']:,.2f}")
    print(f"   Total Return: {metrics['total_return_pct']:+.2f}%")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
    
    print(f"\n📈 TRADING:")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Winning Trades: {results['winning_trades']}")
    print(f"   Losing Trades: {results['losing_trades']}")
    print(f"   Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    
    print(f"\n📊 TRADE ANALYSIS:")
    print(f"   Average Trade: {metrics['avg_trade_pnl']:+.2f}%")
    print(f"   Best Trade: {metrics['best_trade_pct']:+.2f}%")
    print(f"   Worst Trade: {metrics['worst_trade_pct']:+.2f}%")
    print(f"   Avg Duration: {metrics['avg_trade_duration_hours']:.1f} hours")
    
    # Show sample trades
    if len(results['trades']) > 0:
        print(f"\n📋 SAMPLE TRADES (First 10):")
        print(f"{'Entry':<12} {'Exit':<12} {'Dir':<5} {'Entry$':<8} {'Exit$':<8} {'P&L%':<8} {'Dur(h)':<6}")
        print("-" * 65)
        
        for i, trade in enumerate(results['trades'][:10]):
            print(f"{trade['entry_time'][:10]:<12} {trade['exit_time'][:10]:<12} "
                  f"{trade['direction']:<5} {trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
                  f"{trade['pnl_pct']:+<8.2f} {trade['duration_hours']:<6.1f}")
        
        if len(results['trades']) > 10:
            print(f"... and {len(results['trades']) - 10} more trades")

def save_results(results, metrics, initial_capital):
    """Save results to files."""
    
    # Save trades to CSV
    if len(results['trades']) > 0:
        trades_df = pd.DataFrame(results['trades'])
        trades_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/backtest_3months_trades.csv'
        trades_df.to_csv(trades_file, index=False)
        print(f"\n💾 Trades saved to: {trades_file}")
    
    # Save equity curve
    if len(results['equity_curve']) > 0:
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/backtest_3months_equity.csv'
        equity_df.to_csv(equity_file, index=False)
        print(f"💾 Equity curve saved to: {equity_file}")
    
    # Save summary
    summary = {
        'strategy_name': 'MACD+RSI Basic',
        'test_date': datetime.now().strftime('%Y-%m-%d'),
        'period': '2025-06-01 to 2025-08-30',
        'initial_capital': initial_capital,
        'final_capital': results['final_capital'],
        'total_return_pct': metrics['total_return_pct'],
        'total_trades': results['total_trades'],
        'winning_trades': results['winning_trades'],
        'losing_trades': results['losing_trades'],
        'win_rate_pct': metrics['win_rate_pct'],
        'profit_factor': metrics['profit_factor'],
        'max_drawdown_pct': results['max_drawdown'],
        'avg_trade_pnl_pct': metrics['avg_trade_pnl'],
        'best_trade_pct': metrics['best_trade_pct'],
        'worst_trade_pct': metrics['worst_trade_pct'],
        'avg_trade_duration_hours': metrics['avg_trade_duration_hours']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/backtest_3months_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Summary saved to: {summary_file}")

def main():
    """Main execution function."""
    
    print("🚀 Starting Basic 3-Month Backtest (June-August 2025)")
    print("Strategy: MACD+RSI with EMA50 filter")
    
    # Load data
    df = load_data()
    
    # Run backtest
    initial_capital = 10000.0
    results = run_basic_backtest(df)
    
    # Calculate metrics
    metrics = calculate_metrics(results, initial_capital)
    
    # Display results
    display_results(results, metrics, initial_capital)
    
    # Save results
    save_results(results, metrics, initial_capital)
    
    print(f"\n✅ Basic backtest completed!")
    print(f"📊 Strategy tested on {len(df):,} bars over 90 days")

if __name__ == "__main__":
    main()