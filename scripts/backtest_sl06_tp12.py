#!/usr/bin/env python3
"""
Modified backtest for MACD+RSI strategy with SL=0.6%, TP=1.2% (was SL=0.6%, TP=0.8%).
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

def generate_modified_signals(df):
    """Generate trading signals with RSI 80/20 and new SL/TP."""
    
    print(f"🔄 Generating signals with RSI 80/20 and SL=0.6%, TP=1.2%...")
    
    # Create copy for modification
    df_modified = df.copy()
    
    # Reset all signals to HOLD initially
    df_modified['signal_type'] = 'HOLD'
    
    # Track signal generation
    signals_generated = {
        'BUY': 0,
        'SELL': 0,
        'CLOSE_LONG': 0,
        'CLOSE_SHORT': 0
    }
    
    # Track current position for exit signals
    current_position = None
    position_entry_bar = 0
    position_entry_price = 0.0
    
    # Process each bar with indicators ready
    ready_bars = df_modified[df_modified['indicators_ready'] == True].copy()
    
    for idx in range(1, len(ready_bars)):  # Start from 1 to have previous values
        current_row = ready_bars.iloc[idx]
        prev_row = ready_bars.iloc[idx-1]
        
        bar_idx = current_row.name  # Original DataFrame index
        
        # Get indicator values
        current_price = current_row['close']
        macd_line = current_row['macd_line']
        macd_signal = current_row['macd_signal_line']
        prev_macd_line = prev_row['macd_line']
        prev_macd_signal = prev_row['macd_signal_line']
        rsi = current_row['rsi']
        ema50 = current_row['ema50']
        
        # Skip if any indicator is NaN
        if pd.isna(macd_line) or pd.isna(macd_signal) or pd.isna(prev_macd_line) or pd.isna(prev_macd_signal) or pd.isna(rsi) or pd.isna(ema50):
            continue
        
        # Calculate crossovers
        macd_bullish_cross = prev_macd_line <= prev_macd_signal and macd_line > macd_signal
        macd_bearish_cross = prev_macd_line >= prev_macd_signal and macd_line < macd_signal
        
        # Position management (exits) - UPDATED SL/TP
        if current_position is not None:
            # Calculate P&L for stop-loss/take-profit
            if current_position == 'LONG':
                pnl_pct = ((current_price - position_entry_price) / position_entry_price) * 100
                
                # Take profit at +1.2% or stop loss at -0.6%
                if pnl_pct >= 1.2 or pnl_pct <= -0.6:  # CHANGED: TP from 0.8% to 1.2%
                    df_modified.loc[bar_idx, 'signal_type'] = 'CLOSE_LONG'
                    signals_generated['CLOSE_LONG'] += 1
                    current_position = None
                    
            elif current_position == 'SHORT':
                pnl_pct = ((position_entry_price - current_price) / position_entry_price) * 100
                
                # Take profit at +1.2% or stop loss at -0.6%
                if pnl_pct >= 1.2 or pnl_pct <= -0.6:  # CHANGED: TP from 0.8% to 1.2%
                    df_modified.loc[bar_idx, 'signal_type'] = 'CLOSE_SHORT'
                    signals_generated['CLOSE_SHORT'] += 1
                    current_position = None
        
        # Entry signals (only when no position)
        elif current_position is None:
            # LONG: MACD bullish cross + RSI < 80 + Price > EMA50
            if (macd_bullish_cross and 
                rsi < 80 and  # RSI threshold 80
                current_price > ema50):
                
                df_modified.loc[bar_idx, 'signal_type'] = 'BUY'
                signals_generated['BUY'] += 1
                current_position = 'LONG'
                position_entry_bar = current_row['bar']
                position_entry_price = current_price
                
            # SHORT: MACD bearish cross + RSI > 20 + Price < EMA50
            elif (macd_bearish_cross and 
                  rsi > 20 and  # RSI threshold 20
                  current_price < ema50):
                
                df_modified.loc[bar_idx, 'signal_type'] = 'SELL'
                signals_generated['SELL'] += 1
                current_position = 'SHORT'
                position_entry_bar = current_row['bar']
                position_entry_price = current_price
    
    print(f"✅ Signal generation completed:")
    for signal, count in signals_generated.items():
        print(f"   {signal}: {count}")
    
    return df_modified

def run_backtest_with_modified_signals(df):
    """Run backtest with modified SL/TP signals."""
    
    print(f"\n🚀 Starting backtest with SL=0.6%, TP=1.2%...")
    
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
                
                # Determine exit reason
                tp_reason = "TP" if abs(pnl_pct) >= 1.15 else "SL"  # Close to TP threshold
                print(f"🔴 {signal_type} at bar {row['bar']}: ${current_price:.2f} | {tp_reason}: {pnl_pct:+.2f}% (${pnl_dollars:+.2f}) | Capital: ${current_capital:.2f}")
                
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

def display_comparison(results, metrics, initial_capital):
    """Display results with comparison to previous versions."""
    
    print(f"\n" + "="*80)
    print(f"📊 BACKTEST RESULTS - MACD+RSI Strategy (SL=0.6%, TP=1.2%)")
    print(f"="*80)
    
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
    
    print(f"\n🔄 COMPARISON WITH PREVIOUS VERSIONS:")
    print(f"   RSI 70/30, SL=0.6%, TP=0.8%: +1.20% (251 trades, 47.01% WR)")
    print(f"   RSI 80/20, SL=0.6%, TP=0.8%: +1.57% (261 trades, 48.28% WR)")
    print(f"   RSI 80/20, SL=0.6%, TP=1.2%: {metrics['total_return_pct']:+.2f}% ({results['total_trades']} trades, {metrics['win_rate_pct']:.2f}% WR)")
    
    print(f"\n📈 RISK/REWARD ANALYSIS:")
    print(f"   New Risk/Reward Ratio: 1:2.0 (SL 0.6% vs TP 1.2%)")
    print(f"   Previous Risk/Reward: 1:1.33 (SL 0.6% vs TP 0.8%)")

def save_results(results, metrics, initial_capital):
    """Save results with SL06_TP12 suffix."""
    
    # Save trades to CSV
    if len(results['trades']) > 0:
        trades_df = pd.DataFrame(results['trades'])
        trades_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/backtest_3months_trades_sl06_tp12.csv'
        trades_df.to_csv(trades_file, index=False)
        print(f"\n💾 Trades saved to: {trades_file}")
    
    # Save summary
    summary = {
        'strategy_name': 'MACD+RSI SL0.6%/TP1.2%',
        'test_date': datetime.now().strftime('%Y-%m-%d'),
        'period': '2025-06-01 to 2025-08-30',
        'rsi_overbought': 80,
        'rsi_oversold': 20,
        'stop_loss_pct': 0.6,
        'take_profit_pct': 1.2,
        'risk_reward_ratio': 2.0,
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
    summary_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/backtest_3months_summary_sl06_tp12.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Summary saved to: {summary_file}")
    
    return summary

def main():
    """Main execution function."""
    
    print("🚀 Starting MACD+RSI Backtest with SL=0.6%, TP=1.2%")
    print("Strategy: MACD+RSI with RSI 80/20 and improved risk/reward ratio")
    
    # Load data
    df = load_data()
    
    # Generate modified signals with new SL/TP
    df_modified = generate_modified_signals(df)
    
    # Run backtest
    initial_capital = 10000.0
    results = run_backtest_with_modified_signals(df_modified)
    
    # Calculate metrics
    metrics = calculate_metrics(results, initial_capital)
    
    # Display results with comparison
    display_comparison(results, metrics, initial_capital)
    
    # Save results
    summary = save_results(results, metrics, initial_capital)
    
    print(f"\n✅ Modified backtest completed!")
    print(f"📊 Risk/Reward improved from 1:1.33 to 1:2.0")
    print(f"🎯 Impact: {'Better' if metrics['total_return_pct'] > 1.57 else 'Worse'} performance vs previous version")

if __name__ == "__main__":
    main()