#!/usr/bin/env python3
"""
Create short strategy comparison CSV for comparing different parameter sets.
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path

def create_strategy_summary_row():
    """Create a summary row for the current MACD+RSI strategy."""
    
    # Load the detailed backtest results
    json_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/backtests/macd_rsi_detailed_report_data.json'
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    backtest_info = data['backtest_info']
    metrics = data['performance_metrics']
    detailed = data['detailed_analysis']
    
    # Create summary row
    summary_row = {
        # Strategy identification
        'strategy_name': 'MACD+RSI',
        'test_date': datetime.now().strftime('%Y-%m-%d'),
        'period': f"{backtest_info['start_date'][:10]} to {backtest_info['end_date'][:10]}",
        
        # Parameters
        'macd_fast': backtest_info['parameters']['macd_fast'],
        'macd_slow': backtest_info['parameters']['macd_slow'],
        'macd_signal': backtest_info['parameters']['macd_signal'],
        'rsi_period': backtest_info['parameters']['rsi_period'],
        'ema_period': backtest_info['parameters']['ema_period'],
        'rsi_overbought': backtest_info['parameters']['rsi_overbought'],
        'rsi_oversold': backtest_info['parameters']['rsi_oversold'],
        'take_profit_pct': backtest_info['parameters']['take_profit_pct'],
        'stop_loss_pct': backtest_info['parameters']['stop_loss_pct'],
        'position_size_pct': backtest_info['parameters']['position_size_pct'],
        
        # Performance metrics
        'initial_capital': backtest_info['initial_capital'],
        'final_capital': backtest_info['final_capital'],
        'total_return_pct': metrics['total_return_pct'],
        'total_trades': metrics['total_trades'],
        'winning_trades': metrics['winning_trades'],
        'losing_trades': metrics['losing_trades'],
        'win_rate_pct': metrics['win_rate_pct'],
        'profit_factor': metrics['profit_factor'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'sortino_ratio': metrics['sortino_ratio'],
        'max_drawdown_pct': metrics['max_drawdown_pct'],
        'avg_trade_pnl': metrics['avg_trade_pnl'],
        'best_trade_pnl': metrics['best_trade_pnl'],
        'worst_trade_pnl': metrics['worst_trade_pnl'],
        'max_consecutive_losses': metrics['max_consecutive_losses'],
        'avg_trade_duration_min': metrics['avg_trade_duration_minutes'],
        
        # Signal analysis
        'total_bars': detailed['total_bars'],
        'indicators_ready_bars': detailed['indicators_ready_bars'],
        'buy_signals': detailed['buy_signals'],
        'sell_signals': detailed['sell_signals'],
        'macd_bullish_crossovers': detailed['macd_bullish_crossovers'],
        'macd_bearish_crossovers': detailed['macd_bearish_crossovers'],
        'signal_filter_efficiency_pct': round((1 - (detailed['buy_signals'] + detailed['sell_signals']) / (detailed['macd_bullish_crossovers'] + detailed['macd_bearish_crossovers'])) * 100, 1),
        
        # Market conditions
        'avg_rsi': detailed['average_rsi'],
        'time_above_ema_pct': detailed['time_above_ema50'],
        'rsi_overbought_time_pct': detailed['rsi_overbought_time'],
        'rsi_oversold_time_pct': detailed['rsi_oversold_time'],
        
        # Notes
        'notes': 'Baseline MACD+RSI strategy with EMA50 trend filter'
    }
    
    return summary_row

def save_strategy_comparison():
    """Save or append strategy results to comparison CSV."""
    
    # Create summary row
    summary_row = create_strategy_summary_row()
    
    # Path for comparison CSV
    comparison_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/strategy_comparison.csv'
    comparison_path = Path(comparison_file)
    
    # Load existing data or create new DataFrame
    if comparison_path.exists():
        print(f"📄 Loading existing comparison file...")
        df_existing = pd.read_csv(comparison_file)
        
        # Check if this exact configuration already exists
        config_match = (
            (df_existing['strategy_name'] == summary_row['strategy_name']) &
            (df_existing['macd_fast'] == summary_row['macd_fast']) &
            (df_existing['macd_slow'] == summary_row['macd_slow']) &
            (df_existing['macd_signal'] == summary_row['macd_signal']) &
            (df_existing['rsi_period'] == summary_row['rsi_period']) &
            (df_existing['ema_period'] == summary_row['ema_period']) &
            (df_existing['take_profit_pct'] == summary_row['take_profit_pct']) &
            (df_existing['stop_loss_pct'] == summary_row['stop_loss_pct'])
        )
        
        if config_match.any():
            print(f"⚠️  Similar configuration found, updating row...")
            df_existing.loc[config_match, :] = pd.DataFrame([summary_row]).values
            df_comparison = df_existing
        else:
            print(f"➕ Adding new configuration...")
            df_comparison = pd.concat([df_existing, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        print(f"🆕 Creating new comparison file...")
        df_comparison = pd.DataFrame([summary_row])
    
    # Save to CSV
    df_comparison.to_csv(comparison_file, index=False)
    
    print(f"💾 Saved to: {comparison_file}")
    print(f"📊 Total configurations: {len(df_comparison)}")
    
    return comparison_file, df_comparison

def display_comparison_summary(df):
    """Display a summary of the comparison data."""
    
    print(f"\n📋 Strategy Comparison Summary:")
    print(f"=" * 60)
    
    # Key columns for comparison
    display_cols = [
        'strategy_name', 'test_date', 
        'total_return_pct', 'win_rate_pct', 'total_trades',
        'profit_factor', 'sharpe_ratio', 'max_drawdown_pct',
        'macd_fast', 'macd_slow', 'rsi_period', 'ema_period',
        'take_profit_pct', 'stop_loss_pct'
    ]
    
    # Filter to available columns
    available_cols = [col for col in display_cols if col in df.columns]
    
    print(df[available_cols].round(2).to_string(index=False))
    
    # Best performing strategies
    if len(df) > 1:
        print(f"\n🏆 Best Performers:")
        print(f"   Highest Return: {df.loc[df['total_return_pct'].idxmax(), 'strategy_name']} ({df['total_return_pct'].max():.2f}%)")
        print(f"   Highest Win Rate: {df.loc[df['win_rate_pct'].idxmax(), 'strategy_name']} ({df['win_rate_pct'].max():.2f}%)")
        print(f"   Best Sharpe: {df.loc[df['sharpe_ratio'].idxmax(), 'strategy_name']} ({df['sharpe_ratio'].max():.2f})")
        print(f"   Lowest Drawdown: {df.loc[df['max_drawdown_pct'].idxmin(), 'strategy_name']} ({df['max_drawdown_pct'].min():.2f}%)")

def main():
    """Main execution function."""
    
    print("🚀 Creating strategy comparison CSV...")
    
    # Create and save comparison
    comparison_file, df_comparison = save_strategy_comparison()
    
    # Display summary
    display_comparison_summary(df_comparison)
    
    print(f"\n✅ Strategy comparison file ready!")
    print(f"📂 File: {comparison_file}")
    print(f"💡 Use this file to compare results when testing different parameters")
    
    # Show CSV structure
    print(f"\n📊 CSV Structure ({len(df_comparison.columns)} columns):")
    for i, col in enumerate(df_comparison.columns, 1):
        print(f"   {i:2d}. {col}")

if __name__ == "__main__":
    main()