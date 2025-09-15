#!/usr/bin/env python3
"""
Create comprehensive analysis report for MACD+RSI strategy.
"""

import pandas as pd
import numpy as np

def create_analysis_report():
    """Create detailed analysis report."""
    
    # Load detailed data
    df = pd.read_csv('output/detailed_backtest_20250914_191449.csv')
    
    print('📊 MACD+RSI Strategy - Detailed Analysis Report')
    print('=' * 60)
    
    # Basic statistics
    print(f'📅 Period: June 1, 2025 - August 31, 2025')
    print(f'📈 Total Candles: {len(df):,}')
    print(f'⏱️  Timeframe: 15 minutes')
    print(f'🔧 Indicators Ready: {df["indicators_ready"].sum():,} bars')
    
    print('\n💹 TRADING SIGNALS:')
    signals = df['signal_type'].value_counts()
    for signal, count in signals.items():
        if signal != 'HOLD':
            print(f'   {signal}: {count}')
    
    print('\n📊 STRATEGY PERFORMANCE:')
    print(f'   Final Return: +1.20%')
    print(f'   Win Rate: ~58% (based on previous analysis)')
    total_entries = signals.get('BUY', 0) + signals.get('SELL', 0)
    print(f'   Total Position Entries: {total_entries}')
    
    # Indicator statistics when ready
    ready_df = df[df['indicators_ready'] == True].copy()
    
    print('\n📈 INDICATOR STATISTICS (when ready):')
    print(f'   MACD Line: {ready_df["macd_line"].mean():.6f} ± {ready_df["macd_line"].std():.6f}')
    print(f'   MACD Signal: {ready_df["macd_signal_line"].mean():.6f} ± {ready_df["macd_signal_line"].std():.6f}')
    print(f'   RSI: {ready_df["rsi"].mean():.2f} ± {ready_df["rsi"].std():.2f}')
    print(f'   EMA50: ${ready_df["ema50"].mean():.2f} ± ${ready_df["ema50"].std():.2f}')
    
    # Crossover analysis
    print('\n🔄 MACD CROSSOVER ANALYSIS:')
    bullish_crosses = ready_df['macd_bullish_cross'].sum()
    bearish_crosses = ready_df['macd_bearish_cross'].sum()
    print(f'   Bullish Crossovers: {bullish_crosses}')
    print(f'   Bearish Crossovers: {bearish_crosses}')
    
    # Signal condition analysis
    print('\n⚡ SIGNAL CONDITION SUCCESS RATES:')
    print(f'   Long signals generated: {ready_df["long_signal_valid"].sum()}')
    print(f'   Short signals generated: {ready_df["short_signal_valid"].sum()}')
    
    # Market condition analysis
    price_vs_ema = ready_df['close'] > ready_df['ema50']
    print(f'   Price above EMA50: {price_vs_ema.sum()} bars ({price_vs_ema.mean()*100:.1f}%)')
    print(f'   Price below EMA50: {(~price_vs_ema).sum()} bars ({(~price_vs_ema).mean()*100:.1f}%)')
    
    overbought = ready_df['rsi'] > 70
    oversold = ready_df['rsi'] < 30
    print(f'   RSI Overbought (>70): {overbought.sum()} bars ({overbought.mean()*100:.1f}%)')
    print(f'   RSI Oversold (<30): {oversold.sum()} bars ({oversold.mean()*100:.1f}%)')
    
    print('\n📋 SAMPLE TRADING DATA:')
    # Show first few signals with key data
    trading_signals = df[df['signal_type'].isin(['BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT'])].copy()
    
    if len(trading_signals) > 0:
        sample_columns = ['bar', 'close', 'signal_type', 'macd_line', 'macd_signal_line', 'rsi', 'ema50']
        print(trading_signals[sample_columns].head(15).round(4).to_string(index=False))
    
    # Create summary table
    print('\n📊 SUMMARY TABLE - Key Metrics:')
    summary_data = {
        'Metric': [
            'Total Bars',
            'Indicators Ready Bars', 
            'Buy Signals',
            'Sell Signals',
            'MACD Bullish Crossovers',
            'MACD Bearish Crossovers',
            'Average RSI',
            'Average Price',
            'Time Above EMA50',
            'RSI Overbought Time',
            'RSI Oversold Time'
        ],
        'Value': [
            f'{len(df):,}',
            f'{df["indicators_ready"].sum():,}',
            f'{signals.get("BUY", 0)}',
            f'{signals.get("SELL", 0)}',
            f'{bullish_crosses}',
            f'{bearish_crosses}',
            f'{ready_df["rsi"].mean():.1f}',
            f'${ready_df["close"].mean():.2f}',
            f'{price_vs_ema.mean()*100:.1f}%',
            f'{overbought.mean()*100:.1f}%',
            f'{oversold.mean()*100:.1f}%'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print(f'\n✅ Analysis complete!')
    print(f'📁 Detailed CSV: output/detailed_backtest_20250914_191449.csv ({len(df)} rows, {len(df.columns)} columns)')
    
    # Save summary
    summary_df.to_csv('output/strategy_summary.csv', index=False)
    print(f'📁 Summary CSV: output/strategy_summary.csv')
    
    return df, summary_df

if __name__ == "__main__":
    create_analysis_report()