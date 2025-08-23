#!/usr/bin/env python3
"""
Realistic backtest using August 10-17 data with proper position sizing
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

def calculate_technical_indicators(df):
    """Calculate basic technical indicators for the dataset."""
    # EMA indicators
    df['ema_8'] = df['close'].ewm(span=8).mean()
    df['ema_13'] = df['close'].ewm(span=13).mean() 
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_34'] = df['close'].ewm(span=34).mean()
    df['ema_55'] = df['close'].ewm(span=55).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volume ratio
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    return df

def create_ml_signal(row):
    """Simple ML-like signal generation using technical indicators."""
    
    # Trend alignment
    ema_bullish = row['ema_8'] > row['ema_13'] > row['ema_21']
    ema_bearish = row['ema_8'] < row['ema_13'] < row['ema_21']
    
    # RSI conditions
    rsi_oversold = row['rsi'] < 35
    rsi_overbought = row['rsi'] > 65
    
    # MACD signal
    macd_bullish = row['macd'] > row['macd_signal']
    macd_bearish = row['macd'] < row['macd_signal']
    
    # Price vs VWAP
    above_vwap = row['close'] > row['vwap']
    below_vwap = row['close'] < row['vwap']
    
    # Volume confirmation
    volume_high = row['volume_ratio'] > 1.2
    
    # Generate signals
    bullish_signals = sum([ema_bullish, rsi_oversold, macd_bullish, above_vwap, volume_high])
    bearish_signals = sum([ema_bearish, rsi_overbought, macd_bearish, below_vwap, volume_high])
    
    if bullish_signals >= 3:
        return 'buy', 0.8
    elif bearish_signals >= 3:
        return 'sell', 0.8
    else:
        return 'hold', 0.5

def run_realistic_backtest():
    print("🔬 Realistic ML Backtest on August 10-17 data")
    print("=" * 50)
    
    # Load data
    print("📊 Loading August 10-17 dataset...")
    try:
        df = pd.read_csv('SOLUSDT_5m_aug10_17.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        print(f"✅ Loaded {len(df)} records")
        print(f"📅 Period: {df.index[0]} → {df.index[-1]}")
        print(f"💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Calculate indicators
    print("🔧 Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    # Remove NaN values
    df = df.dropna()
    print(f"📊 After indicators: {len(df)} records")
    
    # Backtest settings
    initial_balance = 10000.0
    commission_rate = 0.001  # 0.1%
    max_position_pct = 0.02  # Max 2% of balance per trade (following CLAUDE.md)
    
    balance = initial_balance
    position = 0  # 0 = no position, 1 = long, -1 = short
    position_size_usd = 0
    entry_price = 0
    
    trades = []
    equity_curve = [initial_balance]
    
    print(f"💰 Initial Balance: ${initial_balance:,.2f}")
    print(f"💸 Commission: {commission_rate*100:.1f}%")
    print(f"📊 Max position per trade: {max_position_pct*100:.1f}% of balance")
    print("🚀 Running simulation...")
    
    signals_generated = 0
    trades_executed = 0
    
    for i in range(100, len(df)):  # Start from 100 for indicator stability
        current_data = df.iloc[i]
        current_price = current_data['close']
        
        # Get ML signal
        signal, confidence = create_ml_signal(current_data)
        
        if signal != 'hold':
            signals_generated += 1
        
        # Trading logic
        if position == 0:  # No position
            if signal == 'buy' and confidence > 0.7:
                # Open LONG - fixed position sizing
                position_size_usd = balance * max_position_pct
                position = 1
                entry_price = current_price
                commission_cost = position_size_usd * commission_rate
                balance -= commission_cost
                trades_executed += 1
                
                print(f"📈 LONG ${position_size_usd:.0f} @${current_price:.2f} (conf: {confidence:.3f}) - {current_data.name.strftime('%m-%d %H:%M')}")
                
            elif signal == 'sell' and confidence > 0.7:
                # Open SHORT - fixed position sizing
                position_size_usd = balance * max_position_pct
                position = -1
                entry_price = current_price
                commission_cost = position_size_usd * commission_rate
                balance -= commission_cost
                trades_executed += 1
                
                print(f"📉 SHORT ${position_size_usd:.0f} @${current_price:.2f} (conf: {confidence:.3f}) - {current_data.name.strftime('%m-%d %H:%M')}")
        
        elif position != 0:  # Have position
            should_exit = False
            exit_reason = ""
            
            if position == 1:  # LONG position
                pnl_pct = (current_price - entry_price) / entry_price
                if signal == 'sell' and confidence > 0.6:
                    should_exit = True
                    exit_reason = "ML signal"
                elif pnl_pct <= -0.02:  # 2% stop loss (from CLAUDE.md)
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif pnl_pct >= 0.04:  # 4% take profit
                    should_exit = True
                    exit_reason = "Take Profit"
                    
            elif position == -1:  # SHORT position
                pnl_pct = (entry_price - current_price) / entry_price
                if signal == 'buy' and confidence > 0.6:
                    should_exit = True
                    exit_reason = "ML signal"
                elif pnl_pct <= -0.02:  # 2% stop loss
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif pnl_pct >= 0.04:  # 4% take profit
                    should_exit = True
                    exit_reason = "Take Profit"
            
            if should_exit:
                # Close position with realistic P&L calculation
                if position == 1:  # Close LONG
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # Close SHORT
                    pnl_pct = (entry_price - current_price) / entry_price
                
                gross_pnl = position_size_usd * pnl_pct
                commission_cost = position_size_usd * commission_rate
                net_pnl = gross_pnl - commission_cost
                
                balance += position_size_usd + net_pnl
                
                # Record trade
                trades.append({
                    'entry_time': str(current_data.name),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position_size_usd': position_size_usd,
                    'side': 'LONG' if position == 1 else 'SHORT',
                    'pnl': net_pnl,
                    'pnl_pct': pnl_pct * 100,
                    'exit_reason': exit_reason
                })
                
                print(f"🔄 EXIT @${current_price:.2f} | PnL: ${net_pnl:+.2f} ({pnl_pct*100:+.2f}%) [{exit_reason}]")
                
                position = 0
                position_size_usd = 0
                entry_price = 0
        
        # Update equity curve
        if position == 0:
            current_equity = balance
        else:
            if position == 1:  # LONG
                unrealized_pnl = position_size_usd * ((current_price - entry_price) / entry_price)
            else:  # SHORT
                unrealized_pnl = position_size_usd * ((entry_price - current_price) / entry_price)
            current_equity = balance + unrealized_pnl
            
        equity_curve.append(current_equity)
    
    # Close final position if open
    if position != 0:
        final_price = df.iloc[-1]['close']
        if position == 1:
            pnl_pct = (final_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - final_price) / entry_price
        
        gross_pnl = position_size_usd * pnl_pct
        commission_cost = position_size_usd * commission_rate
        net_pnl = gross_pnl - commission_cost
        balance += position_size_usd + net_pnl
        
        trades.append({
            'entry_time': str(df.index[-1]),
            'entry_price': entry_price,
            'exit_price': final_price,
            'position_size_usd': position_size_usd,
            'side': 'LONG' if position == 1 else 'SHORT',
            'pnl': net_pnl,
            'pnl_pct': pnl_pct * 100,
            'exit_reason': 'End of backtest'
        })
    
    # Results
    final_balance = balance
    total_pnl = final_balance - initial_balance
    total_return_pct = (total_pnl / initial_balance) * 100
    
    print("\n" + "="*50)
    print("📊 BACKTEST RESULTS:")
    print(f"💰 Initial Balance: ${initial_balance:,.2f}")
    print(f"💸 Final Balance: ${final_balance:,.2f}")
    print(f"📈 Total P&L: ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)")
    print(f"📊 Signals Generated: {signals_generated}")
    print(f"🔄 Trades Executed: {trades_executed}")
    print(f"📋 Total Trades: {len(trades)}")
    
    if trades:
        profitable_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (profitable_trades / len(trades)) * 100
        avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
        avg_pnl_pct = sum(t['pnl_pct'] for t in trades) / len(trades)
        
        print(f"✅ Profitable Trades: {profitable_trades} ({win_rate:.1f}%)")
        print(f"❌ Losing Trades: {len(trades) - profitable_trades}")
        print(f"📊 Average PnL: ${avg_pnl:+.2f} ({avg_pnl_pct:+.2f}%)")
        
        if trades:
            best_trade = max(trades, key=lambda x: x['pnl'])
            worst_trade = min(trades, key=lambda x: x['pnl'])
            
            print(f"🏆 Best Trade: ${best_trade['pnl']:+.2f} ({best_trade['pnl_pct']:+.2f}%)")
            print(f"💀 Worst Trade: ${worst_trade['pnl']:+.2f} ({worst_trade['pnl_pct']:+.2f}%)")
    
    # Max drawdown
    max_balance = initial_balance
    max_drawdown = 0
    for equity in equity_curve:
        if equity > max_balance:
            max_balance = equity
        drawdown = (max_balance - equity) / max_balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    print(f"📉 Max Drawdown: {max_drawdown*100:.2f}%")
    
    # Performance metrics
    if len(trades) > 0:
        sharpe_approximation = (total_return_pct / 100) / (max_drawdown + 0.01)  # Rough Sharpe approximation
        print(f"📊 Sharpe Ratio (approx): {sharpe_approximation:.2f}")
    
    # Save results
    results = {
        'backtest_date': datetime.now().isoformat(),
        'model_version': 'realistic_ml_signals',
        'period': 'august_10_17_2025',
        'dataset_records': len(df),
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'signals_generated': signals_generated,
        'trades_executed': trades_executed,
        'total_trades': len(trades),
        'profitable_trades': profitable_trades if trades else 0,
        'win_rate': win_rate if trades else 0,
        'max_drawdown_pct': max_drawdown * 100,
        'max_position_pct': max_position_pct * 100,
        'commission_rate': commission_rate,
        'avg_pnl_per_trade': avg_pnl if trades else 0,
        'trades': trades[:10] if trades else []  # Save only first 10 trades
    }
    
    Path('output').mkdir(exist_ok=True)
    output_file = f"output/realistic_backtest_august_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to {output_file}")
    print("\n🎉 Realistic backtest completed!")

if __name__ == "__main__":
    run_realistic_backtest()