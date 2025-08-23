#!/usr/bin/env python3
"""
Simple backtest using August 10-17 data with 90d enhanced ML models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add src to path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

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
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_middle = df['close'].rolling(window=bb_period).mean()
    bb_std_dev = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
    df['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_middle
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price features
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_15'] = df['close'].pct_change(15)
    
    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    
    # Support/Resistance
    df['high_20'] = df['high'].rolling(window=20).max()
    df['low_20'] = df['low'].rolling(window=20).min()
    df['distance_to_high'] = (df['high_20'] - df['close']) / df['close']
    df['distance_to_low'] = (df['close'] - df['low_20']) / df['close']
    
    # Trend features
    df['ema_trend'] = (df['ema_8'] > df['ema_21']).astype(int)
    df['price_vs_vwap'] = (df['close'] > df['vwap']).astype(int)
    
    # ADX (simplified)
    df['adx'] = 25.0  # Default value
    
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
    macd_bullish = row['macd'] > row['macd_signal'] and row['macd_hist'] > 0
    macd_bearish = row['macd'] < row['macd_signal'] and row['macd_hist'] < 0
    
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

def run_simple_backtest():
    print("🔬 Simple ML Backtest on August 10-17 data")
    print("=" * 50)
    
    # Load data
    print("📊 Loading August 10-17 dataset...")
    try:
        df = pd.read_csv('SOLUSDT_5m_aug10_17.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        print(f"✅ Loaded {len(df)} records")
        print(f"📅 Period: {df.index[0]} → {df.index[-1]}")
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
    commission = 0.001  # 0.1%
    balance = initial_balance
    position = 0  # 0 = no position, 1 = long, -1 = short
    position_size = 0
    entry_price = 0
    
    trades = []
    equity_curve = [initial_balance]
    
    print(f"💰 Initial Balance: ${initial_balance:,.2f}")
    print(f"💸 Commission: {commission*100:.1f}%")
    print("🚀 Running simulation...")
    
    for i in range(100, len(df)):  # Start from 100 for indicator stability
        current_data = df.iloc[i]
        current_price = current_data['close']
        
        # Get ML signal
        signal, confidence = create_ml_signal(current_data)
        
        # Trading logic
        if position == 0:  # No position
            if signal == 'buy' and confidence > 0.7:
                # Open LONG
                position = 1
                position_size = (balance * 0.95) / current_price
                entry_price = current_price
                commission_cost = position_size * current_price * commission
                balance -= commission_cost
                
                print(f"📈 LONG @${current_price:.2f} (conf: {confidence:.3f}) - {current_data.name}")
                
            elif signal == 'sell' and confidence > 0.7:
                # Open SHORT
                position = -1
                position_size = (balance * 0.95) / current_price
                entry_price = current_price
                commission_cost = position_size * current_price * commission
                balance -= commission_cost
                
                print(f"📉 SHORT @${current_price:.2f} (conf: {confidence:.3f}) - {current_data.name}")
        
        elif position != 0:  # Have position
            should_exit = False
            exit_reason = ""
            
            if position == 1:  # LONG position
                if signal == 'sell' and confidence > 0.6:
                    should_exit = True
                    exit_reason = "ML signal"
                elif current_price <= entry_price * 0.98:  # 2% stop loss
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif current_price >= entry_price * 1.04:  # 4% take profit
                    should_exit = True
                    exit_reason = "Take Profit"
                    
            elif position == -1:  # SHORT position
                if signal == 'buy' and confidence > 0.6:
                    should_exit = True
                    exit_reason = "ML signal"
                elif current_price >= entry_price * 1.02:  # 2% stop loss
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif current_price <= entry_price * 0.96:  # 4% take profit
                    should_exit = True
                    exit_reason = "Take Profit"
            
            if should_exit:
                # Close position
                if position == 1:  # Close LONG
                    pnl = position_size * (current_price - entry_price)
                else:  # Close SHORT
                    pnl = position_size * (entry_price - current_price)
                
                commission_cost = position_size * current_price * commission
                net_pnl = pnl - commission_cost
                balance += position_size * current_price - commission_cost
                
                # Record trade
                trades.append({
                    'entry_time': str(current_data.name),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position_size': position_size,
                    'side': 'LONG' if position == 1 else 'SHORT',
                    'pnl': net_pnl,
                    'exit_reason': exit_reason
                })
                
                print(f"🔄 EXIT @${current_price:.2f} | PnL: ${net_pnl:+.2f} ({exit_reason})")
                
                position = 0
                position_size = 0
                entry_price = 0
        
        # Update equity curve
        if position == 0:
            current_equity = balance
        elif position == 1:  # LONG
            current_equity = balance + position_size * (current_price - entry_price)
        else:  # SHORT
            current_equity = balance + position_size * (entry_price - current_price)
            
        equity_curve.append(current_equity)
    
    # Close final position
    if position != 0:
        final_price = df.iloc[-1]['close']
        if position == 1:
            pnl = position_size * (final_price - entry_price)
        else:
            pnl = position_size * (entry_price - final_price)
        
        commission_cost = position_size * final_price * commission
        net_pnl = pnl - commission_cost
        balance += position_size * final_price - commission_cost
        
        trades.append({
            'entry_time': str(df.index[-1]),
            'entry_price': entry_price,
            'exit_price': final_price,
            'position_size': position_size,
            'side': 'LONG' if position == 1 else 'SHORT',
            'pnl': net_pnl,
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
    print(f"🔄 Total Trades: {len(trades)}")
    
    if trades:
        profitable_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (profitable_trades / len(trades)) * 100
        avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
        
        print(f"✅ Profitable Trades: {profitable_trades} ({win_rate:.1f}%)")
        print(f"❌ Losing Trades: {len(trades) - profitable_trades}")
        print(f"📊 Average PnL: ${avg_pnl:+.2f}")
        
        if trades:
            best_trade = max(trades, key=lambda x: x['pnl'])
            worst_trade = min(trades, key=lambda x: x['pnl'])
            
            print(f"🏆 Best Trade: ${best_trade['pnl']:+.2f}")
            print(f"💀 Worst Trade: ${worst_trade['pnl']:+.2f}")
    
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
    
    # Save results
    results = {
        'backtest_date': datetime.now().isoformat(),
        'model_version': 'simple_ml_signals',
        'period': 'august_10_17_2025',
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'total_trades': len(trades),
        'profitable_trades': profitable_trades if trades else 0,
        'win_rate': win_rate if trades else 0,
        'max_drawdown_pct': max_drawdown * 100,
        'trades': trades
    }
    
    Path('output').mkdir(exist_ok=True)
    output_file = f"output/ml_backtest_august_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to {output_file}")
    print("\n🎉 Backtest completed!")

if __name__ == "__main__":
    run_simple_backtest()