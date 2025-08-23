#!/usr/bin/env python3
"""
Quick backtest script with progress logging
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def run_quick_backtest():
    print("🚀 Quick Backtest Started")
    print("=" * 50)
    
    # Load data
    print("📊 Loading 5m data...")
    df = pd.read_csv('data/SOLUSDT_5m_real_2025-08-10_to_2025-08-17.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Filter to single day
    target_date = '2025-08-16'
    df_day = df[df.index.date.astype(str) == target_date]
    
    print(f"📅 Testing on {target_date}: {len(df_day)} candles")
    print(f"💰 Initial balance: $10,000")
    
    if len(df_day) == 0:
        print("❌ No data for target date")
        return
    
    # Simple backtest simulation
    balance = 10000
    position = 0
    trades = 0
    start_time = time.time()
    
    for i, (timestamp, row) in enumerate(df_day.iterrows()):
        # Progress every 50 candles
        if i % 50 == 0:
            elapsed = time.time() - start_time
            progress = i / len(df_day) * 100
            remaining = (len(df_day) - i) * (elapsed / max(i, 1))
            
            print(f"📈 Progress: {i}/{len(df_day)} ({progress:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"ETA: {remaining:.1f}s | "
                  f"Trades: {trades} | "
                  f"Balance: ${balance:.2f}")
        
        # Simple moving average strategy
        if i >= 20:  # Need some history
            recent_closes = df_day['close'].iloc[i-20:i].values
            sma_short = np.mean(recent_closes[-5:])
            sma_long = np.mean(recent_closes[-20:])
            
            # Buy signal
            if sma_short > sma_long and position == 0:
                position = balance / row['close']
                balance = 0
                trades += 1
                print(f"🟢 BUY at ${row['close']:.2f} | Position: {position:.4f} SOL")
            
            # Sell signal
            elif sma_short < sma_long and position > 0:
                balance = position * row['close']
                position = 0
                trades += 1
                print(f"🔴 SELL at ${row['close']:.2f} | Balance: ${balance:.2f}")
    
    # Final results
    final_value = balance + (position * df_day['close'].iloc[-1])
    profit = final_value - 10000
    
    print("\n" + "=" * 50)
    print("📊 BACKTEST RESULTS")
    print("=" * 50)
    print(f"💰 Final Value: ${final_value:.2f}")
    print(f"📈 Profit/Loss: ${profit:.2f} ({profit/10000*100:.2f}%)")
    print(f"🔄 Total Trades: {trades}")
    print(f"⏱️  Total Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    run_quick_backtest()