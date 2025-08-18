#!/usr/bin/env python3
"""Debug script for aggressive momentum strategy."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_blocks import DataBlockManager
from src.utils.types import TimeFrame

def debug_aggressive_signals():
    """Debug why aggressive strategy still produces no signals."""
    print("🔍 Debugging Aggressive Momentum Strategy")
    print("=" * 60)
    
    # Load data
    manager = DataBlockManager()
    data = manager.load_block("august_12_single_day", ["5m"])
    df = data["5m"].copy()
    
    print(f"📊 Loaded {len(df)} candles")
    print(f"📈 Price movement: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    print()
    
    # Add momentum calculations
    df['price_change_1'] = df['close'].pct_change()
    df['price_change_3'] = df['close'].pct_change(periods=3)
    df['volume_ma'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Basic indicators
    df['ema_8'] = df['close'].ewm(span=8).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_55'] = df['close'].ewm(span=55).mean()
    
    # Strategy parameters
    params = {
        'min_confidence': 0.05,
        'min_volume_ratio': 0.4,
        'min_adx': 8.0,
        'immediate_entry_threshold': 0.02,
        'breakout_threshold': 0.008,
    }
    
    print("🔧 Strategy Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # Analyze each candle
    total_analyzed = 0
    immediate_entries = 0
    basic_filter_failures = 0
    confidence_failures = 0
    potential_signals = []
    
    for i in range(55, len(df)):  # Start after EMA stabilization
        row = df.iloc[i]
        total_analyzed += 1
        
        # Check immediate entry condition
        immediate_entry = abs(row['price_change_1']) >= params['immediate_entry_threshold']
        if immediate_entry:
            immediate_entries += 1
            potential_signals.append({
                'timestamp': row['timestamp'],
                'type': 'IMMEDIATE',
                'price': row['close'],
                'change': row['price_change_1'] * 100,
                'volume_ratio': row.get('volume_ratio', 1.0),
                'reasoning': f"Immediate entry: {row['price_change_1']*100:.2f}% move"
            })
            continue
        
        # Basic filtering
        adx_ok = True  # Assume ADX > 8 (very low threshold)
        volume_ok = row.get('volume_ratio', 1.0) >= params['min_volume_ratio']
        
        if not (adx_ok and volume_ok):
            basic_filter_failures += 1
            continue
        
        # Signal scoring
        bullish_score = 0.0
        reasons = []
        
        # Trend alignment (30%)
        if row['close'] > row['ema_21']:
            bullish_score += 0.15
            reasons.append("price>EMA21")
        
        if row['ema_8'] > row['ema_21']:
            bullish_score += 0.15
            reasons.append("EMA8>EMA21")
        
        # Momentum (40%)
        if row['price_change_1'] > 0.002:
            bullish_score += 0.10
            reasons.append("pos_momentum")
        
        if row['price_change_3'] > 0.005:
            bullish_score += 0.10
            reasons.append("strong_momentum_3")
        
        if row['price_change_1'] > params['breakout_threshold']:
            bullish_score += 0.10
            reasons.append("breakout")
        
        # Volume (20%)
        if row.get('volume_ratio', 1.0) > 2.0:
            bullish_score += 0.15
            reasons.append("volume_spike")
        elif row.get('volume_ratio', 1.0) > 1.0:
            bullish_score += 0.05
            reasons.append("good_volume")
        
        # RSI placeholder (10%)
        bullish_score += 0.05  # Assume RSI is OK
        reasons.append("rsi_ok")
        
        # Check confidence threshold
        if bullish_score >= params['min_confidence']:
            potential_signals.append({
                'timestamp': row['timestamp'],
                'type': 'BUY',
                'price': row['close'],
                'confidence': bullish_score,
                'change': row['price_change_1'] * 100,
                'volume_ratio': row.get('volume_ratio', 1.0),
                'reasoning': f"Bullish: {bullish_score:.3f}, {','.join(reasons[:3])}"
            })
        else:
            confidence_failures += 1
    
    # Print analysis
    print("📊 Signal Generation Analysis:")
    print("-" * 50)
    print(f"Total candles analyzed: {total_analyzed}")
    print(f"Immediate entries found: {immediate_entries}")
    print(f"Basic filter failures: {basic_filter_failures}")
    print(f"Confidence too low: {confidence_failures}")
    print(f"Potential signals: {len(potential_signals)}")
    print()
    
    if potential_signals:
        print("🎯 Found Potential Signals:")
        print("-" * 70)
        for i, signal in enumerate(potential_signals[:10], 1):
            print(f"{i:2d}. {signal['timestamp']} | {signal['type']} | "
                  f"${signal['price']:.2f} | Change: {signal['change']:+.2f}% | "
                  f"Vol: {signal['volume_ratio']:.1f}x")
            print(f"    {signal['reasoning']}")
            print()
    
    # Find the biggest moves to see what we're missing
    print("📈 Biggest Price Movements:")
    print("-" * 50)
    
    big_moves = df[abs(df['price_change_1']) > 0.005].copy()  # >0.5% moves
    if len(big_moves) > 0:
        for _, move in big_moves.head(10).iterrows():
            volume_ratio = move.get('volume_ratio', 1.0)
            print(f"⏰ {move['timestamp']} | "
                  f"${move['close']:.2f} | "
                  f"Change: {move['price_change_1']*100:+.2f}% | "
                  f"Vol: {volume_ratio:.1f}x | "
                  f"EMA21: ${move['ema_21']:.2f}")
    
    # Recommendations
    print("\n💡 Debug Recommendations:")
    print("-" * 50)
    
    if immediate_entries == 0:
        max_move = abs(df['price_change_1']).max() * 100
        print(f"🔧 No immediate entries found. Max 5m move was {max_move:.2f}%")
        print(f"    Consider lowering immediate_entry_threshold from 2.0% to {max_move * 0.7:.1f}%")
    
    if len(potential_signals) == 0:
        print("🔧 No signals at all! Try:")
        print("    - Lower min_confidence to 0.02")
        print("    - Lower min_volume_ratio to 0.2")
        print("    - Remove trend alignment requirements")
    
    # Test with even more aggressive parameters
    print("\n🚀 Testing ULTRA-AGGRESSIVE Parameters:")
    print("-" * 50)
    
    ultra_signals = 0
    for i in range(55, len(df)):
        row = df.iloc[i]
        
        # Ultra simple conditions
        simple_bullish = (
            row['close'] > row['ema_21'] and  # Basic trend
            row['price_change_1'] > 0.001     # >0.1% momentum
        )
        
        simple_bearish = (
            row['close'] < row['ema_21'] and  # Basic trend
            row['price_change_1'] < -0.001    # <-0.1% momentum
        )
        
        if simple_bullish or simple_bearish:
            ultra_signals += 1
    
    print(f"Ultra-aggressive signals found: {ultra_signals}")
    
    if ultra_signals == 0:
        print("❌ Even ultra-aggressive parameters produce no signals!")
        print("🔍 This suggests a fundamental issue with signal logic or data processing.")
    else:
        print(f"✅ Ultra-aggressive would generate {ultra_signals} signals")
        print("💡 The issue is with parameter calibration, not fundamental logic.")

if __name__ == "__main__":
    debug_aggressive_signals()