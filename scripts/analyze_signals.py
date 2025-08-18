#!/usr/bin/env python3
"""Diagnostic script to analyze why no signals were generated on 12 August data."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_blocks import DataBlockManager
from src.feature_engine.technical_indicators import TechnicalIndicatorCalculator
from src.utils.types import TimeFrame

def analyze_signal_failures(block_id: str = "august_12_single_day"):
    """Analyze why signals failed for the given block."""
    print(f"🔍 Analyzing signal failures for block: {block_id}")
    print("=" * 60)
    
    # Load data
    manager = DataBlockManager()
    if block_id not in manager.blocks_registry:
        print(f"❌ Block '{block_id}' not found")
        return
    
    data = manager.load_block(block_id, ["5m"])
    if "5m" not in data:
        print("❌ No 5m data available")
        return
    
    df_5m = data["5m"]
    print(f"📊 Loaded {len(df_5m)} 5-minute candles")
    print(f"📈 Price range: ${df_5m['low'].min():.2f} - ${df_5m['high'].max():.2f}")
    print(f"📈 Price movement: {((df_5m['close'].iloc[-1] / df_5m['close'].iloc[0]) - 1) * 100:.2f}%")
    print()
    
    # Calculate indicators
    print("🔧 Calculating technical indicators...")
    calc = TechnicalIndicatorCalculator()
    
    # Calculate key indicators
    df_5m['ema_8'] = df_5m['close'].ewm(span=8).mean()
    df_5m['ema_21'] = df_5m['close'].ewm(span=21).mean()
    df_5m['ema_55'] = df_5m['close'].ewm(span=55).mean()
    
    # RSI
    delta = df_5m['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_5m['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df_5m['close'].ewm(span=12).mean()
    ema_26 = df_5m['close'].ewm(span=26).mean()
    df_5m['macd'] = ema_12 - ema_26
    df_5m['macd_signal'] = df_5m['macd'].ewm(span=9).mean()
    
    # Volume ratio
    df_5m['volume_sma'] = df_5m['volume'].rolling(window=20).mean()
    df_5m['volume_ratio'] = df_5m['volume'] / df_5m['volume_sma']
    
    # ATR
    df_5m['tr'] = np.maximum(
        df_5m['high'] - df_5m['low'],
        np.maximum(
            abs(df_5m['high'] - df_5m['close'].shift(1)),
            abs(df_5m['low'] - df_5m['close'].shift(1))
        )
    )
    df_5m['atr'] = df_5m['tr'].rolling(window=14).mean()
    df_5m['atr_ratio'] = df_5m['atr'] / df_5m['close']
    
    # ADX (simplified)
    df_5m['adx'] = 25.0  # Placeholder - assume sufficient trend strength
    
    # VWAP
    df_5m['vwap'] = (df_5m['close'] * df_5m['volume']).cumsum() / df_5m['volume'].cumsum()
    df_5m['price_vs_vwap'] = (df_5m['close'] - df_5m['vwap']) / df_5m['vwap']
    
    # Bollinger Bands position (simplified)
    bb_middle = df_5m['close'].rolling(window=20).mean()
    bb_std = df_5m['close'].rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    df_5m['bb_position'] = (df_5m['close'] - bb_lower) / (bb_upper - bb_lower)
    
    print("✅ Indicators calculated")
    print()
    
    # Strategy parameters
    strategy_params = {
        'min_confidence': 0.15,
        'min_volume_ratio': 1.2,
        'min_adx': 20.0,
    }
    
    # Analyze each candle
    print("🔍 Signal Analysis Summary")
    print("-" * 60)
    
    total_candles = 0
    adx_failures = 0
    volume_failures = 0
    confidence_failures = 0
    trend_alignment_failures = 0
    signal_opportunities = 0
    potential_signals = []
    
    for i in range(55, len(df_5m)):  # Start after EMAs stabilize
        row = df_5m.iloc[i]
        total_candles += 1
        
        # Check basic filters
        adx_ok = row['adx'] >= strategy_params['min_adx']
        volume_ok = row['volume_ratio'] >= strategy_params['min_volume_ratio']
        
        if not adx_ok:
            adx_failures += 1
        if not volume_ok:
            volume_failures += 1
        
        if not (adx_ok and volume_ok):
            continue
        
        # Check bullish conditions
        bullish_conditions = [
            row['ema_8'] > row['ema_21'],
            row['ema_21'] > row['ema_55'],
            row['close'] > row['ema_8'],
            row['rsi'] > 40 and row['rsi'] < 75,
            row['macd'] > row['macd_signal'],
            row['price_vs_vwap'] > -0.01,
            row['volume_ratio'] > 1.2,
            row['bb_position'] > 0.2 and row['bb_position'] < 0.8,
        ]
        
        bullish_score = sum(bullish_conditions) / len(bullish_conditions)
        
        # Check bearish conditions
        bearish_conditions = [
            row['ema_8'] < row['ema_21'],
            row['ema_21'] < row['ema_55'],
            row['close'] < row['ema_8'],
            row['rsi'] > 25 and row['rsi'] < 60,
            row['macd'] < row['macd_signal'],
            row['price_vs_vwap'] < 0.01,
            row['volume_ratio'] > 1.2,
            row['bb_position'] > 0.2 and row['bb_position'] < 0.8,
        ]
        
        bearish_score = sum(bearish_conditions) / len(bearish_conditions)
        
        # Check signal generation
        signal_type = None
        confidence = 0.0
        
        if bullish_score > 0.6 and bullish_score > bearish_score + 0.2:
            signal_type = "BUY"
            confidence = bullish_score
        elif bearish_score > 0.6 and bearish_score > bullish_score + 0.2:
            signal_type = "SELL"
            confidence = bearish_score
        
        if signal_type and confidence >= strategy_params['min_confidence']:
            signal_opportunities += 1
            potential_signals.append({
                'timestamp': row['timestamp'],
                'type': signal_type,
                'confidence': confidence,
                'price': row['close'],
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'volume_ratio': row['volume_ratio'],
                'rsi': row['rsi'],
                'ema_alignment': row['ema_8'] > row['ema_21'] > row['ema_55']
            })
        elif signal_type:
            confidence_failures += 1
        else:
            trend_alignment_failures += 1
    
    # Print summary
    print(f"📊 Total candles analyzed: {total_candles}")
    print(f"❌ ADX filter failures: {adx_failures} ({adx_failures/total_candles*100:.1f}%)")
    print(f"❌ Volume filter failures: {volume_failures} ({volume_failures/total_candles*100:.1f}%)")
    print(f"❌ Trend alignment failures: {trend_alignment_failures}")
    print(f"❌ Confidence too low: {confidence_failures}")
    print(f"✅ Valid signal opportunities: {signal_opportunities}")
    print()
    
    if potential_signals:
        print("🎯 Potential Signals Found:")
        print("-" * 80)
        for signal in potential_signals[:10]:  # Show first 10
            print(f"⏰ {signal['timestamp']} | {signal['type']} | "
                  f"Price: ${signal['price']:.2f} | Confidence: {signal['confidence']:.3f} | "
                  f"Vol: {signal['volume_ratio']:.1f}x | RSI: {signal['rsi']:.1f}")
    
    # Detailed analysis of key periods
    print("\n🔍 Critical Period Analysis")
    print("-" * 60)
    
    # Find the biggest price moves
    df_5m['price_change'] = df_5m['close'].pct_change()
    big_moves = df_5m[abs(df_5m['price_change']) > 0.01].copy()  # >1% moves
    
    if len(big_moves) > 0:
        print(f"📈 Found {len(big_moves)} candles with >1% price movement:")
        for _, move in big_moves.head(5).iterrows():
            print(f"⏰ {move['timestamp']} | "
                  f"Price: ${move['close']:.2f} | "
                  f"Change: {move['price_change']*100:.2f}% | "
                  f"Vol: {move['volume_ratio']:.1f}x | "
                  f"EMA8: ${move['ema_8']:.2f} | "
                  f"EMA21: ${move['ema_21']:.2f} | "
                  f"RSI: {move['rsi']:.1f}")
    
    # Recommendations
    print("\n💡 Recommendations for Algorithm Improvement")
    print("-" * 60)
    
    recommendations = []
    
    if volume_failures > total_candles * 0.5:
        recommendations.append("🔧 Lower volume_ratio threshold (current: 1.2x → try 1.0x)")
    
    if confidence_failures > signal_opportunities:
        recommendations.append("🔧 Lower min_confidence threshold (current: 0.15 → try 0.10)")
    
    if adx_failures > total_candles * 0.3:
        recommendations.append("🔧 Lower min_adx threshold (current: 20 → try 15)")
    
    if trend_alignment_failures > total_candles * 0.7:
        recommendations.append("🔧 Relax trend alignment requirements (allow 6/8 conditions instead of strict thresholds)")
    
    if not potential_signals:
        recommendations.extend([
            "🔧 Add breakout detection for strong moves",
            "🔧 Implement momentum-based signals",
            "🔧 Consider counter-trend strategies for mean reversion",
        ])
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✅ Algorithm parameters seem reasonable for this dataset")

if __name__ == "__main__":
    analyze_signal_failures()