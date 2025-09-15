#!/usr/bin/env python3
"""
Диагностика Breakout Volume Strategy

Анализирует почему стратегия не генерирует сделки.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append('.')

from examples.strategies.breakout_volume_strategy import BreakoutVolumeStrategy
from modules.backtester import MarketData, Position
from datetime import datetime


def analyze_data():
    """Анализ данных для понимания поведения стратегии"""
    
    print("🔍 Analyzing Breakout Strategy Conditions...")
    
    # Загружаем данные
    data_file = Path("data/raw/SOLUSDT_5m_20250807_20250906.csv")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"📊 Data: {len(df)} candles")
    print(f"📈 Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"📊 Avg volume: {df['volume'].mean():.0f}")
    
    # Создаём стратегию
    strategy = BreakoutVolumeStrategy()
    position = Position(
        direction="NONE", quantity=0, entry_price=0,
        entry_time=datetime.now(), unrealized_pnl=0, duration_minutes=0
    )
    
    # Анализируем первые 500 свечей с детальным логированием
    test_data = df.head(500)
    
    flat_count = 0
    breakout_attempts = 0
    volume_failures = 0
    trend_failures = 0
    price_movement_count = 0
    
    print(f"\n🔍 Analyzing first {len(test_data)} candles...")
    
    for i, row in test_data.iterrows():
        market_data = MarketData(
            timestamp=row['timestamp'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        
        # Процессим свечу в стратегии
        signal = strategy.on_bar(market_data, position)
        
        # После накопления достаточной истории начинаем анализ
        if len(strategy.state['history']) >= 60:  # После часа данных
            
            # Проверим условия для флэта вручную
            flat_data = strategy._detect_flat()
            if flat_data:
                flat_count += 1
                
                # Детальная проверка пробоя
                current_bar = strategy.state['history'][-1]
                avg_volume = current_bar.get('avg_volume', 0)
                min_volume_needed = avg_volume * strategy.parameters['breakout_volume_mult']
                
                # Проверим условия пробоя вручную
                resistance_broken = market_data.close > flat_data['resistance']
                support_broken = market_data.close < flat_data['support']
                volume_ok = market_data.volume >= min_volume_needed
                
                if resistance_broken or support_broken:
                    price_movement_count += 1
                    if price_movement_count <= 3:  # Show first 3 examples
                        print(f"\n📈 Price movement #{price_movement_count} at {market_data.timestamp}")
                        print(f"  Price: {market_data.close:.4f}")
                        print(f"  Flat range: {flat_data['support']:.4f} - {flat_data['resistance']:.4f}")
                        print(f"  Resistance broken: {resistance_broken}")
                        print(f"  Support broken: {support_broken}")
                        print(f"  Volume: {market_data.volume:.0f} (avg: {avg_volume:.0f}, needed: {min_volume_needed:.0f})")
                        print(f"  Volume OK: {volume_ok}")
                    
                    if volume_ok:
                        breakout = strategy._detect_breakout(market_data, flat_data)
                        if breakout:
                            breakout_attempts += 1
                            if price_movement_count <= 3:
                                print(f"  ✅ BREAKOUT DETECTED: {breakout}")
                            
                            # Проверим тренд
                            if not strategy._pass_trend_filter(breakout):
                                trend_failures += 1
                                if price_movement_count <= 3:
                                    print(f"  ❌ Trend filter failed")
                                    
                                    # Детали тренда
                                    ema_fast = current_bar.get('ema_fast', 0)
                                    ema_slow = current_bar.get('ema_slow', 0)
                                    vwap = current_bar.get('vwap', 0)
                                    print(f"    EMA Fast: {ema_fast:.4f}, EMA Slow: {ema_slow:.4f}")
                                    print(f"    VWAP: {vwap:.4f}, Price: {market_data.close:.4f}")
                            else:
                                if price_movement_count <= 3:
                                    print(f"  ✅ Trend filter passed - SHOULD HAVE SIGNAL!")
                        else:
                            if price_movement_count <= 3:
                                print(f"  ❌ Strategy._detect_breakout returned None despite conditions!")
                    else:
                        volume_failures += 1
                        if price_movement_count <= 3:
                            print(f"  ❌ Volume too low")
                
                # Show first few flat examples
                elif flat_count <= 3:
                    print(f"\n📊 Flat #{flat_count} at {market_data.timestamp}")
                    print(f"  Price: {market_data.close:.4f}")
                    print(f"  Flat range: {flat_data['support']:.4f} - {flat_data['resistance']:.4f} (width: {flat_data['range']:.3f})")
                    print(f"  Price is WITHIN flat bounds")
    
    print(f"\n📊 Analysis Summary:")
    print(f"  Flats detected: {flat_count}")
    print(f"  Price movements beyond flat: {price_movement_count}")
    print(f"  Breakout attempts: {breakout_attempts}")
    print(f"  Volume failures: {volume_failures}")
    print(f"  Trend failures: {trend_failures}")
    print(f"  Successful conditions: {breakout_attempts - volume_failures - trend_failures}")
    
    # Анализируем параметры флэта
    print(f"\n📏 Flat Parameters Analysis:")
    print(f"  Min range: {strategy.parameters['breakout_min_range']} USDT")
    print(f"  Max range: {strategy.parameters['breakout_max_range']} USDT")
    
    # Считаем реальные диапазоны в данных
    windows = []
    for i in range(12, len(test_data)):
        window = test_data.iloc[i-12:i]
        range_val = window['high'].max() - window['low'].min()
        windows.append(range_val)
    
    if windows:
        print(f"  Actual ranges in data:")
        print(f"    Min: {min(windows):.3f}")
        print(f"    Max: {max(windows):.3f}")  
        print(f"    Avg: {np.mean(windows):.3f}")
        print(f"    Median: {np.median(windows):.3f}")
        
        # Сколько окон попадают в наш диапазон
        valid_ranges = [r for r in windows if 
                       strategy.parameters['breakout_min_range'] <= r <= 
                       strategy.parameters['breakout_max_range']]
        print(f"    Valid ranges: {len(valid_ranges)}/{len(windows)} ({len(valid_ranges)/len(windows)*100:.1f}%)")


if __name__ == "__main__":
    analyze_data()