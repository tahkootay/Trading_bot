#!/usr/bin/env python3
"""
Детальный анализ рыночных условий

Анализирует точные причины отсутствия сделок и предлагает конкретные параметры.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append('.')

from examples.strategies.breakout_volume_strategy_optimized import OptimizedBreakoutVolumeStrategy
from modules.backtester import MarketData, Position


def analyze_market_conditions():
    """Детальный анализ условий рынка"""
    
    print("🔬 Detailed Market Conditions Analysis...")
    
    # Загружаем данные
    data_file = Path("data/raw/SOLUSDT_5m_20250807_20250906.csv")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"📊 Analyzing {len(df)} candles")
    print(f"📈 Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"📊 Volume stats: min={df['volume'].min():.0f}, max={df['volume'].max():.0f}, avg={df['volume'].mean():.0f}")
    
    # Создаём стратегию
    strategy = OptimizedBreakoutVolumeStrategy()
    position = Position(
        direction="NONE", quantity=0, entry_price=0,
        entry_time=datetime.now(), unrealized_pnl=0, duration_minutes=0
    )
    
    # Анализируем первые 1000 свечей
    test_data = df.head(1000)
    
    # Статистика условий
    stats = {
        'total_bars': 0,
        'sufficient_history': 0,
        'flats_detected': 0,
        'price_breakouts': 0,
        'volume_confirmations': 0,
        'trend_passes': 0,
        'actual_signals': 0
    }
    
    breakout_examples = []
    
    print(f"\n🔍 Processing {len(test_data)} candles step by step...")
    
    for i, row in test_data.iterrows():
        market_data = MarketData(
            timestamp=row['timestamp'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        
        # Процессим свечу
        strategy.on_bar(market_data, position)
        stats['total_bars'] += 1
        
        # Проверяем достаточность истории
        min_required = max(
            strategy.parameters['ema_slow'], 
            strategy.parameters['volume_avg_period'],
            strategy.parameters['flat_periods'],
            strategy.parameters['volatility_lookback']
        )
        
        if len(strategy.state['history']) < min_required:
            continue
            
        stats['sufficient_history'] += 1
        
        # Получаем адаптивные параметры
        adaptive_params = strategy._get_adaptive_parameters()
        
        # Проверяем флэт
        flat_data = strategy._detect_flat(adaptive_params)
        if not flat_data:
            continue
            
        stats['flats_detected'] += 1
        
        # Проверяем пробой цены
        resistance_broken = market_data.close > flat_data['resistance']
        support_broken = market_data.close < flat_data['support']
        
        if not (resistance_broken or support_broken):
            continue
            
        stats['price_breakouts'] += 1
        
        # Проверяем объём
        current_bar = strategy.state['history'][-1]
        avg_volume = current_bar.get('avg_volume', 0)
        min_volume_needed = avg_volume * adaptive_params['volume_mult']
        volume_ok = market_data.volume >= min_volume_needed
        
        if not volume_ok:
            continue
            
        stats['volume_confirmations'] += 1
        
        # Проверяем тренд
        direction = 'LONG' if resistance_broken else 'SHORT'
        trend_ok = strategy._pass_trend_filter(direction)
        
        if not trend_ok:
            continue
            
        stats['trend_passes'] += 1
        
        # Если дошли сюда - должен быть сигнал!
        stats['actual_signals'] += 1
        
        # Сохраняем пример для анализа
        if len(breakout_examples) < 5:
            example = {
                'timestamp': market_data.timestamp,
                'price': market_data.close,
                'direction': direction,
                'flat_range': f"{flat_data['support']:.4f} - {flat_data['resistance']:.4f}",
                'flat_width': flat_data['range'],
                'volume': market_data.volume,
                'avg_volume': avg_volume,
                'volume_mult': adaptive_params['volume_mult'],
                'volatility': strategy.state.get('volatility', 0)
            }
            breakout_examples.append(example)
            
            print(f"\n📈 SIGNAL #{len(breakout_examples)} at {market_data.timestamp}")
            print(f"  Direction: {direction}")
            print(f"  Price: {market_data.close:.4f}")
            print(f"  Flat: {flat_data['support']:.4f} - {flat_data['resistance']:.4f} (width: {flat_data['range']:.3f})")
            print(f"  Volume: {market_data.volume:.0f} (avg: {avg_volume:.0f}, mult: {adaptive_params['volume_mult']:.2f})")
            print(f"  Volatility: {strategy.state.get('volatility', 0):.3f}")
    
    # Выводим статистику фильтрации
    print(f"\n📊 Step-by-step Filtering Analysis:")
    print(f"  1. Total bars processed: {stats['total_bars']}")
    print(f"  2. With sufficient history: {stats['sufficient_history']} ({stats['sufficient_history']/stats['total_bars']*100:.1f}%)")
    print(f"  3. Flats detected: {stats['flats_detected']} ({stats['flats_detected']/stats['sufficient_history']*100:.1f}% of valid bars)")
    print(f"  4. Price breakouts: {stats['price_breakouts']} ({stats['price_breakouts']/stats['flats_detected']*100:.1f}% of flats)")
    if stats['price_breakouts'] > 0:
        print(f"  5. Volume confirmations: {stats['volume_confirmations']} ({stats['volume_confirmations']/stats['price_breakouts']*100:.1f}% of breakouts)")
    else:
        print(f"  5. Volume confirmations: {stats['volume_confirmations']} (no breakouts to confirm)")
        
    if stats['volume_confirmations'] > 0:
        print(f"  6. Trend filter passed: {stats['trend_passes']} ({stats['trend_passes']/stats['volume_confirmations']*100:.1f}% of volume-confirmed)")
    else:
        print(f"  6. Trend filter passed: {stats['trend_passes']} (no volume confirmations to filter)")
    print(f"  7. Actual signals: {stats['actual_signals']}")
    
    # Анализируем проблемы
    print(f"\n🚫 Bottleneck Analysis:")
    if stats['flats_detected'] == 0:
        print("  ❌ MAIN ISSUE: No flats detected with current parameters")
        analyze_flat_parameters(test_data, strategy)
    elif stats['price_breakouts'] == 0:
        print("  ❌ MAIN ISSUE: No price breakouts from detected flats")
        print("     Market is very consolidating during this period")
    elif stats['volume_confirmations'] == 0:
        print("  ❌ MAIN ISSUE: Volume requirements too strict")
        analyze_volume_requirements(test_data, strategy)
    elif stats['trend_passes'] == 0:
        print("  ❌ MAIN ISSUE: Trend filter too restrictive")
    else:
        print("  ✅ All conditions should be generating signals!")
    
    print(f"\n✅ Analysis completed!")


def analyze_flat_parameters(df, strategy):
    """Анализ параметров флэтов"""
    
    print(f"\n📏 Flat Parameters Deep Dive:")
    
    ranges = []
    body_percentages = []
    
    for i in range(12, len(df)):
        window = df.iloc[i-12:i]
        range_val = window['high'].max() - window['low'].min()
        ranges.append(range_val)
        
        # Анализ тел свечей
        big_bodies = 0
        max_body_size = range_val * 0.5
        
        for _, bar in window.iterrows():
            body_size = abs(bar['close'] - bar['open'])
            if body_size > max_body_size:
                big_bodies += 1
        
        body_pct = big_bodies / len(window)
        body_percentages.append(body_pct)
    
    current_params = strategy.parameters
    adaptive_params = strategy._get_adaptive_parameters()
    
    print(f"  Current flat range: {adaptive_params['min_range']:.2f} - {adaptive_params['max_range']:.2f}")
    print(f"  Actual ranges in data:")
    print(f"    Min: {min(ranges):.3f}")
    print(f"    Max: {max(ranges):.3f}")
    print(f"    Mean: {np.mean(ranges):.3f}")
    print(f"    Median: {np.median(ranges):.3f}")
    print(f"    25th percentile: {np.percentile(ranges, 25):.3f}")
    print(f"    75th percentile: {np.percentile(ranges, 75):.3f}")
    
    # Сколько окон попадает в текущий диапазон
    valid_ranges = [r for r in ranges if adaptive_params['min_range'] <= r <= adaptive_params['max_range']]
    print(f"    Windows in range: {len(valid_ranges)}/{len(ranges)} ({len(valid_ranges)/len(ranges)*100:.1f}%)")
    
    # Анализ процента больших тел
    print(f"\n  Max body percentage allowed: {current_params['max_body_pct']}")
    print(f"  Actual body percentages:")
    print(f"    Mean: {np.mean(body_percentages):.3f}")
    print(f"    Median: {np.median(body_percentages):.3f}")
    
    valid_bodies = [bp for bp in body_percentages if bp <= current_params['max_body_pct']]
    print(f"    Windows passing body filter: {len(valid_bodies)}/{len(body_percentages)} ({len(valid_bodies)/len(body_percentages)*100:.1f}%)")
    
    # Рекомендации
    print(f"\n💡 Recommended adjustments:")
    if len(valid_ranges) < len(ranges) * 0.1:  # Менее 10% окон подходят
        suggested_min = np.percentile(ranges, 10)
        suggested_max = np.percentile(ranges, 90)
        print(f"    Suggested range: {suggested_min:.2f} - {suggested_max:.2f} (captures 80% of data)")
    
    if len(valid_bodies) < len(body_percentages) * 0.2:  # Менее 20% окон подходят
        suggested_body_pct = np.percentile(body_percentages, 80)
        print(f"    Suggested max body pct: {suggested_body_pct:.2f} (allows 80% of data)")


def analyze_volume_requirements(df, strategy):
    """Анализ требований к объёму"""
    
    print(f"\n📊 Volume Requirements Analysis:")
    
    # Рассчитываем средний объём за 20 периодов
    df['avg_volume_20'] = df['volume'].rolling(20).mean()
    
    adaptive_params = strategy._get_adaptive_parameters()
    volume_mult = adaptive_params['volume_mult']
    
    valid_data = df[df['avg_volume_20'].notna()]
    required_volumes = valid_data['avg_volume_20'] * volume_mult
    actual_volumes = valid_data['volume']
    
    volume_confirmations = (actual_volumes >= required_volumes).sum()
    
    print(f"  Current volume multiplier: {volume_mult}")
    print(f"  Bars with volume confirmation: {volume_confirmations}/{len(valid_data)} ({volume_confirmations/len(valid_data)*100:.1f}%)")
    
    print(f"  Volume statistics:")
    print(f"    Average volume: {df['volume'].mean():.0f}")
    print(f"    Median volume: {df['volume'].median():.0f}")
    print(f"    Volume std: {df['volume'].std():.0f}")
    
    # Анализируем разные множители
    multipliers = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
    print(f"\n  Volume multiplier analysis:")
    for mult in multipliers:
        required = valid_data['avg_volume_20'] * mult
        confirmations = (actual_volumes >= required).sum()
        pct = confirmations / len(valid_data) * 100
        print(f"    {mult:.2f}x: {confirmations} confirmations ({pct:.1f}%)")
    
    # Рекомендация
    optimal_mult = None
    for mult in [1.0, 1.05, 1.1, 1.15, 1.2]:
        required = valid_data['avg_volume_20'] * mult
        confirmations = (actual_volumes >= required).sum()
        pct = confirmations / len(valid_data) * 100
        if pct >= 15:  # Хотим хотя бы 15% подтверждений
            optimal_mult = mult
    
    if optimal_mult:
        print(f"\n💡 Recommended volume multiplier: {optimal_mult} (gives ~15%+ confirmations)")
    else:
        print(f"\n💡 Consider using 1.0x (no volume filter) for this market")


if __name__ == "__main__":
    analyze_market_conditions()