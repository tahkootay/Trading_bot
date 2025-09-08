#!/usr/bin/env python3
"""
Тестирование оптимизированной Breakout Volume Strategy

Проверяем, генерирует ли стратегия сделки с обновлёнными параметрами.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Добавляем путь к модулям
sys.path.append('.')

from examples.strategies.breakout_volume_strategy_optimized import OptimizedBreakoutVolumeStrategy
from modules.backtester import MarketData, Position


def test_optimized_strategy():
    """Тестирование оптимизированной стратегии"""
    
    print("🧪 Testing Optimized Breakout Volume Strategy...")
    
    # Загружаем тестовые данные
    data_file = Path("data/raw/SOLUSDT_5m_20250807_20250906.csv")
    
    if not data_file.exists():
        print("❌ Test data file not found!")
        return
    
    # Читаем данные
    df = pd.read_csv(data_file)
    print(f"📊 Loaded {len(df)} candles from {data_file.name}")
    
    # Создаём оптимизированную стратегию
    strategy = OptimizedBreakoutVolumeStrategy()
    
    # Показываем ключевые изменения в параметрах
    print(f"\n⚙️ Optimized Parameters:")
    print(f"  Volume multiplier: {strategy.parameters['breakout_volume_mult']} (was 1.3)")
    print(f"  Flat range: {strategy.parameters['breakout_min_range']}-{strategy.parameters['breakout_max_range']} USDT (was 0.5-1.5)")
    print(f"  Max body %: {strategy.parameters['max_body_pct']} (was 0.3)")
    print(f"  Target move: {strategy.parameters['target_move']} USDT (was 2.0)")
    
    # Тестируем на первых 1000 свечах (больше данных для анализа)
    test_data = df.head(1000)
    
    position = Position(
        direction="NONE",
        quantity=0,
        entry_price=0,
        entry_time=datetime.now(),
        unrealized_pnl=0,
        duration_minutes=0
    )
    
    signals_count = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    trades = []
    current_trade = None
    
    print(f"\n📈 Processing {len(test_data)} candles...")
    
    for i, row in test_data.iterrows():
        # Создаём MarketData
        market_data = MarketData(
            timestamp=pd.to_datetime(row['timestamp']),
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        
        # Получаем сигнал от стратегии
        signal = strategy.on_bar(market_data, position)
        signals_count[signal.signal.name] += 1
        
        # Имитируем выполнение сигнала
        if signal.signal.name == 'BUY' and position.quantity == 0:
            position.direction = "LONG"
            position.quantity = 10  # Условная позиция
            position.entry_price = market_data.close
            position.entry_time = market_data.timestamp
            
            current_trade = {
                'entry_time': market_data.timestamp,
                'entry_price': market_data.close,
                'direction': 'LONG',
                'reason': signal.entry_reason,
                'exit_time': None,
                'exit_price': None,
                'pnl': 0
            }
            print(f"🟢 BUY at {market_data.close:.4f} - {signal.entry_reason}")
            
        elif signal.signal.name == 'SELL' and position.quantity > 0:
            pnl = (market_data.close - position.entry_price) * position.quantity
            
            if current_trade:
                current_trade['exit_time'] = market_data.timestamp
                current_trade['exit_price'] = market_data.close  
                current_trade['pnl'] = pnl
                trades.append(current_trade)
                current_trade = None
            
            position.direction = "NONE"
            position.quantity = 0
            position.entry_price = 0
            
            exit_reason = getattr(signal, 'exit_reason', 'Exit signal')
            print(f"🔴 SELL at {market_data.close:.4f} - {exit_reason} | P&L: {pnl:.2f}")
    
    print(f"\n📊 Signal Statistics:")
    for signal_type, count in signals_count.items():
        print(f"  {signal_type}: {count}")
    
    print(f"\n💼 Trading Results:")
    print(f"  Total trades: {len(trades)}")
    
    if trades:
        profitable_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = len(profitable_trades) / len(trades) * 100
        
        print(f"  Profitable: {len(profitable_trades)} ({win_rate:.1f}%)")
        print(f"  Losing: {len(losing_trades)}")
        print(f"  Total P&L: {total_pnl:.2f} USDT")
        
        if profitable_trades:
            avg_win = sum(t['pnl'] for t in profitable_trades) / len(profitable_trades)
            print(f"  Average win: {avg_win:.2f} USDT")
        
        if losing_trades:
            avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
            print(f"  Average loss: {avg_loss:.2f} USDT")
        
        # Показываем последние 3 сделки
        print(f"\n📋 Last 3 trades:")
        for trade in trades[-3:]:
            pnl_sign = "✅" if trade['pnl'] > 0 else "❌"
            print(f"  {pnl_sign} {trade['entry_time'].strftime('%m-%d %H:%M')} -> "
                  f"{trade['exit_time'].strftime('%m-%d %H:%M')} | "
                  f"{trade['entry_price']:.4f} -> {trade['exit_price']:.4f} | "
                  f"P&L: {trade['pnl']:.2f}")
    
    # Показываем состояние волатильности
    if strategy.state['volatility']:
        print(f"\n📊 Market Analysis:")
        print(f"  Current volatility: {strategy.state['volatility']:.3f}")
        
        adaptive_params = strategy._get_adaptive_parameters()
        print(f"  Adaptive volume mult: {adaptive_params['volume_mult']:.2f}")
        print(f"  Adaptive range: {adaptive_params['min_range']:.2f}-{adaptive_params['max_range']:.2f}")
    
    print(f"\n📚 Strategy State:")
    print(f"  History length: {len(strategy.state['history'])}")
    print(f"  Trades today: {strategy.state['trades_today']}")
    print(f"  Consecutive losses: {strategy.state['consecutive_losses']}")
    
    print("\n✅ Test completed!")
    
    # Сравнение с оригинальной стратегией
    if len(trades) > 0:
        print(f"\n🎯 Optimization Result: STRATEGY NOW GENERATES TRADES!")
        print(f"   Original strategy: 0 trades")
        print(f"   Optimized strategy: {len(trades)} trades")
    else:
        print(f"\n⚠️ Still no trades generated. May need further parameter adjustment.")


if __name__ == "__main__":
    test_optimized_strategy()