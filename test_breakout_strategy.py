#!/usr/bin/env python3
"""
Тестирование Breakout Volume Strategy

Скрипт для быстрого тестирования новой стратегии на исторических данных.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Добавляем путь к модулям
sys.path.append('.')

from examples.strategies.breakout_volume_strategy import BreakoutVolumeStrategy
from modules.backtester import MarketData, Position


def test_strategy():
    """Тестирование стратегии на небольшом наборе данных"""
    
    print("🧪 Testing Breakout Volume Strategy...")
    
    # Загружаем тестовые данные
    data_file = Path("data/raw/SOLUSDT_5m_20250807_20250906.csv")
    
    if not data_file.exists():
        print("❌ Test data file not found!")
        return
    
    # Читаем данные
    df = pd.read_csv(data_file)
    print(f"📊 Loaded {len(df)} candles from {data_file.name}")
    
    # Создаём стратегию
    strategy = BreakoutVolumeStrategy()
    
    # Тестируем на первых 100 свечах
    test_data = df.head(100)
    
    position = Position(
        direction="NONE",
        quantity=0,
        entry_price=0,
        entry_time=datetime.now(),
        unrealized_pnl=0,
        duration_minutes=0
    )
    signals_count = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    
    print("\n📈 Processing candles...")
    
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
            print(f"🟢 BUY at {market_data.close:.4f} - {signal.entry_reason}")
            
        elif signal.signal.name == 'SELL' and position.quantity > 0:
            pnl = (market_data.close - position.entry_price) * position.quantity
            position.direction = "NONE"
            position.quantity = 0
            position.entry_price = 0
            print(f"🔴 SELL at {market_data.close:.4f} - {signal.exit_reason} | P&L: {pnl:.2f}")
    
    print(f"\n📊 Signal Statistics:")
    for signal_type, count in signals_count.items():
        print(f"  {signal_type}: {count}")
    
    # Проверяем параметры стратегии
    print(f"\n⚙️ Strategy Parameters:")
    for key, value in strategy.parameters.items():
        print(f"  {key}: {value}")
    
    print(f"\n📚 History length: {len(strategy.state['history'])}")
    print(f"🎯 Trades today: {strategy.state['trades_today']}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_strategy()