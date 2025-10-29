#!/usr/bin/env python3
"""
Convert 15-minute data to 1-hour data for backtesting
Create OHLCV 1h bars from 15m data
"""

import csv
from datetime import datetime, timedelta

print('🔄 КОНВЕРТАЦИЯ 15m → 1h ДАННЫХ')
print('=' * 40)
print('📊 Источник: SOLUSDT_15m_20240901_20250831.csv')
print('🎯 Цель: создать 1h данные за весь период')

# Загрузка 15-минутных данных
data_15m = []
with open('data/raw/SOLUSDT_15m_20240901_20250831.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_15m.append({
            'timestamp': datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })

print(f'✅ Загружено {len(data_15m)} 15-минутных баров')

# Конвертация в 1-часовые бары
data_1h = []
current_hour = None
hour_data = []

for bar in data_15m:
    # Определяем час (округляем до часа)
    hour_timestamp = bar['timestamp'].replace(minute=0, second=0, microsecond=0)
    
    if current_hour != hour_timestamp:
        # Завершаем предыдущий час
        if hour_data:
            # Создаём OHLCV для часа
            hour_bar = {
                'timestamp': current_hour.strftime('%Y-%m-%d %H:%M:%S'),
                'open': hour_data[0]['open'],
                'high': max(bar['high'] for bar in hour_data),
                'low': min(bar['low'] for bar in hour_data),
                'close': hour_data[-1]['close'],
                'volume': sum(bar['volume'] for bar in hour_data)
            }
            data_1h.append(hour_bar)
        
        # Начинаем новый час
        current_hour = hour_timestamp
        hour_data = []
    
    hour_data.append(bar)

# Добавляем последний час
if hour_data:
    hour_bar = {
        'timestamp': current_hour.strftime('%Y-%m-%d %H:%M:%S'),
        'open': hour_data[0]['open'],
        'high': max(bar['high'] for bar in hour_data),
        'low': min(bar['low'] for bar in hour_data),
        'close': hour_data[-1]['close'],
        'volume': sum(bar['volume'] for bar in hour_data)
    }
    data_1h.append(hour_bar)

print(f'✅ Создано {len(data_1h)} 1-часовых баров')
print(f'📅 Период: {data_1h[0]["timestamp"]} → {data_1h[-1]["timestamp"]}')

# Сохранение в файл
output_path = 'data/raw/SOLUSDT_1h_20240901_20250831.csv'
with open(output_path, 'w', newline='') as f:
    fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data_1h)

print(f'💾 Сохранено в: {output_path}')
print('✅ КОНВЕРТАЦИЯ ЗАВЕРШЕНА!')