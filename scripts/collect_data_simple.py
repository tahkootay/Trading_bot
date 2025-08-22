#!/usr/bin/env python3
"""
Простой сбор данных SOL/USDT через публичные API без аутентификации
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import click

def collect_bybit_data(symbol: str, interval: str, days: int, output_dir: str = "data"):
    """
    Сбор данных через публичное API Bybit
    
    Args:
        symbol: Торговая пара (например SOLUSDT)
        interval: Интервал (1, 5, 15, 60, 240, D)
        days: Количество дней назад
        output_dir: Папка для сохранения
    """
    
    # Конвертация интервалов
    interval_map = {
        "1m": "1",
        "5m": "5", 
        "15m": "15",
        "1h": "60",
        "4h": "240",
        "1d": "D"
    }
    
    bybit_interval = interval_map.get(interval, interval)
    
    # Расчет временных рамок
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Bybit использует секунды
    start_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())
    
    print(f"📊 Сбор данных {symbol} ({interval}) за {days} дней...")
    print(f"📅 Период: {start_time.strftime('%Y-%m-%d')} - {end_time.strftime('%Y-%m-%d')}")
    
    all_candles = []
    current_start = start_timestamp
    
    # URL для публичного API Bybit
    base_url = "https://api.bybit.com/v5/market/kline"
    
    while current_start < end_timestamp:
        try:
            params = {
                'category': 'spot',  # или 'linear' для фьючерсов
                'symbol': symbol,
                'interval': bybit_interval,
                'start': current_start * 1000,  # Bybit ожидает миллисекунды
                'end': end_timestamp * 1000,
                'limit': 1000
            }
            
            print(f"🔄 Запрос данных с {datetime.fromtimestamp(current_start)}...")
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data['retCode'] != 0:
                print(f"❌ Ошибка API: {data['retMsg']}")
                break
            
            candles = data['result']['list']
            
            if not candles:
                print("✅ Данные закончились")
                break
                
            # Конвертируем в нужный формат
            processed_candles = []
            for candle in candles:
                # Bybit возвращает [timestamp, open, high, low, close, volume, turnover]
                processed_candles.append({
                    'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
                    'open': float(candle[1]),
                    'high': float(candle[2]), 
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]),
                    'turnover': float(candle[6]) if len(candle) > 6 else 0
                })
            
            all_candles.extend(processed_candles)
            print(f"📈 Собрано {len(processed_candles)} свечей (всего: {len(all_candles)})")
            
            # Обновляем временную метку для следующего запроса
            if processed_candles:
                last_time = processed_candles[-1]['timestamp']
                current_start = int(last_time.timestamp()) + 1
            else:
                break
                
            # Пауза чтобы не превысить лимиты
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка запроса: {e}")
            time.sleep(5)  # Больше задержка при ошибке
            continue
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
            break
    
    if all_candles:
        # Создаем DataFrame и сортируем по времени
        df = pd.DataFrame(all_candles)
        df = df.sort_values('timestamp')
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Сохраняем в CSV
        Path(output_dir).mkdir(exist_ok=True)
        filename = f"{symbol}_{interval}_180d_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = Path(output_dir) / filename
        
        df.to_csv(filepath, index=False)
        
        print(f"✅ Сохранено {len(df)} свечей в {filepath}")
        
        # Создаем метаданные
        metadata = {
            "symbol": symbol,
            "interval": interval,
            "collection_date": datetime.now().isoformat(),
            "days_collected": days,
            "total_candles": len(df),
            "date_range": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat()
            },
            "data_source": "bybit_public_api",
            "filename": filename
        }
        
        metadata_file = Path(output_dir) / f"{symbol}_{interval}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return len(df)
    else:
        print("❌ Не удалось собрать данные")
        return 0

@click.command()
@click.option('--symbol', '-s', default='SOLUSDT', help='Trading pair symbol')
@click.option('--days', '-d', default=180, help='Days to collect')
@click.option('--output-dir', '-o', default='data', help='Output directory')
def main(symbol, days, output_dir):
    """Сбор исторических данных для обучения ML моделей"""
    
    print(f"🚀 Простой сборщик данных Bybit")
    print(f"📊 Символ: {symbol}")
    print(f"📅 Период: {days} дней")
    print(f"📁 Папка: {output_dir}")
    print()
    
    # Собираем основные таймфреймы для ML
    intervals = ['5m', '15m', '1h']  # Фокусируемся на ключевых таймфреймах
    
    total_candles = 0
    successful_intervals = 0
    
    for interval in intervals:
        try:
            candles = collect_bybit_data(symbol, interval, days, output_dir)
            if candles > 0:
                total_candles += candles
                successful_intervals += 1
            print()
        except Exception as e:
            print(f"❌ Ошибка для {interval}: {e}")
            print()
    
    print("="*50)
    print(f"🎉 Сбор данных завершен!")
    print(f"✅ Успешно собрано {successful_intervals} из {len(intervals)} таймфреймов")
    print(f"📊 Общий объем: {total_candles:,} свечей")
    print(f"📁 Данные сохранены в: {Path(output_dir).absolute()}")
    
    if total_candles >= 50000:
        print("🎯 Отлично! Данных достаточно для качественного ML обучения")
    elif total_candles >= 25000:
        print("👍 Хорошо! Данных достаточно для базового ML обучения") 
    else:
        print("⚠️  Данных маловато, но можно попробовать обучение")

if __name__ == "__main__":
    main()