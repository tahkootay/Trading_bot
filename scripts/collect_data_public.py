#!/usr/bin/env python3
"""Сбор исторических данных через публичный API Bybit."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import click
import json
import aiohttp

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

try:
    from src.utils.types import TimeFrame
    from src.utils.logger import setup_logging, TradingLogger
except ImportError:
    # Fallback для совместимости с Python 3.9
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.types import TimeFrame
    from src.utils.logger import setup_logging, TradingLogger


class PublicDataCollector:
    """Сбор данных через публичный API Bybit."""
    
    def __init__(self):
        self.logger = TradingLogger("public_data_collector")
        self.base_url = "https://api.bybit.com"
    
    def _get_interval_code(self, timeframe: TimeFrame) -> str:
        """Конвертировать TimeFrame в код интервала для Bybit API."""
        interval_map = {
            TimeFrame.M1: "1",      # 1 минута
            TimeFrame.M3: "3",      # 3 минуты
            TimeFrame.M5: "5",      # 5 минут
            TimeFrame.M15: "15",    # 15 минут
            TimeFrame.M30: "30",    # 30 минут
            TimeFrame.H1: "60",     # 1 час
            TimeFrame.H4: "240",    # 4 часа
            TimeFrame.D1: "D",      # 1 день
        }
        return interval_map.get(timeframe, "1")  # По умолчанию 1 минута
        
    async def collect_historical_data(
        self,
        symbol: str,
        timeframes: list,
        days: int = 30,
        output_dir: str = "data",
    ) -> dict:
        """Сбор исторических данных через публичный API."""
        
        # Создаем структуру папок
        processed_dir = Path(output_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        collected_data = {}
        
        print(f"🚀 Публичный сбор данных для {symbol}")
        print(f"📅 Период: {days} дней")
        print(f"⏱️  Временные фреймы: {', '.join(timeframes)}")
        print(f"📁 Сохранение в: {processed_dir}")
        print()
        
        async with aiohttp.ClientSession() as session:
            for timeframe_str in timeframes:
                try:
                    timeframe = TimeFrame(timeframe_str)
                    print(f"📊 Сбор {timeframe.value} данных для {symbol}...")
                    print(f"    🔍 TimeFrame: {timeframe}, value: {timeframe.value}, interval_code: {self._get_interval_code(timeframe)}")
                    
                    # Рассчитать временной диапазон
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=days)
                    
                    print(f"    🔍 Временной диапазон: {start_time} -> {end_time}")
                    
                    # Попробуем сначала без временных ограничений для получения текущих данных
                    all_candles = []
                    try:
                        url = f"{self.base_url}/v5/market/kline"
                        params = {
                            "category": "linear",
                            "symbol": symbol,
                            "interval": self._get_interval_code(timeframe),
                            "limit": 1000,  # Максимум для публичного API
                        }
                        
                        print(f"    📡 Запрос текущих данных (без временных ограничений)")
                        
                        async with session.get(url, params=params) as response:
                            if response.status != 200:
                                print(f"    ❌ HTTP ошибка: {response.status}")
                                continue
                            
                            data = await response.json()
                            
                            if data.get("retCode") != 0:
                                print(f"    ❌ API ошибка: {data.get('retMsg')}")
                                continue
                            
                            klines = data.get("result", {}).get("list", [])
                            
                            if klines:
                                print(f"    📈 Получено {len(klines)} текущих свечей")
                                
                                # Конвертировать в удобный формат
                                for kline in klines:
                                    candle_data = {
                                        'timestamp': datetime.fromtimestamp(int(kline[0]) / 1000),
                                        'open': float(kline[1]),
                                        'high': float(kline[2]),
                                        'low': float(kline[3]),
                                        'close': float(kline[4]),
                                        'volume': float(kline[5]),
                                    }
                                    all_candles.append(candle_data)
                                
                                print(f"    ✅ Успешно собрано {len(all_candles)} свечей")
                            else:
                                print(f"    ⚠️  Нет текущих данных")
                    
                    except Exception as e:
                        print(f"    ❌ Ошибка запроса текущих данных: {e}")
                    
                    # Если нет текущих данных, попробуем с временными ограничениями
                    if not all_candles:
                        print(f"    🔄 Пробуем исторические данные с временными ограничениями...")
                        
                        # Конвертировать в миллисекунды
                        start_timestamp = int(start_time.timestamp() * 1000)
                        end_timestamp = int(end_time.timestamp() * 1000)
                        current_start = start_timestamp
                        
                        while current_start < end_timestamp:
                            try:
                                url = f"{self.base_url}/v5/market/kline"
                                params = {
                                    "category": "linear",
                                    "symbol": symbol,
                                    "interval": self._get_interval_code(timeframe),
                                    "start": current_start,
                                    "end": end_timestamp,
                                    "limit": 1000,
                                }
                                
                                print(f"    📡 Запрос: {current_start} -> {end_timestamp}")
                                
                                async with session.get(url, params=params) as response:
                                    if response.status != 200:
                                        print(f"    ❌ HTTP ошибка: {response.status}")
                                        break
                                    
                                    data = await response.json()
                                    
                                    if data.get("retCode") != 0:
                                        print(f"    ❌ API ошибка: {data.get('retMsg')}")
                                        break
                                    
                                    klines = data.get("result", {}).get("list", [])
                                    
                                    if not klines:
                                        print(f"    ⚠️  Нет данных, завершение")
                                        break
                                    
                                    print(f"    📈 Получено {len(klines)} свечей")
                                    
                                    # Конвертировать в удобный формат
                                    for kline in klines:
                                        candle_data = {
                                            'timestamp': datetime.fromtimestamp(int(kline[0]) / 1000),
                                            'open': float(kline[1]),
                                            'high': float(kline[2]),
                                            'low': float(kline[3]),
                                            'close': float(kline[4]),
                                            'volume': float(kline[5]),
                                        }
                                        all_candles.append(candle_data)
                                    
                                    # Обновить время начала для следующего запроса
                                    if klines:
                                        last_candle_time = int(klines[-1][0])
                                        current_start = last_candle_time + 1
                                        print(f"    🔄 Следующий старт: {current_start}")
                                    
                                    # Небольшая задержка для соблюдения лимитов
                                    await asyncio.sleep(0.1)
                                    
                            except Exception as e:
                                print(f"    ❌ Ошибка запроса: {e}")
                                await asyncio.sleep(1)
                                break
                    
                    if all_candles:
                        # Конвертировать в DataFrame
                        df = pd.DataFrame(all_candles)
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Удалить дубликаты
                        df = df[~df.index.duplicated(keep='last')]
                        
                        # Сохранить в файл
                        filename = processed_dir / f"{symbol}_{timeframe.value}_{days}d_public.csv"
                        df.to_csv(filename)
                        
                        collected_data[timeframe] = df
                        
                        print(f"    ✅ Сохранено {len(df)} свечей в {filename}")
                        
                        # Показать диапазон данных
                        if len(df) > 0:
                            start_date = df.index[0].strftime("%Y-%m-%d %H:%M")
                            end_date = df.index[-1].strftime("%Y-%m-%d %H:%M")
                            print(f"    📅 Диапазон: {start_date} → {end_date}")
                    else:
                        print(f"    ❌ Нет данных для {timeframe.value}")
                    
                    print()
                    
                except Exception as e:
                    print(f"  ❌ Ошибка сбора {timeframe_str}: {e}")
                    import traceback
                    print(f"  📍 Traceback: {traceback.format_exc()}")
        
        # Сохранить метаданные
        metadata = {
            "symbol": symbol,
            "collection_date": datetime.now().isoformat(),
            "days_collected": days,
            "timeframes": [tf.value for tf in collected_data.keys()],
            "total_candles": {tf.value: len(df) for tf, df in collected_data.items()},
            "api_type": "public",
        }
        
        with open(processed_dir / f"{symbol}_public_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return collected_data


@click.command()
@click.option("--symbol", default="SOLUSDT", help="Торговый символ")
@click.option("--days", default=30, help="Количество дней для сбора")
@click.option("--timeframes", default="1m,5m,15m,1h", help="Временные фреймы через запятую")
@click.option("--output", default="data", help="Папка для сохранения")
def main(symbol: str, days: int, timeframes: str, output: str):
    """Сбор исторических данных через публичный API Bybit."""
    
    setup_logging(log_level="INFO", log_format="text")
    
    print("🌐 Публичный сборщик данных Bybit")
    print("=" * 50)
    
    collector = PublicDataCollector()
    timeframes_list = [tf.strip() for tf in timeframes.split(",")]
    
    async def run():
        try:
            data = await collector.collect_historical_data(
                symbol=symbol,
                timeframes=timeframes_list,
                days=days,
                output_dir=output,
            )
            
            print(f"🎉 Сбор данных завершен!")
            print(f"📁 Файлы сохранены в: {output}/")
            
            # Показать сводку
            print("\n📊 Сводка:")
            for tf, df in data.items():
                print(f"  {tf.value:>4}: {len(df):>6} свечей")
                if len(df) > 0:
                    start_date = df.index[0].strftime("%Y-%m-%d %H:%M")
                    end_date = df.index[-1].strftime("%Y-%m-%d %H:%M")
                    print(f"        {start_date} → {end_date}")
        
        except Exception as e:
            print(f"❌ Ошибка сбора данных: {e}")
    
    # Запустить асинхронные задачи
    asyncio.run(run())


if __name__ == "__main__":
    main()
