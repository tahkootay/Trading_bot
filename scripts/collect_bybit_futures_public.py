#!/usr/bin/env python3
"""
Сбор данных с Bybit Futures через публичный API (без авторизации)
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import aiohttp
import json
import click

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from src.utils.logger import setup_logging, TradingLogger


class BybitFuturesPublicCollector:
    """Сборщик данных с Bybit Futures через публичный API."""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.session = None
        self.logger = TradingLogger("bybit_futures_collector")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_current_price(self, symbol: str = "SOLUSDT"):
        """Получить текущую цену."""
        try:
            url = f"{self.base_url}/v5/market/tickers"
            params = {
                "category": "linear",
                "symbol": symbol
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data["retCode"] == 0 and data["result"]["list"]:
                        ticker = data["result"]["list"][0]
                        return {
                            "symbol": ticker["symbol"],
                            "price": float(ticker["lastPrice"]),
                            "bid": float(ticker["bid1Price"]),
                            "ask": float(ticker["ask1Price"]),
                            "volume": float(ticker["volume24h"]),
                            "change": float(ticker["price24hPcnt"]) * 100
                        }
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
        return None
    
    async def get_klines_batch(self, symbol: str, interval: str, limit: int = 200, 
                              start_time: int = None, end_time: int = None):
        """Получить batch исторических данных."""
        try:
            url = f"{self.base_url}/v5/market/kline"
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            if start_time:
                params["start"] = start_time
            if end_time:
                params["end"] = end_time
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data["retCode"] == 0:
                        klines = []
                        for kline in data["result"]["list"]:
                            klines.append({
                                "timestamp": pd.to_datetime(int(kline[0]), unit="ms"),
                                "open": float(kline[1]),
                                "high": float(kline[2]),
                                "low": float(kline[3]),
                                "close": float(kline[4]),
                                "volume": float(kline[5])
                            })
                        return klines
        except Exception as e:
            self.logger.error(f"Error getting klines: {e}")
        return []
    
    async def collect_historical_data(self, symbol: str = "SOLUSDT", 
                                    interval: str = "5", days: int = 90):
        """Собрать исторические данные за указанный период."""
        
        print(f"🚀 Сбор данных Bybit Futures: {symbol}")
        print(f"📅 Период: {days} дней")
        print(f"⏱️  Интервал: {interval} минут")
        
        # Показать текущую цену
        current_price = await self.get_current_price(symbol)
        if current_price:
            print(f"💰 Текущая цена: ${current_price['price']:.2f}")
            print(f"📊 Объем 24ч: {current_price['volume']:,.0f}")
            print(f"📈 Изменение 24ч: {current_price['change']:+.2f}%")
        
        # Рассчитаем временные рамки
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Конвертируем в миллисекунды
        end_timestamp = int(end_time.timestamp() * 1000)
        start_timestamp = int(start_time.timestamp() * 1000)
        
        print(f"🔍 Сбор данных: {start_time} → {end_time}")
        
        all_klines = []
        current_end = end_timestamp
        batch_count = 0
        
        # Собираем данные батчами, идя назад во времени
        while current_end > start_timestamp and batch_count < 1000:  # Safety limit
            batch_count += 1
            
            # Рассчитываем start для этого batch
            if interval == "D":
                batch_duration = 200 * 24 * 60 * 60 * 1000  # 200 дней
            else:
                batch_duration = 200 * int(interval) * 60 * 1000  # 200 свечей × интервал
            batch_start = max(start_timestamp, current_end - batch_duration)
            
            print(f"📡 Batch {batch_count}: {datetime.fromtimestamp(batch_start/1000)} → {datetime.fromtimestamp(current_end/1000)}")
            
            # Получаем данные
            klines = await self.get_klines_batch(
                symbol=symbol,
                interval=interval,
                limit=200,
                start_time=batch_start,
                end_time=current_end
            )
            
            if not klines:
                print(f"⚠️  Batch {batch_count}: Нет данных, завершаем сбор")
                break
            
            # Фильтруем данные в нужном диапазоне
            valid_klines = []
            for kline in klines:
                kline_ts = int(kline["timestamp"].timestamp() * 1000)
                if start_timestamp <= kline_ts <= end_timestamp:
                    valid_klines.append(kline)
            
            if valid_klines:
                all_klines.extend(valid_klines)
                print(f"✅ Batch {batch_count}: {len(valid_klines)} свечей (всего: {len(all_klines)})")
                
                # Обновляем current_end для следующего батча
                earliest_ts = min(int(k["timestamp"].timestamp() * 1000) for k in valid_klines)
                current_end = earliest_ts - 1
            else:
                print(f"⚠️  Batch {batch_count}: Нет валидных данных")
                current_end -= batch_duration
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        print(f"📊 Всего собрано: {len(all_klines)} свечей")
        
        if all_klines:
            # Создаем DataFrame
            df = pd.DataFrame(all_klines)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            
            # Удаляем дубликаты
            df = df[~df.index.duplicated(keep='last')]
            
            # Статистика
            actual_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 3600)
            print(f"📅 Фактический период: {actual_days:.1f} дней")
            print(f"💰 Ценовой диапазон: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return df
        
        return pd.DataFrame()


@click.command()
@click.option("--symbol", default="SOLUSDT", help="Торговый символ")
@click.option("--interval", default="5", help="Интервал в минутах (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)")
@click.option("--days", default=7, help="Количество дней для сбора")
@click.option("--output", default="data", help="Папка для сохранения")
def main(symbol: str, interval: str, days: int, output: str):
    """Сбор данных с Bybit Futures через публичный API."""
    
    setup_logging(log_level="INFO", log_format="text")
    
    async def collect_data():
        Path(output).mkdir(exist_ok=True)
        
        async with BybitFuturesPublicCollector() as collector:
            df = await collector.collect_historical_data(
                symbol=symbol,
                interval=interval,
                days=days
            )
            
            if not df.empty:
                # Сохраняем данные
                if interval == "D":
                    filename = f"{output}/{symbol}_{interval}_{days}d_bybit_futures.csv"
                else:
                    filename = f"{output}/{symbol}_{interval}m_{days}d_bybit_futures.csv"
                df.to_csv(filename)
                
                print(f"💾 Данные сохранены: {filename}")
                print(f"📊 Записей: {len(df)}")
                print(f"📅 Период: {df.index[0]} → {df.index[-1]}")
                
                # Создаем metadata
                metadata = {
                    "symbol": symbol,
                    "interval": f"{interval}" if interval == "D" else f"{interval}m",
                    "days_collected": days,
                    "records_count": len(df),
                    "start_time": str(df.index[0]),
                    "end_time": str(df.index[-1]),
                    "source": "bybit_futures_public",
                    "collected_at": datetime.now().isoformat()
                }
                
                if interval == "D":
                    metadata_file = f"{output}/{symbol}_{interval}_{days}d_bybit_futures_metadata.json"
                else:
                    metadata_file = f"{output}/{symbol}_{interval}m_{days}d_bybit_futures_metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                print(f"📋 Метаданные: {metadata_file}")
                return True
            else:
                print("❌ Данные не получены")
                return False
    
    success = asyncio.run(collect_data())
    if success:
        print("\n🎉 Сбор данных с Bybit Futures завершен успешно!")
    else:
        print("\n⚠️  Сбор данных не удался")

if __name__ == "__main__":
    main()