#!/usr/bin/env python3
"""Enhanced data collector with multiple sources and synthetic data generation."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import click
import json
import requests
import aiohttp
import time

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Add scripts to path for importing other collectors  
scripts_path = str(Path(__file__).parent)
sys.path.insert(0, scripts_path)

from src.utils.logger import setup_logging, TradingLogger


class EnhancedDataCollector:
    """Enhanced data collector with multiple sources."""
    
    def __init__(self):
        self.logger = TradingLogger("enhanced_data_collector")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def calculate_expected_candles(self, days: int, timeframe: str) -> int:
        """Calculate expected number of candles."""
        minutes_per_day = 24 * 60
        total_minutes = days * minutes_per_day
        
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }
        
        return total_minutes // interval_minutes.get(timeframe, 1)
    
    async def collect_from_bybit_futures(self, symbol: str, interval: str, days: int = 90) -> pd.DataFrame:
        """Collect data from Bybit Futures public API (PRIORITY)."""
        print(f"🟨 Trying Bybit Futures API for {symbol} {interval}...")
        
        try:
            # Импортируем BybitFuturesPublicCollector
            from collect_bybit_futures_public import BybitFuturesPublicCollector
            
            collector = BybitFuturesPublicCollector()
            async with collector:
                df = await collector.collect_historical_data(symbol, interval, days)
                if not df.empty:
                    print(f"  ✅ Bybit Futures: {len(df)} candles collected")
                    return df
        except Exception as e:
            print(f"  ❌ Bybit Futures error: {e}")
        
        return pd.DataFrame()

    async def collect_from_binance_public(self, symbol: str, interval: str, days: int = 90) -> pd.DataFrame:
        """Collect data from Binance public API (FALLBACK)."""
        print(f"🟦 Trying Binance public API for {symbol} {interval}...")
        
        # Convert symbol format
        binance_symbol = symbol.replace('/', '')  # SOLUSDT -> SOLUSDT
        
        # Calculate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        all_data = []
        current_start = start_ts
        
        try:
            while current_start < end_ts:
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': binance_symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': min(current_start + (1000 * self._get_interval_ms(interval)), end_ts),
                    'limit': 1000
                }
                
                print(f"  📡 Requesting {len(all_data)}+ candles from Binance...")
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data:
                            break
                        
                        for kline in data:
                            all_data.append({
                                'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                                'open': float(kline[1]),
                                'high': float(kline[2]), 
                                'low': float(kline[3]),
                                'close': float(kline[4]),
                                'volume': float(kline[5])
                            })
                        
                        # Update start time for next batch
                        current_start = int(kline[6]) + 1  # Close time + 1ms
                        
                        await asyncio.sleep(0.1)  # Rate limiting
                    else:
                        print(f"  ❌ Binance API error: {response.status}")
                        break
            
            if all_data:
                df = pd.DataFrame(all_data)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                df = df[~df.index.duplicated(keep='last')]
                
                print(f"  ✅ Binance: {len(df)} candles collected")
                return df
                
        except Exception as e:
            print(f"  ❌ Binance error: {e}")
        
        return pd.DataFrame()
    
    def _get_interval_ms(self, interval: str) -> int:
        """Get interval in milliseconds."""
        intervals = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000, 
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return intervals.get(interval, 60 * 1000)
    
    def generate_synthetic_data(self, symbol: str, interval: str, days: int = 90, 
                              base_price: float = 200.0) -> pd.DataFrame:
        """Generate synthetic but realistic market data."""
        print(f"🎲 Generating synthetic data for {symbol} {interval} ({days} days)...")
        
        # Calculate number of candles needed
        expected_candles = self.calculate_expected_candles(days, interval)
        
        # Time parameters
        end_time = datetime(2024, 8, 23, 12, 0)  # Fixed end time
        interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
        interval_mins = interval_minutes.get(interval, 60)
        
        # Generate timestamps
        timestamps = []
        current_time = end_time
        for i in range(expected_candles):
            timestamps.append(current_time - timedelta(minutes=interval_mins * i))
        timestamps.reverse()
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        
        data = []
        current_price = base_price
        
        for i, ts in enumerate(timestamps):
            # Add some trend and volatility
            trend = np.sin(i / 100) * 0.1  # Long-term wave
            volatility = np.random.normal(0, 0.02)  # Random volatility
            price_change = trend + volatility
            
            current_price = current_price * (1 + price_change)
            
            # Generate OHLC for this candle
            daily_volatility = abs(np.random.normal(0, 0.01))
            
            open_price = current_price
            high_price = open_price * (1 + abs(np.random.normal(0, daily_volatility)))
            low_price = open_price * (1 - abs(np.random.normal(0, daily_volatility)))
            close_price = open_price + np.random.normal(0, open_price * daily_volatility * 0.5)
            
            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume
            base_volume = np.random.uniform(500000, 2000000)
            volume = base_volume * (1 + abs(price_change) * 10)  # Higher volume with price movement
            
            data.append({
                'timestamp': ts,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 1)
            })
            
            current_price = close_price
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"  ✅ Generated {len(df)} synthetic candles")
        print(f"  📊 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"  📅 Time range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    async def collect_data(self, symbol: str, timeframes: list, days: int = 90, 
                          output_dir: str = "data", use_synthetic: bool = False) -> dict:
        """Main data collection method."""
        
        Path(output_dir).mkdir(exist_ok=True)
        collected_data = {}
        
        self.logger.log_system_event(
            event_type="enhanced_data_collection_start",
            component="enhanced_data_collector", 
            status="starting",
            details={"symbol": symbol, "days": days, "timeframes": timeframes}
        )
        
        for timeframe in timeframes:
            print(f"\n📊 Collecting {timeframe} data for {symbol}...")
            df = pd.DataFrame()
            
            if not use_synthetic:
                # Try Bybit Futures first (PRIORITY)
                df = await self.collect_from_bybit_futures(symbol, timeframe, days)
                
                # Fallback to Binance if Bybit fails
                if df.empty:
                    df = await self.collect_from_binance_public(symbol, timeframe, days)
            
            # If no data from APIs, generate synthetic
            if df.empty or use_synthetic:
                print(f"  🎲 Using synthetic data for {timeframe}")
                # Get current price from existing data if available
                base_price = self._get_base_price(symbol)
                df = self.generate_synthetic_data(symbol, timeframe, days, base_price)
            
            if not df.empty:
                # Calculate statistics
                expected_candles = self.calculate_expected_candles(days, timeframe)
                coverage_pct = (len(df) / expected_candles) * 100 if expected_candles > 0 else 0
                actual_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 3600)
                
                print(f"  ✅ Final dataset: {len(df)} candles")
                print(f"  📊 Expected: {expected_candles}, Coverage: {coverage_pct:.1f}%") 
                print(f"  📅 Period: {actual_days:.1f} days")
                
                # Save data
                filename = f"{output_dir}/{symbol}_{timeframe}_{days}d_enhanced.csv"
                df.to_csv(filename)
                collected_data[timeframe] = df
                
                print(f"  💾 Saved to {filename}")
            else:
                print(f"  ❌ No data collected for {timeframe}")
        
        # Save metadata
        metadata = {
            "symbol": symbol,
            "collection_date": datetime.now().isoformat(),
            "days_collected": days,
            "timeframes": list(collected_data.keys()),
            "total_candles": {tf: len(df) for tf, df in collected_data.items()},
            "data_source": "enhanced_collector",
            "use_synthetic": use_synthetic
        }
        
        with open(f"{output_dir}/{symbol}_enhanced_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.log_system_event(
            event_type="enhanced_data_collection_complete",
            component="enhanced_data_collector",
            status="success", 
            details=metadata
        )
        
        return collected_data
    
    def _get_base_price(self, symbol: str) -> float:
        """Get base price for synthetic data from existing data or defaults."""
        price_defaults = {
            'SOLUSDT': 200.0,
            'BTCUSDT': 60000.0,
            'ETHUSDT': 3000.0,
            'ADAUSDT': 0.5
        }
        return price_defaults.get(symbol, 100.0)


@click.command()
@click.option("--symbol", default="SOLUSDT", help="Trading symbol")  
@click.option("--days", default=90, help="Number of days to collect")
@click.option("--timeframes", default="1m,5m,15m,1h", help="Comma-separated timeframes")
@click.option("--output", default="data", help="Output directory")
@click.option("--synthetic", is_flag=True, help="Use synthetic data generation")
def main(symbol: str, days: int, timeframes: str, output: str, synthetic: bool):
    """Enhanced data collector with multiple sources."""
    
    setup_logging(log_level="INFO", log_format="text")
    
    print("🚀 Enhanced Data Collector")
    print("=" * 50)
    
    timeframes_list = [tf.strip() for tf in timeframes.split(",")]
    
    async def collect():
        async with EnhancedDataCollector() as collector:
            data = await collector.collect_data(
                symbol=symbol,
                timeframes=timeframes_list,
                days=days,
                output_dir=output,
                use_synthetic=synthetic
            )
            
            print(f"\n🎉 Collection completed!")
            print(f"📁 Files saved to: {output}/")
            
            # Summary
            print(f"\n📊 Summary:")
            for tf, df in data.items():
                print(f"  {tf:>4}: {len(df):>6} candles")
                if len(df) > 0:
                    start_date = df.index[0].strftime("%Y-%m-%d %H:%M")
                    end_date = df.index[-1].strftime("%Y-%m-%d %H:%M") 
                    print(f"        {start_date} → {end_date}")
    
    asyncio.run(collect())


if __name__ == "__main__":
    main()