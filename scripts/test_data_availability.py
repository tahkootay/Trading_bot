#!/usr/bin/env python3
"""Test data availability on Bybit testnet."""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

async def test_data_availability():
    """Test what historical data is available on testnet."""
    
    print("🔍 Testing Bybit Testnet Data Availability")
    print("=" * 50)
    
    # Test different intervals
    intervals = ["1", "5", "15", "60", "240", "D"]  # 1m, 5m, 15m, 1h, 4h, 1d
    
    async with aiohttp.ClientSession() as session:
        for interval in intervals:
            print(f"\n📊 Testing {interval} interval...")
            
            # Try different limits
            for limit in [10, 50, 200]:
                url = "https://api-testnet.bybit.com/v5/market/kline"
                params = {
                    "category": "linear",
                    "symbol": "SOLUSDT",
                    "interval": interval,
                    "limit": limit
                }
                
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("retCode") == 0:
                                klines = data["result"]["list"]
                                if klines:
                                    # Get time range
                                    start_time = int(klines[-1][0])  # Oldest
                                    end_time = int(klines[0][0])     # Newest
                                    
                                    start_dt = datetime.fromtimestamp(start_time / 1000)
                                    end_dt = datetime.fromtimestamp(end_time / 1000)
                                    
                                    print(f"  ✅ {interval} interval, limit {limit}: {len(klines)} candles")
                                    print(f"     Range: {start_dt} → {end_dt}")
                                    print(f"     Duration: {(end_dt - start_dt).total_seconds() / 3600:.1f} hours")
                                    
                                    # Show latest candle
                                    latest = klines[0]
                                    close_price = float(latest[4])
                                    volume = float(latest[5])
                                    print(f"     Latest: ${close_price:.2f}, Volume: {volume:.2f}")
                                    
                                    # Only test first successful limit for each interval
                                    break
                                else:
                                    print(f"  ❌ {interval} interval, limit {limit}: No data returned")
                            else:
                                print(f"  ❌ {interval} interval, limit {limit}: API error {data.get('retMsg')}")
                        else:
                            print(f"  ❌ {interval} interval, limit {limit}: HTTP {response.status}")
                except Exception as e:
                    print(f"  ❌ {interval} interval, limit {limit}: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 Testing different symbols...")
    
    # Test other symbols to see if it's symbol-specific
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in symbols:
        url = "https://api-testnet.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": "5",
            "limit": 10
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            klines = data["result"]["list"]
                            print(f"✅ {symbol}: {len(klines)} candles available")
                        else:
                            print(f"❌ {symbol}: {data.get('retMsg')}")
                    else:
                        print(f"❌ {symbol}: HTTP {response.status}")
            except Exception as e:
                print(f"❌ {symbol}: {e}")

if __name__ == "__main__":
    asyncio.run(test_data_availability())