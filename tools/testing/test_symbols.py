#!/usr/bin/env python3
"""Test symbol availability on Bybit"""

import asyncio
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add src to path  
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

from src.data_collector.bybit_client import BybitHTTPClient
from src.utils.types import TimeFrame

load_dotenv()

async def test_symbols():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    client = BybitHTTPClient(api_key, api_secret, testnet=False)
    
    symbols_to_test = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
    
    async with client:
        for symbol in symbols_to_test:
            print(f"\nTesting {symbol}:")
            
            # Test ticker first
            try:
                ticker = await client.get_ticker(symbol)
                if ticker:
                    print(f"  ✅ Ticker: Price=${ticker.price:.2f}, Volume24h={ticker.volume_24h:.0f}")
                else:
                    print(f"  ❌ No ticker data")
                    continue
            except Exception as e:
                print(f"  ❌ Ticker error: {e}")
                continue
            
            # Test candles
            try:
                candles = await client.get_klines(symbol, TimeFrame.H1, limit=5)
                if candles:
                    print(f"  ✅ Candles: {len(candles)} candles received")
                    print(f"      Latest: {candles[0].timestamp} - ${candles[0].close:.2f}")
                else:
                    print(f"  ❌ No candle data")
            except Exception as e:
                print(f"  ❌ Candles error: {e}")

if __name__ == "__main__":
    asyncio.run(test_symbols())