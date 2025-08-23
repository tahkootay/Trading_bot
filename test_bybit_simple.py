#!/usr/bin/env python3
"""Simple test of Bybit API"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

# Add src to path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

from src.data_collector.bybit_client import BybitHTTPClient
from src.utils.types import TimeFrame

load_dotenv()

async def test_simple():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    client = BybitHTTPClient(api_key, api_secret, testnet=False)
    
    async with client:
        # Try to get recent 1h candles
        print("Testing 1h candles...")
        candles = await client.get_klines("SOLUSDT", TimeFrame.H1, limit=10)
        print(f"Got {len(candles) if candles else 0} candles")
        
        if candles:
            for i, candle in enumerate(candles[:3]):
                print(f"  Candle {i}: {candle.timestamp} - O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close}")
        
        # Try with historical timestamps (August 2024)
        print("\nTesting with historical timestamps...")
        end_time = 1724450000000  # Aug 23, 2024
        start_time = 1724360000000  # Aug 22, 2024
        
        candles_hist = await client.get_klines(
            "SOLUSDT", 
            TimeFrame.H1, 
            limit=50,
            start_time=start_time,
            end_time=end_time
        )
        print(f"Historical: Got {len(candles_hist) if candles_hist else 0} candles")
        
        if candles_hist:
            for i, candle in enumerate(candles_hist[:3]):
                print(f"  Historical {i}: {candle.timestamp} - O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close}")

if __name__ == "__main__":
    asyncio.run(test_simple())