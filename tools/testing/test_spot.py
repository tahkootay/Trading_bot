#!/usr/bin/env python3
"""Test spot category on Bybit"""

import asyncio
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add src to path  
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

from pybit.unified_trading import HTTP

load_dotenv()

async def test_direct_api():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
    )
    
    categories = ["linear", "spot"]
    symbols = ["SOLUSDT", "BTCUSDT"]
    
    for category in categories:
        print(f"\n=== Testing {category.upper()} ===")
        
        for symbol in symbols:
            print(f"\nTesting {symbol} on {category}:")
            
            try:
                # Test klines
                response = session.get_kline(
                    category=category,
                    symbol=symbol,
                    interval="1h",
                    limit=5,
                )
                
                print(f"  Response: {response}")
                
                if response["retCode"] == 0:
                    klines = response["result"]["list"]
                    print(f"  ✅ Got {len(klines)} klines")
                    if klines:
                        for i, kline in enumerate(klines[:2]):
                            print(f"    {i}: Time={kline[0]}, O={kline[1]}, H={kline[2]}, L={kline[3]}, C={kline[4]}")
                else:
                    print(f"  ❌ Error: {response['retMsg']}")
                    
            except Exception as e:
                print(f"  ❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_direct_api())