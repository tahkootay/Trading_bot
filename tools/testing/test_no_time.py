#!/usr/bin/env python3
"""Test without time constraints"""

import asyncio
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime

# Add src to path  
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

from pybit.unified_trading import HTTP

load_dotenv()

async def test_no_constraints():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
    )
    
    # Try different limits and no time constraints
    print("Testing with NO time constraints...")
    
    limits = [1, 5, 10, 200]
    intervals = ["1m", "5m", "1h", "1d"]
    
    for limit in limits:
        for interval in intervals:
            print(f"\nTesting limit={limit}, interval={interval}:")
            
            response = session.get_kline(
                category="linear",
                symbol="SOLUSDT",
                interval=interval,
                limit=limit,
            )
            
            if response["retCode"] == 0:
                klines = response["result"]["list"]
                print(f"  ✅ Response successful, got {len(klines)} klines")
                
                if klines:
                    # Show first kline
                    kline = klines[0]
                    timestamp = datetime.fromtimestamp(int(kline[0]) / 1000)
                    print(f"    Latest: {timestamp} - ${kline[4]}")
                    break  # Found data, move to next limit
                else:
                    print(f"    Empty result")
            else:
                print(f"  ❌ Error: {response['retMsg']}")
        
        if response["retCode"] == 0 and response["result"]["list"]:
            print(f"\n✅ Found data with limit={limit}!")
            break
    else:
        print(f"\n❌ No data found with any configuration")

if __name__ == "__main__":
    asyncio.run(test_no_constraints())