#!/usr/bin/env python3
"""Test with real 2024 timestamps"""

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

async def test_with_real_timestamps():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
    )
    
    # Use real 2024 timestamps (August 2024)
    end_time = 1724500000000    # August 24, 2024 12:26:40 GMT
    start_time = 1724400000000  # August 23, 2024 08:40:00 GMT
    
    print(f"Using REAL 2024 timestamps:")
    print(f"Start: {datetime.fromtimestamp(start_time / 1000)}")
    print(f"End: {datetime.fromtimestamp(end_time / 1000)}")
    
    # Test different intervals
    intervals = ["1h", "4h", "1d"]
    
    for interval in intervals:
        print(f"\nTesting {interval} interval:")
        
        response = session.get_kline(
            category="linear",
            symbol="SOLUSDT",
            interval=interval,
            limit=20,
            start=start_time,
            end=end_time,
        )
        
        print(f"Response code: {response['retCode']}")
        
        if response["retCode"] == 0:
            klines = response["result"]["list"]
            print(f"✅ Got {len(klines)} klines")
            if klines:
                for i, kline in enumerate(klines[:3]):
                    timestamp = datetime.fromtimestamp(int(kline[0]) / 1000)
                    print(f"  {i}: {timestamp} - O=${kline[1]}, H=${kline[2]}, L=${kline[3]}, C=${kline[4]}, V={kline[5]}")
            else:
                print("  No klines in response")
        else:
            print(f"❌ Error: {response['retMsg']}")

if __name__ == "__main__":
    asyncio.run(test_with_real_timestamps())