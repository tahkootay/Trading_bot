#!/usr/bin/env python3
"""Test with server time as reference"""

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

async def test_with_server_time():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
    )
    
    # First, get server time from any response
    response = session.get_kline(
        category="linear",
        symbol="SOLUSDT",
        interval="1h",
        limit=1,
    )
    
    server_time = response.get("time", 0)
    print(f"Server time: {server_time}")
    print(f"Server time as date: {datetime.fromtimestamp(server_time / 1000)}")
    
    # Calculate time ranges based on server time
    end_time = server_time
    start_time = server_time - (24 * 60 * 60 * 1000)  # 24 hours ago
    
    print(f"Using time range: {start_time} to {end_time}")
    print(f"Start: {datetime.fromtimestamp(start_time / 1000)}")
    print(f"End: {datetime.fromtimestamp(end_time / 1000)}")
    
    # Test with time range
    response2 = session.get_kline(
        category="linear",
        symbol="SOLUSDT",
        interval="1h",
        limit=50,
        start=start_time,
        end=end_time,
    )
    
    print(f"\nWith time range: {response2}")
    
    if response2["retCode"] == 0:
        klines = response2["result"]["list"]
        print(f"✅ Got {len(klines)} klines with time range")
        if klines:
            for i, kline in enumerate(klines[:3]):
                print(f"  {i}: Time={kline[0]}, O={kline[1]}, H={kline[2]}, L={kline[3]}, C={kline[4]}")
    else:
        print(f"❌ Error: {response2['retMsg']}")

if __name__ == "__main__":
    asyncio.run(test_with_server_time())