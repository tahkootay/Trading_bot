#!/usr/bin/env python3
"""Check account permissions and capabilities"""

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

async def check_permissions():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
    )
    
    print("🔍 Checking account capabilities...")
    
    # Check wallet balance
    try:
        print("\n💰 Wallet Balance:")
        balance = session.get_wallet_balance(accountType="UNIFIED")
        print(f"  Response: {balance}")
    except Exception as e:
        print(f"  ❌ Wallet balance error: {e}")
    
    # Check positions
    try:
        print("\n📊 Positions:")
        positions = session.get_positions(category="linear", symbol="SOLUSDT")
        print(f"  Response: {positions}")
    except Exception as e:
        print(f"  ❌ Positions error: {e}")
    
    # Check instruments info
    try:
        print("\n📋 Instruments Info for SOLUSDT:")
        instruments = session.get_instruments_info(category="linear", symbol="SOLUSDT")
        print(f"  Response: {instruments}")
    except Exception as e:
        print(f"  ❌ Instruments error: {e}")
    
    # Check if we can get orderbook
    try:
        print("\n📖 Orderbook:")
        orderbook = session.get_orderbook(category="linear", symbol="SOLUSDT")
        print(f"  Response retCode: {orderbook.get('retCode')}")
        if orderbook.get('retCode') == 0:
            bids = orderbook['result']['b'][:3] if orderbook['result']['b'] else []
            asks = orderbook['result']['a'][:3] if orderbook['result']['a'] else []
            print(f"  Bids: {bids}")
            print(f"  Asks: {asks}")
    except Exception as e:
        print(f"  ❌ Orderbook error: {e}")
    
    # Check different symbols
    print("\n🔍 Testing klines for different symbols:")
    symbols_to_test = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT"]
    
    for symbol in symbols_to_test:
        try:
            response = session.get_kline(
                category="linear",
                symbol=symbol,
                interval="1d",
                limit=1,
            )
            
            if response["retCode"] == 0:
                klines = response["result"]["list"]
                status = "✅ Data" if klines else "❌ Empty"
                print(f"  {symbol}: {status} ({len(klines)} klines)")
            else:
                print(f"  {symbol}: ❌ Error - {response['retMsg']}")
                
        except Exception as e:
            print(f"  {symbol}: ❌ Exception - {e}")

if __name__ == "__main__":
    asyncio.run(check_permissions())