#!/usr/bin/env python3
"""Simple test script for Bybit connection."""

import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv


async def test_bybit_connection():
    """Test connection to Bybit testnet."""
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    print("🔗 Testing Bybit Testnet Connection")
    print("=" * 50)
    
    if not api_key or not api_secret:
        print("❌ API credentials not found!")
        print("Please set BYBIT_API_KEY and BYBIT_API_SECRET in .env file")
        return False
    
    print(f"API Key: {api_key[:8]}...")
    print(f"Testnet: True")
    print()
    
    # Test 1: Public API (no auth required)
    print("📊 Testing public API...")
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api-testnet.bybit.com/v5/market/tickers"
            params = {
                "category": "linear",
                "symbol": "SOLUSDT"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("retCode") == 0:
                        ticker = data["result"]["list"][0]
                        price = float(ticker["lastPrice"])
                        volume = float(ticker["volume24h"])
                        change = float(ticker["price24hPcnt"]) * 100
                        
                        print(f"✅ SOL/USDT Price: ${price:.2f}")
                        print(f"   24h Volume: {volume:,.0f}")
                        print(f"   24h Change: {change:+.2f}%")
                    else:
                        print(f"❌ API error: {data.get('retMsg', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ HTTP error: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    print()
    
    # Test 2: Orderbook
    print("📈 Testing orderbook...")
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api-testnet.bybit.com/v5/market/orderbook"
            params = {
                "category": "linear",
                "symbol": "SOLUSDT",
                "limit": 5
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("retCode") == 0:
                        orderbook = data["result"]
                        bids = orderbook["b"][:3]
                        asks = orderbook["a"][:3]
                        
                        print("✅ Orderbook received")
                        print("   Top 3 Bids:")
                        for i, (price, size) in enumerate(bids):
                            print(f"     {i+1}. ${float(price):.2f} × {float(size):.2f}")
                        print("   Top 3 Asks:")
                        for i, (price, size) in enumerate(asks):
                            print(f"     {i+1}. ${float(price):.2f} × {float(size):.2f}")
                    else:
                        print(f"❌ API error: {data.get('retMsg', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ HTTP error: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Orderbook test failed: {e}")
        return False
    
    print()
    
    # Test 3: Historical data
    print("📊 Testing historical data...")
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api-testnet.bybit.com/v5/market/kline"
            params = {
                "category": "linear",
                "symbol": "SOLUSDT",
                "interval": "5",
                "limit": 10
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("retCode") == 0:
                        klines = data["result"]["list"]
                        if klines:
                            latest = klines[0]  # Most recent candle
                            timestamp = int(latest[0])
                            open_price = float(latest[1])
                            high = float(latest[2])
                            low = float(latest[3])
                            close = float(latest[4])
                            volume = float(latest[5])
                            
                            from datetime import datetime
                            time_str = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
                            
                            print(f"✅ Historical data received: {len(klines)} candles")
                            print(f"   Latest 5m candle ({time_str}):")
                            print(f"     OHLC: ${open_price:.2f} ${high:.2f} ${low:.2f} ${close:.2f}")
                            print(f"     Volume: {volume:.2f}")
                        else:
                            print("❌ No historical data received")
                            return False
                    else:
                        print(f"❌ API error: {data.get('retMsg', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ HTTP error: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Historical data test failed: {e}")
        return False
    
    print()
    print("🎉 All public API tests passed!")
    print()
    print("✅ Bybit testnet connection is working properly.")
    print()
    print("📋 Next Steps:")
    print("1. Install full dependencies: pip install -r requirements.txt")
    print("2. Collect historical data: python scripts/collect_data.py")
    print("3. Run backtesting: python scripts/enhanced_backtest.py")
    print("4. Start paper trading: make run-paper")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_bybit_connection())
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit_code = 1
    
    exit(exit_code)