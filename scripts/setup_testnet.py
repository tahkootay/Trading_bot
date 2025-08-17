#!/usr/bin/env python3
"""Setup script for Bybit testnet integration."""

import asyncio
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_collector.bybit_client import BybitHTTPClient
from src.utils.logger import setup_logging


async def test_bybit_connection():
    """Test connection to Bybit testnet."""
    
    # Setup logging
    setup_logging(log_level="INFO", log_format="text")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ API credentials not found!")
        print("Please set BYBIT_API_KEY and BYBIT_API_SECRET in .env file")
        return False
    
    print("🔗 Testing Bybit Testnet Connection")
    print("=" * 50)
    print(f"API Key: {api_key[:8]}...")
    print(f"Testnet: True")
    print()
    
    try:
        client = BybitHTTPClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
            rate_limit=5,
        )
        
        async with client:
            # Test 1: Get ticker
            print("📊 Testing ticker data...")
            ticker = await client.get_ticker("SOLUSDT")
            if ticker:
                print(f"✅ SOL/USDT Price: ${ticker.price:.2f}")
                print(f"   Bid: ${ticker.bid:.2f}, Ask: ${ticker.ask:.2f}")
                print(f"   24h Volume: {ticker.volume_24h:,.0f}")
                print(f"   24h Change: {ticker.change_24h:+.2f}%")
            else:
                print("❌ Failed to get ticker")
                return False
            
            print()
            
            # Test 2: Get orderbook
            print("📈 Testing orderbook data...")
            orderbook = await client.get_orderbook("SOLUSDT", limit=5)
            if orderbook:
                print("✅ Orderbook received")
                print("   Top 3 Bids:")
                for i, (price, size) in enumerate(orderbook.bids[:3]):
                    print(f"     {i+1}. ${price:.2f} × {size:.2f}")
                print("   Top 3 Asks:")
                for i, (price, size) in enumerate(orderbook.asks[:3]):
                    print(f"     {i+1}. ${price:.2f} × {size:.2f}")
            else:
                print("❌ Failed to get orderbook")
                return False
            
            print()
            
            # Test 3: Get account info
            print("💰 Testing account info...")
            try:
                account = await client.get_account_info()
                if account:
                    print("✅ Account info received")
                    print(f"   Total Balance: ${account.total_balance:.2f}")
                    print(f"   Available: ${account.available_balance:.2f}")
                    print(f"   Unrealized P&L: ${account.unrealized_pnl:.2f}")
                    
                    print("   Balances:")
                    for balance in account.balances[:5]:  # Show first 5
                        if balance.total > 0:
                            print(f"     {balance.asset}: {balance.total:.4f}")
                else:
                    print("❌ Failed to get account info")
            except Exception as e:
                print(f"⚠️  Account info failed: {e}")
                print("   This is normal if using read-only API keys")
            
            print()
            
            # Test 4: Get historical data
            print("📊 Testing historical data...")
            try:
                from src.utils.types import TimeFrame
                candles = await client.get_klines(
                    symbol="SOLUSDT",
                    interval=TimeFrame.M5,
                    limit=10,
                )
                if candles:
                    print(f"✅ Historical data received: {len(candles)} candles")
                    latest = candles[-1]
                    print(f"   Latest 5m candle:")
                    print(f"     Time: {latest.timestamp}")
                    print(f"     OHLC: ${latest.open:.2f} ${latest.high:.2f} ${latest.low:.2f} ${latest.close:.2f}")
                    print(f"     Volume: {latest.volume:.2f}")
                else:
                    print("❌ Failed to get historical data")
                    return False
            except Exception as e:
                print(f"❌ Historical data failed: {e}")
                return False
        
        print()
        print("🎉 All tests passed! Bybit testnet connection is working.")
        print()
        print("Next steps:")
        print("1. Collect historical data: python scripts/collect_data.py")
        print("2. Run backtest: python scripts/enhanced_backtest.py")
        print("3. Start paper trading: make run-paper")
        
        return True
    
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_bybit_connection())
    sys.exit(0 if success else 1)