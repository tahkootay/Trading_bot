#!/usr/bin/env python3
"""Script to collect historical data from Bybit for backtesting."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import click
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_collector.bybit_client import BybitHTTPClient
from src.utils.types import TimeFrame
from src.utils.logger import setup_logging, TradingLogger


class DataCollector:
    """Collect historical data from Bybit."""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = BybitHTTPClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit=5,  # Conservative rate limit
        )
        self.logger = TradingLogger("data_collector")
    
    async def collect_historical_data(
        self,
        symbol: str,
        timeframes: list,
        days: int = 30,
        output_dir: str = "data",
    ) -> dict:
        """Collect historical data for multiple timeframes."""
        
        Path(output_dir).mkdir(exist_ok=True)
        collected_data = {}
        
        self.logger.log_system_event(
            event_type="data_collection_start",
            component="data_collector",
            status="starting",
            details={"symbol": symbol, "days": days, "timeframes": timeframes},
        )
        
        async with self.client:
            for timeframe_str in timeframes:
                try:
                    timeframe = TimeFrame(timeframe_str)
                    print(f"📊 Collecting {timeframe.value} data for {symbol}...")
                    
                    # Calculate how many candles we need
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=days)
                    
                    # Convert to milliseconds
                    start_timestamp = int(start_time.timestamp() * 1000)
                    end_timestamp = int(end_time.timestamp() * 1000)
                    
                    all_candles = []
                    current_start = start_timestamp
                    
                    # Bybit limits to 200 candles per request
                    while current_start < end_timestamp:
                        try:
                            candles = await self.client.get_klines(
                                symbol=symbol,
                                interval=timeframe,
                                limit=200,
                                start_time=current_start,
                                end_time=end_timestamp,
                            )
                            
                            if not candles:
                                break
                            
                            all_candles.extend(candles)
                            
                            # Update start time for next batch
                            last_candle_time = candles[-1].timestamp
                            current_start = int(last_candle_time.timestamp() * 1000) + 1
                            
                            print(f"  📈 Collected {len(candles)} candles (total: {len(all_candles)})")
                            
                            # Small delay to respect rate limits
                            await asyncio.sleep(0.2)
                            
                        except Exception as e:
                            self.logger.log_error(
                                error_type="candle_collection_failed",
                                component="data_collector",
                                error_message=str(e),
                                details={"timeframe": timeframe.value, "start": current_start},
                            )
                            await asyncio.sleep(1)
                            continue
                    
                    if all_candles:
                        # Convert to DataFrame
                        df_data = []
                        for candle in all_candles:
                            df_data.append({
                                'timestamp': candle.timestamp,
                                'open': candle.open,
                                'high': candle.high,
                                'low': candle.low,
                                'close': candle.close,
                                'volume': candle.volume,
                            })
                        
                        df = pd.DataFrame(df_data)
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Remove duplicates
                        df = df[~df.index.duplicated(keep='last')]
                        
                        # Save to file
                        filename = f"{output_dir}/{symbol}_{timeframe.value}_{days}d.csv"
                        df.to_csv(filename)
                        
                        collected_data[timeframe] = df
                        
                        print(f"  ✅ Saved {len(df)} candles to {filename}")
                        
                        self.logger.log_system_event(
                            event_type="timeframe_data_collected",
                            component="data_collector",
                            status="success",
                            details={
                                "timeframe": timeframe.value,
                                "candles": len(df),
                                "file": filename,
                            },
                        )
                    
                except Exception as e:
                    self.logger.log_error(
                        error_type="timeframe_collection_failed",
                        component="data_collector",
                        error_message=str(e),
                        details={"timeframe": timeframe_str},
                    )
                    print(f"  ❌ Failed to collect {timeframe_str} data: {e}")
        
        # Save metadata
        metadata = {
            "symbol": symbol,
            "collection_date": datetime.now().isoformat(),
            "days_collected": days,
            "timeframes": list(collected_data.keys()),
            "total_candles": {tf.value: len(df) for tf, df in collected_data.items()},
        }
        
        with open(f"{output_dir}/{symbol}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.log_system_event(
            event_type="data_collection_complete",
            component="data_collector",
            status="success",
            details=metadata,
        )
        
        return collected_data
    
    async def test_connection(self) -> bool:
        """Test connection to Bybit API."""
        try:
            async with self.client:
                ticker = await self.client.get_ticker("SOLUSDT")
                if ticker:
                    print(f"✅ Connection successful! SOL price: ${ticker.price:.2f}")
                    return True
                else:
                    print("❌ Failed to get ticker data")
                    return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def get_account_info(self) -> bool:
        """Get account information (testnet)."""
        try:
            async with self.client:
                account = await self.client.get_account_info()
                if account:
                    print(f"✅ Account info retrieved:")
                    print(f"  💰 Total Balance: ${account.total_balance:.2f}")
                    print(f"  💸 Available: ${account.available_balance:.2f}")
                    print(f"  📊 Unrealized P&L: ${account.unrealized_pnl:.2f}")
                    return True
                else:
                    print("❌ Failed to get account info")
                    return False
        except Exception as e:
            print(f"❌ Account info failed: {e}")
            return False


@click.command()
@click.option("--symbol", default="SOLUSDT", help="Trading symbol")
@click.option("--days", default=30, help="Number of days to collect")
@click.option("--timeframes", default="1m,5m,15m,1h", help="Comma-separated timeframes")
@click.option("--output", default="data", help="Output directory")
@click.option("--test-only", is_flag=True, help="Only test connection")
@click.option("--account-info", is_flag=True, help="Show account information")
def main(symbol: str, days: int, timeframes: str, output: str, test_only: bool, account_info: bool):
    """Collect historical data from Bybit."""
    
    # Setup logging
    setup_logging(log_level="INFO", log_format="text")
    
    print("🚀 Bybit Data Collector")
    print("=" * 50)
    
    # Load API credentials from environment
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in .env file")
        print("Please set BYBIT_API_KEY and BYBIT_API_SECRET")
        return
    
    print(f"🔑 Using API key: {api_key[:8]}...")
    print(f"🌐 Testnet mode: True")
    print()
    
    # Initialize collector
    collector = DataCollector(api_key, api_secret, testnet=True)
    
    async def run_tasks():
        # Test connection
        print("🔌 Testing connection...")
        if not await collector.test_connection():
            return
        
        # Show account info if requested
        if account_info:
            print("\n📊 Getting account info...")
            await collector.get_account_info()
        
        # Exit if test-only
        if test_only:
            print("\n✅ Connection test completed!")
            return
        
        # Collect data
        print(f"\n📈 Collecting data for {symbol}")
        print(f"📅 Period: {days} days")
        print(f"⏱️  Timeframes: {timeframes}")
        print()
        
        timeframes_list = [tf.strip() for tf in timeframes.split(",")]
        
        try:
            data = await collector.collect_historical_data(
                symbol=symbol,
                timeframes=timeframes_list,
                days=days,
                output_dir=output,
            )
            
            print(f"\n🎉 Data collection completed!")
            print(f"📁 Files saved to: {output}/")
            
            # Show summary
            print("\n📊 Summary:")
            for tf, df in data.items():
                print(f"  {tf.value:>4}: {len(df):>6} candles")
                if len(df) > 0:
                    start_date = df.index[0].strftime("%Y-%m-%d %H:%M")
                    end_date = df.index[-1].strftime("%Y-%m-%d %H:%M")
                    print(f"        {start_date} → {end_date}")
        
        except Exception as e:
            print(f"❌ Data collection failed: {e}")
    
    # Run async tasks
    asyncio.run(run_tasks())


if __name__ == "__main__":
    main()