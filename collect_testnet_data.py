#!/usr/bin/env python3
"""Simple testnet data collector."""

import asyncio
import aiohttp
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

async def collect_testnet_data():
    """Collect available data from Bybit testnet."""
    
    print("📊 Collecting Testnet Data for SOL/USDT")
    print("=" * 50)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Timeframes to collect
    timeframes = {
        "1m": "1",
        "5m": "5", 
        "15m": "15",
        "1h": "60"
    }
    
    collected_data = {}
    
    async with aiohttp.ClientSession() as session:
        for tf_name, tf_value in timeframes.items():
            print(f"📈 Collecting {tf_name} data...")
            
            url = "https://api-testnet.bybit.com/v5/market/kline"
            params = {
                "category": "linear",
                "symbol": "SOLUSDT",
                "interval": tf_value,
                "limit": 200  # Maximum available
            }
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("retCode") == 0:
                            klines = data["result"]["list"]
                            
                            if klines:
                                # Convert to DataFrame
                                df_data = []
                                for kline in reversed(klines):  # Reverse to get chronological order
                                    df_data.append({
                                        'timestamp': datetime.fromtimestamp(int(kline[0]) / 1000),
                                        'open': float(kline[1]),
                                        'high': float(kline[2]),
                                        'low': float(kline[3]),
                                        'close': float(kline[4]),
                                        'volume': float(kline[5]),
                                        'turnover': float(kline[6]) if len(kline) > 6 else 0.0
                                    })
                                
                                df = pd.DataFrame(df_data)
                                
                                # Save to CSV
                                filename = f"SOLUSDT_{tf_name}_testnet.csv"
                                filepath = data_dir / filename
                                df.to_csv(filepath, index=False)
                                
                                collected_data[tf_name] = len(df)
                                
                                print(f"  ✅ {tf_name}: {len(df)} candles")
                                print(f"     Range: {df.timestamp.iloc[0]} → {df.timestamp.iloc[-1]}")
                                print(f"     Price range: ${df.low.min():.2f} - ${df.high.max():.2f}")
                                print(f"     Avg volume: {df.volume.mean():.2f}")
                                print(f"     Saved to: {filename}")
                            else:
                                print(f"  ❌ {tf_name}: No data returned")
                        else:
                            print(f"  ❌ {tf_name}: API error {data.get('retMsg')}")
                    else:
                        print(f"  ❌ {tf_name}: HTTP {response.status}")
            except Exception as e:
                print(f"  ❌ {tf_name}: {e}")
    
    # Save metadata
    metadata = {
        "symbol": "SOLUSDT",
        "collection_date": datetime.now().isoformat(),
        "source": "bybit_testnet",
        "timeframes": list(collected_data.keys()),
        "total_candles": collected_data
    }
    
    with open(data_dir / "SOLUSDT_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎉 Data collection completed!")
    print(f"📁 Files saved to: {data_dir}/")
    print(f"📊 Total datasets: {len(collected_data)}")
    
    return collected_data

if __name__ == "__main__":
    asyncio.run(collect_testnet_data())