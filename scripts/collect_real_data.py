#!/usr/bin/env python3
"""Script to collect real historical data from Bybit using public API."""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import click
import json
import time

class RealDataCollector:
    """Collect real historical data from Bybit public API."""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 200):
        """Get klines from Bybit public API."""
        
        url = f"{self.base_url}/v5/market/kline"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'start': start_time,
            'end': end_time,
            'limit': limit
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('retCode') == 0:
                        klines = data.get('result', {}).get('list', [])
                        
                        # Convert to our format
                        candles = []
                        for kline in klines:
                            # Bybit format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
                            candles.append({
                                'timestamp': datetime.fromtimestamp(int(kline[0]) / 1000),
                                'open': float(kline[1]),
                                'high': float(kline[2]),
                                'low': float(kline[3]),
                                'close': float(kline[4]),
                                'volume': float(kline[5]),
                                'turnover': float(kline[6])
                            })
                        
                        return candles
                    else:
                        print(f"❌ API Error: {data.get('retMsg', 'Unknown error')}")
                        return []
                else:
                    print(f"❌ HTTP Error: {response.status}")
                    return []
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return []
    
    async def collect_historical_data(
        self,
        symbol: str,
        timeframes: list,
        days: int = 7,
        output_dir: str = "data",
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """Collect historical data for multiple timeframes."""
        
        Path(output_dir).mkdir(exist_ok=True)
        collected_data = {}
        
        print(f"🚀 Collecting real market data for {symbol}")
        if start_date or end_date:
            print(f"📅 Period: {start_date or 'beginning'} → {end_date or 'now'}")
        else:
            print(f"📅 Period: {days} days")
        print(f"⏱️  Timeframes: {', '.join(timeframes)}")
        print()
        
        # Timeframe mapping for Bybit API
        tf_mapping = {
            '1m': '1',
            '5m': '5', 
            '15m': '15',
            '1h': '60',
            '4h': '240',
            '1d': 'D'
        }
        
        for timeframe_str in timeframes:
            if timeframe_str not in tf_mapping:
                print(f"⚠️  Skipping unsupported timeframe: {timeframe_str}")
                continue
                
            bybit_interval = tf_mapping[timeframe_str]
            print(f"📊 Collecting {timeframe_str} data...")
            
            # Calculate time range
            if start_date or end_date:
                def _parse_date(s: str):
                    try:
                        return datetime.fromisoformat(s)
                    except Exception:
                        return datetime.strptime(s, "%Y-%m-%d")
                start_dt = _parse_date(start_date) if start_date else datetime.fromtimestamp(0)
                end_dt = _parse_date(end_date) if end_date else datetime.now()
                # Make end inclusive to end of day if only date
                if end_date and end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
                    end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999000)
                end_time = int(end_dt.timestamp() * 1000)
                start_time = int(start_dt.timestamp() * 1000)
            else:
                end_time = int(datetime.now().timestamp() * 1000)
                start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            all_candles = []
            current_end = end_time
            
            # Collect data in batches (Bybit returns max 200 candles per request)
            while True:
                try:
                    candles = await self.get_klines(
                        symbol=symbol,
                        interval=bybit_interval,
                        start_time=start_time,
                        end_time=current_end,
                        limit=200
                    )
                    
                    if not candles:
                        break
                    
                    # Sort by timestamp (Bybit returns newest first)
                    candles.sort(key=lambda x: x['timestamp'])
                    
                    # Add new candles
                    for candle in candles:
                        if candle['timestamp'] >= datetime.fromtimestamp(start_time / 1000) and candle['timestamp'] <= datetime.fromtimestamp(end_time / 1000):
                            all_candles.append(candle)
                    
                    print(f"  📈 Collected {len(candles)} candles (total: {len(all_candles)})")
                    
                    # Update for next batch (get older data)
                    if candles:
                        current_end = int(candles[0]['timestamp'].timestamp() * 1000) - 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                    # Stop if we've reached the start boundary
                    if current_end <= start_time:
                        break
                        
                except Exception as e:
                    print(f"  ❌ Error collecting {timeframe_str}: {e}")
                    await asyncio.sleep(1)
                    continue
            
            if all_candles:
                # Remove duplicates and sort
                df_data = []
                seen_timestamps = set()
                
                for candle in sorted(all_candles, key=lambda x: x['timestamp']):
                    if candle['timestamp'] not in seen_timestamps:
                        df_data.append(candle)
                        seen_timestamps.add(candle['timestamp'])
                
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Keep only requested period
                if start_date or end_date:
                    df = df[(df.index >= datetime.fromtimestamp(start_time / 1000)) & (df.index <= datetime.fromtimestamp(end_time / 1000))]
                else:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    df = df[df.index >= cutoff_date]
                
                # Save to file
                if start_date or end_date:
                    start_tag = datetime.fromtimestamp(start_time / 1000).strftime('%Y-%m-%d')
                    end_tag = datetime.fromtimestamp(end_time / 1000).strftime('%Y-%m-%d')
                    filename = f"{output_dir}/{symbol}_{timeframe_str}_real_{start_tag}_to_{end_tag}.csv"
                else:
                    filename = f"{output_dir}/{symbol}_{timeframe_str}_real_{days}d.csv"
                df.to_csv(filename)
                
                collected_data[timeframe_str] = df
                
                print(f"  ✅ Saved {len(df)} candles to {filename}")
                print(f"  📅 Data range: {df.index[0]} → {df.index[-1]}")
            else:
                print(f"  ❌ No data collected for {timeframe_str}")
            
            print()
        
        # Save metadata
        metadata = {
            "symbol": symbol,
            "collection_date": datetime.now().isoformat(),
            "days_collected": days,
            "timeframes": list(collected_data.keys()),
            "total_candles": {tf: len(df) for tf, df in collected_data.items()},
            "data_source": "bybit_public_api",
            "data_type": "real_market_data"
        }
        
        with open(f"{output_dir}/{symbol}_real_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return collected_data

@click.command()
@click.option("--symbol", default="SOLUSDT", help="Trading symbol")
@click.option("--days", default=7, help="Number of days to collect")
@click.option("--timeframes", default="1m,5m,15m,1h", help="Comma-separated timeframes")
@click.option("--output", default="data", help="Output directory")
@click.option("--start-date", help="Start date (YYYY-MM-DD or ISO) to collect from")
@click.option("--end-date", help="End date (YYYY-MM-DD or ISO) to collect to (inclusive)")
def main(symbol: str, days: int, timeframes: str, output: str, start_date: str, end_date: str):
    """Collect real historical data from Bybit."""
    
    print("🚀 Real Market Data Collector")
    print("=" * 50)
    print(f"📊 Symbol: {symbol}")
    print(f"📅 Days: {days}")
    print(f"⏱️  Timeframes: {timeframes}")
    print(f"🌐 Source: Bybit Public API (Real Market Data)")
    print()
    
    async def run_collection():
        async with RealDataCollector() as collector:
            timeframes_list = [tf.strip() for tf in timeframes.split(",")]
            
            try:
                data = await collector.collect_historical_data(
                    symbol=symbol,
                    timeframes=timeframes_list,
                    days=days,
                    output_dir=output,
                    start_date=start_date,
                    end_date=end_date,
                )
                
                if data:
                    print("🎉 Real data collection completed!")
                    print(f"📁 Files saved to: {output}/")
                    
                    # Show summary
                    print("\n📊 Summary:")
                    total_candles = 0
                    for tf, df in data.items():
                        print(f"  {tf:>4}: {len(df):>6} candles")
                        if len(df) > 0:
                            period_start_str = df.index[0].strftime("%Y-%m-%d %H:%M")
                            period_end_str = df.index[-1].strftime("%Y-%m-%d %H:%M")
                            print(f"        {period_start_str} → {period_end_str}")
                            
                            # Show some price statistics
                            price_min = df['low'].min()
                            price_max = df['high'].max()
                            price_start = df['close'].iloc[0]
                            price_end = df['close'].iloc[-1]
                            total_volume = df['volume'].sum()
                            
                            print(f"        Price: ${price_start:.2f} → ${price_end:.2f} (${price_min:.2f}-${price_max:.2f})")
                            print(f"        Volume: {total_volume:,.2f}")
                        total_candles += len(df)
                    
                    print(f"\n📈 Total candles collected: {total_candles:,}")
                    
                    # Calculate some market stats
                    if '5m' in data and len(data['5m']) > 0:
                        df_5m = data['5m']
                        returns = df_5m['close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(288) * 100  # Annualized
                        
                        # Price movements >= $2
                        price_changes = df_5m['close'].diff().abs()
                        movements_over_2 = (price_changes >= 2.0).sum()
                        
                        print(f"📊 Market Statistics (5m data):")
                        print(f"  • Volatility: {volatility:.1f}% (annualized)")
                        print(f"  • Movements ≥$2: {movements_over_2}")
                        print(f"  • Avg 5min return: {returns.mean()*100:.3f}%")
                        print(f"  • Max 5min move: ${price_changes.max():.2f}")
                
                else:
                    print("❌ No data collected")
            
            except Exception as e:
                print(f"❌ Collection failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Run async collection
    asyncio.run(run_collection())

if __name__ == "__main__":
    main()