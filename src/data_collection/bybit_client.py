"""
Simplified Bybit client for historical data collection.
No ML dependencies, only data collection functionality.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from pybit.unified_trading import HTTP


class TimeFrame(Enum):
    """Supported timeframes."""
    M1 = "1"
    M5 = "5" 
    M15 = "15"
    M30 = "30"
    H1 = "60"
    H2 = "120"
    H4 = "240"
    H6 = "360"
    H12 = "720"
    D1 = "D"
    
    @classmethod
    def from_string(cls, timeframe: str) -> 'TimeFrame':
        """Convert string to TimeFrame enum."""
        mapping = {
            "1m": cls.M1, "1": cls.M1,
            "5m": cls.M5, "5": cls.M5,
            "15m": cls.M15, "15": cls.M15,
            "30m": cls.M30, "30": cls.M30,
            "1h": cls.H1, "60": cls.H1, "60m": cls.H1,
            "2h": cls.H2, "120": cls.H2, "120m": cls.H2,
            "4h": cls.H4, "240": cls.H4, "240m": cls.H4,
            "6h": cls.H6, "360": cls.H6, "360m": cls.H6,
            "12h": cls.H12, "720": cls.H12, "720m": cls.H12,
            "1d": cls.D1, "D": cls.D1, "1D": cls.D1,
        }
        
        if timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        return mapping[timeframe]


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str


class BybitDataCollector:
    """Simplified Bybit client for historical data collection."""
    
    def __init__(self, testnet: bool = False, rate_limit: int = 10):
        self.testnet = testnet
        self.rate_limit = rate_limit
        
        # Initialize pybit client (no auth needed for public data)
        self.client = HTTP(testnet=testnet)
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_interval = 1.0 / rate_limit
    
    async def _rate_limit(self) -> None:
        """Ensure rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._request_interval:
            await asyncio.sleep(self._request_interval - time_since_last)
        
        self._last_request_time = time.time()
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit_per_request: int = 1000
    ) -> List[Candle]:
        """
        Collect historical OHLCV data for a symbol using block-based approach.
        
        Args:
            symbol: Trading pair (e.g., SOLUSDT)
            interval: Timeframe string (e.g., "5m", "1h")
            start_time: Start datetime
            end_time: End datetime
            limit_per_request: Max candles per API request
            
        Returns:
            List of Candle objects
        """
        
        # Convert timeframe
        timeframe = TimeFrame.from_string(interval)
        
        # Calculate timeframe in minutes for proper block calculation
        timeframe_minutes = self._get_timeframe_minutes(interval)
        
        # Calculate total expected candles (more precise calculation)
        total_duration_minutes = int((end_time - start_time).total_seconds() / 60)
        # Add 1 to include both start and end boundaries, then subtract 1 if end time is on exact boundary
        expected_candles = (total_duration_minutes // timeframe_minutes)
        
        # Check if we're exactly on timeframe boundary for end time
        if total_duration_minutes % timeframe_minutes == 0:
            expected_candles += 1  # Include the boundary candle
        
        print(f"📊 Collecting {symbol} {interval} data from {start_time} to {end_time}")
        print(f"📈 Expected candles: ~{expected_candles} ({total_duration_minutes} minutes / {timeframe_minutes}m intervals)")
        
        all_candles = []
        current_start = start_time
        block_number = 1
        
        while current_start < end_time:
            # Calculate block end time (limit by end_time)
            block_duration_minutes = limit_per_request * timeframe_minutes
            block_end = current_start + timedelta(minutes=block_duration_minutes)
            if block_end > end_time:
                block_end = end_time
            
            await self._rate_limit()
            
            try:
                print(f"   📦 Block {block_number}: {current_start.strftime('%m-%d %H:%M')} → {block_end.strftime('%m-%d %H:%M')}")
                
                # Convert to milliseconds for API
                start_ms = int(current_start.timestamp() * 1000)
                end_ms = int(block_end.timestamp() * 1000)
                
                response = self.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=timeframe.value,
                    limit=limit_per_request,
                    start=start_ms,
                    end=end_ms,
                )
                
                if response["retCode"] != 0:
                    raise Exception(f"API error: {response['retMsg']}")
                
                klines = response["result"]["list"]
                
                if not klines:
                    print(f"   ⚠️ No data in this block, moving to next")
                    # Move to next block
                    current_start = block_end
                    block_number += 1
                    continue
                
                # Process candles
                batch_candles = []
                for kline in klines:
                    candle_timestamp = datetime.fromtimestamp(int(kline[0]) / 1000)
                    
                    # Only include candles within our requested range
                    if start_time <= candle_timestamp <= end_time:
                        candle = Candle(
                            timestamp=candle_timestamp,
                            open=float(kline[1]),
                            high=float(kline[2]),
                            low=float(kline[3]),
                            close=float(kline[4]),
                            volume=float(kline[5]),
                            symbol=symbol,
                            timeframe=interval,
                        )
                        batch_candles.append(candle)
                
                # Sort by timestamp and add to collection
                batch_candles.sort(key=lambda x: x.timestamp)
                all_candles.extend(batch_candles)
                
                print(f"   ✅ Retrieved {len(batch_candles)} candles (Total: {len(all_candles)})")
                
                # Move to next block based on the last candle timestamp
                if batch_candles:
                    last_candle_time = batch_candles[-1].timestamp
                    # Next block starts from the next interval after last candle
                    current_start = last_candle_time + timedelta(minutes=timeframe_minutes)
                else:
                    # No candles in this block, move to next block
                    current_start = block_end
                
                block_number += 1
                
                # Progress indicator for large collections
                if block_number % 5 == 0:
                    progress = min(100, (len(all_candles) / expected_candles) * 100) if expected_candles > 0 else 0
                    print(f"   📊 Progress: {progress:.1f}% ({len(all_candles)} candles collected)")
                    
            except Exception as e:
                print(f"   ❌ Block {block_number} failed: {e}")
                # Add delay before retry
                await asyncio.sleep(2)
                # Move to next block to avoid getting stuck
                current_start = block_end
                block_number += 1
                continue
        
        # Remove duplicates and sort (enhanced deduplication)
        unique_candles = {}
        duplicates_found = 0
        
        for candle in all_candles:
            key = (candle.timestamp, candle.symbol, candle.timeframe)
            if key not in unique_candles:
                unique_candles[key] = candle
            else:
                duplicates_found += 1
        
        if duplicates_found > 0:
            print(f"   🔄 Removed {duplicates_found} duplicate candles")
        
        final_candles = list(unique_candles.values())
        final_candles.sort(key=lambda x: x.timestamp)
        
        # Final statistics with detailed analysis
        if final_candles and expected_candles > 0:
            # Calculate actual time range from data
            actual_start = min(candle.timestamp for candle in final_candles)
            actual_end = max(candle.timestamp for candle in final_candles)
            actual_duration = int((actual_end - actual_start).total_seconds() / 60)
            actual_expected = (actual_duration // timeframe_minutes) + 1
            
            theoretical_coverage = (len(final_candles) / expected_candles * 100)
            actual_coverage = (len(final_candles) / actual_expected * 100) if actual_expected > 0 else 100
            
            print(f"📈 Collection complete: {len(final_candles)} candles")
            print(f"   📊 Theoretical coverage: {theoretical_coverage:.1f}% ({len(final_candles)}/{expected_candles})")
            print(f"   📊 Actual coverage: {actual_coverage:.1f}% ({len(final_candles)}/{actual_expected}) - based on available data range")
            
            missing_candles = expected_candles - len(final_candles)
            if missing_candles > 0:
                print(f"   ⚠️ Missing {missing_candles} candles - likely due to no trading activity or exchange gaps")
        else:
            coverage_percent = (len(final_candles) / expected_candles * 100) if expected_candles > 0 else 100
            print(f"📈 Collection complete: {len(final_candles)} candles ({coverage_percent:.1f}% coverage)")
            
            if coverage_percent < 95:
                print(f"⚠️ Warning: Only {coverage_percent:.1f}% coverage. Some data may be missing from exchange.")
        
        return final_candles
    
    def _get_timeframe_minutes(self, interval: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe_map = {
            "1m": 1, "1": 1,
            "5m": 5, "5": 5,
            "15m": 15, "15": 15,
            "30m": 30, "30": 30,
            "1h": 60, "60": 60, "60m": 60,
            "2h": 120, "120": 120, "120m": 120,
            "4h": 240, "240": 240, "240m": 240,
            "6h": 360, "360": 360, "360m": 360,
            "12h": 720, "720": 720, "720m": 720,
            "1d": 1440, "D": 1440, "1D": 1440,
        }
        
        if interval not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {interval}")
            
        return timeframe_map[interval]
    
    async def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols."""
        try:
            await self._rate_limit()
            
            response = self.client.get_instruments_info(category="linear")
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
            
            symbols = []
            for instrument in response["result"]["list"]:
                if instrument["status"] == "Trading":
                    symbols.append(instrument["symbol"])
            
            return sorted(symbols)
            
        except Exception as e:
            print(f"❌ Failed to get symbols: {e}")
            return []
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid and tradeable."""
        try:
            await self._rate_limit()
            
            response = self.client.get_instruments_info(
                category="linear",
                symbol=symbol
            )
            
            if response["retCode"] != 0:
                return False
            
            instruments = response["result"]["list"]
            if not instruments:
                return False
            
            return instruments[0]["status"] == "Trading"
            
        except Exception:
            return False