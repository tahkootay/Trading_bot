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
        Collect historical OHLCV data for a symbol.
        
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
        
        # Convert dates to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        all_candles = []
        current_start = start_ms
        
        print(f"📊 Collecting {symbol} {interval} data from {start_time} to {end_time}")
        
        while current_start < end_ms:
            await self._rate_limit()
            
            try:
                print(f"   📡 API request: start={datetime.fromtimestamp(current_start/1000)}, limit={limit_per_request}")
                
                response = self.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=timeframe.value,
                    limit=limit_per_request,
                    start=current_start,
                    end=end_ms,
                )
                
                if response["retCode"] != 0:
                    raise Exception(f"API error: {response['retMsg']}")
                
                klines = response["result"]["list"]
                
                if not klines:
                    print(f"   ⚠️ No more data available")
                    break
                
                # Process candles
                batch_candles = []
                for kline in klines:
                    candle = Candle(
                        timestamp=datetime.fromtimestamp(int(kline[0]) / 1000),
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
                
                print(f"   ✅ Retrieved {len(batch_candles)} candles")
                
                # Update start time for next request
                if batch_candles:
                    last_timestamp = int(batch_candles[-1].timestamp.timestamp() * 1000)
                    current_start = last_timestamp + 1
                else:
                    break
                
                # If we got fewer candles than requested, we've reached the end
                if len(batch_candles) < limit_per_request:
                    break
                    
            except Exception as e:
                print(f"   ❌ API request failed: {e}")
                # Add small delay before retry
                await asyncio.sleep(1)
                # Move to next time window to avoid getting stuck
                current_start += 60000 * limit_per_request  # Estimate next window
                continue
        
        # Remove duplicates and sort
        unique_candles = {}
        for candle in all_candles:
            key = candle.timestamp
            if key not in unique_candles:
                unique_candles[key] = candle
        
        final_candles = list(unique_candles.values())
        final_candles.sort(key=lambda x: x.timestamp)
        
        # Filter by date range (API might return data outside requested range)
        filtered_candles = [
            candle for candle in final_candles
            if start_time <= candle.timestamp <= end_time
        ]
        
        print(f"📈 Total collected: {len(filtered_candles)} candles")
        
        return filtered_candles
    
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