"""Market data collector for aggregating data from multiple sources."""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque
import pandas as pd

from .bybit_client import BybitHTTPClient, BybitWebSocketClient
from ..utils.types import (
    Candle, Ticker, OrderBook, Trade, MarketData, TimeFrame
)
from ..utils.logger import TradingLogger


class MarketDataBuffer:
    """Buffer for storing market data in memory."""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.candles: Dict[TimeFrame, Deque[Candle]] = defaultdict(
            lambda: deque(maxlen=max_size)
        )
        self.trades: Deque[Trade] = deque(maxlen=1000)
        self.orderbooks: Deque[OrderBook] = deque(maxlen=100)
        self.tickers: Deque[Ticker] = deque(maxlen=100)
        
        # Latest data cache
        self.latest_ticker: Optional[Ticker] = None
        self.latest_orderbook: Optional[OrderBook] = None
    
    def add_candle(self, candle: Candle) -> None:
        """Add candle to buffer."""
        self.candles[candle.timeframe].append(candle)
    
    def add_trade(self, trade: Trade) -> None:
        """Add trade to buffer."""
        self.trades.append(trade)
    
    def add_orderbook(self, orderbook: OrderBook) -> None:
        """Add orderbook to buffer."""
        self.orderbooks.append(orderbook)
        self.latest_orderbook = orderbook
    
    def add_ticker(self, ticker: Ticker) -> None:
        """Add ticker to buffer."""
        self.tickers.append(ticker)
        self.latest_ticker = ticker
    
    def get_candles_df(self, timeframe: TimeFrame) -> pd.DataFrame:
        """Get candles as DataFrame."""
        candles = list(self.candles[timeframe])
        if not candles:
            return pd.DataFrame()
        
        data = {
            'timestamp': [c.timestamp for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles],
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_latest_price(self) -> Optional[float]:
        """Get latest price from ticker or candles."""
        if self.latest_ticker:
            return self.latest_ticker.price
        
        # Fallback to latest candle close
        for tf in [TimeFrame.M1, TimeFrame.M3, TimeFrame.M5]:
            if self.candles[tf]:
                return self.candles[tf][-1].close
        
        return None


class MarketDataCollector:
    """Aggregates market data from multiple sources."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        symbols: List[str] = None,
        timeframes: List[TimeFrame] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.symbols = symbols or ["SOLUSDT"]
        self.timeframes = timeframes or [
            TimeFrame.M1, TimeFrame.M3, TimeFrame.M5, 
            TimeFrame.M15, TimeFrame.H1
        ]
        
        self.logger = TradingLogger("market_data_collector")
        
        # Clients
        self.http_client: Optional[BybitHTTPClient] = None
        self.ws_client: Optional[BybitWebSocketClient] = None
        
        # Data buffers per symbol
        self.buffers: Dict[str, MarketDataBuffer] = {}
        for symbol in self.symbols:
            self.buffers[symbol] = MarketDataBuffer()
        
        # State tracking
        self.is_running = False
        self.last_kline_update: Dict[str, Dict[TimeFrame, datetime]] = defaultdict(
            lambda: defaultdict(lambda: datetime.min)
        )
        
        # Tasks
        self.tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start data collection."""
        if self.is_running:
            return
        
        self.logger.log_system_event(
            event_type="data_collection_start",
            component="market_data_collector",
            status="starting",
        )
        
        # Initialize clients
        self.http_client = BybitHTTPClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
        )
        
        self.ws_client = BybitWebSocketClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
        )
        
        # Setup WebSocket callbacks
        self.ws_client.add_callback("kline", self._on_kline_update)
        self.ws_client.add_callback("trade", self._on_trade_update)
        self.ws_client.add_callback("orderbook", self._on_orderbook_update)
        self.ws_client.add_callback("ticker", self._on_ticker_update)
        
        # Start tasks
        await self._start_tasks()
        
        self.is_running = True
        
        self.logger.log_system_event(
            event_type="data_collection_started",
            component="market_data_collector",
            status="running",
            details={"symbols": self.symbols, "timeframes": [tf.value for tf in self.timeframes]},
        )
    
    async def stop(self) -> None:
        """Stop data collection."""
        if not self.is_running:
            return
        
        self.logger.log_system_event(
            event_type="data_collection_stop",
            component="market_data_collector",
            status="stopping",
        )
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close clients
        if self.ws_client:
            await self.ws_client.close()
        
        if self.http_client:
            await self.http_client.__aexit__(None, None, None)
        
        self.is_running = False
        
        self.logger.log_system_event(
            event_type="data_collection_stopped",
            component="market_data_collector",
            status="stopped",
        )
    
    async def _start_tasks(self) -> None:
        """Start all background tasks."""
        # Historical data loading
        self.tasks.append(
            asyncio.create_task(self._load_historical_data())
        )
        
        # WebSocket connections
        self.tasks.append(
            asyncio.create_task(self._start_websocket_connections())
        )
        
        # Periodic data updates
        self.tasks.append(
            asyncio.create_task(self._periodic_data_updates())
        )
        
        # Data synchronization
        self.tasks.append(
            asyncio.create_task(self._sync_missing_data())
        )
    
    async def _load_historical_data(self) -> None:
        """Load initial historical data."""
        async with self.http_client:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    try:
                        # Load last 200 candles
                        candles = await self.http_client.get_klines(
                            symbol=symbol,
                            interval=timeframe,
                            limit=200,
                        )
                        
                        # Add to buffer
                        for candle in candles:
                            self.buffers[symbol].add_candle(candle)
                        
                        if candles:
                            self.last_kline_update[symbol][timeframe] = candles[-1].timestamp
                        
                        self.logger.log_market_data(
                            symbol=symbol,
                            timestamp=str(datetime.now()),
                            price=candles[-1].close if candles else 0.0,
                            volume=sum(c.volume for c in candles[-10:]),
                            data_source="historical_load",
                        )
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
                    
                    except Exception as e:
                        self.logger.log_error(
                            error_type="historical_data_load_failed",
                            component="market_data_collector",
                            error_message=str(e),
                            details={"symbol": symbol, "timeframe": timeframe.value},
                        )
    
    async def _start_websocket_connections(self) -> None:
        """Start WebSocket connections and subscriptions."""
        try:
            # Connect to public stream
            await self.ws_client.connect_public()
            
            # Subscribe to data streams
            for symbol in self.symbols:
                # Klines for all timeframes
                for timeframe in self.timeframes:
                    await self.ws_client.subscribe_kline(symbol, timeframe)
                
                # Other streams
                await self.ws_client.subscribe_trades(symbol)
                await self.ws_client.subscribe_orderbook(symbol, depth=25)
                await self.ws_client.subscribe_ticker(symbol)
                await self.ws_client.subscribe_liquidations(symbol)
            
            # Start listening
            await self.ws_client.listen_public()
        
        except Exception as e:
            self.logger.log_error(
                error_type="websocket_connection_failed",
                component="market_data_collector",
                error_message=str(e),
            )
            
            # Retry after delay
            await asyncio.sleep(5)
            if self.is_running:
                await self._start_websocket_connections()
    
    async def _periodic_data_updates(self) -> None:
        """Periodic updates for data not available via WebSocket."""
        while self.is_running:
            try:
                async with self.http_client:
                    for symbol in self.symbols:
                        # Update funding rate, open interest, etc.
                        funding_rate = await self.http_client.get_funding_rate(symbol)
                        open_interest = await self.http_client.get_open_interest(symbol)
                        long_short_ratio = await self.http_client.get_long_short_ratio(symbol)
                        
                        # Store in buffer metadata (could be extended)
                        buffer = self.buffers[symbol]
                        buffer.funding_rate = funding_rate
                        buffer.open_interest = open_interest
                        buffer.long_short_ratio = long_short_ratio
                
                # Update every 5 minutes
                await asyncio.sleep(300)
            
            except Exception as e:
                self.logger.log_error(
                    error_type="periodic_update_failed",
                    component="market_data_collector",
                    error_message=str(e),
                )
                await asyncio.sleep(60)
    
    async def _sync_missing_data(self) -> None:
        """Sync any missing data periodically."""
        while self.is_running:
            try:
                async with self.http_client:
                    for symbol in self.symbols:
                        for timeframe in self.timeframes:
                            # Check if we're missing recent data
                            last_update = self.last_kline_update[symbol][timeframe]
                            now = datetime.now()
                            
                            # Calculate expected interval
                            intervals = {
                                TimeFrame.M1: timedelta(minutes=1),
                                TimeFrame.M3: timedelta(minutes=3),
                                TimeFrame.M5: timedelta(minutes=5),
                                TimeFrame.M15: timedelta(minutes=15),
                                TimeFrame.M30: timedelta(minutes=30),
                                TimeFrame.H1: timedelta(hours=1),
                                TimeFrame.H4: timedelta(hours=4),
                                TimeFrame.D1: timedelta(days=1),
                            }
                            
                            interval = intervals.get(timeframe, timedelta(minutes=5))
                            
                            if now - last_update > interval * 2:  # Missing more than 2 intervals
                                # Fetch missing data
                                start_time = int(last_update.timestamp() * 1000)
                                candles = await self.http_client.get_klines(
                                    symbol=symbol,
                                    interval=timeframe,
                                    start_time=start_time,
                                    limit=100,
                                )
                                
                                # Add missing candles
                                for candle in candles:
                                    if candle.timestamp > last_update:
                                        self.buffers[symbol].add_candle(candle)
                                        self.last_kline_update[symbol][timeframe] = candle.timestamp
                
                # Check every minute
                await asyncio.sleep(60)
            
            except Exception as e:
                self.logger.log_error(
                    error_type="data_sync_failed",
                    component="market_data_collector",
                    error_message=str(e),
                )
                await asyncio.sleep(60)
    
    async def _on_kline_update(self, candle: Candle) -> None:
        """Handle kline WebSocket update."""
        if candle.symbol not in self.buffers:
            return
        
        self.buffers[candle.symbol].add_candle(candle)
        self.last_kline_update[candle.symbol][candle.timeframe] = candle.timestamp
        
        self.logger.log_market_data(
            symbol=candle.symbol,
            timestamp=str(candle.timestamp),
            price=candle.close,
            volume=candle.volume,
            data_source="websocket_kline",
        )
    
    async def _on_trade_update(self, trade: Trade) -> None:
        """Handle trade WebSocket update."""
        if trade.symbol not in self.buffers:
            return
        
        self.buffers[trade.symbol].add_trade(trade)
    
    async def _on_orderbook_update(self, orderbook: OrderBook) -> None:
        """Handle orderbook WebSocket update."""
        if orderbook.symbol not in self.buffers:
            return
        
        self.buffers[orderbook.symbol].add_orderbook(orderbook)
    
    async def _on_ticker_update(self, ticker: Ticker) -> None:
        """Handle ticker WebSocket update."""
        if ticker.symbol not in self.buffers:
            return
        
        self.buffers[ticker.symbol].add_ticker(ticker)
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get aggregated market data for symbol."""
        if symbol not in self.buffers:
            return None
        
        buffer = self.buffers[symbol]
        
        # Convert candles to DataFrames
        candles_df = {}
        for timeframe in self.timeframes:
            df = buffer.get_candles_df(timeframe)
            if not df.empty:
                candles_df[timeframe] = df
        
        if not candles_df:
            return None
        
        # Get recent trades
        recent_trades = list(buffer.trades)[-100:]  # Last 100 trades
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            candles=candles_df,
            ticker=buffer.latest_ticker,
            orderbook=buffer.latest_orderbook,
            recent_trades=recent_trades,
            funding_rate=getattr(buffer, 'funding_rate', None),
            open_interest=getattr(buffer, 'open_interest', None),
            long_short_ratio=getattr(buffer, 'long_short_ratio', None),
        )
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol."""
        if symbol not in self.buffers:
            return None
        
        return self.buffers[symbol].get_latest_price()
    
    def get_candles(self, symbol: str, timeframe: TimeFrame) -> pd.DataFrame:
        """Get candles DataFrame for symbol and timeframe."""
        if symbol not in self.buffers:
            return pd.DataFrame()
        
        return self.buffers[symbol].get_candles_df(timeframe)
    
    def is_data_ready(self, symbol: str, min_candles: int = 50) -> bool:
        """Check if sufficient data is available for analysis."""
        if symbol not in self.buffers:
            return False
        
        # Check if we have enough candles in primary timeframe
        primary_tf = TimeFrame.M5
        df = self.buffers[symbol].get_candles_df(primary_tf)
        
        return len(df) >= min_candles
    
    def get_health_status(self) -> Dict[str, any]:
        """Get collector health status."""
        status = {
            "is_running": self.is_running,
            "symbols": self.symbols,
            "timeframes": [tf.value for tf in self.timeframes],
            "data_status": {},
            "last_updates": {},
        }
        
        for symbol in self.symbols:
            if symbol in self.buffers:
                buffer = self.buffers[symbol]
                status["data_status"][symbol] = {
                    "has_ticker": buffer.latest_ticker is not None,
                    "has_orderbook": buffer.latest_orderbook is not None,
                    "trades_count": len(buffer.trades),
                    "candles_count": {
                        tf.value: len(buffer.candles[tf]) 
                        for tf in self.timeframes
                    },
                }
                
                status["last_updates"][symbol] = {
                    tf.value: self.last_kline_update[symbol][tf].isoformat()
                    if self.last_kline_update[symbol][tf] != datetime.min
                    else None
                    for tf in self.timeframes
                }
        
        return status