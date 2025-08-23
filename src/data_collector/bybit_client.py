"""Bybit API client for data collection and trading."""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode

import aiohttp
import websockets
from pybit.unified_trading import HTTP

from ..utils.types import (
    Candle, Ticker, OrderBook, Trade, Order, Position, Balance, Account,
    Side, OrderType, OrderStatus, TimeFrame
)
from ..utils.logger import TradingLogger


class BybitHTTPClient:
    """Bybit HTTP API client."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        rate_limit: int = 10,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.rate_limit = rate_limit
        self.logger = TradingLogger("bybit_http")
        
        # Initialize pybit client
        self.client = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_interval = 1.0 / rate_limit
        
        # Session for custom requests
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self) -> None:
        """Ensure rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._request_interval:
            await asyncio.sleep(self._request_interval - time_since_last)
        
        self._last_request_time = time.time()
    
    def _generate_signature(
        self,
        timestamp: str,
        api_key: str,
        recv_window: str,
        query_string: str,
    ) -> str:
        """Generate API signature."""
        param_str = timestamp + api_key + recv_window + query_string
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature
    
    async def get_klines(
        self,
        symbol: str,
        interval: TimeFrame,
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Candle]:
        """Get candlestick data."""
        await self._rate_limit()
        
        try:
            print(f"        🔍 API call params: category=linear, symbol={symbol}, interval={interval.value}, limit={limit}, start={start_time}, end={end_time}")
            
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval.value,
                limit=limit,
                start=start_time,
                end=end_time,
            )
            
            print(f"        📊 API response: {response}")
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
            
            candles = []
            for kline in response["result"]["list"]:
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
                candles.append(candle)
            
            # Sort by timestamp (ascending)
            candles.sort(key=lambda x: x.timestamp)
            
            self.logger.log_market_data(
                symbol=symbol,
                timestamp=str(datetime.now()),
                price=candles[-1].close if candles else 0.0,
                volume=candles[-1].volume if candles else 0.0,
                data_source="bybit_http_klines",
            )
            
            return candles
        
        except Exception as e:
            self.logger.log_error(
                error_type="api_request_failed",
                component="bybit_http",
                error_message=str(e),
                details={"symbol": symbol, "interval": interval.value},
            )
            raise
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker data."""
        await self._rate_limit()
        
        try:
            response = self.client.get_tickers(
                category="linear",
                symbol=symbol,
            )
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
            
            ticker_data = response["result"]["list"][0]
            
            ticker = Ticker(
                symbol=symbol,
                price=float(ticker_data["lastPrice"]),
                bid=float(ticker_data["bid1Price"]),
                ask=float(ticker_data["ask1Price"]),
                volume_24h=float(ticker_data["volume24h"]),
                change_24h=float(ticker_data["price24hPcnt"]) * 100,
                timestamp=datetime.now(),
            )
            
            return ticker
        
        except Exception as e:
            self.logger.log_error(
                error_type="ticker_request_failed",
                component="bybit_http",
                error_message=str(e),
                details={"symbol": symbol},
            )
            raise
    
    async def get_orderbook(self, symbol: str, limit: int = 25) -> OrderBook:
        """Get order book data."""
        await self._rate_limit()
        
        try:
            response = self.client.get_orderbook(
                category="linear",
                symbol=symbol,
                limit=limit,
            )
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
            
            data = response["result"]
            
            bids = [(float(bid[0]), float(bid[1])) for bid in data["b"]]
            asks = [(float(ask[0]), float(ask[1])) for ask in data["a"]]
            
            orderbook = OrderBook(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(int(data["ts"]) / 1000),
                bids=bids,
                asks=asks,
            )
            
            return orderbook
        
        except Exception as e:
            self.logger.log_error(
                error_type="orderbook_request_failed",
                component="bybit_http",
                error_message=str(e),
                details={"symbol": symbol},
            )
            raise
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades."""
        await self._rate_limit()
        
        try:
            response = self.client.get_public_trade_history(
                category="linear",
                symbol=symbol,
                limit=limit,
            )
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
            
            trades = []
            for trade_data in response["result"]["list"]:
                trade = Trade(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(int(trade_data["time"]) / 1000),
                    side=Side(trade_data["side"].upper()),
                    price=float(trade_data["price"]),
                    quantity=float(trade_data["size"]),
                    trade_id=trade_data["execId"],
                )
                trades.append(trade)
            
            return trades
        
        except Exception as e:
            self.logger.log_error(
                error_type="trades_request_failed",
                component="bybit_http",
                error_message=str(e),
                details={"symbol": symbol},
            )
            raise
    
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate."""
        await self._rate_limit()
        
        try:
            response = self.client.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                limit=1,
            )
            
            if response["retCode"] != 0:
                return None
            
            funding_data = response["result"]["list"]
            if not funding_data:
                return None
            
            return float(funding_data[0]["fundingRate"])
        
        except Exception as e:
            self.logger.log_error(
                error_type="funding_rate_failed",
                component="bybit_http",
                error_message=str(e),
                details={"symbol": symbol},
            )
            return None
    
    async def get_open_interest(self, symbol: str) -> Optional[float]:
        """Get open interest."""
        await self._rate_limit()
        
        try:
            response = self.client.get_open_interest(
                category="linear",
                symbol=symbol,
                intervalTime="1h",
                limit=1,
            )
            
            if response["retCode"] != 0:
                return None
            
            oi_data = response["result"]["list"]
            if not oi_data:
                return None
            
            return float(oi_data[0]["openInterest"])
        
        except Exception as e:
            self.logger.log_error(
                error_type="open_interest_failed",
                component="bybit_http",
                error_message=str(e),
                details={"symbol": symbol},
            )
            return None
    
    async def get_long_short_ratio(self, symbol: str) -> Optional[float]:
        """Get long/short ratio."""
        await self._rate_limit()
        
        try:
            response = self.client.get_long_short_ratio(
                category="linear",
                symbol=symbol,
                period="1h",
                limit=1,
            )
            
            if response["retCode"] != 0:
                return None
            
            ratio_data = response["result"]["list"]
            if not ratio_data:
                return None
            
            buy_ratio = float(ratio_data[0]["buyRatio"])
            sell_ratio = float(ratio_data[0]["sellRatio"])
            
            if sell_ratio == 0:
                return None
            
            return buy_ratio / sell_ratio
        
        except Exception as e:
            self.logger.log_error(
                error_type="long_short_ratio_failed",
                component="bybit_http",
                error_message=str(e),
                details={"symbol": symbol},
            )
            return None
    
    async def get_account_info(self) -> Account:
        """Get account information."""
        await self._rate_limit()
        
        try:
            response = self.client.get_wallet_balance(
                accountType="UNIFIED",
            )
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
            
            account_data = response["result"]["list"][0]
            
            balances = []
            for coin in account_data["coin"]:
                balance = Balance(
                    asset=coin["coin"],
                    free=float(coin["availableToWithdraw"]),
                    locked=float(coin["locked"]),
                    total=float(coin["walletBalance"]),
                )
                balances.append(balance)
            
            account = Account(
                total_balance=float(account_data["totalWalletBalance"]),
                available_balance=float(account_data["totalAvailableBalance"]),
                margin_balance=float(account_data["totalMarginBalance"]),
                unrealized_pnl=float(account_data["totalPerpUPL"]),
                margin_ratio=float(account_data["accountMMR"]),
                balances=balances,
                timestamp=datetime.now(),
            )
            
            return account
        
        except Exception as e:
            self.logger.log_error(
                error_type="account_info_failed",
                component="bybit_http",
                error_message=str(e),
            )
            raise
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get positions."""
        await self._rate_limit()
        
        try:
            response = self.client.get_positions(
                category="linear",
                symbol=symbol,
            )
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
            
            positions = []
            for pos_data in response["result"]["list"]:
                if float(pos_data["size"]) == 0:
                    continue  # Skip empty positions
                
                position = Position(
                    symbol=pos_data["symbol"],
                    side=PositionSide(pos_data["side"].upper()),
                    size=float(pos_data["size"]),
                    entry_price=float(pos_data["avgPrice"]),
                    mark_price=float(pos_data["markPrice"]),
                    unrealized_pnl=float(pos_data["unrealisedPnl"]),
                    margin=float(pos_data["positionIM"]),
                    percentage=float(pos_data["unrealisedPnl"]) / float(pos_data["positionValue"]) * 100,
                    created_at=datetime.fromtimestamp(int(pos_data["createdTime"]) / 1000),
                    updated_at=datetime.fromtimestamp(int(pos_data["updatedTime"]) / 1000),
                )
                positions.append(position)
            
            return positions
        
        except Exception as e:
            self.logger.log_error(
                error_type="positions_request_failed",
                component="bybit_http",
                error_message=str(e),
                details={"symbol": symbol},
            )
            raise


class BybitWebSocketClient:
    """Bybit WebSocket client for real-time data."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.logger = TradingLogger("bybit_ws")
        
        # WebSocket URLs
        if testnet:
            self.public_url = "wss://stream-testnet.bybit.com/v5/public/linear"
            self.private_url = "wss://stream-testnet.bybit.com/v5/private"
        else:
            self.public_url = "wss://stream.bybit.com/v5/public/linear"
            self.private_url = "wss://stream.bybit.com/v5/private"
        
        # Connection state
        self.public_ws: Optional[websockets.WebSocketServerProtocol] = None
        self.private_ws: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        
        # Callbacks
        self.callbacks: Dict[str, List[callable]] = {
            "kline": [],
            "orderbook": [],
            "trade": [],
            "ticker": [],
            "liquidation": [],
            "order": [],
            "position": [],
            "execution": [],
        }
    
    def add_callback(self, event_type: str, callback: callable) -> None:
        """Add callback for specific event type."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    async def connect_public(self) -> None:
        """Connect to public WebSocket."""
        try:
            self.public_ws = await websockets.connect(self.public_url)
            self.logger.log_system_event(
                event_type="websocket_connected",
                component="bybit_ws",
                status="success",
                details={"type": "public"},
            )
        except Exception as e:
            self.logger.log_error(
                error_type="websocket_connection_failed",
                component="bybit_ws",
                error_message=str(e),
                details={"type": "public"},
            )
            raise
    
    async def connect_private(self) -> None:
        """Connect to private WebSocket with authentication."""
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials required for private WebSocket")
        
        try:
            self.private_ws = await websockets.connect(self.private_url)
            
            # Authenticate
            expires = int((time.time() + 60) * 1000)
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                f"GET/realtime{expires}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            
            auth_message = {
                "op": "auth",
                "args": [self.api_key, expires, signature],
            }
            
            await self.private_ws.send(json.dumps(auth_message))
            
            # Wait for auth response
            response = await self.private_ws.recv()
            auth_result = json.loads(response)
            
            if not auth_result.get("success"):
                raise Exception(f"Authentication failed: {auth_result}")
            
            self.logger.log_system_event(
                event_type="websocket_connected",
                component="bybit_ws",
                status="success",
                details={"type": "private", "authenticated": True},
            )
        
        except Exception as e:
            self.logger.log_error(
                error_type="websocket_connection_failed",
                component="bybit_ws",
                error_message=str(e),
                details={"type": "private"},
            )
            raise
    
    async def subscribe_kline(self, symbol: str, interval: TimeFrame) -> None:
        """Subscribe to kline updates."""
        if not self.public_ws:
            await self.connect_public()
        
        message = {
            "op": "subscribe",
            "args": [f"kline.{interval.value}.{symbol}"],
        }
        
        await self.public_ws.send(json.dumps(message))
    
    async def subscribe_orderbook(self, symbol: str, depth: int = 25) -> None:
        """Subscribe to orderbook updates."""
        if not self.public_ws:
            await self.connect_public()
        
        message = {
            "op": "subscribe",
            "args": [f"orderbook.{depth}.{symbol}"],
        }
        
        await self.public_ws.send(json.dumps(message))
    
    async def subscribe_trades(self, symbol: str) -> None:
        """Subscribe to trade updates."""
        if not self.public_ws:
            await self.connect_public()
        
        message = {
            "op": "subscribe",
            "args": [f"publicTrade.{symbol}"],
        }
        
        await self.public_ws.send(json.dumps(message))
    
    async def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to ticker updates."""
        if not self.public_ws:
            await self.connect_public()
        
        message = {
            "op": "subscribe",
            "args": [f"tickers.{symbol}"],
        }
        
        await self.public_ws.send(json.dumps(message))
    
    async def subscribe_liquidations(self, symbol: str) -> None:
        """Subscribe to liquidation updates."""
        if not self.public_ws:
            await self.connect_public()
        
        message = {
            "op": "subscribe",
            "args": [f"liquidation.{symbol}"],
        }
        
        await self.public_ws.send(json.dumps(message))
    
    async def subscribe_positions(self) -> None:
        """Subscribe to position updates."""
        if not self.private_ws:
            await self.connect_private()
        
        message = {
            "op": "subscribe",
            "args": ["position"],
        }
        
        await self.private_ws.send(json.dumps(message))
    
    async def subscribe_orders(self) -> None:
        """Subscribe to order updates."""
        if not self.private_ws:
            await self.connect_private()
        
        message = {
            "op": "subscribe",
            "args": ["order"],
        }
        
        await self.private_ws.send(json.dumps(message))
    
    async def listen_public(self) -> None:
        """Listen to public WebSocket messages."""
        if not self.public_ws:
            raise RuntimeError("Public WebSocket not connected")
        
        try:
            async for message in self.public_ws:
                data = json.loads(message)
                await self._handle_public_message(data)
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.log_error(
                error_type="websocket_disconnected",
                component="bybit_ws",
                error_message="Public WebSocket connection closed",
            )
        except Exception as e:
            self.logger.log_error(
                error_type="websocket_error",
                component="bybit_ws",
                error_message=str(e),
                details={"type": "public"},
            )
    
    async def listen_private(self) -> None:
        """Listen to private WebSocket messages."""
        if not self.private_ws:
            raise RuntimeError("Private WebSocket not connected")
        
        try:
            async for message in self.private_ws:
                data = json.loads(message)
                await self._handle_private_message(data)
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.log_error(
                error_type="websocket_disconnected",
                component="bybit_ws",
                error_message="Private WebSocket connection closed",
            )
        except Exception as e:
            self.logger.log_error(
                error_type="websocket_error",
                component="bybit_ws",
                error_message=str(e),
                details={"type": "private"},
            )
    
    async def _handle_public_message(self, data: Dict[str, Any]) -> None:
        """Handle public WebSocket message."""
        topic = data.get("topic", "")
        
        if topic.startswith("kline"):
            await self._handle_kline_update(data)
        elif topic.startswith("orderbook"):
            await self._handle_orderbook_update(data)
        elif topic.startswith("publicTrade"):
            await self._handle_trade_update(data)
        elif topic.startswith("tickers"):
            await self._handle_ticker_update(data)
        elif topic.startswith("liquidation"):
            await self._handle_liquidation_update(data)
    
    async def _handle_private_message(self, data: Dict[str, Any]) -> None:
        """Handle private WebSocket message."""
        topic = data.get("topic", "")
        
        if topic == "position":
            await self._handle_position_update(data)
        elif topic == "order":
            await self._handle_order_update(data)
        elif topic == "execution":
            await self._handle_execution_update(data)
    
    async def _handle_kline_update(self, data: Dict[str, Any]) -> None:
        """Handle kline update."""
        for kline_data in data.get("data", []):
            candle = Candle(
                timestamp=datetime.fromtimestamp(int(kline_data["start"]) / 1000),
                open=float(kline_data["open"]),
                high=float(kline_data["high"]),
                low=float(kline_data["low"]),
                close=float(kline_data["close"]),
                volume=float(kline_data["volume"]),
                symbol=kline_data["symbol"],
                timeframe=TimeFrame(kline_data["interval"]),
            )
            
            for callback in self.callbacks["kline"]:
                await callback(candle)
    
    async def _handle_orderbook_update(self, data: Dict[str, Any]) -> None:
        """Handle orderbook update."""
        orderbook_data = data.get("data", {})
        
        bids = [(float(bid[0]), float(bid[1])) for bid in orderbook_data.get("b", [])]
        asks = [(float(ask[0]), float(ask[1])) for ask in orderbook_data.get("a", [])]
        
        orderbook = OrderBook(
            symbol=orderbook_data["s"],
            timestamp=datetime.fromtimestamp(int(data["ts"]) / 1000),
            bids=bids,
            asks=asks,
        )
        
        for callback in self.callbacks["orderbook"]:
            await callback(orderbook)
    
    async def _handle_trade_update(self, data: Dict[str, Any]) -> None:
        """Handle trade update."""
        for trade_data in data.get("data", []):
            trade = Trade(
                symbol=trade_data["s"],
                timestamp=datetime.fromtimestamp(int(trade_data["T"]) / 1000),
                side=Side(trade_data["S"]),
                price=float(trade_data["p"]),
                quantity=float(trade_data["v"]),
                trade_id=trade_data["i"],
            )
            
            for callback in self.callbacks["trade"]:
                await callback(trade)
    
    async def _handle_ticker_update(self, data: Dict[str, Any]) -> None:
        """Handle ticker update."""
        ticker_data = data.get("data", {})
        
        ticker = Ticker(
            symbol=ticker_data["symbol"],
            price=float(ticker_data["lastPrice"]),
            bid=float(ticker_data["bid1Price"]),
            ask=float(ticker_data["ask1Price"]),
            volume_24h=float(ticker_data["volume24h"]),
            change_24h=float(ticker_data["price24hPcnt"]) * 100,
            timestamp=datetime.now(),
        )
        
        for callback in self.callbacks["ticker"]:
            await callback(ticker)
    
    async def _handle_liquidation_update(self, data: Dict[str, Any]) -> None:
        """Handle liquidation update."""
        for callback in self.callbacks["liquidation"]:
            await callback(data)
    
    async def _handle_position_update(self, data: Dict[str, Any]) -> None:
        """Handle position update."""
        for callback in self.callbacks["position"]:
            await callback(data)
    
    async def _handle_order_update(self, data: Dict[str, Any]) -> None:
        """Handle order update."""
        for callback in self.callbacks["order"]:
            await callback(data)
    
    async def _handle_execution_update(self, data: Dict[str, Any]) -> None:
        """Handle execution update."""
        for callback in self.callbacks["execution"]:
            await callback(data)
    
    async def close(self) -> None:
        """Close WebSocket connections."""
        if self.public_ws:
            await self.public_ws.close()
        if self.private_ws:
            await self.private_ws.close()
        
        self.is_connected = False
        
        self.logger.log_system_event(
            event_type="websocket_disconnected",
            component="bybit_ws",
            status="closed",
        )