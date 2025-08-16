"""Common type definitions and data structures."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd


class Side(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Order status."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class PositionSide(str, Enum):
    """Position side."""
    LONG = "LONG"
    SHORT = "SHORT"


class TimeFrame(str, Enum):
    """Timeframe for candlestick data."""
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class SignalType(str, Enum):
    """Signal type."""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class Candle:
    """Candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: TimeFrame


@dataclass
class Ticker:
    """Ticker data."""
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    timestamp: datetime


@dataclass
class OrderBook:
    """Order book data."""
    symbol: str
    timestamp: datetime
    bids: List[tuple[float, float]]  # [(price, quantity), ...]
    asks: List[tuple[float, float]]  # [(price, quantity), ...]


@dataclass
class Trade:
    """Trade data."""
    symbol: str
    timestamp: datetime
    side: Side
    price: float
    quantity: float
    trade_id: str


@dataclass
class Order:
    """Order data."""
    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    average_price: Optional[float]
    created_at: datetime
    updated_at: Optional[datetime]


@dataclass
class Position:
    """Position data."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    margin: float
    percentage: float
    created_at: datetime
    updated_at: datetime


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    confidence: float  # 0.0 to 1.0
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    features: Dict[str, float]


@dataclass
class Balance:
    """Account balance."""
    asset: str
    free: float
    locked: float
    total: float


@dataclass
class Account:
    """Account information."""
    total_balance: float
    available_balance: float
    margin_balance: float
    unrealized_pnl: float
    margin_ratio: float
    balances: List[Balance]
    timestamp: datetime


@dataclass
class MarketData:
    """Aggregated market data."""
    symbol: str
    timestamp: datetime
    candles: Dict[TimeFrame, pd.DataFrame]
    ticker: Ticker
    orderbook: OrderBook
    recent_trades: List[Trade]
    funding_rate: Optional[float]
    open_interest: Optional[float]
    long_short_ratio: Optional[float]


@dataclass
class TechnicalIndicators:
    """Technical indicators for a symbol."""
    symbol: str
    timestamp: datetime
    timeframe: TimeFrame
    
    # Moving averages
    ema_8: float
    ema_13: float
    ema_21: float
    ema_34: float
    ema_55: float
    sma_20: float
    hull_ma: float
    vwma: float
    
    # Momentum indicators
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    roc: float
    mfi: float
    
    # Volatility indicators
    atr: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    keltner_upper: float
    keltner_middle: float
    keltner_lower: float
    
    # Volume indicators
    vwap: float
    vwap_std1_upper: float
    vwap_std1_lower: float
    vwap_std2_upper: float
    vwap_std2_lower: float
    volume_sma: float
    volume_ratio: float
    
    # Trend indicators
    adx: float
    supertrend: float
    supertrend_direction: int
    parabolic_sar: float
    
    # Support/Resistance
    pivot_point: float
    resistance_1: float
    resistance_2: float
    support_1: float
    support_2: float


@dataclass
class RiskMetrics:
    """Risk management metrics."""
    timestamp: datetime
    account_balance: float
    total_exposure: float
    unrealized_pnl: float
    daily_pnl: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    correlation_matrix: Dict[str, Dict[str, float]]
    position_count: int
    max_position_size: float


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    total_pnl: float
    daily_pnl: float
    monthly_pnl: float
    
    
# Type aliases
Price = Union[float, Decimal]
Quantity = Union[float, Decimal]
Timestamp = Union[datetime, int, float]
Features = Dict[str, Union[float, int, bool]]
MLFeatures = np.ndarray
MLTarget = Union[float, int]