"""
Order Flow Analysis Module

Analyzes market microstructure including delta, CVD, order book imbalance,
and absorption patterns for enhanced trade signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class OrderFlowSignals:
    """Container for order flow analysis results"""
    delta: float
    cvd: float
    cvd_trend: str
    imbalance: float
    large_orders: List[Dict]
    absorption: Optional[str]
    intensity: float
    aggressive_buyers: int
    aggressive_sellers: int
    footprint_data: Dict
    volume_clusters: List[Dict]

@dataclass
class Trade:
    """Individual trade data structure"""
    timestamp: float
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    is_aggressive: bool = True

@dataclass
class OrderBookLevel:
    """Order book level data structure"""
    price: float
    size: float

@dataclass
class OrderBook:
    """Order book snapshot"""
    timestamp: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float

class OrderFlowAnalyzer:
    """
    Advanced order flow analysis for detecting institutional activity,
    absorption, and market microstructure patterns.
    
    Aligned with algorithm specification for complete order flow analysis
    including delta, CVD, order book imbalance, and absorption patterns.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.delta_history = deque(maxlen=1000)
        self.cvd = 0.0
        self.trade_history = deque(maxlen=5000)
        self.orderbook_history = deque(maxlen=100)
        self.volume_profile = {}
        
        # Algorithm-specific parameters
        self.large_order_z_threshold = self.config.get('large_order_z_threshold', 3.0)
        self.absorption_volume_ratio = self.config.get('absorption_volume_ratio', 3.0)
        self.absorption_price_tolerance = self.config.get('absorption_price_tolerance', 0.001)
        
    def analyze(self, trades_data: List[Dict], orderbook_data: Dict) -> OrderFlowSignals:
        """
        Complete order flow analysis
        
        Args:
            trades_data: List of recent trades
            orderbook_data: Current order book snapshot
            
        Returns:
            OrderFlowSignals with comprehensive analysis
        """
        # Convert to internal format
        trades = self._parse_trades(trades_data)
        orderbook = self._parse_orderbook(orderbook_data)
        
        # Update internal state
        self._update_trade_history(trades)
        self._update_orderbook_history(orderbook)
        
        # Calculate delta and CVD
        delta = self._calculate_delta(trades)
        self.cvd = self._update_cvd(delta)
        cvd_trend = self._get_cvd_trend()
        
        # Order book analysis
        imbalance = self._calculate_imbalance(orderbook)
        
        # Large order detection
        large_orders = self._detect_large_orders(trades)
        
        # Absorption detection
        absorption = self._detect_absorption(trades, orderbook)
        
        # Trade intensity analysis
        intensity = self._calculate_trade_intensity(trades)
        
        # Aggressive activity analysis
        aggressive_buyers, aggressive_sellers = self._analyze_aggressive_activity(trades)
        
        # Footprint analysis (time & sales with volume at each price)
        footprint_data = self._create_footprint(trades)
        
        # Volume cluster analysis
        volume_clusters = self._identify_volume_clusters(trades)
        
        return OrderFlowSignals(
            delta=delta,
            cvd=self.cvd,
            cvd_trend=cvd_trend,
            imbalance=imbalance,
            large_orders=large_orders,
            absorption=absorption,
            intensity=intensity,
            aggressive_buyers=aggressive_buyers,
            aggressive_sellers=aggressive_sellers,
            footprint_data=footprint_data,
            volume_clusters=volume_clusters
        )
    
    def _parse_trades(self, trades_data: List[Dict]) -> List[Trade]:
        """Parse raw trade data into Trade objects"""
        trades = []
        for trade_dict in trades_data:
            try:
                trade = Trade(
                    timestamp=trade_dict.get('timestamp', time.time()),
                    price=float(trade_dict.get('price', 0)),
                    size=float(trade_dict.get('size', 0)),
                    side=trade_dict.get('side', 'buy'),
                    is_aggressive=trade_dict.get('is_aggressive', True)
                )
                trades.append(trade)
            except (ValueError, TypeError) as e:
                continue
        
        return trades
    
    def _parse_orderbook(self, orderbook_data: Dict) -> OrderBook:
        """Parse raw orderbook data into OrderBook object"""
        try:
            bids = []
            asks = []
            
            for bid in orderbook_data.get('bids', []):
                bids.append(OrderBookLevel(
                    price=float(bid[0]),
                    size=float(bid[1])
                ))
            
            for ask in orderbook_data.get('asks', []):
                asks.append(OrderBookLevel(
                    price=float(ask[0]),
                    size=float(ask[1])
                ))
            
            # Calculate spread
            spread = 0.0
            if bids and asks:
                spread = asks[0].price - bids[0].price
            
            return OrderBook(
                timestamp=orderbook_data.get('timestamp', time.time()),
                bids=bids,
                asks=asks,
                spread=spread
            )
            
        except (ValueError, TypeError, IndexError):
            return OrderBook(
                timestamp=time.time(),
                bids=[],
                asks=[],
                spread=0.0
            )
    
    def _update_trade_history(self, trades: List[Trade]):
        """Update internal trade history"""
        for trade in trades:
            self.trade_history.append(trade)
    
    def _update_orderbook_history(self, orderbook: OrderBook):
        """Update internal orderbook history"""
        self.orderbook_history.append(orderbook)
    
    def _calculate_delta(self, trades: List[Trade]) -> float:
        """
        Calculate volume delta (buy volume - sell volume)
        
        Args:
            trades: List of trades to analyze
            
        Returns:
            Volume delta
        """
        buy_volume = sum(t.size for t in trades if t.side == 'buy')
        sell_volume = sum(t.size for t in trades if t.side == 'sell')
        
        delta = buy_volume - sell_volume
        self.delta_history.append(delta)
        
        return delta
    
    def _update_cvd(self, delta: float) -> float:
        """Update cumulative volume delta"""
        self.cvd += delta
        return self.cvd
    
    def _get_cvd_trend(self) -> str:
        """
        Determine CVD trend based on recent history
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(self.delta_history) < 10:
            return 'neutral'
        
        recent_deltas = list(self.delta_history)[-10:]
        positive_count = sum(1 for d in recent_deltas if d > 0)
        
        if positive_count >= 7:
            return 'bullish'
        elif positive_count <= 3:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_imbalance(self, orderbook: OrderBook) -> float:
        """
        Calculate order book imbalance ratio
        
        Args:
            orderbook: Current order book
            
        Returns:
            Imbalance ratio (bid_volume / ask_volume)
        """
        if not orderbook.bids or not orderbook.asks:
            return 1.0
        
        # Sum top 10 levels
        bid_volume = sum(level.size for level in orderbook.bids[:10])
        ask_volume = sum(level.size for level in orderbook.asks[:10])
        
        if ask_volume == 0:
            return 10.0  # Heavy bid imbalance
        
        return bid_volume / ask_volume
    
    def _detect_large_orders(self, trades: List[Trade]) -> List[Dict]:
        """
        Detect unusually large orders based on z-score analysis
        
        Args:
            trades: Recent trades
            
        Returns:
            List of large order events
        """
        if len(self.trade_history) < 100:
            return []
        
        # Calculate volume statistics from history
        historical_sizes = [t.size for t in list(self.trade_history)[-500:]]
        mean_size = np.mean(historical_sizes)
        std_size = np.std(historical_sizes)
        
        if std_size == 0:
            return []
        
        large_orders = []
        for trade in trades:
            z_score = (trade.size - mean_size) / std_size
            
            if z_score >= self.large_order_z_threshold:
                large_orders.append({
                    'timestamp': trade.timestamp,
                    'price': trade.price,
                    'size': trade.size,
                    'side': trade.side,
                    'z_score': z_score,
                    'is_aggressive': trade.is_aggressive
                })
        
        return large_orders
    
    def _detect_absorption(self, trades: List[Trade], orderbook: OrderBook) -> Optional[str]:
        """
        Detect order absorption patterns as per algorithm specification
        
        Args:
            trades: Recent trades
            orderbook: Current order book
            
        Returns:
            'bullish_absorption', 'bearish_absorption', or None
        """
        if len(trades) < 5 or not orderbook.bids or not orderbook.asks:
            return None
        
        # Group trades by price levels
        price_volumes = {}
        for trade in trades:
            price_level = round(trade.price, 2)  # Round to nearest cent
            if price_level not in price_volumes:
                price_volumes[price_level] = {'buy': 0, 'sell': 0}
            
            price_volumes[price_level][trade.side] += trade.size
        
        # Check for absorption patterns
        for price, volumes in price_volumes.items():
            total_volume = volumes['buy'] + volumes['sell']
            
            # Find corresponding orderbook level
            bid_size = 0
            ask_size = 0
            
            for bid in orderbook.bids:
                if abs(bid.price - price) <= self.absorption_price_tolerance:
                    bid_size = bid.size
                    break
            
            for ask in orderbook.asks:
                if abs(ask.price - price) <= self.absorption_price_tolerance:
                    ask_size = ask.size
                    break
            
            # Check for bullish absorption (heavy selling absorbed at support)
            if (volumes['sell'] > volumes['buy'] * 2 and  # Heavy selling
                bid_size > 0 and  # Still bid support
                total_volume > self._get_average_volume() * self.absorption_volume_ratio):
                return 'bullish_absorption'
            
            # Check for bearish absorption (heavy buying absorbed at resistance)
            if (volumes['buy'] > volumes['sell'] * 2 and  # Heavy buying
                ask_size > 0 and  # Still ask resistance
                total_volume > self._get_average_volume() * self.absorption_volume_ratio):
                return 'bearish_absorption'
        
        return None
    
    def _calculate_trade_intensity(self, trades: List[Trade]) -> float:
        """
        Calculate trade intensity as per algorithm specification
        
        Args:
            trades: Recent trades
            
        Returns:
            Trade intensity multiplier (1.0 = normal, >1.0 = high intensity)
        """
        if not trades:
            return 1.0
        
        # Calculate current period metrics
        current_volume = sum(t.size for t in trades)
        current_count = len(trades)
        
        # Get historical averages
        if len(self.trade_history) < 100:
            return 1.0
        
        # Group historical trades by time windows
        historical_windows = self._group_trades_by_windows(list(self.trade_history), window_size=60)
        
        if not historical_windows:
            return 1.0
        
        avg_volume = np.mean([w['volume'] for w in historical_windows])
        avg_count = np.mean([w['count'] for w in historical_windows])
        
        # Calculate intensity scores
        volume_intensity = current_volume / avg_volume if avg_volume > 0 else 1.0
        count_intensity = current_count / avg_count if avg_count > 0 else 1.0
        
        # Weighted combination
        intensity = (volume_intensity * 0.7) + (count_intensity * 0.3)
        
        return intensity
    
    def _analyze_aggressive_activity(self, trades: List[Trade]) -> Tuple[int, int]:
        """
        Analyze aggressive buying vs selling activity
        
        Args:
            trades: Recent trades
            
        Returns:
            Tuple of (aggressive_buyers, aggressive_sellers)
        """
        aggressive_buyers = sum(1 for t in trades if t.side == 'buy' and t.is_aggressive)
        aggressive_sellers = sum(1 for t in trades if t.side == 'sell' and t.is_aggressive)
        
        return aggressive_buyers, aggressive_sellers
    
    def _create_footprint(self, trades: List[Trade]) -> Dict:
        """
        Create footprint chart data (volume at each price level)
        
        Args:
            trades: Recent trades
            
        Returns:
            Footprint data dictionary
        """
        footprint = {}
        
        for trade in trades:
            price_level = round(trade.price, 2)
            if price_level not in footprint:
                footprint[price_level] = {
                    'buy_volume': 0,
                    'sell_volume': 0,
                    'buy_count': 0,
                    'sell_count': 0
                }
            
            if trade.side == 'buy':
                footprint[price_level]['buy_volume'] += trade.size
                footprint[price_level]['buy_count'] += 1
            else:
                footprint[price_level]['sell_volume'] += trade.size
                footprint[price_level]['sell_count'] += 1
        
        return footprint
    
    def _identify_volume_clusters(self, trades: List[Trade]) -> List[Dict]:
        """
        Identify significant volume clusters
        
        Args:
            trades: Recent trades
            
        Returns:
            List of volume cluster data
        """
        if not trades:
            return []
        
        footprint = self._create_footprint(trades)
        clusters = []
        
        for price, data in footprint.items():
            total_volume = data['buy_volume'] + data['sell_volume']
            
            # Only consider significant volume levels
            if total_volume > self._get_average_volume() * 2:
                imbalance = abs(data['buy_volume'] - data['sell_volume']) / total_volume
                
                clusters.append({
                    'price': price,
                    'total_volume': total_volume,
                    'buy_volume': data['buy_volume'],
                    'sell_volume': data['sell_volume'],
                    'imbalance': imbalance,
                    'dominant_side': 'buy' if data['buy_volume'] > data['sell_volume'] else 'sell'
                })
        
        # Sort by volume descending
        clusters.sort(key=lambda x: x['total_volume'], reverse=True)
        
        return clusters[:10]  # Return top 10 clusters
    
    def _get_average_volume(self) -> float:
        """Get average trade volume from history"""
        if len(self.trade_history) < 10:
            return 1000.0  # Default fallback
        
        return np.mean([t.size for t in list(self.trade_history)[-100:]])
    
    def _group_trades_by_windows(self, trades: List[Trade], window_size: int = 60) -> List[Dict]:
        """
        Group trades into time windows for analysis
        
        Args:
            trades: List of trades
            window_size: Window size in seconds
            
        Returns:
            List of window statistics
        """
        if not trades:
            return []
        
        windows = []
        current_window = {
            'start_time': trades[0].timestamp,
            'volume': 0,
            'count': 0
        }
        
        for trade in trades:
            if trade.timestamp - current_window['start_time'] > window_size:
                # Close current window and start new one
                if current_window['count'] > 0:
                    windows.append(current_window)
                
                current_window = {
                    'start_time': trade.timestamp,
                    'volume': 0,
                    'count': 0
                }
            
            current_window['volume'] += trade.size
            current_window['count'] += 1
        
        # Add final window
        if current_window['count'] > 0:
            windows.append(current_window)
        
        return windows
    
    def get_signals_summary(self) -> Dict:
        """Get summary of current order flow signals"""
        return {
            'cvd': self.cvd,
            'cvd_trend': self._get_cvd_trend(),
            'delta_ma_10': np.mean(list(self.delta_history)[-10:]) if len(self.delta_history) >= 10 else 0,
            'trade_count': len(self.trade_history),
            'orderbook_count': len(self.orderbook_history),
            'recent_large_orders': len(self._detect_large_orders(list(self.trade_history)[-100:])),
            'avg_trade_size': self._get_average_volume()
        }
        
        # Large orders detection
        large_orders = self._detect_large_orders(trades)
        
        # Absorption detection
        absorption = self._detect_absorption(trades, orderbook)
        
        # Trade intensity
        intensity = self._calculate_trade_intensity(trades)
        
        # Aggressive flow analysis
        aggressive_buyers, aggressive_sellers = self._analyze_aggressive_flow(trades)
        
        # Footprint analysis
        footprint_data = self._generate_footprint(trades)
        
        # Volume clustering
        volume_clusters = self._identify_volume_clusters(trades)
        
        return OrderFlowSignals(
            delta=delta,
            cvd=self.cvd,
            cvd_trend=cvd_trend,
            imbalance=imbalance,
            large_orders=large_orders,
            absorption=absorption,
            intensity=intensity,
            aggressive_buyers=aggressive_buyers,
            aggressive_sellers=aggressive_sellers,
            footprint_data=footprint_data,
            volume_clusters=volume_clusters
        )
    
    def _parse_trades(self, trades_data: List[Dict]) -> List[Trade]:
        """Convert raw trade data to Trade objects"""
        trades = []
        for trade_data in trades_data:
            try:
                trade = Trade(
                    timestamp=trade_data.get('timestamp', time.time()),
                    price=float(trade_data['price']),
                    size=float(trade_data['size']),
                    side=trade_data['side'].lower(),
                    is_aggressive=trade_data.get('is_aggressive', True)
                )
                trades.append(trade)
            except (KeyError, ValueError, TypeError):
                continue
        return trades
    
    def _parse_orderbook(self, orderbook_data: Dict) -> OrderBook:
        """Convert raw orderbook data to OrderBook object"""
        try:
            bids = [
                OrderBookLevel(price=float(bid[0]), size=float(bid[1]))
                for bid in orderbook_data.get('bids', [])[:20]
            ]
            asks = [
                OrderBookLevel(price=float(ask[0]), size=float(ask[1]))
                for ask in orderbook_data.get('asks', [])[:20]
            ]
            
            spread = 0.0
            if bids and asks:
                spread = asks[0].price - bids[0].price
            
            return OrderBook(
                timestamp=orderbook_data.get('timestamp', time.time()),
                bids=bids,
                asks=asks,
                spread=spread
            )
        except (KeyError, ValueError, TypeError, IndexError):
            return OrderBook(timestamp=time.time(), bids=[], asks=[], spread=0.0)
    
    def _update_trade_history(self, trades: List[Trade]):
        """Update internal trade history"""
        for trade in trades:
            self.trade_history.append(trade)
    
    def _update_orderbook_history(self, orderbook: OrderBook):
        """Update internal orderbook history"""
        self.orderbook_history.append(orderbook)
    
    def _calculate_delta(self, trades: List[Trade]) -> float:
        """Calculate buy volume - sell volume"""
        buy_volume = sum(trade.size for trade in trades if trade.side == 'buy')
        sell_volume = sum(trade.size for trade in trades if trade.side == 'sell')
        return buy_volume - sell_volume
    
    def _update_cvd(self, delta: float) -> float:
        """Update Cumulative Volume Delta"""
        self.cvd += delta
        self.delta_history.append(delta)
        return self.cvd
    
    def _get_cvd_trend(self) -> str:
        """Determine CVD trend based on recent deltas"""
        if len(self.delta_history) < 20:
            return 'neutral'
        
        recent_deltas = list(self.delta_history)[-20:]
        recent_sum = sum(recent_deltas[-10:])
        older_sum = sum(recent_deltas[-20:-10])
        
        if recent_sum > older_sum * 1.2:
            return 'bullish'
        elif recent_sum < older_sum * 0.8:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_imbalance(self, orderbook: OrderBook, levels: int = 5) -> float:
        """Calculate order book imbalance ratio"""
        if not orderbook.bids or not orderbook.asks:
            return 1.0
        
        bid_volume = sum(bid.size for bid in orderbook.bids[:levels])
        ask_volume = sum(ask.size for ask in orderbook.asks[:levels])
        
        if ask_volume == 0:
            return 10.0  # Max bullish imbalance
        
        imbalance = bid_volume / ask_volume
        return min(max(imbalance, 0.1), 10.0)  # Cap between 0.1 and 10
    
    def _detect_large_orders(self, trades: List[Trade], window: int = 100) -> List[Dict]:
        """Detect unusually large orders using z-score analysis"""
        if len(self.trade_history) < window:
            return []
        
        # Get recent trade sizes for statistical analysis
        recent_trades = list(self.trade_history)[-window:]
        sizes = [trade.size for trade in recent_trades]
        
        if not sizes:
            return []
        
        avg_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        if std_size == 0:
            return []
        
        large_orders = []
        for trade in trades[-20:]:  # Check last 20 trades
            z_score = (trade.size - avg_size) / std_size
            if z_score > 3:  # 3 standard deviations above mean
                large_orders.append({
                    'timestamp': trade.timestamp,
                    'size': trade.size,
                    'side': trade.side,
                    'price': trade.price,
                    'z_score': z_score,
                    'significance': 'very_large' if z_score > 5 else 'large'
                })
        
        return large_orders
    
    def _detect_absorption(self, trades: List[Trade], orderbook: OrderBook) -> Optional[str]:
        """
        Detect absorption - high volume with minimal price movement
        indicating institutional absorption of retail flow
        """
        if len(trades) < 50:
            return None
        
        recent_trades = trades[-50:]
        total_volume = sum(trade.size for trade in recent_trades)
        
        if not recent_trades:
            return None
        
        # Calculate price range
        prices = [trade.price for trade in recent_trades]
        price_range = max(prices) - min(prices)
        avg_price = np.mean(prices)
        
        # Get baseline volume for comparison
        if len(self.trade_history) >= 200:
            baseline_trades = list(self.trade_history)[-200:-50]
            baseline_volume = sum(trade.size for trade in baseline_trades[-50:])
            volume_ratio = total_volume / (baseline_volume / 50 * 50) if baseline_volume > 0 else 1
        else:
            volume_ratio = 1
        
        # High volume with small price change indicates absorption
        price_change_pct = price_range / avg_price if avg_price > 0 else 0
        
        if volume_ratio > 3.0 and price_change_pct < 0.001:  # 0.1% price change
            # Determine absorption direction based on where price is holding
            if orderbook.bids and orderbook.asks:
                mid_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
                last_price = recent_trades[-1].price
                
                if abs(last_price - orderbook.bids[0].price) < abs(last_price - orderbook.asks[0].price):
                    return 'bid_absorption'  # Sellers being absorbed at bid
                else:
                    return 'ask_absorption'  # Buyers being absorbed at ask
        
        return None
    
    def _calculate_trade_intensity(self, trades: List[Trade], window_seconds: int = 60) -> float:
        """Calculate trade intensity (trades per second)"""
        if len(trades) < 2:
            return 0.0
        
        current_time = time.time()
        recent_trades = [
            trade for trade in trades 
            if current_time - trade.timestamp <= window_seconds
        ]
        
        if len(recent_trades) < 2:
            return 0.0
        
        time_span = max(trade.timestamp for trade in recent_trades) - \
                   min(trade.timestamp for trade in recent_trades)
        
        if time_span == 0:
            return 0.0
        
        return len(recent_trades) / time_span
    
    def _analyze_aggressive_flow(self, trades: List[Trade]) -> Tuple[int, int]:
        """Count aggressive buyers vs sellers"""
        aggressive_buyers = sum(
            1 for trade in trades 
            if trade.side == 'buy' and trade.is_aggressive
        )
        aggressive_sellers = sum(
            1 for trade in trades 
            if trade.side == 'sell' and trade.is_aggressive
        )
        
        return aggressive_buyers, aggressive_sellers
    
    def _generate_footprint(self, trades: List[Trade], price_levels: int = 20) -> Dict:
        """
        Generate market footprint showing volume at each price level
        """
        if not trades:
            return {}
        
        # Determine price range and levels
        prices = [trade.price for trade in trades]
        min_price = min(prices)
        max_price = max(prices)
        
        if min_price == max_price:
            return {min_price: {'buy_volume': 0, 'sell_volume': 0, 'total_volume': 0}}
        
        price_step = (max_price - min_price) / price_levels
        footprint = {}
        
        # Initialize price levels
        for i in range(price_levels + 1):
            level_price = min_price + i * price_step
            footprint[round(level_price, 4)] = {
                'buy_volume': 0,
                'sell_volume': 0,
                'total_volume': 0,
                'trade_count': 0
            }
        
        # Aggregate volume by price level
        for trade in trades:
            # Find closest price level
            level_index = int((trade.price - min_price) / price_step)
            level_index = min(level_index, price_levels)
            level_price = round(min_price + level_index * price_step, 4)
            
            if level_price in footprint:
                footprint[level_price]['total_volume'] += trade.size
                footprint[level_price]['trade_count'] += 1
                
                if trade.side == 'buy':
                    footprint[level_price]['buy_volume'] += trade.size
                else:
                    footprint[level_price]['sell_volume'] += trade.size
        
        return footprint
    
    def _identify_volume_clusters(self, trades: List[Trade], min_cluster_volume: float = None) -> List[Dict]:
        """
        Identify significant volume clusters that may act as support/resistance
        """
        footprint = self._generate_footprint(trades)
        
        if not footprint:
            return []
        
        # Calculate average volume to identify significant clusters
        total_volume = sum(level['total_volume'] for level in footprint.values())
        avg_volume = total_volume / len(footprint) if footprint else 0
        
        if min_cluster_volume is None:
            min_cluster_volume = avg_volume * 2  # 2x average volume
        
        clusters = []
        for price, data in footprint.items():
            if data['total_volume'] >= min_cluster_volume:
                # Calculate dominance (buy vs sell imbalance)
                total_vol = data['total_volume']
                buy_dominance = (data['buy_volume'] / total_vol) if total_vol > 0 else 0.5
                
                cluster = {
                    'price': price,
                    'total_volume': data['total_volume'],
                    'buy_volume': data['buy_volume'],
                    'sell_volume': data['sell_volume'],
                    'trade_count': data['trade_count'],
                    'buy_dominance': buy_dominance,
                    'sell_dominance': 1 - buy_dominance,
                    'significance': data['total_volume'] / avg_volume if avg_volume > 0 else 1,
                    'type': 'buy_cluster' if buy_dominance > 0.6 else 'sell_cluster' if buy_dominance < 0.4 else 'balanced'
                }
                clusters.append(cluster)
        
        # Sort by volume significance
        clusters.sort(key=lambda x: x['significance'], reverse=True)
        
        return clusters[:10]  # Return top 10 clusters
    
    def get_volume_profile_summary(self) -> Dict:
        """Get summary of current volume profile"""
        if not self.trade_history:
            return {}
        
        trades = list(self.trade_history)[-500:]  # Last 500 trades
        footprint = self._generate_footprint(trades)
        
        if not footprint:
            return {}
        
        # Find Point of Control (highest volume price)
        poc_price = max(footprint.keys(), key=lambda x: footprint[x]['total_volume'])
        poc_volume = footprint[poc_price]['total_volume']
        
        # Calculate Value Area (70% of total volume)
        total_volume = sum(level['total_volume'] for level in footprint.values())
        target_va_volume = total_volume * 0.7
        
        # Sort levels by volume
        sorted_levels = sorted(
            footprint.items(), 
            key=lambda x: x[1]['total_volume'], 
            reverse=True
        )
        
        va_volume = 0
        va_levels = []
        for price, data in sorted_levels:
            va_volume += data['total_volume']
            va_levels.append(price)
            if va_volume >= target_va_volume:
                break
        
        vah = max(va_levels) if va_levels else poc_price  # Value Area High
        val = min(va_levels) if va_levels else poc_price  # Value Area Low
        
        return {
            'poc': poc_price,
            'poc_volume': poc_volume,
            'vah': vah,
            'val': val,
            'total_volume': total_volume,
            'value_area_volume': va_volume,
            'levels_count': len(footprint)
        }
    
    def calculate_order_flow_score(self, signals: OrderFlowSignals) -> float:
        """
        Calculate overall order flow score (-1 to 1)
        Positive = Bullish, Negative = Bearish
        """
        score = 0.0
        
        # CVD trend contribution (30%)
        if signals.cvd_trend == 'bullish':
            score += 0.3
        elif signals.cvd_trend == 'bearish':
            score -= 0.3
        
        # Delta contribution (20%)
        delta_normalized = np.tanh(signals.delta / 1000)  # Normalize delta
        score += 0.2 * delta_normalized
        
        # Imbalance contribution (20%)
        if signals.imbalance > 1.5:
            score += 0.2 * min((signals.imbalance - 1) / 4, 1)  # Cap at 5:1 ratio
        elif signals.imbalance < 0.67:
            score += 0.2 * max((signals.imbalance - 1) / 4, -1)
        
        # Aggressive flow contribution (15%)
        if signals.aggressive_buyers > signals.aggressive_sellers:
            score += 0.15 * min((signals.aggressive_buyers - signals.aggressive_sellers) / 10, 1)
        elif signals.aggressive_sellers > signals.aggressive_buyers:
            score -= 0.15 * min((signals.aggressive_sellers - signals.aggressive_buyers) / 10, 1)
        
        # Absorption contribution (15%)
        if signals.absorption == 'bid_absorption':
            score += 0.15  # Bullish - sellers being absorbed
        elif signals.absorption == 'ask_absorption':
            score -= 0.15  # Bearish - buyers being absorbed
        
        return max(-1.0, min(1.0, score))  # Ensure score stays within bounds