"""
Volume Profile Analysis for SOL/USDT Trading Algorithm

Implements volume profile calculation with POC, VAH, VAL identification
and liquidity pools detection as specified in the algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

@dataclass
class VolumeProfileNode:
    """Volume profile price level node"""
    price: float
    volume: float
    buy_volume: float
    sell_volume: float
    trades_count: int
    percentage: float

@dataclass
class VolumeProfile:
    """Complete volume profile data"""
    nodes: List[VolumeProfileNode]
    poc: float  # Point of Control
    vah: float  # Value Area High
    val: float  # Value Area Low
    total_volume: float
    value_area_volume: float
    volume_distribution: Dict[str, float]

@dataclass
class LiquidityPool:
    """Liquidity pool identification"""
    price: float
    volume: float
    pool_type: str  # 'support', 'resistance', 'neutral'
    strength: float  # 0-100 score
    orders_count: int
    avg_order_size: float
    time_at_level: float  # seconds
    breaks_count: int
    holds_count: int

class VolumeProfileAnalyzer:
    """
    Volume Profile and Liquidity Pool analyzer
    
    Implements algorithm specification for:
    - Volume profile calculation (TPO style)
    - POC, VAH, VAL identification
    - Liquidity pools detection
    - Volume clustering analysis
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Algorithm constants
        self.VALUE_AREA_PERCENTAGE = 0.68  # 68% of volume for value area
        self.MIN_POOL_VOLUME = self.config.get('min_pool_volume', 1000)
        self.PRICE_RESOLUTION = self.config.get('price_resolution', 0.01)
        self.LIQUIDITY_THRESHOLD = self.config.get('liquidity_threshold', 0.02)  # 2% of total volume
        
    def calculate_volume_profile(
        self, 
        trades_data: List[Dict], 
        price_levels: int = 50
    ) -> VolumeProfile:
        """
        Calculate volume profile from trades data as per algorithm specification
        
        Args:
            trades_data: List of trade executions with price, size, side
            price_levels: Number of price levels for profile calculation
            
        Returns:
            VolumeProfile with POC, VAH, VAL and detailed nodes
        """
        try:
            if not trades_data:
                return self._empty_volume_profile()
            
            # Extract price and volume data
            prices = [float(trade['price']) for trade in trades_data]
            volumes = [float(trade['size']) for trade in trades_data]
            sides = [trade.get('side', 'buy') for trade in trades_data]
            
            if not prices or not volumes:
                return self._empty_volume_profile()
            
            # Determine price range
            min_price = min(prices)
            max_price = max(prices)
            
            if min_price == max_price:
                return self._single_price_profile(min_price, sum(volumes))
            
            # Create price levels
            price_step = (max_price - min_price) / price_levels
            profile_levels = []
            
            for i in range(price_levels + 1):
                level_price = min_price + i * price_step
                profile_levels.append(level_price)
            
            # Aggregate volume by price levels
            volume_by_level = defaultdict(lambda: {
                'total_volume': 0.0,
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'trades_count': 0
            })
            
            for price, volume, side in zip(prices, volumes, sides):
                # Find closest price level
                level_index = min(
                    int((price - min_price) / price_step),
                    price_levels
                )
                level_price = profile_levels[level_index]
                
                volume_by_level[level_price]['total_volume'] += volume
                volume_by_level[level_price]['trades_count'] += 1
                
                if side.lower() == 'buy':
                    volume_by_level[level_price]['buy_volume'] += volume
                else:
                    volume_by_level[level_price]['sell_volume'] += volume
            
            # Calculate total volume
            total_volume = sum(level['total_volume'] for level in volume_by_level.values())
            
            # Create profile nodes
            nodes = []
            for price, data in volume_by_level.items():
                if data['total_volume'] > 0:
                    percentage = (data['total_volume'] / total_volume) * 100
                    
                    node = VolumeProfileNode(
                        price=price,
                        volume=data['total_volume'],
                        buy_volume=data['buy_volume'],
                        sell_volume=data['sell_volume'],
                        trades_count=data['trades_count'],
                        percentage=percentage
                    )
                    nodes.append(node)
            
            # Sort nodes by price
            nodes.sort(key=lambda x: x.price)
            
            if not nodes:
                return self._empty_volume_profile()
            
            # Find POC (Point of Control) - highest volume level
            poc_node = max(nodes, key=lambda x: x.volume)
            poc = poc_node.price
            
            # Calculate Value Area (68% of total volume around POC)
            vah, val, value_area_volume = self._calculate_value_area(nodes, poc, total_volume)
            
            # Calculate volume distribution
            volume_distribution = self._calculate_volume_distribution(nodes, total_volume)
            
            return VolumeProfile(
                nodes=nodes,
                poc=poc,
                vah=vah,
                val=val,
                total_volume=total_volume,
                value_area_volume=value_area_volume,
                volume_distribution=volume_distribution
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return self._empty_volume_profile()
    
    def _calculate_value_area(
        self, 
        nodes: List[VolumeProfileNode], 
        poc: float, 
        total_volume: float
    ) -> Tuple[float, float, float]:
        """Calculate Value Area High (VAH) and Value Area Low (VAL)"""
        try:
            target_volume = total_volume * self.VALUE_AREA_PERCENTAGE
            
            # Sort nodes by price to find POC node
            nodes_by_price = sorted(nodes, key=lambda x: x.price)
            poc_index = next(
                i for i, node in enumerate(nodes_by_price) 
                if node.price == poc
            )
            
            # Start from POC and expand outward
            included_volume = nodes_by_price[poc_index].volume
            low_index = poc_index
            high_index = poc_index
            
            # Expand value area until we reach target volume
            while included_volume < target_volume:
                # Determine which direction to expand
                can_expand_up = high_index < len(nodes_by_price) - 1
                can_expand_down = low_index > 0
                
                if not can_expand_up and not can_expand_down:
                    break
                
                volume_up = nodes_by_price[high_index + 1].volume if can_expand_up else 0
                volume_down = nodes_by_price[low_index - 1].volume if can_expand_down else 0
                
                # Expand in direction with higher volume
                if can_expand_up and (not can_expand_down or volume_up >= volume_down):
                    high_index += 1
                    included_volume += nodes_by_price[high_index].volume
                elif can_expand_down:
                    low_index -= 1
                    included_volume += nodes_by_price[low_index].volume
                else:
                    break
            
            vah = nodes_by_price[high_index].price
            val = nodes_by_price[low_index].price
            
            return vah, val, included_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating value area: {e}")
            # Fallback to simple percentage of price range
            price_range = max(node.price for node in nodes) - min(node.price for node in nodes)
            return poc + price_range * 0.1, poc - price_range * 0.1, total_volume * 0.68
    
    def _calculate_volume_distribution(
        self, 
        nodes: List[VolumeProfileNode], 
        total_volume: float
    ) -> Dict[str, float]:
        """Calculate volume distribution statistics"""
        try:
            if not nodes:
                return {}
            
            # Calculate various distribution metrics
            volumes = [node.volume for node in nodes]
            
            distribution = {
                'max_volume': max(volumes),
                'min_volume': min(volumes),
                'avg_volume': np.mean(volumes),
                'std_volume': np.std(volumes),
                'median_volume': np.median(volumes),
                'volume_concentration': max(volumes) / total_volume,  # POC concentration
                'price_range': max(node.price for node in nodes) - min(node.price for node in nodes)
            }
            
            # Calculate skewness (where is most volume concentrated)
            prices = [node.price for node in nodes]
            weighted_avg_price = sum(node.price * node.volume for node in nodes) / total_volume
            min_price = min(prices)
            max_price = max(prices)
            
            # Skewness: -1 (bottom heavy) to +1 (top heavy)
            distribution['price_skewness'] = (weighted_avg_price - min_price) / (max_price - min_price) * 2 - 1
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error calculating volume distribution: {e}")
            return {}
    
    def detect_liquidity_pools(
        self, 
        orderbook_data: Dict, 
        volume_profile: VolumeProfile = None
    ) -> List[LiquidityPool]:
        """
        Detect liquidity pools from orderbook data as per algorithm specification
        
        Args:
            orderbook_data: Current orderbook snapshot
            volume_profile: Optional volume profile for additional context
            
        Returns:
            List of identified liquidity pools sorted by strength
        """
        try:
            pools = []
            
            if not orderbook_data:
                return pools
            
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            if not bids or not asks:
                return pools
            
            # Analyze bid side (support pools)
            bid_pools = self._analyze_orderbook_side(bids, 'support')
            pools.extend(bid_pools)
            
            # Analyze ask side (resistance pools)
            ask_pools = self._analyze_orderbook_side(asks, 'resistance')
            pools.extend(ask_pools)
            
            # Enhance with volume profile data if available
            if volume_profile:
                pools = self._enhance_pools_with_volume_profile(pools, volume_profile)
            
            # Sort by strength and return top pools
            pools.sort(key=lambda x: x.strength, reverse=True)
            
            return pools[:20]  # Top 20 pools
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity pools: {e}")
            return []
    
    def _analyze_orderbook_side(self, orders: List[List], pool_type: str) -> List[LiquidityPool]:
        """Analyze one side of orderbook for liquidity pools"""
        try:
            pools = []
            
            if len(orders) < 2:
                return pools
            
            # Calculate total volume for this side
            total_side_volume = sum(float(order[1]) for order in orders)
            
            if total_side_volume == 0:
                return pools
            
            # Group nearby price levels
            price_groups = self._group_nearby_prices(orders)
            
            for group in price_groups:
                if not group:
                    continue
                
                # Calculate group statistics
                group_volume = sum(float(order[1]) for order in group)
                group_price = sum(float(order[0]) * float(order[1]) for order in group) / group_volume
                orders_count = len(group)
                avg_order_size = group_volume / orders_count
                
                # Calculate strength based on volume relative to total
                volume_strength = (group_volume / total_side_volume) * 100
                
                # Size consistency bonus
                order_sizes = [float(order[1]) for order in group]
                size_std = np.std(order_sizes) / np.mean(order_sizes) if np.mean(order_sizes) > 0 else 1
                consistency_bonus = max(0, (1 - size_std) * 20)  # Up to 20 points
                
                # Final strength calculation
                strength = min(100, volume_strength + consistency_bonus)
                
                # Only include significant pools
                if strength >= 5.0 and group_volume >= self.MIN_POOL_VOLUME:
                    pool = LiquidityPool(
                        price=group_price,
                        volume=group_volume,
                        pool_type=pool_type,
                        strength=strength,
                        orders_count=orders_count,
                        avg_order_size=avg_order_size,
                        time_at_level=0.0,  # Would be calculated from historical data
                        breaks_count=0,     # Would be calculated from historical data
                        holds_count=0       # Would be calculated from historical data
                    )
                    pools.append(pool)
            
            return pools
            
        except Exception as e:
            self.logger.error(f"Error analyzing orderbook side: {e}")
            return []
    
    def _group_nearby_prices(self, orders: List[List], tolerance_pct: float = 0.001) -> List[List]:
        """Group orders with similar prices (within tolerance)"""
        try:
            if not orders:
                return []
            
            # Sort orders by price
            sorted_orders = sorted(orders, key=lambda x: float(x[0]))
            groups = []
            current_group = [sorted_orders[0]]
            
            for order in sorted_orders[1:]:
                current_price = float(order[0])
                group_price = float(current_group[0][0])
                
                # Check if price is within tolerance of group
                price_diff = abs(current_price - group_price) / group_price
                
                if price_diff <= tolerance_pct:
                    current_group.append(order)
                else:
                    # Start new group
                    groups.append(current_group)
                    current_group = [order]
            
            # Add last group
            if current_group:
                groups.append(current_group)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Error grouping nearby prices: {e}")
            return []
    
    def _enhance_pools_with_volume_profile(
        self, 
        pools: List[LiquidityPool], 
        volume_profile: VolumeProfile
    ) -> List[LiquidityPool]:
        """Enhance liquidity pools with volume profile context"""
        try:
            enhanced_pools = []
            
            for pool in pools:
                enhanced_pool = pool
                
                # Find nearest volume profile node
                nearest_node = min(
                    volume_profile.nodes,
                    key=lambda node: abs(node.price - pool.price)
                )
                
                # Calculate distance to nearest node
                distance = abs(nearest_node.price - pool.price)
                price_tolerance = pool.price * 0.005  # 0.5% tolerance
                
                if distance <= price_tolerance:
                    # Pool is near significant volume area
                    volume_bonus = min(30, nearest_node.percentage)  # Up to 30 points
                    enhanced_pool.strength = min(100, pool.strength + volume_bonus)
                
                # Check proximity to POC, VAH, VAL
                key_levels = [volume_profile.poc, volume_profile.vah, volume_profile.val]
                for level in key_levels:
                    if abs(pool.price - level) <= price_tolerance:
                        enhanced_pool.strength = min(100, enhanced_pool.strength + 15)
                        break
                
                enhanced_pools.append(enhanced_pool)
            
            return enhanced_pools
            
        except Exception as e:
            self.logger.error(f"Error enhancing pools with volume profile: {e}")
            return pools
    
    def get_key_levels(self, volume_profile: VolumeProfile) -> Dict[str, float]:
        """Extract key levels from volume profile for trading decisions"""
        try:
            levels = {
                'poc': volume_profile.poc,
                'vah': volume_profile.vah,
                'val': volume_profile.val
            }
            
            # Add high volume nodes as additional levels
            high_volume_nodes = sorted(
                volume_profile.nodes,
                key=lambda x: x.volume,
                reverse=True
            )[:5]  # Top 5 volume levels
            
            for i, node in enumerate(high_volume_nodes):
                levels[f'hvn_{i+1}'] = node.price
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error extracting key levels: {e}")
            return {}
    
    def _empty_volume_profile(self) -> VolumeProfile:
        """Return empty volume profile"""
        return VolumeProfile(
            nodes=[],
            poc=0.0,
            vah=0.0,
            val=0.0,
            total_volume=0.0,
            value_area_volume=0.0,
            volume_distribution={}
        )
    
    def _single_price_profile(self, price: float, volume: float) -> VolumeProfile:
        """Return volume profile for single price level"""
        node = VolumeProfileNode(
            price=price,
            volume=volume,
            buy_volume=volume / 2,
            sell_volume=volume / 2,
            trades_count=1,
            percentage=100.0
        )
        
        return VolumeProfile(
            nodes=[node],
            poc=price,
            vah=price,
            val=price,
            total_volume=volume,
            value_area_volume=volume,
            volume_distribution={
                'max_volume': volume,
                'min_volume': volume,
                'avg_volume': volume,
                'std_volume': 0.0,
                'median_volume': volume,
                'volume_concentration': 1.0,
                'price_range': 0.0,
                'price_skewness': 0.0
            }
        )
    
    def get_volume_profile_summary(self, volume_profile: VolumeProfile) -> Dict[str, Any]:
        """Generate summary statistics for volume profile"""
        try:
            if not volume_profile.nodes:
                return {}
            
            summary = {
                'total_volume': volume_profile.total_volume,
                'poc': volume_profile.poc,
                'vah': volume_profile.vah,
                'val': volume_profile.val,
                'value_area_volume': volume_profile.value_area_volume,
                'value_area_percentage': (volume_profile.value_area_volume / volume_profile.total_volume) * 100,
                'price_levels': len(volume_profile.nodes),
                'price_range': max(node.price for node in volume_profile.nodes) - min(node.price for node in volume_profile.nodes),
                'volume_distribution': volume_profile.volume_distribution
            }
            
            # Add top volume levels
            top_nodes = sorted(volume_profile.nodes, key=lambda x: x.volume, reverse=True)[:3]
            summary['top_volume_levels'] = [
                {'price': node.price, 'volume': node.volume, 'percentage': node.percentage}
                for node in top_nodes
            ]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating volume profile summary: {e}")
            return {}