"""
Setup Detection Engine for SOL/USDT Trading Algorithm

Detects breakout, momentum, mean reversion, and liquidity hunt setups
as specified in the algorithm document.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradingSetup:
    """Container for detected trading setup"""
    setup_type: str  # 'breakout', 'momentum', 'mean_reversion', 'liquidity_hunt'
    direction: str   # 'long' or 'short'
    entry: float
    stop_loss: float
    targets: List[Dict[str, float]]  # [{'price': float, 'allocation': float}]
    confidence: float
    reasoning: str
    validation_score: float
    metadata: Dict[str, Any]

class SetupDetector:
    """
    Setup detection engine that identifies multiple setup types
    and ranks them by confidence as per algorithm specification.
    """
    
    def __init__(self, indicators: Dict, order_flow: Dict, regime: str, config: Dict = None):
        self.indicators = indicators
        self.order_flow = order_flow
        self.regime = regime
        self.config = config or {}
        
        # Algorithm constants
        self.BREAKOUT_RANGE_ATR_MAX = 0.75  # Dynamic range threshold
        self.MIN_VALIDATION_SCORE = 0.50
        self.MIN_CONFIDENCE = 0.30
        
    def detect_all_setups(self) -> Optional[TradingSetup]:
        """
        Detect all possible setups and return the highest confidence one
        
        Returns:
            TradingSetup with highest confidence or None if no valid setups
        """
        setups = []
        
        # Check each setup type as per algorithm
        breakout_setup = self.detect_breakout()
        if breakout_setup:
            setups.append(breakout_setup)
            
        momentum_setup = self.detect_momentum()
        if momentum_setup:
            setups.append(momentum_setup)
            
        mean_reversion_setup = self.detect_mean_reversion()
        if mean_reversion_setup:
            setups.append(mean_reversion_setup)
            
        liquidity_hunt_setup = self.detect_liquidity_hunt()
        if liquidity_hunt_setup:
            setups.append(liquidity_hunt_setup)
        
        # Sort by confidence and return best
        if setups:
            setups.sort(key=lambda x: x.confidence, reverse=True)
            return setups[0]
        
        return None
    
    def detect_breakout(self) -> Optional[TradingSetup]:
        """
        Enhanced breakout detection with false breakout filter
        Implements algorithm specification exactly
        """
        # Get required data
        atr = self.indicators.get('atr', 0)
        close_data = self.indicators.get('close', [])
        high_data = self.indicators.get('high', [])
        low_data = self.indicators.get('low', [])
        
        if not all([atr, close_data, high_data, low_data]) or len(close_data) < 100:
            return None
        
        # Dynamic range threshold based on ATR (algorithm requirement)
        range_max = self.BREAKOUT_RANGE_ATR_MAX * atr
        
        # Find consolidation range (last 100 periods)
        recent_highs = high_data[-100:] if len(high_data) >= 100 else high_data
        recent_lows = low_data[-100:] if len(low_data) >= 100 else low_data
        
        high_100 = max(recent_highs)
        low_100 = min(recent_lows)
        range_size = high_100 - low_100
        
        # Check if range is suitable for breakout
        if range_size > range_max:
            return None
        
        current_price = close_data[-1] if close_data else 0
        
        # Determine breakout direction
        direction = None
        breakout_level = None
        
        if current_price > high_100:
            direction = 'long'
            breakout_level = high_100
        elif current_price < low_100:
            direction = 'short'
            breakout_level = low_100
        else:
            return None
        
        # Validate breakout (prevent false breakouts)
        validation_score = 0.0
        reasoning_parts = []
        
        # 1. Volume confirmation
        zvol = self.indicators.get('zvol', 0)
        if zvol >= 2.0:
            validation_score += 0.25
            reasoning_parts.append(f"Strong volume (z={zvol:.1f})")
        elif zvol >= 1.0:
            validation_score += 0.15
            reasoning_parts.append(f"Good volume (z={zvol:.1f})")
        
        # 2. Multiple closes beyond level
        recent_closes = close_data[-3:] if len(close_data) >= 3 else close_data
        closes_beyond = sum([
            1 for c in recent_closes 
            if (c > breakout_level if direction == 'long' else c < breakout_level)
        ])
        validation_score += closes_beyond * 0.15
        if closes_beyond > 1:
            reasoning_parts.append(f"{closes_beyond} closes beyond level")
        
        # 3. Order flow confirmation
        cvd_trend = self.order_flow.get('cvd_trend', 'neutral')
        if (direction == 'long' and cvd_trend == 'bullish') or \
           (direction == 'short' and cvd_trend == 'bearish'):
            validation_score += 0.20
            reasoning_parts.append(f"CVD {cvd_trend}")
        
        # 4. No immediate rejection (small wick)
        wick_ratio = self._calculate_wick_ratio(direction)
        if wick_ratio < 0.3:
            validation_score += 0.15
            reasoning_parts.append("Clean breakout")
        
        # 5. Market regime alignment
        if self.regime in ['trending', 'strong_trend']:
            validation_score += 0.10
            reasoning_parts.append(f"Regime: {self.regime}")
        
        # Check minimum validation
        if validation_score < self.MIN_VALIDATION_SCORE:
            return None
        
        # Calculate entry and levels
        entry = current_price * (1 - 0.0005 if direction == 'long' else 1 + 0.0005)
        
        # Adaptive stop loss based on regime
        sl_multiplier = 1.0 if self.regime == 'strong_trend' else 1.5
        stop_loss = entry - sl_multiplier * atr if direction == 'long' else entry + sl_multiplier * atr
        
        # Dynamic targets based on structure
        targets = self._calculate_dynamic_targets(entry, direction, atr)
        
        return TradingSetup(
            setup_type='breakout',
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            targets=targets,
            confidence=validation_score,
            reasoning=f"Breakout: {', '.join(reasoning_parts)}",
            validation_score=validation_score,
            metadata={
                'range_size': range_size,
                'breakout_level': breakout_level,
                'atr': atr
            }
        )
    
    def detect_momentum(self) -> Optional[TradingSetup]:
        """
        Momentum setup with multiple triggers as per algorithm
        """
        triggers = []
        close_data = self.indicators.get('close', [])
        ema21 = self.indicators.get('ema_21', 0)
        adx = self.indicators.get('adx', 0)
        zvol = self.indicators.get('zvol', 0)
        rsi = self.indicators.get('rsi', 50)
        
        if not close_data or len(close_data) < 6:
            return None
        
        price = close_data[-1]
        prev_price = close_data[-2] if len(close_data) >= 2 else price
        
        # 1. Pullback to EMA with continuation
        if abs(price - ema21) / ema21 < 0.002:  # Within 0.2% of EMA21
            if price > prev_price and price > ema21:
                triggers.append(('ema_bounce', 'long', 0.20))
            elif price < prev_price and price < ema21:
                triggers.append(('ema_bounce', 'short', 0.20))
        
        # 2. Volume spike with directional move
        if zvol >= 3.0:
            price_change = (price - close_data[-6]) / close_data[-6] if len(close_data) >= 6 else 0
            if abs(price_change) > 0.01:  # 1% move
                direction = 'long' if price_change > 0 else 'short'
                triggers.append(('volume_spike', direction, 0.25))
        
        # 3. RSI momentum shift
        rsi_history = self.indicators.get('rsi_history', [])
        if len(rsi_history) >= 2:
            rsi_prev = rsi_history[-2]
            if rsi > 50 and rsi_prev <= 50 and adx > 25:
                triggers.append(('rsi_cross', 'long', 0.15))
            elif rsi < 50 and rsi_prev >= 50 and adx > 25:
                triggers.append(('rsi_cross', 'short', 0.15))
        
        # 4. Order flow momentum
        intensity = self.order_flow.get('intensity', 0)
        delta = self.order_flow.get('delta', 0)
        cvd_trend = self.order_flow.get('cvd_trend', 'neutral')
        
        if intensity > 2.0:  # 2x normal intensity
            if delta > 0 and cvd_trend == 'bullish':
                triggers.append(('flow_momentum', 'long', 0.30))
            elif delta < 0 and cvd_trend == 'bearish':
                triggers.append(('flow_momentum', 'short', 0.30))
        
        if not triggers:
            return None
        
        # Combine triggers and calculate confidence
        direction_scores = {'long': 0, 'short': 0}
        for trigger_type, direction, score in triggers:
            direction_scores[direction] += score
        
        final_direction = max(direction_scores, key=direction_scores.get)
        confidence = direction_scores[final_direction]
        
        if confidence < self.MIN_CONFIDENCE:
            return None
        
        # Setup parameters
        atr = self.indicators.get('atr', price * 0.015)
        entry = price * (1 - 0.0005 if final_direction == 'long' else 1 + 0.0005)
        
        # Tighter stop for momentum trades
        stop_loss = entry - 1.0 * atr if final_direction == 'long' else entry + 1.0 * atr
        
        # Quick targets for momentum
        targets = [
            {'price': entry + atr * 1.0 if final_direction == 'long' else entry - atr * 1.0, 'allocation': 0.5},
            {'price': entry + atr * 1.5 if final_direction == 'long' else entry - atr * 1.5, 'allocation': 0.3},
            {'price': entry + atr * 2.0 if final_direction == 'long' else entry - atr * 2.0, 'allocation': 0.2}
        ]
        
        trigger_names = [t[0] for t in triggers]
        
        return TradingSetup(
            setup_type='momentum',
            direction=final_direction,
            entry=entry,
            stop_loss=stop_loss,
            targets=targets,
            confidence=confidence,
            reasoning=f"Momentum: {', '.join(trigger_names)}",
            validation_score=confidence,
            metadata={
                'triggers': triggers,
                'atr': atr
            }
        )
    
    def detect_mean_reversion(self) -> Optional[TradingSetup]:
        """
        Mean reversion from extremes as per algorithm
        """
        close_data = self.indicators.get('close', [])
        vwap = self.indicators.get('vwap', 0)
        vwap_upper_2sigma = self.indicators.get('vwap_upper_2sigma', 0)
        vwap_lower_2sigma = self.indicators.get('vwap_lower_2sigma', 0)
        rsi = self.indicators.get('rsi', 50)
        zvol = self.indicators.get('zvol', 0)
        
        if not close_data or not vwap:
            return None
        
        price = close_data[-1]
        
        # Check for extreme deviation
        direction = None
        distance_from_mean = 0
        
        if price > vwap_upper_2sigma:
            direction = 'short'
            distance_from_mean = (price - vwap) / vwap
        elif price < vwap_lower_2sigma:
            direction = 'long'
            distance_from_mean = (vwap - price) / vwap
        else:
            return None
        
        # Validate reversal signals
        validation_score = 0.0
        reasoning_parts = []
        
        # 1. RSI not at extreme
        if direction == 'short' and rsi < 70:
            validation_score += 0.20
            reasoning_parts.append(f"RSI not extreme ({rsi:.1f})")
        elif direction == 'long' and rsi > 30:
            validation_score += 0.20
            reasoning_parts.append(f"RSI not extreme ({rsi:.1f})")
        
        # 2. Volume declining at extremes (exhaustion)
        if zvol < 0:  # Below average volume
            validation_score += 0.15
            reasoning_parts.append("Volume exhaustion")
        
        # 3. Order flow divergence
        cvd_trend = self.order_flow.get('cvd_trend', 'neutral')
        if (direction == 'short' and cvd_trend != 'bullish') or \
           (direction == 'long' and cvd_trend != 'bearish'):
            validation_score += 0.25
            reasoning_parts.append(f"CVD divergence ({cvd_trend})")
        
        # 4. Rejection wick
        wick_ratio = self._calculate_wick_ratio(direction)
        if wick_ratio > 0.5:
            validation_score += 0.20
            reasoning_parts.append("Rejection wick")
        
        # 5. Market regime suitable
        if self.regime in ['normal_range', 'low_volatility_range']:
            validation_score += 0.20
            reasoning_parts.append(f"Suitable regime ({self.regime})")
        
        if validation_score < 0.40:  # Higher threshold for mean reversion
            return None
        
        # Setup parameters
        atr = self.indicators.get('atr', price * 0.015)
        entry = price * (1 - 0.0005 if direction == 'long' else 1 + 0.0005)
        
        # Tight stop for mean reversion
        stop_loss = entry - 1.0 * atr if direction == 'long' else entry + 1.0 * atr
        
        # Conservative targets - back to mean
        targets = [
            {'price': vwap, 'allocation': 0.5},
            {'price': vwap + (0.5 * atr if direction == 'long' else -0.5 * atr), 'allocation': 0.3},
            {'price': vwap + (1.0 * atr if direction == 'long' else -1.0 * atr), 'allocation': 0.2}
        ]
        
        return TradingSetup(
            setup_type='mean_reversion',
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            targets=targets,
            confidence=validation_score,
            reasoning=f"Mean reversion: {', '.join(reasoning_parts)}",
            validation_score=validation_score,
            metadata={
                'distance_from_mean': distance_from_mean,
                'vwap': vwap,
                'atr': atr
            }
        )
    
    def detect_liquidity_hunt(self) -> Optional[TradingSetup]:
        """
        Detect liquidity sweep and reversal as per algorithm
        """
        liquidity_pools = self.indicators.get('liquidity_pools', [])
        close_data = self.indicators.get('close', [])
        high_data = self.indicators.get('high', [])
        low_data = self.indicators.get('low', [])
        
        if not all([liquidity_pools, close_data, high_data, low_data]):
            return None
        
        price = close_data[-1]
        high = high_data[-1]
        low = low_data[-1]
        
        # Check for sweep of liquidity pools
        for pool in liquidity_pools[:5]:  # Check top 5 pools
            level = pool['level']
            pool_type = pool['type']
            
            swept = False
            reclaimed = False
            direction = None
            
            if pool_type == 'resistance':
                # Check if price swept above and came back
                if high > level * 1.001:  # Swept by 0.1%
                    swept = True
                    if price < level:  # Reclaimed below
                        reclaimed = True
                        direction = 'short'
            
            elif pool_type == 'support':
                # Check if price swept below and came back
                if low < level * 0.999:  # Swept by 0.1%
                    swept = True
                    if price > level:  # Reclaimed above
                        reclaimed = True
                        direction = 'long'
            
            if swept and reclaimed:
                # Validate with order flow
                validation_score = 0.40  # Base score for sweep and reclaim
                reasoning_parts = ['Liquidity sweep + reclaim']
                
                # Check order flow confirmation
                delta = self.order_flow.get('delta', 0)
                if (direction == 'long' and delta > 0) or (direction == 'short' and delta < 0):
                    validation_score += 0.30
                    reasoning_parts.append('Order flow confirmation')
                
                # Check for absorption at the level
                absorption = self.order_flow.get('absorption', None)
                if absorption:
                    validation_score += 0.20
                    reasoning_parts.append(f'Absorption: {absorption}')
                
                # Pool strength
                pool_strength = pool.get('strength', 0)
                validation_score += min(pool_strength / 100, 0.10)
                
                if validation_score < self.MIN_VALIDATION_SCORE:
                    continue
                
                # Setup parameters
                atr = self.indicators.get('atr', price * 0.015)
                entry = level * (1 + 0.0005 if direction == 'long' else 1 - 0.0005)
                
                # Stop beyond the sweep
                stop_loss = low - 0.5 * atr if direction == 'long' else high + 0.5 * atr
                
                # Target opposite liquidity or fixed move
                opposite_pools = [p for p in liquidity_pools if p['type'] != pool_type]
                if opposite_pools:
                    target_level = opposite_pools[0]['level']
                else:
                    target_level = entry + (3.0 if direction == 'long' else -3.0)  # $3 move
                
                targets = [
                    {'price': entry + (1.5 if direction == 'long' else -1.5), 'allocation': 0.4},
                    {'price': entry + (2.5 if direction == 'long' else -2.5), 'allocation': 0.3},
                    {'price': target_level, 'allocation': 0.3}
                ]
                
                return TradingSetup(
                    setup_type='liquidity_hunt',
                    direction=direction,
                    entry=entry,
                    stop_loss=stop_loss,
                    targets=targets,
                    confidence=validation_score,
                    reasoning=f"Liquidity hunt: {', '.join(reasoning_parts)}",
                    validation_score=validation_score,
                    metadata={
                        'pool_level': level,
                        'pool_strength': pool_strength,
                        'atr': atr
                    }
                )
        
        return None
    
    def _calculate_wick_ratio(self, direction: str) -> float:
        """Calculate wick ratio for rejection detection"""
        high_data = self.indicators.get('high', [])
        low_data = self.indicators.get('low', [])
        open_data = self.indicators.get('open', [])
        close_data = self.indicators.get('close', [])
        
        if not all([high_data, low_data, open_data, close_data]):
            return 0.0
        
        high = high_data[-1]
        low = low_data[-1]
        open_price = open_data[-1]
        close = close_data[-1]
        
        body = abs(close - open_price)
        if body == 0:
            return 0.0
        
        if direction == 'long':
            lower_wick = min(open_price, close) - low
            return lower_wick / body
        else:
            upper_wick = high - max(open_price, close)
            return upper_wick / body
    
    def _calculate_dynamic_targets(self, entry: float, direction: str, atr: float) -> List[Dict[str, float]]:
        """
        Calculate dynamic targets based on market structure
        Simplified version of algorithm specification
        """
        # Basic ATR-based targets for now
        targets = []
        
        multipliers = [1.0, 1.5, 2.5]
        allocations = [0.4, 0.3, 0.3]  # Adjusted for 3 targets
        
        for i, (mult, alloc) in enumerate(zip(multipliers, allocations)):
            if direction == 'long':
                target_price = entry + (atr * mult)
            else:
                target_price = entry - (atr * mult)
            
            targets.append({
                'price': target_price,
                'allocation': alloc
            })
        
        return targets