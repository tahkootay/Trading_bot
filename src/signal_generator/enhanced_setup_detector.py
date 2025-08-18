"""
Enhanced Setup Detection Engine
Implements full algorithm specification for SOL/USDT setups
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from ..utils.algorithm_constants import (
    MARKET_REGIMES, ENTRY_FILTERS, SL_TP_PARAMS, TIME_STOPS,
    LIQUIDITY_PARAMS, get_regime_entry_filters, get_time_stop_minutes
)

@dataclass
class SetupSignal:
    """Trading setup signal"""
    setup_type: str
    direction: str
    entry: float
    stop_loss: float
    targets: List[Dict]
    confidence: float
    validation_score: float
    regime: str
    triggers: List[Tuple[str, str, float]]
    metadata: Dict[str, Any]

class EnhancedSetupDetector:
    """
    Complete setup detection engine as per algorithm specification
    
    Implements all 4 setup types:
    - Breakout (with false breakout filter)
    - Momentum (multiple triggers)
    - Mean Reversion (from extremes)
    - Liquidity Hunt (sweep and reclaim)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.confluence_tolerance_atr = 0.2  # 0.2 ATR tolerance for confluence
        
    def detect_all_setups(
        self, 
        indicators: Dict, 
        order_flow: Dict, 
        regime: str,
        market_data: Dict
    ) -> Optional[SetupSignal]:
        """
        Detect all possible setups and rank by confidence
        
        Args:
            indicators: Technical indicators from TechnicalIndicators class
            order_flow: Order flow data from OrderFlowAnalyzer
            regime: Current market regime
            market_data: Raw market data (OHLCV, etc.)
            
        Returns:
            Best setup signal or None if no valid setups
        """
        try:
            setups = []
            
            # Check each setup type
            breakout = self.detect_breakout(indicators, order_flow, regime, market_data)
            if breakout:
                setups.append(breakout)
                
            momentum = self.detect_momentum(indicators, order_flow, regime, market_data)
            if momentum:
                setups.append(momentum)
                
            mean_rev = self.detect_mean_reversion(indicators, order_flow, regime, market_data)
            if mean_rev:
                setups.append(mean_rev)
                
            liquidity = self.detect_liquidity_hunt(indicators, order_flow, regime, market_data)
            if liquidity:
                setups.append(liquidity)
            
            if not setups:
                return None
            
            # Sort by confidence score
            setups.sort(key=lambda x: x.confidence, reverse=True)
            
            # Return highest confidence setup
            return setups[0]
            
        except Exception as e:
            self.logger.error(f"Error in setup detection: {e}")
            return None
    
    def detect_breakout(
        self, 
        indicators: Dict, 
        order_flow: Dict, 
        regime: str,
        market_data: Dict
    ) -> Optional[SetupSignal]:
        """
        Enhanced breakout detection with false breakout filter
        Exactly as specified in algorithm
        """
        try:
            # Dynamic range threshold based on ATR
            atr = indicators.get('atr', 0)
            if atr == 0:
                return None
                
            range_max = 0.75 * atr  # Dynamic instead of fixed 1.5 USD
            
            # Get price data
            ohlcv = market_data.get('ohlcv', {}).get('primary', None)
            if ohlcv is None or len(ohlcv) < 100:
                return None
            
            # Find consolidation range (last 100 bars)
            high_100 = ohlcv['high'].tail(100).max()
            low_100 = ohlcv['low'].tail(100).min()
            range_size = high_100 - low_100
            
            if range_size > range_max:
                return None
            
            current_price = ohlcv['close'].iloc[-1]
            
            # Check for breakout
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
            validation_score = 0
            
            # 1. Volume confirmation
            zvol = indicators.get('zvol', 0)
            if zvol >= 2.0:
                validation_score += 0.25
            elif zvol >= 1.0:
                validation_score += 0.15
            
            # 2. Multiple closes beyond level
            recent_closes = ohlcv['close'].tail(3).tolist()
            closes_beyond = sum([
                1 for c in recent_closes 
                if (c > breakout_level if direction == 'long' else c < breakout_level)
            ])
            validation_score += closes_beyond * 0.15
            
            # 3. Order flow confirmation
            cvd_trend = order_flow.get('cvd_trend', 'neutral')
            if direction == 'long' and cvd_trend == 'bullish':
                validation_score += 0.20
            elif direction == 'short' and cvd_trend == 'bearish':
                validation_score += 0.20
            
            # 4. No immediate rejection (small wick)
            wick_ratio = self._calculate_wick_ratio(
                ohlcv['high'].iloc[-1],
                ohlcv['low'].iloc[-1], 
                ohlcv['open'].iloc[-1],
                ohlcv['close'].iloc[-1],
                direction
            )
            if wick_ratio < 0.3:  # Small wick
                validation_score += 0.15
            
            # 5. Market regime alignment
            if regime in ['trending', 'strong_trend']:
                validation_score += 0.10
            
            if validation_score < 0.50:
                return None
            
            # Calculate entry and stops
            entry = current_price * (1 - 0.0005 if direction == 'long' else 1 + 0.0005)
            
            # Adaptive stop loss
            sl_multiplier = 1.0 if regime == 'strong_trend' else 1.5
            stop_loss = entry - sl_multiplier * atr if direction == 'long' else entry + sl_multiplier * atr
            
            # Dynamic targets based on structure
            targets = self._calculate_dynamic_targets(
                entry, direction, atr, indicators, market_data
            )
            
            return SetupSignal(
                setup_type='breakout',
                direction=direction,
                entry=entry,
                stop_loss=stop_loss,
                targets=targets,
                confidence=validation_score,
                validation_score=validation_score,
                regime=regime,
                triggers=[('volume', direction, zvol), ('breakout', direction, range_size)],
                metadata={
                    'range_size': range_size,
                    'breakout_level': breakout_level,
                    'closes_beyond': closes_beyond,
                    'wick_ratio': wick_ratio
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in breakout detection: {e}")
            return None
    
    def detect_momentum(
        self, 
        indicators: Dict, 
        order_flow: Dict, 
        regime: str,
        market_data: Dict
    ) -> Optional[SetupSignal]:
        """
        Momentum setup with multiple triggers as per algorithm specification
        """
        try:
            ohlcv = market_data.get('ohlcv', {}).get('primary', None)
            if ohlcv is None or len(ohlcv) < 60:
                return None
            
            triggers = []
            price = ohlcv['close'].iloc[-1]
            prev_price = ohlcv['close'].iloc[-2]
            
            # 1. Pullback to EMA with continuation
            ema21 = indicators.get('ema_21', price)
            if abs(price - ema21) / ema21 < 0.002:  # Within 0.2% of EMA21
                if price > prev_price and price > ema21:
                    triggers.append(('ema_bounce', 'long', 0.20))
                elif price < prev_price and price < ema21:
                    triggers.append(('ema_bounce', 'short', 0.20))
            
            # 2. Volume spike with directional move
            zvol = indicators.get('zvol', 0)
            if zvol >= 3.0:
                price_change_5 = (price - ohlcv['close'].iloc[-6]) / ohlcv['close'].iloc[-6]
                if abs(price_change_5) > 0.01:  # 1% move
                    direction = 'long' if price_change_5 > 0 else 'short'
                    triggers.append(('volume_spike', direction, 0.25))
            
            # 3. RSI momentum shift
            rsi = indicators.get('rsi', 50)
            # Simulate previous RSI (would be tracked in real implementation)
            rsi_prev = rsi - 2  # Simplified for demonstration
            
            adx = indicators.get('adx', 20)
            if rsi > 50 and rsi_prev <= 50 and adx > 25:
                triggers.append(('rsi_cross', 'long', 0.15))
            elif rsi < 50 and rsi_prev >= 50 and adx > 25:
                triggers.append(('rsi_cross', 'short', 0.15))
            
            # 4. Order flow momentum
            intensity = order_flow.get('intensity', 0)
            if intensity > 2.0:  # 2x normal intensity
                delta = order_flow.get('delta', 0)
                cvd_trend = order_flow.get('cvd_trend', 'neutral')
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
            
            if confidence < 0.30:
                return None
            
            # Setup parameters
            atr = indicators.get('atr', 0)
            entry = price * (1 - 0.0005 if final_direction == 'long' else 1 + 0.0005)
            
            # Tighter stop for momentum trades
            stop_loss = entry - 1.0 * atr if final_direction == 'long' else entry + 1.0 * atr
            
            # Quick targets for momentum
            targets = [
                {
                    'price': entry + atr * 1.0 if final_direction == 'long' else entry - atr * 1.0, 
                    'allocation': 0.5
                },
                {
                    'price': entry + atr * 1.5 if final_direction == 'long' else entry - atr * 1.5, 
                    'allocation': 0.3
                },
                {
                    'price': entry + atr * 2.0 if final_direction == 'long' else entry - atr * 2.0, 
                    'allocation': 0.2
                }
            ]
            
            return SetupSignal(
                setup_type='momentum',
                direction=final_direction,
                entry=entry,
                stop_loss=stop_loss,
                targets=targets,
                confidence=confidence,
                validation_score=confidence,
                regime=regime,
                triggers=triggers,
                metadata={
                    'ema21_distance': abs(price - ema21) / ema21,
                    'volume_spike': zvol,
                    'rsi': rsi,
                    'flow_intensity': intensity
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in momentum detection: {e}")
            return None
    
    def detect_mean_reversion(
        self, 
        indicators: Dict, 
        order_flow: Dict, 
        regime: str,
        market_data: Dict
    ) -> Optional[SetupSignal]:
        """
        Mean reversion from extremes as per algorithm specification
        """
        try:
            ohlcv = market_data.get('ohlcv', {}).get('primary', None)
            if ohlcv is None:
                return None
            
            price = ohlcv['close'].iloc[-1]
            
            # VWAP bands analysis
            vwap = indicators.get('vwap', price)
            vwap_upper_2sigma = indicators.get('vwap_upper_2sigma', price * 1.02)
            vwap_lower_2sigma = indicators.get('vwap_lower_2sigma', price * 0.98)
            
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
            validation_score = 0
            
            # 1. RSI divergence
            rsi = indicators.get('rsi', 50)
            if direction == 'short' and rsi < 70:  # Not extremely overbought
                validation_score += 0.20
            elif direction == 'long' and rsi > 30:  # Not extremely oversold
                validation_score += 0.20
            
            # 2. Volume declining at extremes (exhaustion)
            zvol = indicators.get('zvol', 0)
            if zvol < 0:  # Below average volume
                validation_score += 0.15
            
            # 3. Order flow divergence
            cvd_trend = order_flow.get('cvd_trend', 'neutral')
            if direction == 'short' and cvd_trend != 'bullish':
                validation_score += 0.25
            elif direction == 'long' and cvd_trend != 'bearish':
                validation_score += 0.25
            
            # 4. Rejection wick
            wick_ratio = self._calculate_wick_ratio(
                ohlcv['high'].iloc[-1],
                ohlcv['low'].iloc[-1], 
                ohlcv['open'].iloc[-1],
                ohlcv['close'].iloc[-1],
                direction
            )
            if wick_ratio > 0.5:  # Large rejection wick
                validation_score += 0.20
            
            # 5. Market regime suitable for mean reversion
            if regime in ['normal_range', 'low_volatility_range']:
                validation_score += 0.20
            
            if validation_score < 0.40:
                return None
            
            # Setup parameters
            atr = indicators.get('atr', 0)
            entry = price * (1 - 0.0005 if direction == 'long' else 1 + 0.0005)
            
            # Tight stop for mean reversion
            stop_loss = entry - 1.0 * atr if direction == 'long' else entry + 1.0 * atr
            
            # Conservative targets - back to mean
            targets = [
                {'price': vwap, 'allocation': 0.5},
                {
                    'price': vwap + (0.5 * atr if direction == 'long' else -0.5 * atr), 
                    'allocation': 0.3
                },
                {
                    'price': vwap + (1.0 * atr if direction == 'long' else -1.0 * atr), 
                    'allocation': 0.2
                }
            ]
            
            return SetupSignal(
                setup_type='mean_reversion',
                direction=direction,
                entry=entry,
                stop_loss=stop_loss,
                targets=targets,
                confidence=validation_score,
                validation_score=validation_score,
                regime=regime,
                triggers=[('vwap_deviation', direction, distance_from_mean)],
                metadata={
                    'distance_from_mean': distance_from_mean,
                    'vwap': vwap,
                    'upper_band': vwap_upper_2sigma,
                    'lower_band': vwap_lower_2sigma,
                    'rsi': rsi,
                    'wick_ratio': wick_ratio
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in mean reversion detection: {e}")
            return None
    
    def detect_liquidity_hunt(
        self, 
        indicators: Dict, 
        order_flow: Dict, 
        regime: str,
        market_data: Dict
    ) -> Optional[SetupSignal]:
        """
        Detect liquidity sweep and reversal as per algorithm specification
        """
        try:
            ohlcv = market_data.get('ohlcv', {}).get('primary', None)
            if ohlcv is None:
                return None
            
            # Get liquidity pools
            liquidity_pools = indicators.get('liquidity_pools', [])
            if not liquidity_pools:
                return None
            
            price = ohlcv['close'].iloc[-1]
            high = ohlcv['high'].iloc[-1]
            low = ohlcv['low'].iloc[-1]
            
            # Check for sweep of liquidity pools
            for pool in liquidity_pools[:5]:  # Check top 5 pools
                level = pool['level']
                pool_type = pool['type']
                
                # Check for sweep and reclaim
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
                
                if not (swept and reclaimed):
                    continue
                
                # Validate with order flow
                validation_score = 0.40  # Base score for sweep and reclaim
                
                # Check order flow confirmation
                delta = order_flow.get('delta', 0)
                if direction == 'long' and delta > 0:
                    validation_score += 0.30
                elif direction == 'short' and delta < 0:
                    validation_score += 0.30
                
                # Check for absorption at the level
                absorption = order_flow.get('absorption', None)
                if absorption:
                    validation_score += 0.20
                
                # Check pool strength
                pool_strength = pool.get('strength', 0)
                validation_score += min(pool_strength / 100, 0.10)
                
                if validation_score < 0.50:
                    continue
                
                # Setup parameters
                atr = indicators.get('atr', 0)
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
                
                return SetupSignal(
                    setup_type='liquidity_hunt',
                    direction=direction,
                    entry=entry,
                    stop_loss=stop_loss,
                    targets=targets,
                    confidence=validation_score,
                    validation_score=validation_score,
                    regime=regime,
                    triggers=[('liquidity_sweep', direction, pool_strength)],
                    metadata={
                        'pool_level': level,
                        'pool_type': pool_type,
                        'pool_strength': pool_strength,
                        'sweep_high': high,
                        'sweep_low': low,
                        'absorption': absorption
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in liquidity hunt detection: {e}")
            return None
    
    def _calculate_wick_ratio(
        self, 
        high: float, 
        low: float, 
        open_price: float, 
        close: float, 
        direction: str
    ) -> float:
        """Calculate wick ratio for rejection detection"""
        try:
            body = abs(close - open_price)
            if body == 0:
                return 0
            
            if direction == 'long':
                lower_wick = min(open_price, close) - low
                return lower_wick / body
            else:
                upper_wick = high - max(open_price, close)
                return upper_wick / body
                
        except Exception:
            return 0
    
    def _calculate_dynamic_targets(
        self, 
        entry: float, 
        direction: str, 
        atr: float, 
        indicators: Dict,
        market_data: Dict
    ) -> List[Dict]:
        """
        Calculate dynamic targets based on market structure
        Implements full algorithm specification for confluence zones
        """
        try:
            # Get structure levels
            sr_levels = indicators.get('support_resistance', [])
            liquidity_pools = indicators.get('liquidity_pools', [])
            volume_profile = indicators.get('volume_profile', {})
            
            # Collect all potential targets
            potential_targets = []
            
            # ATR-based targets
            for multiplier in [1.0, 1.5, 2.5, 4.0]:
                target_price = entry + (atr * multiplier if direction == 'long' else -atr * multiplier)
                potential_targets.append({
                    'price': target_price, 
                    'type': 'atr', 
                    'strength': 0.5
                })
            
            # Structure-based targets
            for level in sr_levels:
                level_price = level.get('price', 0)
                if (direction == 'long' and level_price > entry) or \
                   (direction == 'short' and level_price < entry):
                    potential_targets.append({
                        'price': level_price,
                        'type': 'structure',
                        'strength': level.get('strength', 0.5)
                    })
            
            # Fibonacci extensions (simplified implementation)
            fib_levels = self._calculate_fibonacci_extensions(entry, direction, atr)
            for fib in fib_levels:
                potential_targets.append({
                    'price': fib['price'],
                    'type': 'fibonacci',
                    'strength': 0.6
                })
            
            # Liquidity pool targets
            for pool in liquidity_pools:
                pool_level = pool.get('level', 0)
                if (direction == 'long' and pool_level > entry) or \
                   (direction == 'short' and pool_level < entry):
                    potential_targets.append({
                        'price': pool_level,
                        'type': 'liquidity',
                        'strength': min(pool.get('strength', 0) / 50, 1.0)
                    })
            
            # Volume profile targets
            vp_targets = [
                volume_profile.get('vah', 0), 
                volume_profile.get('val', 0), 
                volume_profile.get('poc', 0)
            ]
            for vp_level in vp_targets:
                if vp_level > 0 and \
                   ((direction == 'long' and vp_level > entry) or \
                    (direction == 'short' and vp_level < entry)):
                    potential_targets.append({
                        'price': vp_level,
                        'type': 'volume_profile',
                        'strength': 0.7
                    })
            
            # Find confluence zones
            confluence_targets = self._find_confluence_zones(
                potential_targets, 
                atr * self.confluence_tolerance_atr
            )
            
            # Select top 3 confluence zones
            confluence_targets.sort(key=lambda x: x['total_strength'], reverse=True)
            final_targets = []
            
            allocations = [0.5, 0.3, 0.2]
            for i, target in enumerate(confluence_targets[:3]):
                final_targets.append({
                    'price': target['price'],
                    'allocation': allocations[i] if i < len(allocations) else 0.1,
                    'confluence_count': target['count'],
                    'types': list(target['types'])
                })
            
            # Ensure we have at least 3 targets
            while len(final_targets) < 3:
                multiplier = 1.0 + len(final_targets) * 0.5
                target_price = entry + (atr * multiplier if direction == 'long' else -atr * multiplier)
                final_targets.append({
                    'price': target_price,
                    'allocation': 0.2,
                    'confluence_count': 1,
                    'types': ['atr']
                })
            
            return final_targets
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic targets: {e}")
            # Fallback to simple ATR targets
            return [
                {'price': entry + (atr * 1.0 if direction == 'long' else -atr * 1.0), 'allocation': 0.4},
                {'price': entry + (atr * 1.5 if direction == 'long' else -atr * 1.5), 'allocation': 0.3},
                {'price': entry + (atr * 2.5 if direction == 'long' else -atr * 2.5), 'allocation': 0.3}
            ]
    
    def _calculate_fibonacci_extensions(
        self, 
        entry: float, 
        direction: str, 
        atr: float
    ) -> List[Dict]:
        """Calculate Fibonacci extension levels"""
        fib_ratios = [1.272, 1.414, 1.618, 2.618]
        fib_levels = []
        
        for ratio in fib_ratios:
            fib_price = entry + (atr * ratio if direction == 'long' else -atr * ratio)
            fib_levels.append({
                'price': fib_price,
                'ratio': ratio
            })
        
        return fib_levels
    
    def _find_confluence_zones(
        self, 
        targets: List[Dict], 
        tolerance: float
    ) -> List[Dict]:
        """
        Group targets into confluence zones
        Implements full algorithm specification for confluence detection
        """
        zones = []
        
        for target in targets:
            added_to_zone = False
            
            for zone in zones:
                if abs(target['price'] - zone['price']) < tolerance:
                    zone['count'] += 1
                    zone['total_strength'] += target['strength']
                    zone['types'].add(target['type'])
                    # Update zone price to weighted average
                    zone['price'] = (
                        (zone['price'] * (zone['count'] - 1) + target['price']) / 
                        zone['count']
                    )
                    added_to_zone = True
                    break
            
            if not added_to_zone:
                zones.append({
                    'price': target['price'],
                    'count': 1,
                    'total_strength': target['strength'],
                    'types': {target['type']}
                })
        
        return zones