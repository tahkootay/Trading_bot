"""
Trade Management Engine for SOL/USDT Trading Algorithm

Manages active positions with trailing stops, partial exits, and setup invalidation
as specified in the algorithm document.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class TradeAction:
    """Action to be taken on a trade"""
    action: str  # 'hold', 'close', 'adjust_stop', 'partial_close'
    reason: str
    data: Optional[Dict] = None

@dataclass
class TargetLevel:
    """Take profit target level"""
    price: float
    allocation: float
    executed: bool = False
    execution_time: Optional[datetime] = None

class TradeManager:
    """
    Manages individual trade lifecycle from entry to exit
    with sophisticated exit strategies as per algorithm specification.
    
    Implements:
    - Multi-level take profits with automatic partial closes
    - Multiple trailing stop methods (Parabolic SAR, structure-based, ATR-based)
    - Time stops adaptive by setup type
    - Setup invalidation checks
    - Volatility expansion handling
    - Profit lock mechanism
    """
    
    def __init__(self, position: Dict, config: Dict = None):
        self.position = position
        self.config = config or {}
        self.entry_time = time.time()
        self.max_profit = 0.0
        self.max_adverse = 0.0
        self.max_profit_price = 0.0
        self.tp_executed = []
        self.trailing_active = False
        self.break_even_set = False
        self.entry_atr = 0.0
        
        # Algorithm constants from specification
        from ..utils.algorithm_constants import SL_TP_PARAMS, TIME_STOPS
        self.SL_TP_PARAMS = SL_TP_PARAMS
        self.TIME_STOPS = TIME_STOPS
        
        # Setup targets from position
        self.targets = [
            TargetLevel(price=target['price'], allocation=target['allocation'])
            for target in position.get('targets', [])
        ]
        
        # Initialize profit lock mechanism
        self.profit_locked = False
        self.locked_profit_level = 0.0
        
        self.logger = logging.getLogger(__name__)
        
        # Algorithm constants
        self.PROFIT_LOCK_PCT = self.config.get('profit_lock_pct', 0.70)
        self.TRAILING_ACTIVATION_ATR = self.config.get('trailing_activation', 1.0)
    
    def update(self, current_price: float, market_data: Dict) -> TradeAction:
        """
        Main update loop for position management as per algorithm specification
        
        Args:
            current_price: Current market price
            market_data: Current market state including ATR, indicators, etc.
            
        Returns:
            TradeAction to be taken
        """
        try:
            # Store initial ATR if not set
            if self.entry_atr == 0.0:
                self.entry_atr = market_data.get('atr', current_price * 0.015)
            
            # Update profit/loss tracking
            self._update_profit_tracking(current_price)
            
            # Check take profits (partial closes)
            tp_action = self._check_take_profits(current_price)
            if tp_action.action == 'partial_close':
                return tp_action
            
            # Update stop loss (trailing, break-even, profit lock)
            self._update_stop_loss(current_price, market_data)
            
            # Check time stop
            time_action = self._check_time_stop(market_data)
            if time_action.action == 'close':
                return time_action
            
            # Check volatility expansion
            vol_action = self._check_volatility_expansion(market_data)
            if vol_action.action == 'adjust_stop':
                return vol_action
            
            # Check for setup invalidation
            invalidation_action = self._check_setup_invalidation(market_data)
            if invalidation_action.action == 'close':
                return invalidation_action
            
            # Check if stop loss hit
            if self._is_stop_loss_hit(current_price):
                return TradeAction(
                    action='close',
                    reason='stop_loss',
                    data={'price': current_price, 'stop_level': self.position['stop_loss']}
                )
            
            return TradeAction(action='hold', reason='all_checks_passed')
            
        except Exception as e:
            self.logger.error(f"Error in trade update: {e}")
            return TradeAction(action='hold', reason='error_in_update')
    
    def _update_profit_tracking(self, current_price: float):
        """Update maximum profit and adverse movement tracking"""
        entry_price = self.position['entry']
        direction = self.position['direction']
        
        if direction == 'long':
            profit = current_price - entry_price
            adverse = entry_price - current_price if current_price < entry_price else 0
        else:
            profit = entry_price - current_price
            adverse = current_price - entry_price if current_price > entry_price else 0
        
        # Update maximums
        if profit > self.max_profit:
            self.max_profit = profit
            self.max_profit_price = current_price
        
        if adverse > self.max_adverse:
            self.max_adverse = adverse
    
    def _check_take_profits(self, current_price: float) -> TradeAction:
        """
        Execute partial take profits as per algorithm specification
        """
        remaining_targets = [tp for tp in self.targets if not tp.executed]
        
        for target in remaining_targets:
            hit = False
            
            if self.position['direction'] == 'long':
                hit = current_price >= target.price
            else:
                hit = current_price <= target.price
            
            if hit:
                # Execute partial close
                close_size = self.position['size'] * target.allocation
                target.executed = True
                target.execution_time = datetime.now()
                self.tp_executed.append(target)
                
                # Update position size
                self.position['size'] -= close_size
                
                # Move to break-even after first TP
                if len(self.tp_executed) == 1:
                    self.position['stop_loss'] = self.position['entry']
                    self.trailing_active = True
                    self.break_even_set = True
                
                self.logger.info(f"Take profit {len(self.tp_executed)} executed at {target.price}")
                
                return TradeAction(
                    action='partial_close',
                    reason=f'take_profit_{len(self.tp_executed)}',
                    data={
                        'size': close_size,
                        'price': target.price,
                        'remaining_size': self.position['size'],
                        'target_level': len(self.tp_executed)
                    }
                )
        
        return TradeAction(action='hold', reason='no_take_profits_hit')
    
    def _update_stop_loss(self, current_price: float, market_data: Dict):
        """
        Update stop loss using multiple trailing methods as per algorithm
        """
        if not self.trailing_active:
            return
        
        atr = market_data.get('atr', self.entry_atr)
        
        # Check if we should activate trailing (1 ATR profit reached)
        if not self.trailing_active and self.max_profit >= self.TRAILING_ACTIVATION_ATR * atr:
            self.trailing_active = True
            self.logger.info("Trailing stop activated")
        
        if not self.trailing_active:
            return
        
        # Multiple trailing methods - use the most protective
        potential_stops = []
        
        # 1. Parabolic SAR
        sar_stop = self._calculate_parabolic_sar(market_data)
        if sar_stop:
            potential_stops.append(('sar', sar_stop))
        
        # 2. Structure-based (swing points)
        structure_stop = self._get_structure_stop(market_data)
        if structure_stop:
            potential_stops.append(('structure', structure_stop))
        
        # 3. Percentage of max profit lock
        if self.max_profit > atr:
            profit_stop = self._calculate_profit_lock_stop()
            if profit_stop:
                potential_stops.append(('profit_lock', profit_stop))
        
        # 4. ATR-based trailing
        atr_stop = self._calculate_atr_trailing_stop(current_price, atr)
        potential_stops.append(('atr', atr_stop))
        
        # Select most protective stop
        if potential_stops:
            new_stop = self._select_best_stop(potential_stops)
            if self._is_stop_improvement(new_stop):
                old_stop = self.position['stop_loss']
                self.position['stop_loss'] = new_stop
                self.logger.info(f"Stop loss updated: {old_stop:.4f} -> {new_stop:.4f}")
    
    def _calculate_parabolic_sar(self, market_data: Dict) -> Optional[float]:
        """
        Calculate Parabolic SAR for trailing stop
        Simplified implementation - in production would use full PSAR calculation
        """
        try:
            # For now, use a simplified version based on recent highs/lows
            # In production, implement full PSAR with acceleration factor
            recent_prices = market_data.get('close', [])
            if len(recent_prices) < 5:
                return None
            
            if self.position['direction'] == 'long':
                # Find recent low for long position
                recent_lows = market_data.get('low', recent_prices)[-10:]
                return min(recent_lows) if recent_lows else None
            else:
                # Find recent high for short position
                recent_highs = market_data.get('high', recent_prices)[-10:]
                return max(recent_highs) if recent_highs else None
                
        except Exception:
            return None
    
    def _get_structure_stop(self, market_data: Dict) -> Optional[float]:
        """
        Get stop based on market structure (swing points) as per algorithm
        """
        try:
            swing_points = market_data.get('swing_points', [])
            if not swing_points:
                return None
            
            if self.position['direction'] == 'long':
                # Find most recent swing low
                recent_lows = [s['price'] for s in swing_points if s['type'] == 'low']
                return max(recent_lows) if recent_lows else None
            else:
                # Find most recent swing high
                recent_highs = [s['price'] for s in swing_points if s['type'] == 'high']
                return min(recent_highs) if recent_highs else None
                
        except Exception:
            return None
    
    def _calculate_profit_lock_stop(self) -> Optional[float]:
        """
        Calculate stop to lock in percentage of maximum profit
        """
        try:
            if self.max_profit <= 0:
                return None
            
            locked_profit = self.max_profit * self.PROFIT_LOCK_PCT
            entry_price = self.position['entry']
            
            if self.position['direction'] == 'long':
                return entry_price + locked_profit
            else:
                return entry_price - locked_profit
                
        except Exception:
            return None
    
    def _calculate_atr_trailing_stop(self, current_price: float, atr: float) -> float:
        """
        Calculate ATR-based trailing stop
        """
        multiplier = 1.5  # 1.5 ATR trailing distance
        
        if self.position['direction'] == 'long':
            return current_price - (multiplier * atr)
        else:
            return current_price + (multiplier * atr)
    
    def _select_best_stop(self, potential_stops: List[Tuple[str, float]]) -> float:
        """
        Select most protective stop from multiple methods
        """
        if not potential_stops:
            return self.position['stop_loss']
        
        direction = self.position['direction']
        
        if direction == 'long':
            # For long positions, highest stop is most protective
            best_stop = max(stop for _, stop in potential_stops if stop is not None)
        else:
            # For short positions, lowest stop is most protective
            best_stop = min(stop for _, stop in potential_stops if stop is not None)
        
        return best_stop
    
    def _is_stop_improvement(self, new_stop: float) -> bool:
        """
        Check if new stop is an improvement (more protective)
        """
        current_stop = self.position['stop_loss']
        direction = self.position['direction']
        
        if direction == 'long':
            return new_stop > current_stop
        else:
            return new_stop < current_stop
    
    def _check_time_stop(self, market_data: Dict) -> TradeAction:
        """
        Check if position exceeded time limit as per algorithm
        """
        try:
            time_in_position = (time.time() - self.entry_time) / 60  # minutes
            
            # Get appropriate time stop based on setup type
            setup_type = self.position.get('setup_type', 'default')
            time_limit = self.TIME_STOPS.get(setup_type, self.TIME_STOPS['default'])
            
            if time_in_position >= time_limit:
                # Only close if not in significant profit
                current_profit_atr = self.max_profit / market_data.get('atr', 1)
                
                if current_profit_atr < 0.2:  # Less than 0.2 ATR profit
                    return TradeAction(
                        action='close',
                        reason='time_stop',
                        data={
                            'time_in_position': time_in_position,
                            'time_limit': time_limit,
                            'setup_type': setup_type
                        }
                    )
            
            return TradeAction(action='hold', reason='time_limit_not_reached')
            
        except Exception as e:
            self.logger.error(f"Error checking time stop: {e}")
            return TradeAction(action='hold', reason='time_check_error')
    
    def _check_volatility_expansion(self, market_data: Dict) -> TradeAction:
        """
        Check for sudden volatility expansion as per algorithm
        """
        try:
            current_atr = market_data.get('atr', self.entry_atr)
            
            # If volatility doubled, widen stop
            if current_atr > self.entry_atr * 2.0:
                entry_price = self.position['entry']
                direction = self.position['direction']
                
                # Widen stop to 1.5x current ATR
                if direction == 'long':
                    new_stop = entry_price - (1.5 * current_atr)
                else:
                    new_stop = entry_price + (1.5 * current_atr)
                
                return TradeAction(
                    action='adjust_stop',
                    reason='volatility_expansion',
                    data={
                        'new_stop': new_stop,
                        'old_atr': self.entry_atr,
                        'new_atr': current_atr
                    }
                )
            
            return TradeAction(action='hold', reason='volatility_normal')
            
        except Exception as e:
            self.logger.error(f"Error checking volatility expansion: {e}")
            return TradeAction(action='hold', reason='volatility_check_error')
    
    def _check_setup_invalidation(self, market_data: Dict) -> TradeAction:
        """
        Check if original setup conditions are invalidated as per algorithm
        """
        try:
            # Check trend disappearance
            adx = market_data.get('adx', 25)
            if adx < 20:
                return TradeAction(
                    action='close',
                    reason='trend_disappeared',
                    data={'adx': adx}
                )
            
            # Check regime change to choppy for trend setups
            regime = market_data.get('regime', 'normal_range')
            setup_type = self.position.get('setup_type', 'momentum')
            
            if regime == 'volatile_choppy' and setup_type != 'mean_reversion':
                return TradeAction(
                    action='close',
                    reason='regime_invalidation',
                    data={'regime': regime, 'setup_type': setup_type}
                )
            
            # Check ML confidence drop
            ml_confidence = market_data.get('ml_confidence', 0.7)
            if ml_confidence < 0.5:
                return TradeAction(
                    action='close',
                    reason='ml_confidence_drop',
                    data={'ml_confidence': ml_confidence}
                )
            
            # Check BTC correlation spike
            btc_correlation = market_data.get('btc_correlation', 0.7)
            if abs(btc_correlation) > 0.85:
                return TradeAction(
                    action='close',
                    reason='high_btc_correlation',
                    data={'btc_correlation': btc_correlation}
                )
            
            return TradeAction(action='hold', reason='setup_still_valid')
            
        except Exception as e:
            self.logger.error(f"Error checking setup invalidation: {e}")
            return TradeAction(action='hold', reason='invalidation_check_error')
    
    def _is_stop_loss_hit(self, current_price: float) -> bool:
        """Check if stop loss has been hit"""
        stop_loss = self.position['stop_loss']
        direction = self.position['direction']
        
        if direction == 'long':
            return current_price <= stop_loss
        else:
            return current_price >= stop_loss
    
    def get_trade_stats(self) -> Dict[str, Any]:
        """Get current trade statistics"""
        return {
            'entry_time': self.entry_time,
            'time_in_position_min': (time.time() - self.entry_time) / 60,
            'max_profit': self.max_profit,
            'max_adverse': self.max_adverse,
            'max_profit_price': self.max_profit_price,
            'tp_executed_count': len(self.tp_executed),
            'trailing_active': self.trailing_active,
            'break_even_set': self.break_even_set,
            'profit_locked': self.profit_locked,
            'current_stop': self.position['stop_loss'],
            'remaining_size': self.position['size'],
            'setup_type': self.position.get('setup_type', 'unknown')
        }
        self.TIME_STOPS = self.config.get('time_stops', {
            'scalp': 30,
            'momentum': 60,
            'breakout': 180,
            'mean_reversion': 240,
            'position': 360,
            'default': 120
        })
        
        self.logger = logging.getLogger(__name__)
        
    def update(self, current_price: float, market_data: Dict) -> TradeAction:
        """
        Main update loop for position management
        
        Args:
            current_price: Current market price
            market_data: Current market state and indicators
            
        Returns:
            TradeAction specifying what to do with the position
        """
        try:
            # Update profit/loss tracking
            self._update_profit_tracking(current_price)
            
            # Check for take profit executions
            tp_action = self._check_take_profits(current_price)
            if tp_action:
                return tp_action
            
            # Update stop loss (trailing or break-even)
            self._update_stop_loss(current_price, market_data)
            
            # Check time-based stops
            time_action = self._check_time_stop(current_price, market_data)
            if time_action:
                return time_action
            
            # Check for volatility expansion
            vol_action = self._check_volatility_expansion(market_data)
            if vol_action:
                return vol_action
            
            # Check for setup invalidation
            invalidation_action = self._check_setup_invalidation(market_data)
            if invalidation_action:
                return invalidation_action
            
            # Check for adverse movement limits
            adverse_action = self._check_adverse_movement()
            if adverse_action:
                return adverse_action
            
            return TradeAction(action='hold', reason='all_checks_passed')
            
        except Exception as e:
            self.logger.error(f"Error in trade update: {e}")
            return TradeAction(action='hold', reason=f'error: {str(e)}')
    
    def _update_profit_tracking(self, current_price: float):
        """Update maximum profit and adverse movement tracking"""
        direction = self.position['direction']
        entry_price = self.position['entry']
        
        if direction == 'long':
            profit = current_price - entry_price
            adverse = entry_price - current_price
        else:
            profit = entry_price - current_price
            adverse = current_price - entry_price
        
        self.max_profit = max(self.max_profit, profit)
        self.max_adverse = max(self.max_adverse, adverse)
    
    def _check_take_profits(self, current_price: float) -> Optional[TradeAction]:
        """Execute partial take profits when targets are hit"""
        direction = self.position['direction']
        
        for i, target in enumerate(self.targets):
            if target.executed:
                continue
                
            target_hit = False
            if direction == 'long':
                target_hit = current_price >= target.price
            else:
                target_hit = current_price <= target.price
            
            if target_hit:
                return self._execute_partial_close(target, i)
        
        return None
    
    def _execute_partial_close(self, target: TargetLevel, target_index: int) -> TradeAction:
        """Execute partial position close at target"""
        close_size = self.position['size'] * target.allocation
        
        # Mark target as executed
        target.executed = True
        target.execution_time = datetime.now()
        self.tp_executed.append(target_index)
        
        # Update position size
        self.position['size'] -= close_size
        
        # Set break-even stop after first TP
        if len(self.tp_executed) == 1 and not self.break_even_set:
            self.position['stop_loss'] = self.position['entry']
            self.trailing_active = True
            self.break_even_set = True
            self.logger.info(f"Set break-even stop after first TP")
        
        return TradeAction(
            action='partial_close',
            reason=f'take_profit_{target_index + 1}',
            data={
                'size': close_size,
                'price': target.price,
                'remaining_size': self.position['size']
            }
        )
    
    def _update_stop_loss(self, current_price: float, market_data: Dict):
        """Update stop loss using multiple trailing methods"""
        if not self.trailing_active:
            return
        
        atr = market_data.get('atr', 0)
        if atr <= 0:
            return
        
        direction = self.position['direction']
        current_stop = self.position['stop_loss']
        
        # Multiple trailing stop methods - use most protective
        potential_stops = []
        
        # 1. Parabolic SAR
        sar_value = market_data.get('parabolic_sar', None)
        if sar_value:
            potential_stops.append(('sar', sar_value))
        
        # 2. Structure-based (swing points)
        structure_stop = self._get_structure_stop(market_data)
        if structure_stop:
            potential_stops.append(('structure', structure_stop))
        
        # 3. Percentage of max profit lock
        if self.max_profit > atr:
            profit_lock_price = self.position['entry'] + (
                self.max_profit * self.PROFIT_LOCK_PCT * 
                (1 if direction == 'long' else -1)
            )
            potential_stops.append(('profit_lock', profit_lock_price))
        
        # 4. ATR-based trailing
        atr_trail = current_price - (
            1.5 * atr if direction == 'long' else -1.5 * atr
        )
        potential_stops.append(('atr_trail', atr_trail))
        
        # Select most protective stop
        if direction == 'long':
            # For long, want highest stop (closest to current price)
            valid_stops = [price for _, price in potential_stops if price > current_stop]
            if valid_stops:
                new_stop = max(valid_stops)
                if new_stop > current_stop:
                    old_stop = current_stop
                    self.position['stop_loss'] = new_stop
                    self.logger.info(f"Updated trailing stop: {old_stop:.4f} -> {new_stop:.4f}")
        else:
            # For short, want lowest stop (closest to current price)
            valid_stops = [price for _, price in potential_stops if price < current_stop]
            if valid_stops:
                new_stop = min(valid_stops)
                if new_stop < current_stop:
                    old_stop = current_stop
                    self.position['stop_loss'] = new_stop
                    self.logger.info(f"Updated trailing stop: {old_stop:.4f} -> {new_stop:.4f}")
    
    def _check_time_stop(self, current_price: float, market_data: Dict) -> Optional[TradeAction]:
        """Check if position exceeded time limit"""
        time_in_position = (time.time() - self.entry_time) / 60  # minutes
        
        # Get appropriate time stop based on setup type
        setup_type = self.position.get('setup_type', 'default')
        time_limit = self.TIME_STOPS.get(setup_type, self.TIME_STOPS['default'])
        
        if time_in_position >= time_limit:
            # Check if we're in meaningful profit
            atr = market_data.get('atr', 0)
            current_pnl = self._calculate_current_pnl(current_price)
            
            # If profit is less than 0.2 ATR, close position
            min_profit_threshold = 0.2 * atr if atr > 0 else 0
            
            if current_pnl < min_profit_threshold:
                return TradeAction(
                    action='close',
                    reason='time_stop',
                    data={'time_in_position': time_in_position, 'pnl': current_pnl}
                )
        
        return None
    
    def _check_volatility_expansion(self, market_data: Dict) -> Optional[TradeAction]:
        """Check for sudden volatility expansion requiring stop adjustment"""
        current_atr = market_data.get('atr', 0)
        entry_atr = self.position.get('entry_atr', current_atr)
        
        if current_atr > entry_atr * 2.0:
            # Significant volatility expansion - widen stop to avoid premature exit
            direction = self.position['direction']
            
            new_stop = self.position['entry'] - (
                1.5 * current_atr if direction == 'long' else -1.5 * current_atr
            )
            
            # Only widen stop, never tighten due to volatility
            if ((direction == 'long' and new_stop < self.position['stop_loss']) or
                (direction == 'short' and new_stop > self.position['stop_loss'])):
                
                old_stop = self.position['stop_loss']
                self.position['stop_loss'] = new_stop
                
                return TradeAction(
                    action='adjust_stop',
                    reason='volatility_expansion',
                    data={
                        'old_stop': old_stop,
                        'new_stop': new_stop,
                        'atr_ratio': current_atr / entry_atr
                    }
                )
        
        return None
    
    def _check_setup_invalidation(self, market_data: Dict) -> Optional[TradeAction]:
        """Check if original setup conditions are invalidated"""
        reasons = []
        
        # Check if trend disappeared
        adx = market_data.get('adx', 0)
        if adx < 20:
            reasons.append('trend_disappeared')
        
        # Check if regime became unfavorable
        regime = market_data.get('regime', 'normal_range')
        setup_type = self.position.get('setup_type', 'momentum')
        
        if regime == 'volatile_choppy' and setup_type != 'mean_reversion':
            reasons.append('unfavorable_regime')
        
        # Check if ML confidence dropped significantly
        ml_confidence = market_data.get('ml_confidence', 0.5)
        if ml_confidence < 0.4:  # Significant drop in confidence
            reasons.append('ml_confidence_drop')
        
        # Check if correlation with BTC became extreme
        btc_correlation = market_data.get('btc_correlation', 0)
        if abs(btc_correlation) > 0.90:
            reasons.append('extreme_btc_correlation')
        
        # Close if multiple invalidation factors
        if len(reasons) >= 2:
            return TradeAction(
                action='close',
                reason='setup_invalidated',
                data={'invalidation_reasons': reasons}
            )
        
        return None
    
    def _check_adverse_movement(self) -> Optional[TradeAction]:
        """Check for excessive adverse movement"""
        # If adverse movement exceeds 2x expected (based on ATR), consider closing
        atr = self.position.get('entry_atr', 0)
        if atr > 0 and self.max_adverse > 2.0 * atr:
            # But only if we haven't hit any take profits yet
            if not self.tp_executed:
                return TradeAction(
                    action='close',
                    reason='excessive_adverse_movement',
                    data={'max_adverse': self.max_adverse, 'atr': atr}
                )
        
        return None
    
    def _get_structure_stop(self, market_data: Dict) -> Optional[float]:
        """Get stop based on market structure (swing points)"""
        swing_points = market_data.get('swing_points', [])
        direction = self.position['direction']
        
        if not swing_points:
            return None
        
        if direction == 'long':
            # Find most recent swing low
            recent_lows = [
                sp['price'] for sp in swing_points 
                if sp['type'] == 'low' and sp['price'] < self.position['entry']
            ]
            return max(recent_lows) if recent_lows else None
        else:
            # Find most recent swing high  
            recent_highs = [
                sp['price'] for sp in swing_points
                if sp['type'] == 'high' and sp['price'] > self.position['entry']
            ]
            return min(recent_highs) if recent_highs else None
    
    def _calculate_current_pnl(self, current_price: float = None) -> float:
        """Calculate current P&L of the position"""
        if current_price is None:
            return 0.0
        
        direction = self.position['direction']
        entry_price = self.position['entry']
        position_size = self.position['size']
        
        if direction == 'long':
            pnl_per_unit = current_price - entry_price
        else:
            pnl_per_unit = entry_price - current_price
        
        return pnl_per_unit * position_size
    
    def get_position_metrics(self, current_price: float) -> Dict[str, Any]:
        """Get comprehensive position metrics"""
        time_in_position = (time.time() - self.entry_time) / 60  # minutes
        current_pnl = self._calculate_current_pnl(current_price)
        
        return {
            'time_in_position_minutes': time_in_position,
            'current_pnl': current_pnl,
            'max_profit': self.max_profit,
            'max_adverse': self.max_adverse,
            'targets_hit': len(self.tp_executed),
            'total_targets': len(self.targets),
            'trailing_active': self.trailing_active,
            'break_even_set': self.break_even_set,
            'current_stop': self.position['stop_loss'],
            'remaining_size': self.position['size']
        }
    
    def force_close(self, reason: str = 'manual') -> TradeAction:
        """Force close the position immediately"""
        return TradeAction(
            action='close',
            reason=f'force_close_{reason}',
            data={'remaining_size': self.position['size']}
        )
    
    def adjust_stop_manual(self, new_stop: float, reason: str = 'manual') -> TradeAction:
        """Manually adjust stop loss"""
        old_stop = self.position['stop_loss']
        self.position['stop_loss'] = new_stop
        
        return TradeAction(
            action='adjust_stop',
            reason=f'manual_{reason}',
            data={'old_stop': old_stop, 'new_stop': new_stop}
        )
    
    def is_position_profitable(self, current_price: float, min_profit_atr: float = 0.5) -> bool:
        """Check if position is meaningfully profitable"""
        current_pnl = self._calculate_current_pnl(current_price)
        atr = self.position.get('entry_atr', 0)
        
        if atr > 0:
            return current_pnl > (min_profit_atr * atr)
        else:
            return current_pnl > 0
    
    def get_risk_metrics(self, current_price: float) -> Dict[str, float]:
        """Get current risk metrics for the position"""
        direction = self.position['direction']
        entry_price = self.position['entry']
        stop_loss = self.position['stop_loss']
        position_size = self.position['size']
        
        # Calculate distances
        if direction == 'long':
            distance_to_stop = current_price - stop_loss
            unrealized_risk = (entry_price - stop_loss) * position_size
        else:
            distance_to_stop = stop_loss - current_price
            unrealized_risk = (stop_loss - entry_price) * position_size
        
        return {
            'distance_to_stop': distance_to_stop,
            'unrealized_risk': unrealized_risk,
            'risk_reward_ratio': self.max_profit / max(unrealized_risk, 0.01),
            'stop_loss_price': stop_loss,
            'break_even_distance': abs(current_price - entry_price)
        }
    
    def update_dynamic_targets(self, market_data: Dict):
        """Update targets based on evolving market structure"""
        try:
            entry = self.position['entry']
            direction = self.position['direction']
            atr = market_data.get('atr', 0)
            
            # Get updated targets
            new_targets = self._calculate_dynamic_targets(entry, direction, atr, market_data)
            
            # Update only unexecuted targets
            for i, target in enumerate(self.targets):
                if not target.executed and i < len(new_targets):
                    target.price = new_targets[i]['price']
                    
        except Exception as e:
            self.logger.error(f"Error updating dynamic targets: {e}")
    
    def _calculate_dynamic_targets(self, entry: float, direction: str, atr: float, market_data: Dict) -> List[Dict]:
        """
        Calculate dynamic targets based on market structure as per algorithm specification
        Including Fibonacci extensions, volume profile, and confluence zones
        """
        targets = []
        
        try:
            # Get market structure data
            volume_profile = self._get_volume_profile_targets(entry, direction, market_data)
            fibonacci_targets = self._get_fibonacci_targets(entry, direction, atr, market_data)
            structure_targets = self._get_structure_targets(entry, direction, market_data)
            
            # Combine all target types with confluence scoring
            all_targets = []
            
            # Add volume profile targets
            for vp_target in volume_profile:
                all_targets.append({
                    'price': vp_target['price'],
                    'type': 'volume_profile',
                    'strength': vp_target['strength'],
                    'confluence_score': 1.0
                })
            
            # Add Fibonacci targets
            for fib_target in fibonacci_targets:
                all_targets.append({
                    'price': fib_target['price'],
                    'type': 'fibonacci',
                    'strength': fib_target['strength'],
                    'confluence_score': 1.0
                })
            
            # Add structure targets
            for struct_target in structure_targets:
                all_targets.append({
                    'price': struct_target['price'],
                    'type': 'structure',
                    'strength': struct_target['strength'],
                    'confluence_score': 1.0
                })
            
            # Calculate confluence zones
            confluence_targets = self._calculate_confluence_zones(all_targets, entry)
            
            # Select best targets based on confluence and distance
            selected_targets = self._select_optimal_targets(confluence_targets, entry, direction)
            
            # Fall back to ATR-based targets if no structure targets found
            if not selected_targets:
                selected_targets = self._get_atr_based_targets(entry, direction, atr)
            
            return selected_targets
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic targets: {e}")
            # Fall back to simple ATR-based targets
            return self._get_atr_based_targets(entry, direction, atr)
    
    def _get_volume_profile_targets(self, entry: float, direction: str, market_data: Dict) -> List[Dict]:
        """Get targets based on volume profile (POC, VAH, VAL)"""
        targets = []
        
        # Get volume profile from market data if available
        volume_summary = market_data.get('volume_profile', {})
        
        if volume_summary:
            poc = volume_summary.get('poc', entry)
            vah = volume_summary.get('vah', entry * 1.02)
            val = volume_summary.get('val', entry * 0.98)
            
            if direction == 'long':
                # Long targets above entry
                if poc > entry:
                    targets.append({'price': poc, 'strength': 0.8})
                if vah > entry:
                    targets.append({'price': vah, 'strength': 0.6})
            else:
                # Short targets below entry
                if poc < entry:
                    targets.append({'price': poc, 'strength': 0.8})
                if val < entry:
                    targets.append({'price': val, 'strength': 0.6})
        
        return targets
    
    def _get_fibonacci_targets(self, entry: float, direction: str, atr: float, market_data: Dict) -> List[Dict]:
        """Calculate Fibonacci extension targets as per algorithm"""
        targets = []
        
        # Get swing points for Fibonacci calculation
        swing_points = market_data.get('swing_points', [])
        
        if len(swing_points) >= 2:
            # Use last significant swing for Fibonacci calculation
            recent_swings = sorted(swing_points, key=lambda x: x.get('timestamp', 0))[-2:]
            
            if len(recent_swings) == 2:
                swing_low = min(recent_swings, key=lambda x: x['price'])
                swing_high = max(recent_swings, key=lambda x: x['price'])
                
                swing_range = swing_high['price'] - swing_low['price']
                
                # Fibonacci extension levels: 1.618, 2.618, 4.236
                fib_levels = [1.618, 2.618, 4.236]
                
                for level in fib_levels:
                    if direction == 'long':
                        # Extend from swing high
                        target_price = swing_high['price'] + (swing_range * (level - 1))
                        if target_price > entry:
                            strength = 1.0 - (level - 1.618) * 0.2  # Decreasing strength
                            targets.append({'price': target_price, 'strength': max(0.3, strength)})
                    else:
                        # Extend from swing low
                        target_price = swing_low['price'] - (swing_range * (level - 1))
                        if target_price < entry:
                            strength = 1.0 - (level - 1.618) * 0.2
                            targets.append({'price': target_price, 'strength': max(0.3, strength)})
        
        return targets
    
    def _get_structure_targets(self, entry: float, direction: str, market_data: Dict) -> List[Dict]:
        """Get targets based on market structure (S/R levels)"""
        targets = []
        
        # Get key levels from market data
        key_levels = market_data.get('key_levels', [])
        
        for level in key_levels:
            level_price = level.get('price', 0)
            level_type = level.get('type', 'unknown')
            level_strength = level.get('strength', 0.5)
            
            # Select appropriate levels based on direction
            if direction == 'long' and level_type == 'resistance' and level_price > entry:
                targets.append({
                    'price': level_price,
                    'strength': level_strength,
                    'level_type': level_type
                })
            elif direction == 'short' and level_type == 'support' and level_price < entry:
                targets.append({
                    'price': level_price,
                    'strength': level_strength,
                    'level_type': level_type
                })
        
        return targets
    
    def _calculate_confluence_zones(self, all_targets: List[Dict], entry: float) -> List[Dict]:
        """Calculate confluence zones where multiple targets cluster"""
        confluence_targets = []
        price_tolerance = entry * 0.002  # 0.2% tolerance for confluence
        
        # Group targets by proximity
        target_groups = []
        for target in all_targets:
            added_to_group = False
            
            for group in target_groups:
                group_avg_price = sum(t['price'] for t in group) / len(group)
                if abs(target['price'] - group_avg_price) <= price_tolerance:
                    group.append(target)
                    added_to_group = True
                    break
            
            if not added_to_group:
                target_groups.append([target])
        
        # Calculate confluence score for each group
        for group in target_groups:
            if len(group) >= 1:  # Minimum confluence
                avg_price = sum(t['price'] for t in group) / len(group)
                total_strength = sum(t['strength'] for t in group)
                confluence_score = len(group) * total_strength / len(group)
                
                confluence_targets.append({
                    'price': avg_price,
                    'confluence_score': confluence_score,
                    'target_count': len(group),
                    'types': [t['type'] for t in group]
                })
        
        return confluence_targets
    
    def _select_optimal_targets(self, confluence_targets: List[Dict], entry: float, direction: str) -> List[Dict]:
        """Select optimal targets based on confluence and distance"""
        if not confluence_targets:
            return []
        
        # Filter targets by direction
        valid_targets = []
        for target in confluence_targets:
            if direction == 'long' and target['price'] > entry:
                valid_targets.append(target)
            elif direction == 'short' and target['price'] < entry:
                valid_targets.append(target)
        
        if not valid_targets:
            return []
        
        # Sort by confluence score
        valid_targets.sort(key=lambda x: x['confluence_score'], reverse=True)
        
        # Select top 3 targets with progressive allocations
        selected = valid_targets[:3]
        allocations = [0.5, 0.3, 0.2]  # Algorithm specification
        
        targets = []
        for i, target in enumerate(selected):
            targets.append({
                'price': target['price'],
                'allocation': allocations[i] if i < len(allocations) else 0.1,
                'confluence_score': target['confluence_score'],
                'target_count': target['target_count']
            })
        
        return targets
    
    def _get_atr_based_targets(self, entry: float, direction: str, atr: float) -> List[Dict]:
        """Fallback ATR-based targets"""
        targets = []
        
        # Algorithm specification: 1.0, 1.5, 2.5 ATR targets
        multipliers = [1.0, 1.5, 2.5]
        allocations = [0.4, 0.3, 0.3]
        
        for i, (mult, alloc) in enumerate(zip(multipliers, allocations)):
            if direction == 'long':
                target_price = entry + (atr * mult)
            else:
                target_price = entry - (atr * mult)
            
            targets.append({
                'price': target_price,
                'allocation': alloc,
                'type': 'atr_based'
            })
        
        return targets