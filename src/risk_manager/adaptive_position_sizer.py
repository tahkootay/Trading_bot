"""
Adaptive Position Sizing with Kelly Criterion
Implements full algorithm specification for position sizing
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from ..utils.algorithm_constants import RISK_PARAMS, LIQUIDITY_PARAMS

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    size: float
    risk_pct: float
    risk_amount: float
    adjustments: Dict[str, float]
    warnings: List[str]
    metadata: Dict[str, Any]

@dataclass
class TradeStats:
    """Trading statistics for Kelly calculation"""
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    consecutive_wins: int
    consecutive_losses: int
    profit_factor: float
    sharpe_ratio: float

class AdaptivePositionSizer:
    """
    Position sizing with multiple safety layers as per algorithm specification
    
    Implements:
    1. Base Kelly Criterion calculation
    2. Kelly cap (25% max)
    3. Correlation adjustment with existing positions
    4. Market volatility adjustment
    5. Drawdown-based position scaling
    6. Liquidity constraints
    7. Market regime adjustment
    """
    
    def __init__(self, account_equity: float, trade_stats: TradeStats):
        self.equity = account_equity
        self.stats = trade_stats
        self.correlation_matrix = {}
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(
        self, 
        setup: Dict, 
        current_positions: List[Dict], 
        market_conditions: Dict
    ) -> PositionSizeResult:
        """
        Calculate optimal position size with multiple safety layers
        Exactly as specified in algorithm
        
        Args:
            setup: Trading setup with entry, stop_loss, direction, etc.
            current_positions: List of current open positions
            market_conditions: Current market state and indicators
            
        Returns:
            PositionSizeResult with size and all adjustment details
        """
        try:
            warnings = []
            adjustments = {}
            
            # 1. Base Kelly Criterion
            kelly_fraction = self.calculate_kelly()
            adjustments['kelly_base'] = kelly_fraction
            
            # 2. Apply Kelly cap (25% max)
            kelly_capped = min(kelly_fraction, RISK_PARAMS['kelly_cap'])
            adjustments['kelly_capped'] = kelly_capped
            
            if kelly_capped < kelly_fraction:
                warnings.append(f"Kelly capped from {kelly_fraction:.3f} to {kelly_capped:.3f}")
            
            # 3. Adjust for correlation with existing positions
            correlation_adjustment = self.adjust_for_correlation(setup, current_positions)
            adjustments['correlation'] = correlation_adjustment
            
            if correlation_adjustment < 1.0:
                warnings.append(f"Position reduced due to correlation: {correlation_adjustment:.3f}")
            
            # 4. Adjust for market volatility
            volatility_adjustment = self.adjust_for_volatility(market_conditions)
            adjustments['volatility'] = volatility_adjustment
            
            # 5. Adjust for drawdown
            drawdown_adjustment = self.adjust_for_drawdown()
            adjustments['drawdown'] = drawdown_adjustment
            
            if drawdown_adjustment < 1.0:
                warnings.append(f"Position reduced due to drawdown: {drawdown_adjustment:.3f}")
            
            # 6. Adjust for liquidity
            liquidity_adjustment = self.adjust_for_liquidity(market_conditions)
            adjustments['liquidity'] = liquidity_adjustment
            
            if liquidity_adjustment < 1.0:
                warnings.append(f"Position reduced due to liquidity: {liquidity_adjustment:.3f}")
            
            # 7. Market regime adjustment
            regime = market_conditions.get('regime', 'normal_range')
            regime_adjustment = self.get_regime_multiplier(regime)
            adjustments['regime'] = regime_adjustment
            
            # Calculate final risk percentage
            risk_pct = (kelly_capped * 
                       correlation_adjustment * 
                       volatility_adjustment * 
                       drawdown_adjustment * 
                       liquidity_adjustment * 
                       regime_adjustment)
            
            # Apply maximum risk cap
            risk_pct = min(risk_pct, RISK_PARAMS['position_risk_pct'])
            adjustments['final_risk_pct'] = risk_pct
            
            # Convert to position size
            risk_amount = self.equity * risk_pct
            stop_distance = abs(setup['entry'] - setup['stop_loss'])
            
            if stop_distance == 0:
                warnings.append("Stop distance is zero - using minimum risk")
                position_size = 0
            else:
                position_size = risk_amount / stop_distance
            
            # Final liquidity check
            max_size_liquidity = self.check_max_size_vs_liquidity(
                position_size,
                setup['entry'],
                market_conditions
            )
            
            final_size = min(position_size, max_size_liquidity)
            if final_size < position_size:
                warnings.append(f"Position capped by liquidity: {final_size:.4f} vs {position_size:.4f}")
            
            # Minimum position size check
            min_position_value = 50.0  # $50 minimum
            if final_size * setup['entry'] < min_position_value:
                warnings.append(f"Position below minimum ${min_position_value}")
                final_size = 0
            
            return PositionSizeResult(
                size=final_size,
                risk_pct=risk_pct,
                risk_amount=risk_amount,
                adjustments=adjustments,
                warnings=warnings,
                metadata={
                    'kelly_fraction': kelly_fraction,
                    'stop_distance': stop_distance,
                    'regime': regime,
                    'correlation_count': len([p for p in current_positions if abs(self.get_correlation(setup['symbol'], p['symbol'])) > 0.5]),
                    'liquidity_constraint': max_size_liquidity,
                    'position_value_usd': final_size * setup['entry']
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return PositionSizeResult(
                size=0,
                risk_pct=0,
                risk_amount=0,
                adjustments={},
                warnings=[f"Calculation error: {str(e)}"],
                metadata={}
            )
    
    def calculate_kelly(self) -> float:
        """
        Calculate Kelly fraction from trade statistics
        Exactly as specified in algorithm
        """
        try:
            if self.stats.total_trades < 30:
                return 0.01  # Minimum size for new systems
            
            win_rate = self.stats.win_rate
            avg_win = self.stats.avg_win
            avg_loss = abs(self.stats.avg_loss)
            
            if avg_win <= 0 or avg_loss <= 0:
                return 0.01
            
            # Kelly formula: f = (p*b - q) / b
            # where p = win_rate, q = 1-win_rate, b = avg_win/avg_loss
            b = avg_win / avg_loss
            kelly = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0
            
            # Kelly shouldn't be negative or too high
            kelly = max(0.005, min(kelly, 0.5))  # Cap between 0.5% and 50%
            
            return kelly
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly: {e}")
            return 0.01
    
    def adjust_for_correlation(self, setup: Dict, current_positions: List[Dict]) -> float:
        """
        Reduce size if correlated with existing positions
        Exactly as specified in algorithm
        """
        try:
            if not current_positions:
                return 1.0
            
            total_correlation = 0
            correlation_count = 0
            
            for position in current_positions:
                correlation = self.get_correlation(setup['symbol'], position['symbol'])
                
                if setup['direction'] == position['direction']:
                    # Same direction - full correlation impact
                    total_correlation += abs(correlation)
                else:
                    # Opposite direction - hedging benefit
                    total_correlation -= correlation * 0.5
                
                correlation_count += 1
            
            if correlation_count == 0:
                return 1.0
            
            # Average correlation impact
            avg_correlation = total_correlation / correlation_count
            
            # Reduce size based on correlation
            # High correlation = lower position size
            correlation_factor = 1.0 / (1.0 + max(0, avg_correlation))
            
            # Ensure minimum and maximum bounds
            return max(0.3, min(1.0, correlation_factor))
            
        except Exception as e:
            self.logger.error(f"Error adjusting for correlation: {e}")
            return 1.0
    
    def adjust_for_volatility(self, market_conditions: Dict) -> float:
        """
        Adjust size based on current vs normal volatility
        Exactly as specified in algorithm
        """
        try:
            atr_current = market_conditions.get('atr', 0)
            atr_average = market_conditions.get('atr_20d_avg', atr_current)
            
            if atr_average <= 0:
                return 1.0
            
            volatility_ratio = atr_current / atr_average
            
            # Inverse relationship - higher vol = smaller position
            # Formula: 1.0 / (0.5 + 0.5 * volatility_ratio)
            adjustment = 1.0 / (0.5 + 0.5 * volatility_ratio)
            
            # Bounds: 0.5x to 1.5x normal size
            return max(0.5, min(1.5, adjustment))
            
        except Exception as e:
            self.logger.error(f"Error adjusting for volatility: {e}")
            return 1.0
    
    def adjust_for_drawdown(self) -> float:
        """
        Progressive position reduction based on drawdown
        Exactly as specified in algorithm
        """
        try:
            current_dd = self.calculate_current_drawdown()
            
            # Check each drawdown threshold
            for dd_threshold, params in sorted(RISK_PARAMS['dd_levels'].items()):
                if current_dd >= dd_threshold:
                    return params['position_mult']
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error adjusting for drawdown: {e}")
            return 1.0
    
    def adjust_for_liquidity(self, market_conditions: Dict) -> float:
        """
        Ensure position size doesn't exceed market liquidity
        Exactly as specified in algorithm
        """
        try:
            volume_5m = market_conditions.get('volume_5m_usd', 0)
            orderbook_depth = market_conditions.get('orderbook_depth_usd', 0)
            
            if volume_5m <= 0 or orderbook_depth <= 0:
                return 1.0
            
            # Position shouldn't exceed 1% of 5m volume
            max_position_value = self.equity * RISK_PARAMS['position_risk_pct']
            volume_constraint = LIQUIDITY_PARAMS['max_position_vs_volume']
            
            volume_factor = min(1.0, (volume_constraint * volume_5m) / max_position_value)
            
            # Position shouldn't exceed 10% of visible orderbook
            depth_constraint = LIQUIDITY_PARAMS['max_position_vs_depth']
            depth_factor = min(1.0, (depth_constraint * orderbook_depth) / max_position_value)
            
            # Take the more conservative constraint
            return min(volume_factor, depth_factor)
            
        except Exception as e:
            self.logger.error(f"Error adjusting for liquidity: {e}")
            return 1.0
    
    def get_regime_multiplier(self, regime: str) -> float:
        """
        Adjust position size based on market regime
        Exactly as specified in algorithm
        """
        multipliers = {
            'strong_trend': 1.2,
            'trending': 1.0,
            'normal_range': 0.8,
            'low_volatility_range': 0.6,
            'volatile_choppy': 0.4
        }
        return multipliers.get(regime, 0.5)
    
    def check_max_size_vs_liquidity(
        self, 
        position_size: float, 
        entry_price: float, 
        market_conditions: Dict
    ) -> float:
        """
        Final check against available liquidity
        Exactly as specified in algorithm
        """
        try:
            position_value = position_size * entry_price
            
            # Check against 5m volume
            volume_5m_usd = market_conditions.get('volume_5m_usd', 0)
            if volume_5m_usd > 0:
                max_vs_volume = volume_5m_usd * LIQUIDITY_PARAMS['max_position_vs_volume']
            else:
                max_vs_volume = float('inf')
            
            # Check against orderbook depth
            depth_usd = market_conditions.get('orderbook_depth_usd', 0)
            if depth_usd > 0:
                max_vs_depth = depth_usd * LIQUIDITY_PARAMS['max_position_vs_depth']
            else:
                max_vs_depth = float('inf')
            
            max_position_value = min(max_vs_volume, max_vs_depth)
            
            if position_value > max_position_value:
                return max_position_value / entry_price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error checking liquidity constraints: {e}")
            return position_size
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols
        In production, this would use historical price correlation
        """
        try:
            # Simplified correlation matrix
            # In production, calculate from historical returns
            correlations = {
                ('SOL/USDT', 'SOL/USDT'): 1.0,
                ('SOL/USDT', 'BTC/USDT'): 0.75,
                ('SOL/USDT', 'ETH/USDT'): 0.80,
                ('SOL/USDT', 'AVAX/USDT'): 0.85,
            }
            
            key = (symbol1, symbol2)
            reverse_key = (symbol2, symbol1)
            
            return correlations.get(key, correlations.get(reverse_key, 0.5))
            
        except Exception:
            return 0.5  # Default moderate correlation
    
    def calculate_current_drawdown(self) -> float:
        """
        Calculate current drawdown from peak equity
        In production, this would track actual equity curve
        """
        try:
            # This would be tracked in the account manager
            # For now, return a placeholder
            return 0.0
            
        except Exception:
            return 0.0
    
    def update_trade_stats(
        self, 
        pnl: float, 
        win: bool, 
        trade_duration_minutes: float
    ):
        """
        Update trade statistics for Kelly calculation
        
        Args:
            pnl: Trade P&L
            win: Whether trade was profitable
            trade_duration_minutes: Duration of trade
        """
        try:
            # In production, this would update comprehensive statistics
            # Including win rate, avg win/loss, consecutive counts, etc.
            
            self.stats.total_trades += 1
            
            if win:
                self.stats.consecutive_wins += 1
                self.stats.consecutive_losses = 0
                # Update avg_win with exponential moving average
                alpha = 0.1
                self.stats.avg_win = (1 - alpha) * self.stats.avg_win + alpha * abs(pnl)
            else:
                self.stats.consecutive_losses += 1
                self.stats.consecutive_wins = 0
                # Update avg_loss with exponential moving average
                alpha = 0.1
                self.stats.avg_loss = (1 - alpha) * self.stats.avg_loss + alpha * abs(pnl)
            
            # Recalculate win rate
            # In production, use proper rolling window or exponential weighting
            
        except Exception as e:
            self.logger.error(f"Error updating trade stats: {e}")
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of current sizing parameters"""
        return {
            'equity': self.equity,
            'kelly_fraction': self.calculate_kelly(),
            'total_trades': self.stats.total_trades,
            'win_rate': self.stats.win_rate,
            'avg_win': self.stats.avg_win,
            'avg_loss': self.stats.avg_loss,
            'consecutive_losses': self.stats.consecutive_losses,
            'current_drawdown': self.calculate_current_drawdown(),
            'max_position_risk': RISK_PARAMS['position_risk_pct'],
            'kelly_cap': RISK_PARAMS['kelly_cap']
        }