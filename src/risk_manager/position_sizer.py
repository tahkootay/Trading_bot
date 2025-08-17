"""
Position Sizing with Adaptive Kelly Criterion

Implements the sophisticated position sizing algorithm from the specification
including Kelly criterion with multiple safety layers and regime adjustments.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    size: float
    risk_pct: float
    risk_amount: float
    adjustments: Dict[str, float]
    reasoning: str
    kelly_base: float
    final_multiplier: float

@dataclass
class TradeStats:
    """Trading statistics for Kelly calculation"""
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float

class PositionSizer:
    """
    Advanced position sizing using adaptive Kelly criterion
    with multiple safety layers as per algorithm specification.
    """
    
    def __init__(self, account_equity: float, trade_stats: TradeStats, config: Dict = None):
        self.equity = account_equity
        self.stats = trade_stats
        self.config = config or {}
        self.correlation_matrix = {}
        
        # Algorithm constants
        self.KELLY_CAP = self.config.get('kelly_cap', 0.25)  # 25% maximum
        self.BASE_RISK_PCT = self.config.get('base_risk_pct', 0.02)  # 2% base
        self.MIN_TRADES_FOR_KELLY = 30
        
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(
        self, 
        setup: Dict, 
        current_positions: List[Dict], 
        market_conditions: Dict
    ) -> PositionSizeResult:
        """
        Calculate optimal position size with multiple safety layers
        
        Args:
            setup: Trading setup details
            current_positions: List of current open positions
            market_conditions: Current market state including regime, volatility, etc.
            
        Returns:
            PositionSizeResult with detailed breakdown
        """
        reasoning_parts = []
        
        # 1. Base Kelly Criterion
        kelly_fraction = self.calculate_kelly()
        reasoning_parts.append(f"Kelly base: {kelly_fraction:.3f}")
        
        # 2. Apply Kelly cap (25% max)
        kelly_capped = min(kelly_fraction, self.KELLY_CAP)
        if kelly_capped != kelly_fraction:
            reasoning_parts.append(f"Kelly capped at {self.KELLY_CAP:.1%}")
        
        # 3. Multiple adjustment layers
        adjustments = {}
        
        # Correlation adjustment
        correlation_adj = self.adjust_for_correlation(setup, current_positions)
        adjustments['correlation'] = correlation_adj
        reasoning_parts.append(f"Correlation adj: {correlation_adj:.2f}")
        
        # Volatility adjustment
        volatility_adj = self.adjust_for_volatility(market_conditions)
        adjustments['volatility'] = volatility_adj
        reasoning_parts.append(f"Volatility adj: {volatility_adj:.2f}")
        
        # Drawdown adjustment
        drawdown_adj = self.adjust_for_drawdown()
        adjustments['drawdown'] = drawdown_adj
        if drawdown_adj < 1.0:
            reasoning_parts.append(f"Drawdown adj: {drawdown_adj:.2f}")
        
        # Liquidity adjustment
        liquidity_adj = self.adjust_for_liquidity(market_conditions)
        adjustments['liquidity'] = liquidity_adj
        if liquidity_adj < 1.0:
            reasoning_parts.append(f"Liquidity adj: {liquidity_adj:.2f}")
        
        # Market regime adjustment
        regime_adj = self.get_regime_multiplier(market_conditions.get('regime', 'normal_range'))
        adjustments['regime'] = regime_adj
        reasoning_parts.append(f"Regime adj: {regime_adj:.2f}")
        
        # Setup confidence adjustment
        confidence_adj = self.adjust_for_setup_confidence(setup)
        adjustments['confidence'] = confidence_adj
        reasoning_parts.append(f"Confidence adj: {confidence_adj:.2f}")
        
        # 4. Calculate final risk percentage
        final_multiplier = (
            correlation_adj * 
            volatility_adj * 
            drawdown_adj * 
            liquidity_adj * 
            regime_adj * 
            confidence_adj
        )
        
        risk_pct = kelly_capped * final_multiplier
        
        # Apply maximum risk cap
        risk_pct = min(risk_pct, self.BASE_RISK_PCT)
        
        # 5. Convert to position size
        risk_amount = self.equity * risk_pct
        stop_distance = abs(setup['entry'] - setup['stop_loss'])
        
        if stop_distance <= 0:
            self.logger.warning("Invalid stop distance, using 1% of entry price")
            stop_distance = setup['entry'] * 0.01
        
        position_size = risk_amount / stop_distance
        
        # 6. Final liquidity check
        max_size_liquidity = self.check_max_size_vs_liquidity(
            position_size,
            setup['entry'],
            market_conditions
        )
        
        final_size = min(position_size, max_size_liquidity)
        
        # If size was reduced by liquidity, adjust risk amount
        if final_size < position_size:
            risk_amount = final_size * stop_distance
            risk_pct = risk_amount / self.equity
            reasoning_parts.append(f"Size limited by liquidity")
        
        return PositionSizeResult(
            size=final_size,
            risk_pct=risk_pct,
            risk_amount=risk_amount,
            adjustments=adjustments,
            reasoning=" | ".join(reasoning_parts),
            kelly_base=kelly_fraction,
            final_multiplier=final_multiplier
        )
    
    def calculate_kelly(self) -> float:
        """
        Calculate Kelly fraction from trade statistics
        
        Returns:
            Kelly fraction (0.0 to 1.0)
        """
        if self.stats.total_trades < self.MIN_TRADES_FOR_KELLY:
            # Return conservative size for new systems
            return min(0.01, self.BASE_RISK_PCT / 2)
        
        win_rate = self.stats.win_rate
        avg_win = self.stats.avg_win
        avg_loss = abs(self.stats.avg_loss)
        
        if avg_win <= 0 or avg_loss <= 0:
            return 0.01
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win_rate, q = 1-win_rate, b = avg_win/avg_loss
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b if b > 0 else 0
        
        # Additional safety checks
        if kelly < 0:
            return 0.005  # Minimum position size
        
        # Reduce Kelly if Sharpe ratio is low
        if hasattr(self.stats, 'sharpe_ratio') and self.stats.sharpe_ratio < 1.0:
            kelly *= max(0.5, self.stats.sharpe_ratio)
        
        # Reduce Kelly if max drawdown is high
        if hasattr(self.stats, 'max_drawdown') and self.stats.max_drawdown > 0.10:
            kelly *= max(0.5, 1 - self.stats.max_drawdown)
        
        return max(0.005, min(kelly, self.KELLY_CAP))
    
    def adjust_for_correlation(self, setup: Dict, current_positions: List[Dict]) -> float:
        """
        Reduce size if correlated with existing positions
        
        Args:
            setup: Trading setup
            current_positions: List of current positions
            
        Returns:
            Adjustment factor (0.3 to 1.0)
        """
        if not current_positions:
            return 1.0
        
        total_correlation = 0.0
        symbol = setup.get('symbol', 'SOLUSDT')
        direction = setup.get('direction', 'long')
        
        for position in current_positions:
            pos_symbol = position.get('symbol', 'SOLUSDT')
            pos_direction = position.get('direction', 'long')
            
            # Get correlation between symbols
            correlation = self.get_correlation(symbol, pos_symbol)
            
            if direction == pos_direction:
                # Same direction - full correlation impact
                total_correlation += abs(correlation)
            else:
                # Opposite direction - hedging benefit
                total_correlation -= correlation * 0.5
        
        # Reduce size based on total correlation
        correlation_factor = 1.0 / (1.0 + total_correlation)
        
        return max(0.3, min(1.0, correlation_factor))
    
    def adjust_for_volatility(self, market_conditions: Dict) -> float:
        """
        Adjust size based on current vs normal volatility
        
        Args:
            market_conditions: Market state including volatility metrics
            
        Returns:
            Adjustment factor (0.5 to 1.5)
        """
        atr_current = market_conditions.get('atr', 0)
        atr_average = market_conditions.get('atr_20d_avg', atr_current)
        
        if atr_average <= 0:
            return 1.0
        
        volatility_ratio = atr_current / atr_average
        
        # Inverse relationship - higher volatility = smaller position
        # Formula: 1 / (0.5 + 0.5 * vol_ratio)
        adjustment = 1.0 / (0.5 + 0.5 * volatility_ratio)
        
        return max(0.5, min(1.5, adjustment))
    
    def adjust_for_drawdown(self) -> float:
        """
        Progressive position reduction based on current drawdown
        
        Returns:
            Adjustment factor (0.0 to 1.0)
        """
        current_dd = self.calculate_current_drawdown()
        
        # Progressive reduction levels from config
        dd_levels = self.config.get('dd_levels', {
            0.05: {'position_mult': 0.5},   # 5% DD
            0.08: {'position_mult': 0.3},   # 8% DD  
            0.10: {'position_mult': 0.0}    # 10% DD - stop trading
        })
        
        # Check each drawdown level
        for dd_threshold in sorted(dd_levels.keys(), reverse=True):
            if current_dd >= dd_threshold:
                return dd_levels[dd_threshold]['position_mult']
        
        return 1.0
    
    def adjust_for_liquidity(self, market_conditions: Dict) -> float:
        """
        Ensure position size doesn't exceed market liquidity
        
        Args:
            market_conditions: Market liquidity metrics
            
        Returns:
            Adjustment factor (0.1 to 1.0)
        """
        volume_5m_usd = market_conditions.get('volume_5m_usd', 0)
        orderbook_depth_usd = market_conditions.get('orderbook_depth_usd', 0)
        
        if volume_5m_usd <= 0 or orderbook_depth_usd <= 0:
            return 1.0
        
        # Position shouldn't exceed 1% of 5m volume
        volume_constraint = 0.01
        max_position_value_volume = volume_5m_usd * volume_constraint
        
        # Position shouldn't exceed 10% of visible orderbook
        depth_constraint = 0.10
        max_position_value_depth = orderbook_depth_usd * depth_constraint
        
        # Calculate what fraction of max position we can take
        max_risk_amount = self.equity * self.BASE_RISK_PCT
        
        volume_factor = min(1.0, max_position_value_volume / max_risk_amount)
        depth_factor = min(1.0, max_position_value_depth / max_risk_amount)
        
        return max(0.1, min(volume_factor, depth_factor))
    
    def get_regime_multiplier(self, regime: str) -> float:
        """
        Adjust position size based on market regime
        
        Args:
            regime: Current market regime
            
        Returns:
            Regime multiplier (0.4 to 1.2)
        """
        multipliers = {
            'strong_trend': 1.2,
            'trending': 1.0,
            'normal_range': 0.8,
            'low_volatility_range': 0.6,
            'volatile_choppy': 0.4
        }
        
        return multipliers.get(regime, 0.8)  # Default to conservative
    
    def adjust_for_setup_confidence(self, setup: Dict) -> float:
        """
        Adjust size based on setup confidence and validation score
        
        Args:
            setup: Trading setup with confidence metrics
            
        Returns:
            Confidence adjustment factor (0.5 to 1.2)
        """
        confidence = setup.get('confidence', 0.6)
        validation_score = setup.get('validation_score', confidence)
        
        # Higher confidence = larger position (up to 20% boost)
        if confidence > 0.8:
            return min(1.2, 1.0 + (confidence - 0.8) * 1.0)
        elif confidence > 0.6:
            return 1.0
        else:
            # Lower confidence = smaller position
            return max(0.5, confidence / 0.6)
    
    def check_max_size_vs_liquidity(
        self, 
        position_size: float, 
        entry_price: float, 
        market_conditions: Dict
    ) -> float:
        """
        Final check against available liquidity
        
        Args:
            position_size: Calculated position size
            entry_price: Entry price
            market_conditions: Market liquidity data
            
        Returns:
            Maximum allowable position size
        """
        position_value = position_size * entry_price
        
        # Check against 5m volume
        volume_5m_usd = market_conditions.get('volume_5m_usd', float('inf'))
        max_vs_volume = volume_5m_usd * 0.01  # 1% of volume
        
        # Check against orderbook depth
        depth_usd = market_conditions.get('orderbook_depth_usd', float('inf'))
        max_vs_depth = depth_usd * 0.10  # 10% of depth
        
        max_position_value = min(max_vs_volume, max_vs_depth)
        
        if position_value > max_position_value:
            return max_position_value / entry_price
        
        return position_size
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Correlation coefficient (-1.0 to 1.0)
        """
        if symbol1 == symbol2:
            return 1.0
        
        # For SOL/USDT, assume high correlation with other crypto
        # In production, this would be calculated from price data
        if 'USDT' in symbol1 and 'USDT' in symbol2:
            return 0.7  # High crypto correlation
        elif 'BTC' in symbol1 or 'BTC' in symbol2:
            return 0.85  # Very high BTC correlation
        else:
            return 0.3  # Default moderate correlation
    
    def calculate_current_drawdown(self) -> float:
        """
        Calculate current drawdown from peak equity
        
        Returns:
            Current drawdown as decimal (0.0 to 1.0)
        """
        # This would be calculated from account history
        # For now, return a conservative estimate
        peak_equity = self.config.get('peak_equity', self.equity)
        
        if peak_equity <= 0:
            return 0.0
        
        return max(0.0, (peak_equity - self.equity) / peak_equity)
    
    def get_max_position_count(self) -> int:
        """
        Get maximum number of positions based on Kelly and risk limits
        
        Returns:
            Maximum position count
        """
        # Conservative approach - limit based on correlation risk
        base_max = self.config.get('max_positions', 5)
        kelly_based_max = int(1.0 / max(0.01, self.calculate_kelly()))
        
        return min(base_max, kelly_based_max)
    
    def validate_position_size(self, size_result: PositionSizeResult, setup: Dict) -> bool:
        """
        Final validation of calculated position size
        
        Args:
            size_result: Calculated position size result
            setup: Trading setup
            
        Returns:
            True if position size is valid
        """
        # Minimum size check
        min_position_value = self.config.get('min_position_value', 10.0)
        position_value = size_result.size * setup['entry']
        
        if position_value < min_position_value:
            return False
        
        # Maximum risk check
        if size_result.risk_pct > self.BASE_RISK_PCT * 1.1:  # 10% tolerance
            return False
        
        # Sanity check - position size should be reasonable
        if size_result.size <= 0 or size_result.size > self.equity:
            return False
        
        return True