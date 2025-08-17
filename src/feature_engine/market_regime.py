"""
Market Regime Classification Module

Classifies market conditions into distinct regimes for adaptive trading strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    """Market regime types"""
    LOW_VOLATILITY_RANGE = "low_volatility_range"
    NORMAL_RANGE = "normal_range"
    TRENDING = "trending"
    STRONG_TREND = "strong_trend"
    VOLATILE_CHOPPY = "volatile_choppy"

@dataclass
class RegimeSignals:
    """Container for regime classification signals"""
    adx: float
    atr_ratio: float
    volume_ratio: float
    ema_alignment: str
    whipsaw_count: int
    regime: MarketRegime
    confidence: float
    
class MarketRegimeClassifier:
    """
    Classifies market conditions into distinct regimes for adaptive strategy
    
    Regimes:
    - LOW_VOLATILITY_RANGE: Low ADX, low ATR, low volume
    - NORMAL_RANGE: Moderate values, no clear trend
    - TRENDING: Clear directional movement with aligned EMAs
    - STRONG_TREND: Very strong directional movement with high volume
    - VOLATILE_CHOPPY: High volatility with frequent direction changes
    """
    
    def __init__(self, config: Dict):
        self.config = config.get('risk', {}).get('market_regimes', {})
        self.ema_periods = config.get('trading', {}).get('indicator_params', {}).get('ema_ribbon', [8, 13, 21, 34, 55])
        
    def classify(self, market_data: Dict) -> RegimeSignals:
        """
        Classify current market regime based on multiple indicators
        Aligned with algorithm specification for regime classification.
        
        Args:
            market_data: Dictionary containing OHLCV data and indicators
            
        Returns:
            RegimeSignals object with regime classification and confidence
        """
        # Extract required indicators (aligned with algorithm)
        adx = market_data.get('adx', 0)
        atr_current = market_data.get('atr', 0)
        atr_avg_20d = market_data.get('atr_20d_avg', atr_current)
        volume = market_data.get('volume', 0)
        volume_avg_50 = market_data.get('volume_avg_50', volume)
        
        # Calculate ratios (as per algorithm)
        atr_ratio = atr_current / atr_avg_20d if atr_avg_20d > 0 else 1.0
        volume_ratio = volume / volume_avg_50 if volume_avg_50 > 0 else 1.0
        
        # Get EMA alignment strength
        ema_alignment = self._check_ema_alignment(market_data)
        
        # Count whipsaws (direction changes)
        price_data = market_data.get('close', [])
        if isinstance(price_data, pd.Series):
            price_data = price_data.tolist()
        whipsaw_count = self._count_whipsaws(price_data, window=20)
        
        # Check volume trending (for strong trend detection)
        volume_trending = self._check_volume_trending(market_data)
        
        # Classify regime using algorithm logic
        regime, confidence = self._classify_regime(
            adx, atr_ratio, volume_ratio, ema_alignment, whipsaw_count, volume_trending
        )
        
        return RegimeSignals(
            adx=adx,
            atr_ratio=atr_ratio,
            volume_ratio=volume_ratio,
            ema_alignment=ema_alignment,
            whipsaw_count=whipsaw_count,
            regime=regime,
            confidence=confidence
        )
    
    def _classify_regime(
        self, 
        adx: float, 
        atr_ratio: float, 
        volume_ratio: float, 
        ema_alignment: str, 
        whipsaw_count: int,
        volume_trending: bool
    ) -> Tuple[MarketRegime, float]:
        """
        Main regime classification logic aligned with algorithm specification
        
        Returns:
            Tuple of (regime, confidence_score)
        """
        
        # Strong Trend - Exact algorithm criteria
        strong_trend_config = self.config.get('strong_trend', {})
        if (adx >= strong_trend_config.get('adx_min', 40) and 
            volume_trending and 
            ema_alignment == 'strong'):
            return MarketRegime.STRONG_TREND, 0.9
        
        # Normal Trending - Algorithm criteria
        trending_config = self.config.get('trending', {})
        if (adx >= trending_config.get('adx_min', 25) and 
            adx <= trending_config.get('adx_max', 40) and
            ema_alignment in ['bullish', 'bearish']):
            confidence = 0.85 if volume_ratio > 1.2 else 0.75
            return MarketRegime.TRENDING, confidence
        
        # Volatile Choppy - Algorithm criteria
        choppy_config = self.config.get('volatile_choppy', {})
        if (adx < choppy_config.get('adx_max', 25) and 
            atr_ratio >= choppy_config.get('atr_ratio_min', 1.5) and 
            whipsaw_count >= choppy_config.get('whipsaw_count_min', 3)):
            return MarketRegime.VOLATILE_CHOPPY, 0.8
        
        # Low Volatility Range - Algorithm criteria
        low_vol_config = self.config.get('low_volatility_range', {})
        if (adx < low_vol_config.get('adx_max', 20) and 
            atr_ratio < low_vol_config.get('atr_ratio_max', 0.8) and 
            volume_ratio < low_vol_config.get('volume_ratio_max', 0.7)):
            return MarketRegime.LOW_VOLATILITY_RANGE, 0.75
        
        # Normal Range - Algorithm criteria (fallback)
        normal_config = self.config.get('normal_range', {})
        adx_range = normal_config.get('adx_range', [20, 25])
        atr_range = normal_config.get('atr_ratio_range', [0.8, 1.2])
        
        if (adx_range[0] <= adx <= adx_range[1] and
            atr_range[0] <= atr_ratio <= atr_range[1]):
            return MarketRegime.NORMAL_RANGE, 0.7
        
        # Default fallback
        return MarketRegime.NORMAL_RANGE, 0.6
    
    def _check_ema_alignment(self, market_data: Dict) -> str:
        """
        Check EMA ribbon alignment strength
        
        Returns:
            'strong', 'bullish', 'bearish', or 'neutral'
        """
        # Extract EMA values
        emas = []
        for period in self.ema_periods:
            ema_key = f'ema_{period}'
            if ema_key in market_data:
                emas.append(market_data[ema_key])
            else:
                # If EMAs not available, return neutral
                return 'neutral'
        
        if len(emas) < 5:
            return 'neutral'
        
        ema8, ema13, ema21, ema34, ema55 = emas[:5]
        
        # Check for bullish alignment (ascending order)
        if ema8 > ema13 > ema21 > ema34 > ema55:
            # Calculate spacing to determine strength
            spacings = [
                (ema8 - ema13) / ema13,
                (ema13 - ema21) / ema21,
                (ema21 - ema34) / ema34,
                (ema34 - ema55) / ema55
            ]
            min_spacing = min(spacings)
            return 'strong' if min_spacing > 0.002 else 'bullish'
        
        # Check for bearish alignment (descending order)
        elif ema8 < ema13 < ema21 < ema34 < ema55:
            # Calculate spacing to determine strength
            spacings = [
                (ema13 - ema8) / ema8,
                (ema21 - ema13) / ema13,
                (ema34 - ema21) / ema21,
                (ema55 - ema34) / ema34
            ]
            min_spacing = min(spacings)
            return 'strong' if min_spacing > 0.002 else 'bearish'
        
        return 'neutral'
    
    def _count_whipsaws(self, prices: List[float], window: int = 20) -> int:
        """
        Count price direction changes (whipsaws) in the given window
        
        Args:
            prices: List of price values
            window: Number of periods to analyze
            
        Returns:
            Number of direction changes
        """
        if len(prices) < 3:
            return 0
        
        # Limit to window size
        recent_prices = prices[-window:] if len(prices) > window else prices
        
        if len(recent_prices) < 3:
            return 0
        
        changes = 0
        for i in range(2, len(recent_prices)):
            # Check if direction changed
            prev_direction = recent_prices[i-1] > recent_prices[i-2]
            curr_direction = recent_prices[i] > recent_prices[i-1]
            
            if prev_direction != curr_direction:
                changes += 1
        
        return changes
    
    def _check_volume_trending(self, market_data: Dict) -> bool:
        """
        Check if volume is in a trending pattern (increasing over time)
        Used for strong trend detection as per algorithm.
        
        Args:
            market_data: Dictionary containing volume data
            
        Returns:
            True if volume is trending higher
        """
        volume_data = market_data.get('volume', [])
        if isinstance(volume_data, pd.Series):
            volume_data = volume_data.tolist()
        
        if len(volume_data) < 10:
            return False
        
        # Check if recent volume (last 5 periods) > older volume (5-10 periods back)
        recent_volume = volume_data[-5:] if len(volume_data) >= 5 else volume_data
        older_volume = volume_data[-10:-5] if len(volume_data) >= 10 else volume_data[:-5]
        
        if not recent_volume or not older_volume:
            return False
        
        recent_avg = sum(recent_volume) / len(recent_volume)
        older_avg = sum(older_volume) / len(older_volume)
        
        # Volume is trending if recent average is significantly higher
        return recent_avg > older_avg * 1.2  # 20% increase required
    
    def get_regime_multipliers(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get position sizing and filter multipliers for the given regime
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary of multipliers for various parameters
        """
        multipliers = {
            MarketRegime.STRONG_TREND: {
                'position_size': 1.2,
                'confidence_mult': 0.9,
                'margin_mult': 0.8,
                'agreement_mult': 0.9
            },
            MarketRegime.TRENDING: {
                'position_size': 1.0,
                'confidence_mult': 0.95,
                'margin_mult': 0.9,
                'agreement_mult': 0.95
            },
            MarketRegime.NORMAL_RANGE: {
                'position_size': 0.8,
                'confidence_mult': 1.0,
                'margin_mult': 1.0,
                'agreement_mult': 1.0
            },
            MarketRegime.LOW_VOLATILITY_RANGE: {
                'position_size': 0.6,
                'confidence_mult': 1.1,
                'margin_mult': 1.2,
                'agreement_mult': 1.1
            },
            MarketRegime.VOLATILE_CHOPPY: {
                'position_size': 0.4,
                'confidence_mult': 1.2,
                'margin_mult': 1.3,
                'agreement_mult': 1.2
            }
        }
        
        return multipliers.get(regime, multipliers[MarketRegime.NORMAL_RANGE])
    
    def get_regime_description(self, regime: MarketRegime) -> str:
        """Get human-readable description of the regime"""
        descriptions = {
            MarketRegime.STRONG_TREND: "Strong directional movement with high conviction",
            MarketRegime.TRENDING: "Clear directional bias with moderate momentum",
            MarketRegime.NORMAL_RANGE: "Balanced conditions without clear bias",
            MarketRegime.LOW_VOLATILITY_RANGE: "Low volatility consolidation phase",
            MarketRegime.VOLATILE_CHOPPY: "High volatility with frequent direction changes"
        }
        
        return descriptions.get(regime, "Unknown regime")