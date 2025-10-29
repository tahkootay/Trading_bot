"""
KDJ Indicator - exact TradingView/Bybit implementation
Based on @iamaltcoin's TradingView script
"""

from typing import List, Dict
import numpy as np
from .base import PriceBasedIndicator


class KDJTradingView(PriceBasedIndicator):
    """
    KDJ Indicator - exact TradingView/Bybit implementation
    
    This is a precise replica of the KDJ indicator used on Bybit and TradingView.
    Uses the bcwsma (Bitcoin Wisdom SMA) function for smoothing.
    
    Parameters:
    - ilong (period): Lookback period for highest/lowest (default: 9)
    - isig (signal): Signal period for smoothing (default: 3)
    
    Formula:
    1. RSV = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    2. pK = bcwsma(RSV, isig, 1)
    3. pD = bcwsma(pK, isig, 1)  
    4. pJ = 3 * pK - 2 * pD
    
    bcwsma formula: (m*s + (l-m)*prev_bcwsma) / l
    where m=1, s=current_value, l=period, prev_bcwsma=previous_value
    """
    
    def __init__(self, ilong: int = 9, isig: int = 3):
        super().__init__(ilong)
        self.ilong = ilong      # period for highest/lowest
        self.isig = isig        # signal period for smoothing
        
        # State variables for bcwsma calculations
        self.prev_pk = None     # Previous pK value
        self.prev_pd = None     # Previous pD value
        
        self.current_pk = 50.0
        self.current_pd = 50.0
        self.current_pj = 50.0
        
        self.is_initialized = False
    
    def bcwsma(self, current_value: float, period: int, m: int, prev_value: float = None) -> float:
        """
        Bitcoin Wisdom SMA (bcwsma) function
        Formula: (m*s + (l-m)*prev_bcwsma) / l
        
        Args:
            current_value (s): Current input value
            period (l): Period length
            m: Weight factor (typically 1)
            prev_value: Previous bcwsma value
        
        Returns:
            Smoothed value
        """
        if prev_value is None:
            return current_value
        
        return (m * current_value + (period - m) * prev_value) / period
    
    def calculate_ohlc(self, ohlc_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate KDJ values from OHLC data using exact TradingView algorithm."""
        if len(ohlc_data) < self.ilong:
            return {'k': 50.0, 'd': 50.0, 'j': 50.0}
        
        # Get recent data for the lookback period
        recent_data = ohlc_data[-self.ilong:]
        
        # Calculate highest high and lowest low over the period
        highs = [bar['high'] for bar in recent_data]
        lows = [bar['low'] for bar in recent_data]
        current_close = recent_data[-1]['close']
        
        highest_high = max(highs)
        lowest_low = min(lows)
        
        # Calculate RSV (Raw Stochastic Value)
        if highest_high == lowest_low:
            rsv = 50.0  # Avoid division by zero
        else:
            rsv = 100.0 * ((current_close - lowest_low) / (highest_high - lowest_low))
        
        # Calculate pK using bcwsma
        pk = self.bcwsma(rsv, self.isig, 1, self.prev_pk)
        
        # Calculate pD using bcwsma of pK
        pd = self.bcwsma(pk, self.isig, 1, self.prev_pd)
        
        # Calculate pJ
        pj = 3 * pk - 2 * pd
        
        # Update previous values for next calculation
        self.prev_pk = pk
        self.prev_pd = pd
        
        # Update current values
        self.current_pk = pk
        self.current_pd = pd
        self.current_pj = pj
        
        self.is_initialized = True
        
        return {
            'k': float(pk),
            'd': float(pd),
            'j': float(pj)
        }
    
    def reset(self):
        """Reset indicator state."""
        super().reset()
        self.prev_pk = None
        self.prev_pd = None
        self.current_pk = 50.0
        self.current_pd = 50.0
        self.current_pj = 50.0
        self.is_initialized = False
    
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return self.is_initialized
    
    def calculate(self, data):
        """Calculate KDJ from price data (required by base class)."""
        # This method is required by the base class but not used in our implementation
        # We use calculate_ohlc instead
        return 50.0
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current KDJ values."""
        return {
            'k': self.current_pk,
            'd': self.current_pd,
            'j': self.current_pj
        }