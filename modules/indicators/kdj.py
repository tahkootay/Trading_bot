"""
KDJ Indicator implementation with EMA smoothing to match major exchanges
"""

from typing import List, Dict
import numpy as np
from .base import PriceBasedIndicator


class KDJ(PriceBasedIndicator):
    """
    KDJ Indicator (Enhanced Stochastic)
    
    KDJ is a variant of the Stochastic Oscillator with an additional J line.
    This implementation uses EMA smoothing to better match exchange calculations.
    
    Formula:
    - Raw %K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
    - Smoothed %K = EMA(Raw %K, smooth_k_period) 
    - %D = EMA(Smoothed %K, d_period)
    - %J = 3 * %K - 2 * %D
    """
    
    def __init__(self, k_period: int = 9, d_period: int = 3, smooth_k_period: int = 3):
        super().__init__(k_period)
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k_period = smooth_k_period
        
        # EMA smoothing parameters
        self.k_alpha = 2.0 / (smooth_k_period + 1)
        self.d_alpha = 2.0 / (d_period + 1)
        
        # State variables
        self.raw_k_values: List[float] = []
        self.smoothed_k_values: List[float] = []
        self.d_values: List[float] = []
        
        self.current_k: float = 50.0
        self.current_d: float = 50.0
        self.is_initialized: bool = False
    
    def calculate_ohlc(self, ohlc_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate KDJ values from OHLC data."""
        if len(ohlc_data) < self.k_period:
            return {'k': 50.0, 'd': 50.0, 'j': 50.0}
        
        # Get recent data for K calculation
        recent_data = ohlc_data[-self.k_period:]
        
        # Find highest high and lowest low in the period
        highs = [bar['high'] for bar in recent_data]
        lows = [bar['low'] for bar in recent_data]
        current_close = recent_data[-1]['close']
        
        highest_high = max(highs)
        lowest_low = min(lows)
        
        # Calculate raw %K
        if highest_high == lowest_low:
            raw_k = 50.0
        else:
            raw_k = 100.0 * (current_close - lowest_low) / (highest_high - lowest_low)
        
        # Store raw K value
        self.raw_k_values.append(raw_k)
        if len(self.raw_k_values) > self.smooth_k_period * 3:
            self.raw_k_values = self.raw_k_values[-self.smooth_k_period * 3:]
        
        # Calculate smoothed %K using EMA
        if len(self.raw_k_values) == 1:
            # First value
            smoothed_k = raw_k
        else:
            # EMA smoothing
            if len(self.smoothed_k_values) == 0:
                # Initialize with SMA of first smooth_k_period values
                if len(self.raw_k_values) >= self.smooth_k_period:
                    smoothed_k = np.mean(self.raw_k_values[-self.smooth_k_period:])
                else:
                    smoothed_k = np.mean(self.raw_k_values)
            else:
                # Apply EMA formula
                prev_smoothed_k = self.smoothed_k_values[-1]
                smoothed_k = self.k_alpha * raw_k + (1 - self.k_alpha) * prev_smoothed_k
        
        # Store smoothed K value
        self.smoothed_k_values.append(smoothed_k)
        if len(self.smoothed_k_values) > self.d_period * 3:
            self.smoothed_k_values = self.smoothed_k_values[-self.d_period * 3:]
        
        # Calculate %D using EMA of smoothed %K
        if len(self.smoothed_k_values) == 1:
            # First value
            d_value = smoothed_k
        else:
            if len(self.d_values) == 0:
                # Initialize with SMA of first d_period values
                if len(self.smoothed_k_values) >= self.d_period:
                    d_value = np.mean(self.smoothed_k_values[-self.d_period:])
                else:
                    d_value = np.mean(self.smoothed_k_values)
            else:
                # Apply EMA formula
                prev_d = self.d_values[-1]
                d_value = self.d_alpha * smoothed_k + (1 - self.d_alpha) * prev_d
        
        # Store D value
        self.d_values.append(d_value)
        if len(self.d_values) > self.d_period * 3:
            self.d_values = self.d_values[-self.d_period * 3:]
        
        # Calculate %J
        j_value = 3 * smoothed_k - 2 * d_value
        
        # Update current values
        self.current_k = smoothed_k
        self.current_d = d_value
        self.is_initialized = True
        
        return {
            'k': float(smoothed_k),
            'd': float(d_value),
            'j': float(j_value)
        }
    
    def reset(self):
        """Reset indicator state."""
        super().reset()
        self.raw_k_values.clear()
        self.smoothed_k_values.clear()
        self.d_values.clear()
        self.current_k = 50.0
        self.current_d = 50.0
        self.is_initialized = False
    
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return self.is_initialized and len(self.raw_k_values) >= self.k_period
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current KDJ values."""
        if not self.is_ready():
            return {'k': 50.0, 'd': 50.0, 'j': 50.0}
        
        j_value = 3 * self.current_k - 2 * self.current_d
        return {
            'k': self.current_k,
            'd': self.current_d,
            'j': j_value
        }
    
    def calculate(self, data):
        """Calculate KDJ from price data (required by base class)."""
        # This method is required by the base class but not used in our implementation
        # We use calculate_ohlc instead
        return 50.0