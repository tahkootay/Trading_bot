"""Base classes and utilities for technical indicators."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd


class IndicatorBase(ABC):
    """Base class for all technical indicators."""
    
    def __init__(self, period: int = 14):
        """Initialize indicator with period."""
        self.period = period
        self.values: List[float] = []
        self._initialized = False
    
    @abstractmethod
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate indicator value for given data."""
        pass
    
    def update(self, value: float) -> Optional[float]:
        """Update indicator with new value and return current indicator value."""
        self.values.append(value)
        
        # Keep only necessary history
        if len(self.values) > self.period * 2:
            self.values = self.values[-self.period * 2:]
        
        if len(self.values) >= self.period:
            self._initialized = True
            return self.calculate(self.values)
        
        return None
    
    def is_ready(self) -> bool:
        """Check if indicator has enough data to produce valid values."""
        return self._initialized and len(self.values) >= self.period
    
    def reset(self):
        """Reset indicator state."""
        self.values = []
        self._initialized = False


class PriceBasedIndicator(IndicatorBase):
    """Base class for price-based indicators (OHLC)."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.ohlc_data: List[Dict[str, float]] = []
    
    def update_ohlc(self, open_price: float, high: float, low: float, close: float, volume: float = 0) -> Optional[float]:
        """Update with OHLC data."""
        ohlc = {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }
        
        self.ohlc_data.append(ohlc)
        
        # Keep only necessary history
        if len(self.ohlc_data) > self.period * 2:
            self.ohlc_data = self.ohlc_data[-self.period * 2:]
        
        if len(self.ohlc_data) >= self.period:
            self._initialized = True
            return self.calculate_ohlc(self.ohlc_data)
        
        return None
    
    @abstractmethod
    def calculate_ohlc(self, ohlc_data: List[Dict[str, float]]) -> float:
        """Calculate indicator value using OHLC data."""
        pass


def validate_data(data: Union[List[float], np.ndarray, pd.Series], min_length: int = 1) -> np.ndarray:
    """Validate and convert input data to numpy array."""
    if isinstance(data, pd.Series):
        data = data.values
    elif isinstance(data, list):
        data = np.array(data)
    
    if len(data) < min_length:
        raise ValueError(f"Insufficient data: need at least {min_length} points, got {len(data)}")
    
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError("Data contains NaN or infinite values")
    
    return data


def true_range(high: float, low: float, prev_close: float) -> float:
    """Calculate True Range for a single bar."""
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )


def typical_price(high: float, low: float, close: float) -> float:
    """Calculate Typical Price (HLC/3)."""
    return (high + low + close) / 3.0


def weighted_close(high: float, low: float, close: float) -> float:
    """Calculate Weighted Close (HLCC/4)."""
    return (high + low + close + close) / 4.0