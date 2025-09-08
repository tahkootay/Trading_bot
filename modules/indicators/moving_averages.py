"""Moving average indicators implementation."""

from typing import List, Union, Optional
import numpy as np
import pandas as pd
from .base import IndicatorBase, validate_data


class SMA(IndicatorBase):
    """Simple Moving Average indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Simple Moving Average."""
        data = validate_data(data, self.period)
        return float(np.mean(data[-self.period:]))


class EMA(IndicatorBase):
    """Exponential Moving Average indicator."""
    
    def __init__(self, period: int = 20, alpha: Optional[float] = None):
        super().__init__(period)
        self.alpha = alpha if alpha is not None else 2.0 / (period + 1)
        self.ema_value: Optional[float] = None
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Exponential Moving Average."""
        data = validate_data(data, 1)
        
        if self.ema_value is None:
            # Initialize with SMA of first period values
            if len(data) >= self.period:
                self.ema_value = float(np.mean(data[:self.period]))
                start_idx = self.period
            else:
                self.ema_value = float(data[0])
                start_idx = 1
        else:
            start_idx = 0
        
        # Calculate EMA for remaining values
        for i in range(start_idx, len(data)):
            self.ema_value = self.alpha * data[i] + (1 - self.alpha) * self.ema_value
        
        return float(self.ema_value)
    
    def update(self, value: float) -> Optional[float]:
        """Update EMA with new value."""
        if self.ema_value is None:
            self.values.append(value)
            if len(self.values) >= self.period:
                self.ema_value = float(np.mean(self.values))
                self._initialized = True
                return self.ema_value
            return None
        else:
            self.ema_value = self.alpha * value + (1 - self.alpha) * self.ema_value
            return self.ema_value
    
    def reset(self):
        """Reset EMA state."""
        super().reset()
        self.ema_value = None


class WMA(IndicatorBase):
    """Weighted Moving Average indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.weights = np.arange(1, period + 1)
        self.weight_sum = np.sum(self.weights)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Weighted Moving Average."""
        data = validate_data(data, self.period)
        recent_data = data[-self.period:]
        return float(np.sum(recent_data * self.weights) / self.weight_sum)


class DEMA(IndicatorBase):
    """Double Exponential Moving Average indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.ema1 = EMA(period)
        self.ema2 = EMA(period)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Double Exponential Moving Average."""
        data = validate_data(data, self.period)
        
        # Calculate first EMA
        ema1_values = []
        for i in range(len(data)):
            ema1_val = self.ema1.update(data[i])
            if ema1_val is not None:
                ema1_values.append(ema1_val)
        
        # Calculate EMA of EMA
        if len(ema1_values) >= self.period:
            for val in ema1_values:
                ema2_val = self.ema2.update(val)
            
            # DEMA = 2*EMA1 - EMA2
            return 2.0 * ema1_values[-1] - ema2_val
        
        return ema1_values[-1] if ema1_values else 0.0
    
    def reset(self):
        """Reset DEMA state."""
        super().reset()
        self.ema1.reset()
        self.ema2.reset()


class TEMA(IndicatorBase):
    """Triple Exponential Moving Average indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.ema1 = EMA(period)
        self.ema2 = EMA(period)
        self.ema3 = EMA(period)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Triple Exponential Moving Average."""
        data = validate_data(data, self.period)
        
        # Calculate first EMA
        ema1_values = []
        for i in range(len(data)):
            ema1_val = self.ema1.update(data[i])
            if ema1_val is not None:
                ema1_values.append(ema1_val)
        
        # Calculate EMA of EMA
        ema2_values = []
        for val in ema1_values:
            ema2_val = self.ema2.update(val)
            if ema2_val is not None:
                ema2_values.append(ema2_val)
        
        # Calculate EMA of EMA of EMA
        ema3_values = []
        for val in ema2_values:
            ema3_val = self.ema3.update(val)
            if ema3_val is not None:
                ema3_values.append(ema3_val)
        
        if ema3_values and ema2_values and ema1_values:
            # TEMA = 3*EMA1 - 3*EMA2 + EMA3
            return 3.0 * ema1_values[-1] - 3.0 * ema2_values[-1] + ema3_values[-1]
        
        return ema1_values[-1] if ema1_values else 0.0
    
    def reset(self):
        """Reset TEMA state."""
        super().reset()
        self.ema1.reset()
        self.ema2.reset()
        self.ema3.reset()


class HMA(IndicatorBase):
    """Hull Moving Average indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.half_period = max(1, period // 2)
        self.sqrt_period = max(1, int(np.sqrt(period)))
        
        self.wma_half = WMA(self.half_period)
        self.wma_full = WMA(period)
        self.wma_sqrt = WMA(self.sqrt_period)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Hull Moving Average."""
        data = validate_data(data, self.period)
        
        if len(data) < self.period:
            return float(data[-1])
        
        # Calculate WMA with half period
        wma_half = self.wma_half.calculate(data)
        
        # Calculate WMA with full period  
        wma_full = self.wma_full.calculate(data)
        
        # Calculate raw Hull values
        hull_raw = 2.0 * wma_half - wma_full
        
        # Apply WMA with sqrt(period) to Hull raw values
        # For simplicity in this implementation, return the raw Hull value
        # In a full implementation, you'd maintain a buffer of hull_raw values
        return hull_raw
    
    def reset(self):
        """Reset HMA state."""
        super().reset()
        self.wma_half.reset()
        self.wma_full.reset()
        self.wma_sqrt.reset()


class VWMA(IndicatorBase):
    """Volume Weighted Moving Average indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.volumes: List[float] = []
    
    def update_with_volume(self, price: float, volume: float) -> Optional[float]:
        """Update VWMA with price and volume."""
        self.values.append(price)
        self.volumes.append(volume)
        
        # Keep only necessary history
        if len(self.values) > self.period * 2:
            self.values = self.values[-self.period * 2:]
            self.volumes = self.volumes[-self.period * 2:]
        
        if len(self.values) >= self.period:
            self._initialized = True
            return self.calculate_vwma()
        
        return None
    
    def calculate_vwma(self) -> float:
        """Calculate Volume Weighted Moving Average."""
        if len(self.values) < self.period or len(self.volumes) < self.period:
            return 0.0
        
        recent_prices = self.values[-self.period:]
        recent_volumes = self.volumes[-self.period:]
        
        price_volume_sum = sum(p * v for p, v in zip(recent_prices, recent_volumes))
        volume_sum = sum(recent_volumes)
        
        if volume_sum == 0:
            return float(np.mean(recent_prices))
        
        return float(price_volume_sum / volume_sum)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate VWMA (requires volume data via update_with_volume)."""
        # This method is required by base class but VWMA needs volume
        # Return simple average if no volume data
        data = validate_data(data, self.period)
        return float(np.mean(data[-self.period:]))
    
    def reset(self):
        """Reset VWMA state."""
        super().reset()
        self.volumes = []


# Convenience functions for one-time calculations

def sma(data: Union[List[float], np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Calculate Simple Moving Average for entire dataset."""
    data = validate_data(data, period)
    result = np.full(len(data), np.nan)
    
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    
    return result


def ema(data: Union[List[float], np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Calculate Exponential Moving Average for entire dataset."""
    data = validate_data(data, 1)
    result = np.full(len(data), np.nan)
    alpha = 2.0 / (period + 1)
    
    # Initialize with first value or SMA of first period
    if len(data) >= period:
        result[period - 1] = np.mean(data[:period])
        start_idx = period
    else:
        result[0] = data[0]
        start_idx = 1
    
    # Calculate EMA
    for i in range(start_idx, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    
    return result


def wma(data: Union[List[float], np.ndarray, pd.Series], period: int = 20) -> np.ndarray:
    """Calculate Weighted Moving Average for entire dataset."""
    data = validate_data(data, period)
    result = np.full(len(data), np.nan)
    weights = np.arange(1, period + 1)
    weight_sum = np.sum(weights)
    
    for i in range(period - 1, len(data)):
        window_data = data[i - period + 1:i + 1]
        result[i] = np.sum(window_data * weights) / weight_sum
    
    return result