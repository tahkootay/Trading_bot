"""Volume-based indicators implementation."""

from typing import List, Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from .base import IndicatorBase, PriceBasedIndicator, validate_data
from .moving_averages import SMA, EMA


class VolumeIndicatorBase(IndicatorBase):
    """Base class for volume-based indicators."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.volumes: List[float] = []
    
    def update_price_volume(self, price: float, volume: float) -> Optional[float]:
        """Update indicator with price and volume data."""
        self.values.append(price)
        self.volumes.append(volume)
        
        # Keep only necessary history
        if len(self.values) > self.period * 2:
            self.values = self.values[-self.period * 2:]
            self.volumes = self.volumes[-self.period * 2:]
        
        if len(self.values) >= self.period:
            self._initialized = True
            return self.calculate_volume_indicator()
        
        return None
    
    def calculate_volume_indicator(self) -> float:
        """Calculate volume-based indicator value."""
        return 0.0
    
    def reset(self):
        """Reset volume indicator state."""
        super().reset()
        self.volumes = []


class OBV(IndicatorBase):
    """On-Balance Volume indicator."""
    
    def __init__(self):
        super().__init__(1)  # OBV doesn't use a traditional period
        self.obv_value: float = 0.0
        self.prev_close: Optional[float] = None
    
    def update_price_volume(self, price: float, volume: float) -> float:
        """Update OBV with price and volume."""
        if self.prev_close is not None:
            if price > self.prev_close:
                self.obv_value += volume
            elif price < self.prev_close:
                self.obv_value -= volume
            # If price equals prev_close, OBV remains unchanged
        
        self.prev_close = price
        self._initialized = True
        return self.obv_value
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate OBV (requires volume data via update_price_volume)."""
        return self.obv_value
    
    def reset(self):
        """Reset OBV state."""
        super().reset()
        self.obv_value = 0.0
        self.prev_close = None


class VWAP(VolumeIndicatorBase):
    """Volume Weighted Average Price indicator."""
    
    def __init__(self, period: Optional[int] = None):
        # VWAP can be calculated for entire session or rolling period
        super().__init__(period if period else 1)
        self.cumulative_pv: float = 0.0  # Cumulative price * volume
        self.cumulative_volume: float = 0.0  # Cumulative volume
        self.use_rolling = period is not None
    
    def calculate_volume_indicator(self) -> float:
        """Calculate VWAP value."""
        if self.use_rolling:
            # Rolling VWAP
            recent_prices = self.values[-self.period:]
            recent_volumes = self.volumes[-self.period:]
            
            pv_sum = sum(p * v for p, v in zip(recent_prices, recent_volumes))
            volume_sum = sum(recent_volumes)
            
            if volume_sum == 0:
                return float(np.mean(recent_prices))
            
            return pv_sum / volume_sum
        else:
            # Cumulative VWAP
            if self.cumulative_volume == 0:
                return self.values[-1] if self.values else 0.0
            
            return self.cumulative_pv / self.cumulative_volume
    
    def update_price_volume(self, price: float, volume: float) -> Optional[float]:
        """Update VWAP with price and volume."""
        if not self.use_rolling:
            # Update cumulative values
            self.cumulative_pv += price * volume
            self.cumulative_volume += volume
            self._initialized = True
            return self.calculate_volume_indicator()
        else:
            # Use rolling calculation
            return super().update_price_volume(price, volume)
    
    def reset(self):
        """Reset VWAP state."""
        super().reset()
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0


class PVI(IndicatorBase):
    """Positive Volume Index indicator."""
    
    def __init__(self, base_value: float = 1000.0):
        super().__init__(1)
        self.pvi_value: float = base_value
        self.prev_close: Optional[float] = None
        self.prev_volume: Optional[float] = None
    
    def update_price_volume(self, price: float, volume: float) -> Optional[float]:
        """Update PVI with price and volume."""
        if self.prev_close is not None and self.prev_volume is not None:
            if volume > self.prev_volume:
                # Update PVI only when volume increases
                price_change = (price - self.prev_close) / self.prev_close
                self.pvi_value = self.pvi_value * (1 + price_change)
        
        self.prev_close = price
        self.prev_volume = volume
        self._initialized = True
        return self.pvi_value
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate PVI (requires volume data via update_price_volume)."""
        return self.pvi_value
    
    def reset(self):
        """Reset PVI state."""
        super().reset()
        self.pvi_value = 1000.0
        self.prev_close = None
        self.prev_volume = None


class NVI(IndicatorBase):
    """Negative Volume Index indicator."""
    
    def __init__(self, base_value: float = 1000.0):
        super().__init__(1)
        self.nvi_value: float = base_value
        self.prev_close: Optional[float] = None
        self.prev_volume: Optional[float] = None
    
    def update_price_volume(self, price: float, volume: float) -> Optional[float]:
        """Update NVI with price and volume."""
        if self.prev_close is not None and self.prev_volume is not None:
            if volume < self.prev_volume:
                # Update NVI only when volume decreases
                price_change = (price - self.prev_close) / self.prev_close
                self.nvi_value = self.nvi_value * (1 + price_change)
        
        self.prev_close = price
        self.prev_volume = volume
        self._initialized = True
        return self.nvi_value
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate NVI (requires volume data via update_price_volume)."""
        return self.nvi_value
    
    def reset(self):
        """Reset NVI state."""
        super().reset()
        self.nvi_value = 1000.0
        self.prev_close = None
        self.prev_volume = None


class AccumulationDistributionLine(VolumeIndicatorBase):
    """Accumulation/Distribution Line indicator."""
    
    def __init__(self):
        super().__init__(1)
        self.ad_value: float = 0.0
    
    def update_hlcv(self, high: float, low: float, close: float, volume: float) -> float:
        """Update A/D Line with HLCV data."""
        if high == low:
            money_flow_multiplier = 0.0
        else:
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        
        money_flow_volume = money_flow_multiplier * volume
        self.ad_value += money_flow_volume
        
        self._initialized = True
        return self.ad_value
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate A/D Line (requires HLCV data via update_hlcv)."""
        return self.ad_value
    
    def reset(self):
        """Reset A/D Line state."""
        super().reset()
        self.ad_value = 0.0


class ChaikinMoneyFlow(VolumeIndicatorBase):
    """Chaikin Money Flow indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.money_flow_volumes: List[float] = []
    
    def update_hlcv(self, high: float, low: float, close: float, volume: float) -> Optional[float]:
        """Update CMF with HLCV data."""
        if high == low:
            money_flow_multiplier = 0.0
        else:
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        
        money_flow_volume = money_flow_multiplier * volume
        
        self.money_flow_volumes.append(money_flow_volume)
        self.volumes.append(volume)
        
        # Keep only necessary history
        if len(self.money_flow_volumes) > self.period * 2:
            self.money_flow_volumes = self.money_flow_volumes[-self.period * 2:]
            self.volumes = self.volumes[-self.period * 2:]
        
        if len(self.money_flow_volumes) >= self.period:
            self._initialized = True
            return self.calculate_cmf()
        
        return None
    
    def calculate_cmf(self) -> float:
        """Calculate Chaikin Money Flow."""
        recent_mfv = self.money_flow_volumes[-self.period:]
        recent_volumes = self.volumes[-self.period:]
        
        mfv_sum = sum(recent_mfv)
        volume_sum = sum(recent_volumes)
        
        if volume_sum == 0:
            return 0.0
        
        return mfv_sum / volume_sum
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate CMF (requires HLCV data via update_hlcv)."""
        return 0.0
    
    def reset(self):
        """Reset CMF state."""
        super().reset()
        self.money_flow_volumes = []


class VolumeOscillator(VolumeIndicatorBase):
    """Volume Oscillator indicator."""
    
    def __init__(self, short_period: int = 5, long_period: int = 10):
        super().__init__(max(short_period, long_period))
        self.short_period = short_period
        self.long_period = long_period
        
        self.short_ma = SMA(short_period)
        self.long_ma = SMA(long_period)
    
    def calculate_volume_indicator(self) -> float:
        """Calculate Volume Oscillator."""
        if len(self.volumes) < self.long_period:
            return 0.0
        
        short_avg = self.short_ma.calculate(self.volumes)
        long_avg = self.long_ma.calculate(self.volumes)
        
        if long_avg == 0:
            return 0.0
        
        volume_osc = ((short_avg - long_avg) / long_avg) * 100.0
        return volume_osc
    
    def update_volume_only(self, volume: float) -> Optional[float]:
        """Update Volume Oscillator with volume only."""
        self.volumes.append(volume)
        
        # Keep only necessary history
        if len(self.volumes) > self.long_period * 2:
            self.volumes = self.volumes[-self.long_period * 2:]
        
        if len(self.volumes) >= self.long_period:
            self._initialized = True
            return self.calculate_volume_indicator()
        
        return None


class PriceVolumeRank(VolumeIndicatorBase):
    """Price Volume Rank indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
    
    def calculate_volume_indicator(self) -> float:
        """Calculate Price Volume Rank."""
        if len(self.volumes) < self.period:
            return 50.0
        
        recent_volumes = self.volumes[-self.period:]
        current_volume = recent_volumes[-1]
        
        # Count how many volumes in the period are less than current
        lower_count = sum(1 for vol in recent_volumes[:-1] if vol < current_volume)
        
        # Calculate rank as percentage
        rank = (lower_count / (len(recent_volumes) - 1)) * 100.0
        return rank


class VolumeWeightedMomentum(VolumeIndicatorBase):
    """Volume Weighted Momentum indicator."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
    
    def calculate_volume_indicator(self) -> float:
        """Calculate Volume Weighted Momentum."""
        if len(self.values) < self.period + 1 or len(self.volumes) < self.period + 1:
            return 0.0
        
        # Calculate price changes weighted by volume
        recent_prices = self.values[-self.period-1:]
        recent_volumes = self.volumes[-self.period:]
        
        price_changes = np.diff(recent_prices)
        weighted_changes = price_changes * recent_volumes
        
        # Calculate momentum as sum of weighted changes
        momentum = np.sum(weighted_changes)
        
        return float(momentum)


# Convenience functions for one-time calculations

def obv(prices: Union[List[float], np.ndarray], 
        volumes: Union[List[float], np.ndarray]) -> np.ndarray:
    """Calculate On-Balance Volume for entire dataset."""
    prices = validate_data(prices, 1)
    volumes = validate_data(volumes, 1)
    
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes arrays must have the same length")
    
    result = np.zeros(len(prices))
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            result[i] = result[i-1] + volumes[i]
        elif prices[i] < prices[i-1]:
            result[i] = result[i-1] - volumes[i]
        else:
            result[i] = result[i-1]
    
    return result


def vwap(prices: Union[List[float], np.ndarray], 
         volumes: Union[List[float], np.ndarray],
         period: Optional[int] = None) -> np.ndarray:
    """Calculate VWAP for entire dataset."""
    prices = validate_data(prices, 1)
    volumes = validate_data(volumes, 1)
    
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes arrays must have the same length")
    
    result = np.full(len(prices), np.nan)
    
    if period is None:
        # Cumulative VWAP
        cumulative_pv = 0.0
        cumulative_volume = 0.0
        
        for i in range(len(prices)):
            cumulative_pv += prices[i] * volumes[i]
            cumulative_volume += volumes[i]
            
            if cumulative_volume > 0:
                result[i] = cumulative_pv / cumulative_volume
            else:
                result[i] = prices[i]
    else:
        # Rolling VWAP
        for i in range(period - 1, len(prices)):
            window_prices = prices[i - period + 1:i + 1]
            window_volumes = volumes[i - period + 1:i + 1]
            
            pv_sum = np.sum(window_prices * window_volumes)
            volume_sum = np.sum(window_volumes)
            
            if volume_sum > 0:
                result[i] = pv_sum / volume_sum
            else:
                result[i] = np.mean(window_prices)
    
    return result


def accumulation_distribution_line(highs: Union[List[float], np.ndarray],
                                 lows: Union[List[float], np.ndarray],
                                 closes: Union[List[float], np.ndarray],
                                 volumes: Union[List[float], np.ndarray]) -> np.ndarray:
    """Calculate Accumulation/Distribution Line for entire dataset."""
    highs = validate_data(highs, 1)
    lows = validate_data(lows, 1)
    closes = validate_data(closes, 1)
    volumes = validate_data(volumes, 1)
    
    if not (len(highs) == len(lows) == len(closes) == len(volumes)):
        raise ValueError("All HLCV arrays must have the same length")
    
    result = np.zeros(len(closes))
    ad_value = 0.0
    
    for i in range(len(closes)):
        if highs[i] == lows[i]:
            money_flow_multiplier = 0.0
        else:
            money_flow_multiplier = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
        
        money_flow_volume = money_flow_multiplier * volumes[i]
        ad_value += money_flow_volume
        result[i] = ad_value
    
    return result