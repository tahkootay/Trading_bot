"""Volatility indicators implementation (ATR, Bollinger Bands, etc.)."""

from typing import List, Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from .base import IndicatorBase, PriceBasedIndicator, validate_data, true_range
from .moving_averages import SMA, EMA


class ATR(PriceBasedIndicator):
    """Average True Range indicator."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.true_ranges: List[float] = []
        self.atr_ema = EMA(period)
    
    def calculate_ohlc(self, ohlc_data: List[Dict[str, float]]) -> float:
        """Calculate ATR using OHLC data."""
        if len(ohlc_data) < 2:
            return 0.0
        
        # Calculate true range for the latest bar
        current = ohlc_data[-1]
        previous = ohlc_data[-2]
        
        tr = true_range(current['high'], current['low'], previous['close'])
        self.true_ranges.append(tr)
        
        # Keep only necessary history
        if len(self.true_ranges) > self.period * 2:
            self.true_ranges = self.true_ranges[-self.period * 2:]
        
        # Calculate ATR (Wilder's smoothing)
        if len(self.true_ranges) >= self.period:
            # Use EMA for ATR calculation
            atr_val = self.atr_ema.update(tr)
            return float(atr_val) if atr_val is not None else 0.0
        
        return 0.0
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate ATR (requires OHLC data)."""
        return 0.0
    
    def reset(self):
        """Reset ATR state."""
        super().reset()
        self.true_ranges = []
        self.atr_ema.reset()


class BollingerBands(IndicatorBase):
    """Bollinger Bands indicator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(period)
        self.std_dev = std_dev
        self.sma = SMA(period)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate Bollinger Bands (Upper, Middle, Lower)."""
        data = validate_data(data, self.period)
        
        if len(data) < self.period:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}
        
        recent_data = data[-self.period:]
        
        # Middle band (SMA)
        middle = float(np.mean(recent_data))
        
        # Standard deviation
        std = float(np.std(recent_data, ddof=0))
        
        # Upper and lower bands
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)
        
        return {
            'upper': float(upper),
            'middle': float(middle),
            'lower': float(lower)
        }
    
    def update(self, value: float) -> Optional[Dict[str, float]]:
        """Update Bollinger Bands with new value."""
        self.values.append(value)
        
        # Keep only necessary history
        if len(self.values) > self.period * 2:
            self.values = self.values[-self.period * 2:]
        
        if len(self.values) >= self.period:
            self._initialized = True
            return self.calculate(self.values)
        
        return None
    
    def get_position(self, price: float) -> str:
        """Get position relative to Bollinger Bands."""
        if not self.is_ready():
            return "unknown"
        
        bands = self.calculate(self.values)
        
        if price > bands['upper']:
            return "above_upper"
        elif price < bands['lower']:
            return "below_lower"
        elif price > bands['middle']:
            return "above_middle"
        else:
            return "below_middle"


class KeltnerChannels(PriceBasedIndicator):
    """Keltner Channels indicator."""
    
    def __init__(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        super().__init__(period)
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.ema = EMA(period)
        self.atr = ATR(atr_period)
    
    def calculate_ohlc(self, ohlc_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate Keltner Channels using OHLC data."""
        if len(ohlc_data) < max(self.period, self.atr_period):
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}
        
        # Calculate EMA of typical price
        typical_prices = [(bar['high'] + bar['low'] + bar['close']) / 3.0 for bar in ohlc_data]
        middle = self.ema.calculate(typical_prices)
        
        # Calculate ATR
        atr_val = self.atr.calculate_ohlc(ohlc_data)
        
        # Calculate channels
        upper = middle + (self.multiplier * atr_val)
        lower = middle - (self.multiplier * atr_val)
        
        return {
            'upper': float(upper),
            'middle': float(middle),
            'lower': float(lower)
        }
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate Keltner Channels (requires OHLC data)."""
        return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}


class DonchianChannels(PriceBasedIndicator):
    """Donchian Channels indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
    
    def calculate_ohlc(self, ohlc_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate Donchian Channels using OHLC data."""
        if len(ohlc_data) < self.period:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}
        
        recent_data = ohlc_data[-self.period:]
        
        # Highest high and lowest low
        highs = [bar['high'] for bar in recent_data]
        lows = [bar['low'] for bar in recent_data]
        
        upper = max(highs)
        lower = min(lows)
        middle = (upper + lower) / 2.0
        
        return {
            'upper': float(upper),
            'middle': float(middle),
            'lower': float(lower)
        }
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate Donchian Channels (requires OHLC data)."""
        return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}


class StandardDeviation(IndicatorBase):
    """Standard Deviation indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Standard Deviation."""
        data = validate_data(data, self.period)
        
        if len(data) < self.period:
            return 0.0
        
        recent_data = data[-self.period:]
        return float(np.std(recent_data, ddof=0))


class HistoricalVolatility(IndicatorBase):
    """Historical Volatility indicator."""
    
    def __init__(self, period: int = 20, annualize_periods: int = 252):
        super().__init__(period)
        self.annualize_periods = annualize_periods
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Historical Volatility (annualized)."""
        data = validate_data(data, self.period + 1)
        
        if len(data) < self.period + 1:
            return 0.0
        
        # Calculate log returns
        log_returns = np.log(data[1:] / data[:-1])
        
        # Calculate standard deviation of returns
        if len(log_returns) < self.period:
            return 0.0
        
        recent_returns = log_returns[-self.period:]
        vol = np.std(recent_returns, ddof=1)
        
        # Annualize the volatility
        annualized_vol = vol * np.sqrt(self.annualize_periods)
        
        return float(annualized_vol)


class VIX_Style_Volatility(IndicatorBase):
    """VIX-style volatility indicator."""
    
    def __init__(self, period: int = 30, annualize_periods: int = 252):
        super().__init__(period)
        self.annualize_periods = annualize_periods
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate VIX-style volatility."""
        data = validate_data(data, self.period + 1)
        
        if len(data) < self.period + 1:
            return 0.0
        
        # Calculate returns
        returns = np.diff(data) / data[:-1]
        
        if len(returns) < self.period:
            return 0.0
        
        recent_returns = returns[-self.period:]
        
        # Calculate variance
        variance = np.var(recent_returns, ddof=1)
        
        # Annualize and convert to percentage
        annualized_variance = variance * self.annualize_periods
        volatility = np.sqrt(annualized_variance) * 100
        
        return float(volatility)


class PVOL(PriceBasedIndicator):
    """Price Volatility indicator."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
    
    def calculate_ohlc(self, ohlc_data: List[Dict[str, float]]) -> float:
        """Calculate Price Volatility using OHLC data."""
        if len(ohlc_data) < self.period:
            return 0.0
        
        recent_data = ohlc_data[-self.period:]
        
        # Calculate true ranges
        true_ranges = []
        for i in range(1, len(recent_data)):
            current = recent_data[i]
            previous = recent_data[i-1]
            tr = true_range(current['high'], current['low'], previous['close'])
            true_ranges.append(tr)
        
        if not true_ranges:
            return 0.0
        
        # Average true range as volatility measure
        volatility = np.mean(true_ranges)
        
        return float(volatility)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Price Volatility (requires OHLC data)."""
        return 0.0


class VolatilityRatio(IndicatorBase):
    """Volatility Ratio indicator."""
    
    def __init__(self, short_period: int = 10, long_period: int = 30):
        super().__init__(long_period)
        self.short_period = short_period
        self.long_period = long_period
        
        self.short_vol = StandardDeviation(short_period)
        self.long_vol = StandardDeviation(long_period)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Volatility Ratio."""
        data = validate_data(data, self.long_period)
        
        if len(data) < self.long_period:
            return 1.0
        
        short_volatility = self.short_vol.calculate(data)
        long_volatility = self.long_vol.calculate(data)
        
        if long_volatility == 0:
            return 1.0
        
        ratio = short_volatility / long_volatility
        return float(ratio)


# Convenience functions for one-time calculations

def atr(high: Union[List[float], np.ndarray], 
        low: Union[List[float], np.ndarray], 
        close: Union[List[float], np.ndarray], 
        period: int = 14) -> np.ndarray:
    """Calculate ATR for entire dataset."""
    high = validate_data(high, period)
    low = validate_data(low, period)
    close = validate_data(close, period)
    
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, Low, and Close arrays must have the same length")
    
    result = np.full(len(close), np.nan)
    
    if len(close) < 2:
        return result
    
    # Calculate true ranges
    true_ranges = []
    for i in range(1, len(close)):
        tr = true_range(high[i], low[i], close[i-1])
        true_ranges.append(tr)
    
    if len(true_ranges) < period:
        return result
    
    # Calculate ATR using Wilder's smoothing (similar to EMA with alpha = 1/period)
    alpha = 1.0 / period
    atr_val = np.mean(true_ranges[:period])
    result[period] = atr_val
    
    for i in range(period + 1, len(close)):
        tr_idx = i - 1  # true_ranges is offset by 1
        atr_val = alpha * true_ranges[tr_idx] + (1 - alpha) * atr_val
        result[i] = atr_val
    
    return result


def bollinger_bands(data: Union[List[float], np.ndarray, pd.Series], 
                   period: int = 20, 
                   std_dev: float = 2.0) -> Dict[str, np.ndarray]:
    """Calculate Bollinger Bands for entire dataset."""
    data = validate_data(data, period)
    
    upper = np.full(len(data), np.nan)
    middle = np.full(len(data), np.nan)
    lower = np.full(len(data), np.nan)
    
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        sma = np.mean(window)
        std = np.std(window, ddof=0)
        
        middle[i] = sma
        upper[i] = sma + (std_dev * std)
        lower[i] = sma - (std_dev * std)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def donchian_channels(high: Union[List[float], np.ndarray], 
                     low: Union[List[float], np.ndarray], 
                     period: int = 20) -> Dict[str, np.ndarray]:
    """Calculate Donchian Channels for entire dataset."""
    high = validate_data(high, period)
    low = validate_data(low, period)
    
    if len(high) != len(low):
        raise ValueError("High and Low arrays must have the same length")
    
    upper = np.full(len(high), np.nan)
    lower = np.full(len(low), np.nan)
    middle = np.full(len(high), np.nan)
    
    for i in range(period - 1, len(high)):
        high_window = high[i - period + 1:i + 1]
        low_window = low[i - period + 1:i + 1]
        
        upper[i] = np.max(high_window)
        lower[i] = np.min(low_window)
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }