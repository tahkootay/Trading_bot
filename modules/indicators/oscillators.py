"""Oscillator indicators implementation (RSI, MACD, Stochastic, etc.)."""

from typing import List, Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from .base import IndicatorBase, PriceBasedIndicator, validate_data
from .moving_averages import EMA, SMA


class RSI(IndicatorBase):
    """Relative Strength Index indicator."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.gains: List[float] = []
        self.losses: List[float] = []
        self.avg_gain: float = 0.0
        self.avg_loss: float = 0.0
        self.prev_close: Optional[float] = None
    
    def is_ready(self) -> bool:
        """RSI is ready when we have enough price changes (period gains/losses)."""
        return len(self.gains) >= self.period and self._initialized
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate RSI value."""
        data = validate_data(data, self.period + 1)
        
        # Calculate price changes
        changes = np.diff(data)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        if len(gains) < self.period:
            return 50.0
        
        # Initial average gain and loss (SMA)
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])
        
        # Calculate RSI for subsequent periods using Wilder's smoothing
        for i in range(self.period, len(gains)):
            avg_gain = (avg_gain * (self.period - 1) + gains[i]) / self.period
            avg_loss = (avg_loss * (self.period - 1) + losses[i]) / self.period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return float(rsi)
    
    def update(self, value: float) -> Optional[float]:
        """Update RSI with new value."""
        # Always add to values for is_ready() tracking
        self.values.append(value)
        
        if self.prev_close is not None:
            change = value - self.prev_close
            gain = max(change, 0)
            loss = max(-change, 0)
            
            self.gains.append(gain)
            self.losses.append(loss)
            
            # Keep only necessary history
            if len(self.gains) > self.period * 2:
                self.gains = self.gains[-self.period:]
                self.losses = self.losses[-self.period:]
            
            # Keep values history in sync
            if len(self.values) > self.period * 2:
                self.values = self.values[-self.period * 2:]
            
            if len(self.gains) >= self.period:
                if self.avg_gain == 0.0 and self.avg_loss == 0.0:
                    # Initialize with SMA
                    self.avg_gain = np.mean(self.gains[:self.period])
                    self.avg_loss = np.mean(self.losses[:self.period])
                else:
                    # Use Wilder's smoothing
                    self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
                    self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
                
                self._initialized = True
                
                if self.avg_loss == 0:
                    return 100.0
                
                rs = self.avg_gain / self.avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
                
                self.prev_close = value
                return float(rsi)
        
        self.prev_close = value
        return None
    
    def reset(self):
        """Reset RSI state."""
        super().reset()
        self.gains = []
        self.losses = []
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.prev_close = None


class MACD(IndicatorBase):
    """Moving Average Convergence Divergence indicator."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(max(fast_period, slow_period))
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.signal_ema = EMA(signal_period)
        
        self.macd_line: List[float] = []
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate MACD, Signal, and Histogram."""
        data = validate_data(data, self.slow_period)
        
        # Calculate EMAs
        fast_ema_val = self.fast_ema.calculate(data)
        slow_ema_val = self.slow_ema.calculate(data)
        
        # MACD line
        macd = fast_ema_val - slow_ema_val
        
        # Signal line (EMA of MACD)
        self.macd_line.append(macd)
        if len(self.macd_line) >= self.signal_period:
            signal = self.signal_ema.calculate(self.macd_line)
        else:
            signal = 0.0
        
        # Histogram
        histogram = macd - signal
        
        return {
            'macd': float(macd),
            'signal': float(signal),
            'histogram': float(histogram)
        }
    
    def update(self, value: float) -> Optional[Dict[str, float]]:
        """Update MACD with new value."""
        # Add value to self.values for is_ready() check
        self.values.append(value)
        
        # Keep only necessary history
        if len(self.values) > self.period * 2:
            self.values = self.values[-self.period * 2:]
        
        fast_ema = self.fast_ema.update(value)
        slow_ema = self.slow_ema.update(value)
        
        if fast_ema is not None and slow_ema is not None:
            macd = fast_ema - slow_ema
            self.macd_line.append(macd)
            
            # Keep only necessary history
            if len(self.macd_line) > self.signal_period * 2:
                self.macd_line = self.macd_line[-self.signal_period * 2:]
            
            if len(self.macd_line) >= self.signal_period:
                signal = self.signal_ema.update(macd)
                if signal is not None:
                    self._initialized = True
                    histogram = macd - signal
                    return {
                        'macd': float(macd),
                        'signal': float(signal),
                        'histogram': float(histogram)
                    }
            
            return {
                'macd': float(macd),
                'signal': 0.0,
                'histogram': float(macd)
            }
        
        return None
    
    def reset(self):
        """Reset MACD state."""
        super().reset()
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()
        self.macd_line = []


class StochasticOscillator(PriceBasedIndicator):
    """Stochastic Oscillator indicator."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
        super().__init__(k_period)
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        
        self.k_values: List[float] = []
        self.d_sma = SMA(d_period)
    
    def calculate_ohlc(self, ohlc_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate Stochastic %K and %D."""
        if len(ohlc_data) < self.k_period:
            return {'k': 50.0, 'd': 50.0}
        
        recent_data = ohlc_data[-self.k_period:]
        
        # Find highest high and lowest low in the period
        highs = [bar['high'] for bar in recent_data]
        lows = [bar['low'] for bar in recent_data]
        current_close = recent_data[-1]['close']
        
        highest_high = max(highs)
        lowest_low = min(lows)
        
        # Calculate %K
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = 100.0 * (current_close - lowest_low) / (highest_high - lowest_low)
        
        # Apply smoothing to %K if requested
        if self.smooth_k > 1:
            self.k_values.append(k_percent)
            if len(self.k_values) >= self.smooth_k:
                k_percent = np.mean(self.k_values[-self.smooth_k:])
            
            # Keep only necessary history
            if len(self.k_values) > self.smooth_k * 2:
                self.k_values = self.k_values[-self.smooth_k * 2:]
        
        # Calculate %D (SMA of %K)
        d_percent = self.d_sma.update(k_percent)
        if d_percent is None:
            d_percent = k_percent
        
        return {
            'k': float(k_percent),
            'd': float(d_percent)
        }
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate Stochastic (requires OHLC data)."""
        # This is a fallback - Stochastic requires OHLC data
        return {'k': 50.0, 'd': 50.0}
    
    def reset(self):
        """Reset Stochastic state."""
        super().reset()
        self.k_values = []
        self.d_sma.reset()


class Williams_R(PriceBasedIndicator):
    """Williams %R indicator."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
    
    def calculate_ohlc(self, ohlc_data: List[Dict[str, float]]) -> float:
        """Calculate Williams %R."""
        if len(ohlc_data) < self.period:
            return -50.0
        
        recent_data = ohlc_data[-self.period:]
        
        # Find highest high and lowest low
        highs = [bar['high'] for bar in recent_data]
        lows = [bar['low'] for bar in recent_data]
        current_close = recent_data[-1]['close']
        
        highest_high = max(highs)
        lowest_low = min(lows)
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = -100.0 * (highest_high - current_close) / (highest_high - lowest_low)
        return float(williams_r)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Williams %R (requires OHLC data)."""
        return -50.0


class CCI(IndicatorBase):
    """Commodity Channel Index indicator."""
    
    def __init__(self, period: int = 20, constant: float = 0.015):
        super().__init__(period)
        self.constant = constant
        self.typical_prices: List[float] = []
    
    def update_hlc(self, high: float, low: float, close: float) -> Optional[float]:
        """Update CCI with High, Low, Close values."""
        typical_price = (high + low + close) / 3.0
        self.typical_prices.append(typical_price)
        
        # Keep only necessary history
        if len(self.typical_prices) > self.period * 2:
            self.typical_prices = self.typical_prices[-self.period * 2:]
        
        if len(self.typical_prices) >= self.period:
            self._initialized = True
            return self.calculate_cci()
        
        return None
    
    def calculate_cci(self) -> float:
        """Calculate CCI value."""
        recent_tp = self.typical_prices[-self.period:]
        sma_tp = np.mean(recent_tp)
        
        # Calculate mean deviation
        mean_deviation = np.mean([abs(tp - sma_tp) for tp in recent_tp])
        
        if mean_deviation == 0:
            return 0.0
        
        current_tp = recent_tp[-1]
        cci = (current_tp - sma_tp) / (self.constant * mean_deviation)
        
        return float(cci)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate CCI (requires HLC data via update_hlc)."""
        return 0.0
    
    def reset(self):
        """Reset CCI state."""
        super().reset()
        self.typical_prices = []


class MomentumOscillator(IndicatorBase):
    """Momentum Oscillator indicator."""
    
    def __init__(self, period: int = 10):
        super().__init__(period)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Momentum (current price / price n periods ago)."""
        data = validate_data(data, self.period + 1)
        
        if len(data) < self.period + 1:
            return 100.0
        
        current_price = data[-1]
        past_price = data[-(self.period + 1)]
        
        if past_price == 0:
            return 100.0
        
        momentum = 100.0 * (current_price / past_price)
        return float(momentum)


class ROC(IndicatorBase):
    """Rate of Change indicator."""
    
    def __init__(self, period: int = 10):
        super().__init__(period)
    
    def calculate(self, data: Union[List[float], np.ndarray, pd.Series]) -> float:
        """Calculate Rate of Change percentage."""
        data = validate_data(data, self.period + 1)
        
        if len(data) < self.period + 1:
            return 0.0
        
        current_price = data[-1]
        past_price = data[-(self.period + 1)]
        
        if past_price == 0:
            return 0.0
        
        roc = 100.0 * ((current_price - past_price) / past_price)
        return float(roc)


# Convenience functions for one-time calculations

def rsi(data: Union[List[float], np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
    """Calculate RSI for entire dataset."""
    data = validate_data(data, period + 1)
    result = np.full(len(data), np.nan)
    
    # Calculate price changes
    changes = np.diff(data)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    
    if len(gains) < period:
        return result
    
    # Initial RSI using SMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(period + 1, len(data)):
        gain = gains[i - 1]
        loss = losses[i - 1]
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


def macd(data: Union[List[float], np.ndarray, pd.Series], 
         fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """Calculate MACD for entire dataset."""
    from .moving_averages import ema
    
    data = validate_data(data, slow)
    
    # Calculate EMAs
    fast_ema = ema(data, fast)
    slow_ema = ema(data, slow)
    
    # MACD line
    macd_line = fast_ema - slow_ema
    
    # Signal line
    signal_line = ema(macd_line[~np.isnan(macd_line)], signal)
    
    # Align signal line with MACD line
    signal_aligned = np.full(len(macd_line), np.nan)
    valid_macd_start = slow - 1
    signal_start = valid_macd_start + signal - 1
    signal_aligned[signal_start:signal_start + len(signal_line)] = signal_line
    
    # Histogram
    histogram = macd_line - signal_aligned
    
    return {
        'macd': macd_line,
        'signal': signal_aligned,
        'histogram': histogram
    }