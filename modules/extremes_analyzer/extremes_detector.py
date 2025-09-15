"""
Extremes detection and analysis engine.

This module implements algorithms for detecting local price extremes
and analyzing their characteristics with technical indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings


class ExtremesDetector:
    """Detects and analyzes price extremes in OHLCV data."""
    
    def __init__(self, min_threshold_usdt: float = 3.0, window_size: int = 5):
        """
        Initialize the extremes detector.
        
        Args:
            min_threshold_usdt: Minimum price movement in USDT to qualify as extreme
            window_size: Window size for local extrema detection
        """
        self.min_threshold_usdt = min_threshold_usdt
        self.window_size = window_size
        
    def detect_extremes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect local extremes in price data.
        
        Args:
            df: DataFrame with OHLCV data (columns: timestamp, open, high, low, close, volume)
            
        Returns:
            DataFrame with detected extremes and their analysis
        """
        if len(df) < self.window_size * 2:
            raise ValueError(f"Data too short. Need at least {self.window_size * 2} rows.")
            
        # Find local minima and maxima
        local_maxima = self._find_local_maxima(df['high'].values)
        local_minima = self._find_local_minima(df['low'].values)
        
        extremes = []
        
        # Process maxima
        for idx in local_maxima:
            if self._validate_extreme_movement(df, idx, 'max'):
                extreme_data = self._analyze_extreme(df, idx, 'max')
                extremes.append(extreme_data)
        
        # Process minima
        for idx in local_minima:
            if self._validate_extreme_movement(df, idx, 'min'):
                extreme_data = self._analyze_extreme(df, idx, 'min')
                extremes.append(extreme_data)
        
        if not extremes:
            return self._create_empty_extremes_df()
            
        extremes_df = pd.DataFrame(extremes)
        return extremes_df.sort_values('timestamp').reset_index(drop=True)
    
    def _find_local_maxima(self, prices: np.ndarray) -> np.ndarray:
        """Find local maxima using simple sliding window approach."""
        maxima = []
        n = len(prices)
        
        for i in range(self.window_size, n - self.window_size):
            # Check if current point is higher than all points in the window
            is_maximum = True
            current_price = prices[i]
            
            # Check left side
            for j in range(max(0, i - self.window_size), i):
                if prices[j] >= current_price:
                    is_maximum = False
                    break
            
            # Check right side if still a potential maximum
            if is_maximum:
                for j in range(i + 1, min(n, i + self.window_size + 1)):
                    if prices[j] >= current_price:
                        is_maximum = False
                        break
            
            if is_maximum:
                maxima.append(i)
        
        return np.array(maxima)
    
    def _find_local_minima(self, prices: np.ndarray) -> np.ndarray:
        """Find local minima using simple sliding window approach."""
        minima = []
        n = len(prices)
        
        for i in range(self.window_size, n - self.window_size):
            # Check if current point is lower than all points in the window
            is_minimum = True
            current_price = prices[i]
            
            # Check left side
            for j in range(max(0, i - self.window_size), i):
                if prices[j] <= current_price:
                    is_minimum = False
                    break
            
            # Check right side if still a potential minimum
            if is_minimum:
                for j in range(i + 1, min(n, i + self.window_size + 1)):
                    if prices[j] <= current_price:
                        is_minimum = False
                        break
            
            if is_minimum:
                minima.append(i)
        
        return np.array(minima)
    
    def _validate_extreme_movement(self, df: pd.DataFrame, idx: int, extreme_type: str) -> bool:
        """
        Check if price movement after extreme exceeds threshold.
        
        Args:
            df: Price data DataFrame
            idx: Index of the extreme point
            extreme_type: 'max' or 'min'
            
        Returns:
            True if movement exceeds threshold
        """
        if idx >= len(df) - 1:
            return False
            
        # Use numpy arrays for faster computation
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        extreme_price = high_prices[idx] if extreme_type == 'max' else low_prices[idx]
        
        if extreme_type == 'max':
            # For maxima, check if price dropped by at least threshold
            min_future_price = np.min(low_prices[idx + 1:])
            movement = extreme_price - min_future_price
        else:
            # For minima, check if price rose by at least threshold
            max_future_price = np.max(high_prices[idx + 1:])
            movement = max_future_price - extreme_price
        
        return movement >= self.min_threshold_usdt
    
    def _analyze_extreme(self, df: pd.DataFrame, idx: int, extreme_type: str) -> Dict:
        """
        Analyze an extreme point and calculate indicators.
        
        Args:
            df: Price data DataFrame
            idx: Index of the extreme point
            extreme_type: 'max' or 'min'
            
        Returns:
            Dictionary with extreme analysis data
        """
        row = df.iloc[idx]
        extreme_price = row['high'] if extreme_type == 'max' else row['low']
        
        # Calculate EMAs
        ema20 = self._calculate_ema(df['close'].iloc[:idx+1], 20)
        ema50 = self._calculate_ema(df['close'].iloc[:idx+1], 50)
        ema200 = self._calculate_ema(df['close'].iloc[:idx+1], 200)
        
        # Calculate RSI
        rsi14 = self._calculate_rsi(df['close'].iloc[:idx+1], 14)
        
        # Calculate volume metrics
        volume = row['volume']
        volume_avg20 = df['volume'].iloc[max(0, idx-19):idx+1].mean()
        
        # Calculate future price movement using numpy for speed
        if idx < len(df) - 1:
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            min_price_after = np.min(low_prices[idx + 1:])
            max_price_after = np.max(high_prices[idx + 1:])
            
            if extreme_type == 'max':
                price_change = extreme_price - min_price_after
                direction = -1  # Downward from maximum
            else:
                price_change = max_price_after - extreme_price
                direction = 1   # Upward from minimum
        else:
            min_price_after = extreme_price
            max_price_after = extreme_price
            price_change = 0.0
            direction = 0
        
        return {
            'id': f"{row['timestamp']}_{extreme_type}",
            'timestamp': row['timestamp'],
            'symbol': df.attrs.get('symbol', 'UNKNOWN'),
            'timeframe': df.attrs.get('timeframe', 'UNKNOWN'),
            'extreme_type': extreme_type,
            'extreme_price': extreme_price,
            'ema20': ema20,
            'ema50': ema50,
            'ema200': ema200,
            'rsi14': rsi14,
            'bb_upper': np.nan,  # Will be implemented in v1.1
            'bb_middle': np.nan,
            'bb_lower': np.nan,
            'macd': np.nan,      # Will be implemented in v1.1
            'macd_signal': np.nan,
            'macd_hist': np.nan,
            'volume': volume,
            'volume_avg20': volume_avg20,
            'atr': np.nan,       # Will be implemented in v1.1
            'max_price_after': max_price_after,
            'min_price_after': min_price_after,
            'price_change': price_change,
            'direction': direction,
            'strength': abs(price_change),
            'hit_threshold': price_change >= self.min_threshold_usdt,
            'time_to_hit': np.nan,  # Will be implemented in v1.1
            'trend_context': self._get_trend_context(extreme_price, ema200),
            'volatility_context': 'unknown'  # Will be implemented in v1.1
        }
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.nan
        # Use more efficient calculation for large datasets
        if len(prices) > 1000:  # For large datasets, use simplified calculation
            return prices.tail(period).mean()  # Simple moving average as approximation
        return prices.ewm(span=period, adjust=False).mean().iloc[-1]
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return np.nan
        
        # Use only recent data for faster computation
        recent_prices = prices.tail(period + 10) if len(prices) > period + 10 else prices
        
        delta = recent_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean().iloc[-1]
        avg_loss = loss.rolling(window=period, min_periods=1).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _get_trend_context(self, price: float, ema200: float) -> str:
        """Determine trend context relative to EMA200."""
        if np.isnan(ema200):
            return 'unknown'
        return 'above' if price > ema200 else 'below'
    
    def _create_empty_extremes_df(self) -> pd.DataFrame:
        """Create empty DataFrame with correct structure."""
        columns = [
            "id", "timestamp", "symbol", "timeframe",
            "extreme_type", "extreme_price",
            "ema20", "ema50", "ema200",
            "rsi14",
            "bb_upper", "bb_middle", "bb_lower",
            "macd", "macd_signal", "macd_hist",
            "volume", "volume_avg20",
            "atr",
            "max_price_after", "min_price_after",
            "price_change", "direction", "strength",
            "hit_threshold", "time_to_hit",
            "trend_context", "volatility_context"
        ]
        return pd.DataFrame(columns=columns)