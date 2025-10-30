#!/usr/bin/env python3
"""
Technical Indicators Module for Data Collector

Provides technical analysis indicators that can be calculated during data collection
or applied to existing datasets. All indicators follow a consistent API.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations."""
    rsi_period: int = 14
    sma_period: int = 20
    ema_period: int = 12
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    kdj_period: int = 14
    kdj_k_period: int = 3
    kdj_d_period: int = 3


class TechnicalIndicators:
    """Technical indicators calculator."""
    
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
    
    def calculate_sma(self, prices: List[float], period: int = None) -> List[float]:
        """Calculate Simple Moving Average."""
        if period is None:
            period = self.config.sma_period
            
        if len(prices) < period:
            return [None] * len(prices)
        
        sma_values = []
        for i in range(len(prices)):
            if i < period - 1:
                sma_values.append(None)
            else:
                sma_values.append(sum(prices[i-period+1:i+1]) / period)
        
        return sma_values
    
    def calculate_ema(self, prices: List[float], period: int = None) -> List[float]:
        """Calculate Exponential Moving Average."""
        if period is None:
            period = self.config.ema_period
            
        if len(prices) < period:
            return [None] * len(prices)
        
        alpha = 2 / (period + 1)
        ema_values = [None] * (period - 1)
        
        # First EMA value is SMA
        ema_values.append(sum(prices[:period]) / period)
        
        # Calculate subsequent EMA values
        for i in range(period, len(prices)):
            ema_values.append(alpha * prices[i] + (1 - alpha) * ema_values[-1])
        
        return ema_values
    
    def calculate_rsi(self, prices: List[float], period: int = None) -> List[float]:
        """Calculate Relative Strength Index."""
        if period is None:
            period = self.config.rsi_period
            
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [max(delta, 0) for delta in deltas]
        losses = [abs(min(delta, 0)) for delta in deltas]
        
        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = [None] * (period)
        
        # Calculate RSI for first valid period
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        else:
            rsi_values.append(100)
        
        # Calculate subsequent RSI values using Wilder's smoothing
        for i in range(period + 1, len(prices)):
            current_gain = gains[i - 1]
            current_loss = losses[i - 1]
            
            avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
            avg_loss = ((avg_loss * (period - 1)) + current_loss) / period
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
            else:
                rsi_values.append(100)
        
        return rsi_values
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = None, std_dev: float = None) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if period is None:
            period = self.config.bollinger_period
        if std_dev is None:
            std_dev = self.config.bollinger_std
            
        if len(prices) < period:
            return ([None] * len(prices), [None] * len(prices), [None] * len(prices))
        
        sma_values = self.calculate_sma(prices, period)
        upper_bands = []
        lower_bands = []
        
        for i in range(len(prices)):
            if i < period - 1:
                upper_bands.append(None)
                lower_bands.append(None)
            else:
                # Calculate standard deviation
                period_prices = prices[i-period+1:i+1]
                std = math.sqrt(sum((p - sma_values[i]) ** 2 for p in period_prices) / period)
                
                upper_bands.append(sma_values[i] + (std_dev * std))
                lower_bands.append(sma_values[i] - (std_dev * std))
        
        return upper_bands, sma_values, lower_bands
    
    def calculate_macd(self, prices: List[float], fast: int = None, slow: int = None, signal: int = None) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD (macd_line, signal_line, histogram)."""
        if fast is None:
            fast = self.config.macd_fast
        if slow is None:
            slow = self.config.macd_slow
        if signal is None:
            signal = self.config.macd_signal
            
        if len(prices) < slow:
            return ([None] * len(prices), [None] * len(prices), [None] * len(prices))
        
        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        # Calculate MACD line
        macd_line = []
        for i in range(len(prices)):
            if ema_fast[i] is None or ema_slow[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        # Calculate signal line (EMA of MACD line)
        valid_macd = [val for val in macd_line if val is not None]
        if len(valid_macd) < signal:
            signal_line = [None] * len(prices)
            histogram = [None] * len(prices)
        else:
            signal_ema = self.calculate_ema(valid_macd, signal)
            
            # Align signal line with original data
            signal_line = [None] * (len(macd_line) - len(signal_ema)) + signal_ema
            
            # Calculate histogram
            histogram = []
            for i in range(len(macd_line)):
                if macd_line[i] is None or signal_line[i] is None:
                    histogram.append(None)
                else:
                    histogram.append(macd_line[i] - signal_line[i])
        
        return macd_line, signal_line, histogram
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = None) -> List[float]:
        """Calculate Average True Range."""
        if period is None:
            period = self.config.atr_period
            
        if len(highs) < period + 1:
            return [None] * len(highs)
        
        # Calculate True Range
        true_ranges = [None]  # First value is None
        
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        # Calculate ATR
        atr_values = [None] * period
        
        # First ATR is simple average
        first_atr = sum(true_ranges[1:period+1]) / period
        atr_values.append(first_atr)
        
        # Subsequent ATR values using Wilder's smoothing
        for i in range(period + 1, len(true_ranges)):
            current_atr = ((atr_values[-1] * (period - 1)) + true_ranges[i]) / period
            atr_values.append(current_atr)
        
        return atr_values
    
    def calculate_kdj(self, highs: List[float], lows: List[float], closes: List[float], 
                      period: int = None, k_period: int = None, d_period: int = None) -> Tuple[List[float], List[float], List[float]]:
        """Calculate KDJ indicator (K%, D%, J%)."""
        if period is None:
            period = self.config.kdj_period
        if k_period is None:
            k_period = self.config.kdj_k_period
        if d_period is None:
            d_period = self.config.kdj_d_period
            
        if len(highs) < period:
            return ([None] * len(highs), [None] * len(highs), [None] * len(highs))
        
        # Calculate %K (raw stochastic)
        k_raw = []
        for i in range(len(highs)):
            if i < period - 1:
                k_raw.append(None)
            else:
                highest_high = max(highs[i-period+1:i+1])
                lowest_low = min(lows[i-period+1:i+1])
                if highest_high == lowest_low:
                    k_raw.append(50.0)  # Avoid division by zero
                else:
                    k_raw.append(((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100)
        
        # Calculate %K (smoothed)
        k_values = self.calculate_sma([val for val in k_raw if val is not None], k_period)
        k_aligned = [None] * (len(k_raw) - len(k_values)) + k_values
        
        # Calculate %D (smoothed %K)
        d_values = self.calculate_sma([val for val in k_aligned if val is not None], d_period)
        d_aligned = [None] * (len(k_aligned) - len(d_values)) + d_values
        
        # Calculate %J
        j_values = []
        for i in range(len(k_aligned)):
            if k_aligned[i] is None or d_aligned[i] is None:
                j_values.append(None)
            else:
                j_values.append(3 * k_aligned[i] - 2 * d_aligned[i])
        
        return k_aligned, d_aligned, j_values
    
    def add_indicators_to_dataframe(self, df: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
        """Add technical indicators to a dataframe with OHLCV data."""
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'bollinger', 'macd', 'atr', 'kdj']
        
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ['close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        closes = df['close'].tolist()
        
        if 'sma' in indicators:
            df['sma'] = self.calculate_sma(closes)
        
        if 'ema' in indicators:
            df['ema'] = self.calculate_ema(closes)
        
        if 'rsi' in indicators:
            df['rsi'] = self.calculate_rsi(closes)
        
        if 'bollinger' in indicators:
            upper, middle, lower = self.calculate_bollinger_bands(closes)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
        
        if 'macd' in indicators:
            macd_line, signal_line, histogram = self.calculate_macd(closes)
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_histogram'] = histogram
        
        if 'atr' in indicators and all(col in df.columns for col in ['high', 'low']):
            highs = df['high'].tolist()
            lows = df['low'].tolist()
            df['atr'] = self.calculate_atr(highs, lows, closes)
        
        if 'kdj' in indicators and all(col in df.columns for col in ['high', 'low']):
            highs = df['high'].tolist()
            lows = df['low'].tolist()
            k_values, d_values, j_values = self.calculate_kdj(highs, lows, closes)
            df['kdj_k'] = k_values
            df['kdj_d'] = d_values
            df['kdj_j'] = j_values
        
        return df


def calculate_indicators_for_file(input_file: str, output_file: str = None, indicators: List[str] = None, config: IndicatorConfig = None) -> str:
    """
    Calculate indicators for a CSV file containing OHLCV data.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        indicators: List of indicators to calculate (optional)
        config: Indicator configuration (optional)
    
    Returns:
        Path to output file
    """
    if output_file is None:
        output_file = input_file.replace('.csv', '_with_indicators.csv')
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Calculate indicators
    calculator = TechnicalIndicators(config)
    df_with_indicators = calculator.add_indicators_to_dataframe(df, indicators)
    
    # Save result
    df_with_indicators.to_csv(output_file, index=False)
    
    return output_file


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python indicators.py <input_file.csv> [output_file.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result_file = calculate_indicators_for_file(input_file, output_file)
    print(f"Indicators calculated and saved to: {result_file}")