#!/usr/bin/env python3
"""
Advanced Technical Indicators Module

Comprehensive technical analysis indicators for trading systems.
Supports configuration-based indicator selection and calculation.
"""

import math
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any


class AdvancedIndicators:
    """Advanced technical indicators calculator with configuration support."""
    
    def __init__(self, config_path: str = None):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.indicators_config = self.config.get('indicators', {})
        self.output_config = self.config.get('output', {})
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "indicators.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'indicators': {
                'sma_multiple': {'enabled': True, 'periods': [5, 10, 20]},
                'ema_multiple': {'enabled': True, 'periods': [5, 10, 20, 50, 100]},
                'rsi_14': {'enabled': True, 'period': 14},
                'macd': {'enabled': True, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                'bollinger_bands': {'enabled': True, 'period': 20, 'std_dev': 2.0},
                'atr_14': {'enabled': True, 'period': 14}
            },
            'output': {'round_decimals': 6, 'fill_na_method': 'none'}
        }

    # === Трендовые индикаторы ===
    
    def calculate_sma_multiple(self, prices: List[float], periods: List[int]) -> Dict[str, List[float]]:
        """Calculate multiple SMA periods."""
        results = {}
        for period in periods:
            results[f'sma_{period}'] = self._calculate_sma(prices, period)
        return results
    
    def calculate_ema_multiple(self, prices: List[float], periods: List[int]) -> Dict[str, List[float]]:
        """Calculate multiple EMA periods."""
        results = {}
        for period in periods:
            results[f'ema_{period}'] = self._calculate_ema(prices, period)
        return results
    
    def calculate_slope_ema(self, prices: List[float], period: int, lookback: int = 3) -> List[float]:
        """Calculate EMA slope (rate of change)."""
        ema_values = self._calculate_ema(prices, period)
        slopes = [None] * len(ema_values)
        
        for i in range(lookback, len(ema_values)):
            if ema_values[i] is not None and ema_values[i - lookback] is not None:
                slopes[i] = (ema_values[i] - ema_values[i - lookback]) / lookback
            else:
                slopes[i] = None
        
        return slopes
    
    def calculate_ema_diff(self, prices: List[float], short_period: int, long_period: int) -> List[float]:
        """Calculate difference between two EMAs."""
        ema_short = self._calculate_ema(prices, short_period)
        ema_long = self._calculate_ema(prices, long_period)
        
        diff = []
        for i in range(len(prices)):
            if ema_short[i] is not None and ema_long[i] is not None:
                diff.append(ema_short[i] - ema_long[i])
            else:
                diff.append(None)
        
        return diff

    # === Осцилляторы ===
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(delta, 0) for delta in deltas]
        losses = [abs(min(delta, 0)) for delta in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = [None] * (period)
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        else:
            rsi_values.append(100)
        
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
    
    def calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], 
                           k_period: int = 14, d_period: int = 3, smooth_k: int = 1) -> Tuple[List[float], List[float]]:
        """Calculate Stochastic %K and %D."""
        if len(highs) < k_period:
            return ([None] * len(highs), [None] * len(highs))
        
        # Calculate raw %K
        k_raw = []
        for i in range(len(highs)):
            if i < k_period - 1:
                k_raw.append(None)
            else:
                highest_high = max(highs[i-k_period+1:i+1])
                lowest_low = min(lows[i-k_period+1:i+1])
                if highest_high == lowest_low:
                    k_raw.append(50.0)
                else:
                    k_raw.append(((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100)
        
        # Smooth %K if needed
        if smooth_k > 1:
            k_values = self._calculate_sma([val for val in k_raw if val is not None], smooth_k)
            k_aligned = [None] * (len(k_raw) - len(k_values)) + k_values
        else:
            k_aligned = k_raw
        
        # Calculate %D (SMA of %K)
        d_values = self._calculate_sma([val for val in k_aligned if val is not None], d_period)
        d_aligned = [None] * (len(k_aligned) - len(d_values)) + d_values
        
        return k_aligned, d_aligned
    
    def calculate_cci(self, highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> List[float]:
        """Calculate Commodity Channel Index."""
        if len(highs) < period:
            return [None] * len(highs)
        
        # Calculate Typical Price
        typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(highs))]
        
        # Calculate SMA of Typical Price
        sma_tp = self._calculate_sma(typical_prices, period)
        
        cci_values = []
        for i in range(len(typical_prices)):
            if i < period - 1 or sma_tp[i] is None:
                cci_values.append(None)
            else:
                # Calculate Mean Deviation
                period_tp = typical_prices[i-period+1:i+1]
                mean_dev = sum(abs(tp - sma_tp[i]) for tp in period_tp) / period
                
                if mean_dev != 0:
                    cci = (typical_prices[i] - sma_tp[i]) / (0.015 * mean_dev)
                    cci_values.append(cci)
                else:
                    cci_values.append(0)
        
        return cci_values

    # === Волатильность ===
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """Calculate Average True Range."""
        if len(highs) < period + 1:
            return [None] * len(highs)
        
        true_ranges = [None]
        
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        atr_values = [None] * period
        first_atr = sum(true_ranges[1:period+1]) / period
        atr_values.append(first_atr)
        
        for i in range(period + 1, len(true_ranges)):
            current_atr = ((atr_values[-1] * (period - 1)) + true_ranges[i]) / period
            atr_values.append(current_atr)
        
        return atr_values
    
    def calculate_high_low_range(self, highs: List[float], lows: List[float]) -> List[float]:
        """Calculate High-Low range for each candle."""
        return [highs[i] - lows[i] for i in range(len(highs))]
    
    def calculate_body_to_range(self, opens: List[float], highs: List[float], 
                               lows: List[float], closes: List[float]) -> List[float]:
        """Calculate body to range ratio."""
        ratios = []
        for i in range(len(opens)):
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            
            if total_range != 0:
                ratios.append(body_size / total_range)
            else:
                ratios.append(0)
        
        return ratios
    
    def calculate_volatility_ratio(self, atr_values: List[float], ema_values: List[float]) -> List[float]:
        """Calculate volatility ratio (ATR / EMA)."""
        ratios = []
        for i in range(len(atr_values)):
            if atr_values[i] is not None and ema_values[i] is not None and ema_values[i] != 0:
                ratios.append(atr_values[i] / ema_values[i])
            else:
                ratios.append(None)
        
        return ratios

    # === Bollinger Bands расширенные ===
    
    def calculate_bollinger_bands_extended(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, List[float]]:
        """Calculate extended Bollinger Bands indicators."""
        if len(prices) < period:
            empty = [None] * len(prices)
            return {
                'bb_upper': empty, 'bb_middle': empty, 'bb_lower': empty,
                'bb_width': empty, 'bb_position': empty, 
                'bb_touch_upper': empty, 'bb_touch_lower': empty
            }
        
        # Basic Bollinger Bands
        sma_values = self._calculate_sma(prices, period)
        upper_bands = []
        lower_bands = []
        
        for i in range(len(prices)):
            if i < period - 1:
                upper_bands.append(None)
                lower_bands.append(None)
            else:
                period_prices = prices[i-period+1:i+1]
                std = math.sqrt(sum((p - sma_values[i]) ** 2 for p in period_prices) / period)
                
                upper_bands.append(sma_values[i] + (std_dev * std))
                lower_bands.append(sma_values[i] - (std_dev * std))
        
        # Extended indicators
        bb_width = []
        bb_position = []
        bb_touch_upper = []
        bb_touch_lower = []
        
        for i in range(len(prices)):
            if upper_bands[i] is None or lower_bands[i] is None:
                bb_width.append(None)
                bb_position.append(None)
                bb_touch_upper.append(None)
                bb_touch_lower.append(None)
            else:
                # BB Width: (upper - lower) / middle
                width = (upper_bands[i] - lower_bands[i]) / sma_values[i] if sma_values[i] != 0 else 0
                bb_width.append(width)
                
                # BB Position: (close - lower) / (upper - lower)
                band_range = upper_bands[i] - lower_bands[i]
                if band_range != 0:
                    position = (prices[i] - lower_bands[i]) / band_range
                    bb_position.append(position)
                else:
                    bb_position.append(0.5)
                
                # BB Touch (within 1% of bands)
                touch_threshold = 0.01
                upper_touch = abs(prices[i] - upper_bands[i]) / upper_bands[i] < touch_threshold
                lower_touch = abs(prices[i] - lower_bands[i]) / lower_bands[i] < touch_threshold
                
                bb_touch_upper.append(1 if upper_touch else 0)
                bb_touch_lower.append(1 if lower_touch else 0)
        
        return {
            'bb_upper': upper_bands,
            'bb_middle': sma_values,
            'bb_lower': lower_bands,
            'bb_width': bb_width,
            'bb_position': bb_position,
            'bb_touch_upper': bb_touch_upper,
            'bb_touch_lower': bb_touch_lower
        }

    # === Объёмы ===
    
    def calculate_volume_indicators(self, volumes: List[float]) -> Dict[str, List[float]]:
        """Calculate volume-based indicators."""
        # Volume change
        volume_change = [None]
        for i in range(1, len(volumes)):
            if volumes[i-1] != 0:
                change = (volumes[i] - volumes[i-1]) / volumes[i-1]
                volume_change.append(change)
            else:
                volume_change.append(None)
        
        # Volume SMA
        volume_sma_20 = self._calculate_sma(volumes, 20)
        
        # Relative volume
        rel_volume = []
        for i in range(len(volumes)):
            if volume_sma_20[i] is not None and volume_sma_20[i] != 0:
                rel_volume.append(volumes[i] / volume_sma_20[i])
            else:
                rel_volume.append(None)
        
        return {
            'volume_change': volume_change,
            'volume_sma_20': volume_sma_20,
            'relative_volume': rel_volume
        }

    # === Вспомогательные признаки ===
    
    def calculate_candle_features(self, opens: List[float], highs: List[float], 
                                 lows: List[float], closes: List[float]) -> Dict[str, List[float]]:
        """Calculate candle-based features."""
        candle_type = []
        candle_ratio = []
        
        for i in range(len(opens)):
            # Candle type: 1 for bullish, 0 for bearish
            candle_type.append(1 if closes[i] >= opens[i] else 0)
            
            # Candle ratio: (close - open) / (high - low)
            body_size = closes[i] - opens[i]
            total_range = highs[i] - lows[i]
            
            if total_range != 0:
                candle_ratio.append(body_size / total_range)
            else:
                candle_ratio.append(0)
        
        return {
            'candle_type': candle_type,
            'candle_ratio': candle_ratio
        }
    
    def calculate_momentum_multiple(self, prices: List[float], periods: List[int]) -> Dict[str, List[float]]:
        """Calculate momentum for multiple periods."""
        results = {}
        
        for period in periods:
            momentum = []
            for i in range(len(prices)):
                if i < period:
                    momentum.append(None)
                else:
                    momentum.append(prices[i] - prices[i - period])
            
            results[f'momentum_{period}'] = momentum
        
        return results

    # === MACD ===
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """Calculate MACD."""
        if len(prices) < slow:
            empty = [None] * len(prices)
            return {'macd_line': empty, 'macd_signal': empty, 'macd_histogram': empty}
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = []
        for i in range(len(prices)):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line.append(ema_fast[i] - ema_slow[i])
            else:
                macd_line.append(None)
        
        valid_macd = [val for val in macd_line if val is not None]
        if len(valid_macd) < signal:
            signal_line = [None] * len(prices)
            histogram = [None] * len(prices)
        else:
            signal_ema = self._calculate_ema(valid_macd, signal)
            signal_line = [None] * (len(macd_line) - len(signal_ema)) + signal_ema
            
            histogram = []
            for i in range(len(macd_line)):
                if macd_line[i] is not None and signal_line[i] is not None:
                    histogram.append(macd_line[i] - signal_line[i])
                else:
                    histogram.append(None)
        
        return {
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }

    # === Helper methods ===
    
    def _calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return [None] * len(prices)
        
        sma_values = []
        for i in range(len(prices)):
            if i < period - 1:
                sma_values.append(None)
            else:
                sma_values.append(sum(prices[i-period+1:i+1]) / period)
        
        return sma_values
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return [None] * len(prices)
        
        alpha = 2 / (period + 1)
        ema_values = [None] * (period - 1)
        
        ema_values.append(sum(prices[:period]) / period)
        
        for i in range(period, len(prices)):
            ema_values.append(alpha * prices[i] + (1 - alpha) * ema_values[-1])
        
        return ema_values
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all enabled indicators for a dataframe."""
        result_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Extract data
        opens = result_df['open'].tolist()
        highs = result_df['high'].tolist()
        lows = result_df['low'].tolist()
        closes = result_df['close'].tolist()
        volumes = result_df['volume'].tolist()
        
        # Calculate indicators based on configuration
        for indicator_name, config in self.indicators_config.items():
            if not config.get('enabled', False):
                continue
            
            try:
                if indicator_name == 'sma_multiple':
                    periods = config.get('periods', [5, 10, 20])
                    sma_results = self.calculate_sma_multiple(closes, periods)
                    for col_name, values in sma_results.items():
                        result_df[col_name] = values
                
                elif indicator_name == 'ema_multiple':
                    periods = config.get('periods', [5, 10, 20, 50, 100])
                    ema_results = self.calculate_ema_multiple(closes, periods)
                    for col_name, values in ema_results.items():
                        result_df[col_name] = values
                
                elif indicator_name == 'slope_ema_20':
                    period = config.get('period', 20)
                    lookback = config.get('lookback', 3)
                    result_df['slope_ema_20'] = self.calculate_slope_ema(closes, period, lookback)
                
                elif indicator_name == 'ema_diff_10_50':
                    short = config.get('short_period', 10)
                    long = config.get('long_period', 50)
                    result_df['ema_diff_10_50'] = self.calculate_ema_diff(closes, short, long)
                
                elif indicator_name == 'rsi_14':
                    period = config.get('period', 14)
                    result_df['rsi_14'] = self.calculate_rsi(closes, period)
                
                elif indicator_name == 'stochastic':
                    k_period = config.get('k_period', 14)
                    d_period = config.get('d_period', 3)
                    smooth_k = config.get('smooth_k', 1)
                    stoch_k, stoch_d = self.calculate_stochastic(highs, lows, closes, k_period, d_period, smooth_k)
                    result_df['stoch_k'] = stoch_k
                    result_df['stoch_d'] = stoch_d
                
                elif indicator_name == 'macd':
                    fast = config.get('fast_period', 12)
                    slow = config.get('slow_period', 26)
                    signal = config.get('signal_period', 9)
                    macd_results = self.calculate_macd(closes, fast, slow, signal)
                    for col_name, values in macd_results.items():
                        result_df[col_name] = values
                
                elif indicator_name == 'cci_20':
                    period = config.get('period', 20)
                    result_df['cci_20'] = self.calculate_cci(highs, lows, closes, period)
                
                elif indicator_name == 'atr_14':
                    period = config.get('period', 14)
                    atr_values = self.calculate_atr(highs, lows, closes, period)
                    result_df['atr_14'] = atr_values
                
                elif indicator_name == 'high_low_range':
                    result_df['high_low_range'] = self.calculate_high_low_range(highs, lows)
                
                elif indicator_name == 'body_to_range':
                    result_df['body_to_range'] = self.calculate_body_to_range(opens, highs, lows, closes)
                
                elif indicator_name == 'volatility_ratio':
                    if 'atr_14' in result_df.columns and 'ema_20' in result_df.columns:
                        result_df['volatility_ratio'] = self.calculate_volatility_ratio(
                            result_df['atr_14'].tolist(), result_df['ema_20'].tolist()
                        )
                
                elif indicator_name == 'bollinger_bands':
                    period = config.get('period', 20)
                    std_dev = config.get('std_dev', 2.0)
                    bb_results = self.calculate_bollinger_bands_extended(closes, period, std_dev)
                    for col_name, values in bb_results.items():
                        result_df[col_name] = values
                
                elif indicator_name in ['bb_width', 'bb_position', 'bb_touch']:
                    # These are calculated within bollinger_bands
                    pass
                
                elif indicator_name in ['volume_change', 'volume_sma_20', 'relative_volume']:
                    volume_results = self.calculate_volume_indicators(volumes)
                    for col_name, values in volume_results.items():
                        result_df[col_name] = values
                
                elif indicator_name in ['candle_type', 'candle_ratio']:
                    candle_results = self.calculate_candle_features(opens, highs, lows, closes)
                    for col_name, values in candle_results.items():
                        result_df[col_name] = values
                
                elif indicator_name == 'momentum_multiple':
                    periods = config.get('periods', [3, 10])
                    momentum_results = self.calculate_momentum_multiple(closes, periods)
                    for col_name, values in momentum_results.items():
                        result_df[col_name] = values
                
            except Exception as e:
                print(f"Error calculating {indicator_name}: {e}")
                continue
        
        # Apply output settings
        decimals = self.output_config.get('round_decimals', 6)
        if decimals is not None:
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            result_df[numeric_cols] = result_df[numeric_cols].round(decimals)
        
        return result_df


def calculate_indicators_for_file_advanced(input_file: str, output_file: str = None, config_path: str = None) -> str:
    """
    Calculate advanced indicators for a CSV file using configuration.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        config_path: Path to indicators configuration file (optional)
    
    Returns:
        Path to output file
    """
    if output_file is None:
        output_file = input_file.replace('.csv', '_advanced_indicators.csv')
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Calculate indicators
    calculator = AdvancedIndicators(config_path)
    df_with_indicators = calculator.calculate_all_indicators(df)
    
    # Save result
    df_with_indicators.to_csv(output_file, index=False)
    
    return output_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python advanced_indicators.py <input_file.csv> [output_file.csv] [config_file.yaml]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    config_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    result_file = calculate_indicators_for_file_advanced(input_file, output_file, config_file)
    print(f"Advanced indicators calculated and saved to: {result_file}")