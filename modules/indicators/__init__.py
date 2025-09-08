"""
Technical Indicators Module

This module provides a comprehensive collection of technical analysis indicators
for trading systems. All indicators follow a consistent API and can be used both
for real-time updates and batch calculations.

Usage Examples:
    # Real-time indicator updates
    rsi_indicator = RSI(period=14)
    rsi_value = rsi_indicator.update(close_price)
    
    # Batch calculation
    rsi_values = rsi(price_data, period=14)
    
    # OHLCV indicators
    atr_indicator = ATR(period=14)
    atr_value = atr_indicator.update_ohlc(open, high, low, close, volume)
"""

from .base import (
    IndicatorBase,
    PriceBasedIndicator,
    validate_data,
    true_range,
    typical_price,
    weighted_close
)

# Moving Averages
from .moving_averages import (
    SMA, EMA, WMA, DEMA, TEMA, HMA, VWMA,
    sma, ema, wma
)

# Oscillators
from .oscillators import (
    RSI, MACD, StochasticOscillator, Williams_R, CCI,
    MomentumOscillator, ROC,
    rsi, macd
)

# Volatility Indicators
from .volatility import (
    ATR, BollingerBands, KeltnerChannels, DonchianChannels,
    StandardDeviation, HistoricalVolatility, VIX_Style_Volatility,
    PVOL, VolatilityRatio,
    atr, bollinger_bands, donchian_channels
)

# Volume Indicators
from .volume import (
    OBV, VWAP, PVI, NVI, AccumulationDistributionLine,
    ChaikinMoneyFlow, VolumeOscillator, PriceVolumeRank,
    VolumeWeightedMomentum,
    obv, vwap, accumulation_distribution_line
)

# Indicator categories for easy discovery
MOVING_AVERAGES = [
    'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'HMA', 'VWMA'
]

OSCILLATORS = [
    'RSI', 'MACD', 'StochasticOscillator', 'Williams_R', 'CCI',
    'MomentumOscillator', 'ROC'
]

VOLATILITY_INDICATORS = [
    'ATR', 'BollingerBands', 'KeltnerChannels', 'DonchianChannels',
    'StandardDeviation', 'HistoricalVolatility', 'VIX_Style_Volatility',
    'PVOL', 'VolatilityRatio'
]

VOLUME_INDICATORS = [
    'OBV', 'VWAP', 'PVI', 'NVI', 'AccumulationDistributionLine',
    'ChaikinMoneyFlow', 'VolumeOscillator', 'PriceVolumeRank',
    'VolumeWeightedMomentum'
]

ALL_INDICATORS = (
    MOVING_AVERAGES + 
    OSCILLATORS + 
    VOLATILITY_INDICATORS + 
    VOLUME_INDICATORS
)

# Convenience functions
FUNCTIONS = [
    'sma', 'ema', 'wma',
    'rsi', 'macd',
    'atr', 'bollinger_bands', 'donchian_channels',
    'obv', 'vwap', 'accumulation_distribution_line'
]

__version__ = "1.0.0"
__author__ = "Trading Bot System"

__all__ = [
    # Base classes
    'IndicatorBase', 'PriceBasedIndicator',
    
    # Utility functions
    'validate_data', 'true_range', 'typical_price', 'weighted_close',
    
    # Moving Averages
    'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'HMA', 'VWMA',
    'sma', 'ema', 'wma',
    
    # Oscillators
    'RSI', 'MACD', 'StochasticOscillator', 'Williams_R', 'CCI',
    'MomentumOscillator', 'ROC',
    'rsi', 'macd',
    
    # Volatility
    'ATR', 'BollingerBands', 'KeltnerChannels', 'DonchianChannels',
    'StandardDeviation', 'HistoricalVolatility', 'VIX_Style_Volatility',
    'PVOL', 'VolatilityRatio',
    'atr', 'bollinger_bands', 'donchian_channels',
    
    # Volume
    'OBV', 'VWAP', 'PVI', 'NVI', 'AccumulationDistributionLine',
    'ChaikinMoneyFlow', 'VolumeOscillator', 'PriceVolumeRank',
    'VolumeWeightedMomentum',
    'obv', 'vwap', 'accumulation_distribution_line',
    
    # Categories
    'MOVING_AVERAGES', 'OSCILLATORS', 'VOLATILITY_INDICATORS',
    'VOLUME_INDICATORS', 'ALL_INDICATORS', 'FUNCTIONS'
]


def get_indicator(name: str, **kwargs) -> IndicatorBase:
    """
    Factory function to create indicators by name.
    
    Args:
        name: Name of the indicator (case-insensitive)
        **kwargs: Parameters for the indicator
        
    Returns:
        Indicator instance
        
    Example:
        rsi_ind = get_indicator('RSI', period=14)
        sma_ind = get_indicator('sma', period=20)
    """
    name = name.upper()
    
    # Moving Averages
    if name == 'SMA':
        return SMA(**kwargs)
    elif name == 'EMA':
        return EMA(**kwargs)
    elif name == 'WMA':
        return WMA(**kwargs)
    elif name == 'DEMA':
        return DEMA(**kwargs)
    elif name == 'TEMA':
        return TEMA(**kwargs)
    elif name == 'HMA':
        return HMA(**kwargs)
    elif name == 'VWMA':
        return VWMA(**kwargs)
    
    # Oscillators
    elif name == 'RSI':
        return RSI(**kwargs)
    elif name == 'MACD':
        return MACD(**kwargs)
    elif name == 'STOCHASTIC':
        return StochasticOscillator(**kwargs)
    elif name == 'WILLIAMS_R':
        return Williams_R(**kwargs)
    elif name == 'CCI':
        return CCI(**kwargs)
    elif name == 'MOMENTUM':
        return MomentumOscillator(**kwargs)
    elif name == 'ROC':
        return ROC(**kwargs)
    
    # Volatility
    elif name == 'ATR':
        return ATR(**kwargs)
    elif name == 'BOLLINGER' or name == 'BB':
        return BollingerBands(**kwargs)
    elif name == 'KELTNER':
        return KeltnerChannels(**kwargs)
    elif name == 'DONCHIAN':
        return DonchianChannels(**kwargs)
    elif name == 'STDDEV':
        return StandardDeviation(**kwargs)
    
    # Volume
    elif name == 'OBV':
        return OBV(**kwargs)
    elif name == 'VWAP':
        return VWAP(**kwargs)
    elif name == 'PVI':
        return PVI(**kwargs)
    elif name == 'NVI':
        return NVI(**kwargs)
    elif name == 'ADL':
        return AccumulationDistributionLine(**kwargs)
    elif name == 'CMF':
        return ChaikinMoneyFlow(**kwargs)
    elif name == 'VOLUME_OSC':
        return VolumeOscillator(**kwargs)
    
    else:
        raise ValueError(f"Unknown indicator: {name}. Available: {', '.join(ALL_INDICATORS)}")


def list_indicators():
    """Print all available indicators organized by category."""
    print("Available Technical Indicators:")
    print("\nMoving Averages:")
    for indicator in MOVING_AVERAGES:
        print(f"  - {indicator}")
    
    print("\nOscillators:")
    for indicator in OSCILLATORS:
        print(f"  - {indicator}")
    
    print("\nVolatility Indicators:")
    for indicator in VOLATILITY_INDICATORS:
        print(f"  - {indicator}")
    
    print("\nVolume Indicators:")
    for indicator in VOLUME_INDICATORS:
        print(f"  - {indicator}")
    
    print(f"\nTotal: {len(ALL_INDICATORS)} indicators")
    print(f"Functions: {len(FUNCTIONS)} batch calculation functions")


# Module-level convenience for common indicators
def create_rsi(period: int = 14) -> RSI:
    """Create RSI indicator with specified period."""
    return RSI(period=period)

def create_macd(fast: int = 12, slow: int = 26, signal: int = 9) -> MACD:
    """Create MACD indicator with specified periods."""
    return MACD(fast_period=fast, slow_period=slow, signal_period=signal)

def create_bollinger_bands(period: int = 20, std_dev: float = 2.0) -> BollingerBands:
    """Create Bollinger Bands indicator with specified parameters."""
    return BollingerBands(period=period, std_dev=std_dev)

def create_atr(period: int = 14) -> ATR:
    """Create ATR indicator with specified period."""
    return ATR(period=period)