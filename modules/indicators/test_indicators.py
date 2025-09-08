"""Comprehensive tests for technical indicators module."""

import pytest
import numpy as np
from typing import List

# Import all indicators for testing
from . import (
    SMA, EMA, WMA, RSI, MACD, ATR, BollingerBands, 
    OBV, VWAP, StochasticOscillator, Williams_R,
    sma, ema, rsi, macd, bollinger_bands, atr
)


class TestDataGenerator:
    """Generate test data for indicators."""
    
    @staticmethod
    def generate_price_series(length: int = 100, start: float = 100.0, volatility: float = 0.02) -> List[float]:
        """Generate random price series."""
        np.random.seed(42)  # For reproducible tests
        prices = [start]
        
        for _ in range(length - 1):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        return prices
    
    @staticmethod
    def generate_ohlcv_data(length: int = 100) -> dict:
        """Generate OHLCV data for testing."""
        closes = TestDataGenerator.generate_price_series(length)
        opens = [c * np.random.uniform(0.99, 1.01) for c in closes]
        
        ohlcv = {
            'open': [],
            'high': [],
            'low': [],
            'close': closes,
            'volume': []
        }
        
        for i, (o, c) in enumerate(zip(opens, closes)):
            # Generate high and low
            high = max(o, c) * np.random.uniform(1.0, 1.02)
            low = min(o, c) * np.random.uniform(0.98, 1.0)
            
            # Generate volume
            volume = np.random.uniform(1000, 10000)
            
            ohlcv['open'].append(o)
            ohlcv['high'].append(high)
            ohlcv['low'].append(low)
            ohlcv['volume'].append(volume)
        
        return ohlcv


class TestMovingAverages:
    """Test moving average indicators."""
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Test with period 3
        sma_indicator = SMA(period=3)
        
        # First two values should return None
        assert sma_indicator.update(1) is None
        assert sma_indicator.update(2) is None
        
        # Third value should return SMA
        result = sma_indicator.update(3)
        assert result == 2.0  # (1+2+3)/3
        
        result = sma_indicator.update(4)
        assert result == 3.0  # (2+3+4)/3
        
        # Test batch function
        sma_values = sma(prices, period=3)
        assert np.isnan(sma_values[0])
        assert np.isnan(sma_values[1])
        assert sma_values[2] == 2.0
        assert sma_values[3] == 3.0
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        prices = [2, 4, 6, 8, 12, 14, 16, 18, 20]
        
        ema_indicator = EMA(period=3)
        
        # Test real-time updates
        results = []
        for price in prices:
            result = ema_indicator.update(price)
            if result is not None:
                results.append(result)
        
        assert len(results) > 0
        assert all(r > 0 for r in results)
        
        # Test batch function
        ema_values = ema(prices, period=3)
        assert not np.isnan(ema_values[-1])
    
    def test_wma_calculation(self):
        """Test Weighted Moving Average calculation."""
        prices = [1, 2, 3, 4, 5]
        
        wma_indicator = WMA(period=3)
        
        # Should return None until enough data
        assert wma_indicator.update(1) is None
        assert wma_indicator.update(2) is None
        
        # Test calculation: (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6 ≈ 2.33
        result = wma_indicator.update(3)
        expected = (1*1 + 2*2 + 3*3) / (1+2+3)
        assert abs(result - expected) < 0.01


class TestOscillators:
    """Test oscillator indicators."""
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        # Test with known values that should produce specific RSI
        prices = TestDataGenerator.generate_price_series(50)
        
        rsi_indicator = RSI(period=14)
        
        # Update with all prices
        rsi_values = []
        for price in prices:
            result = rsi_indicator.update(price)
            if result is not None:
                rsi_values.append(result)
        
        # RSI should be between 0 and 100
        assert all(0 <= rsi <= 100 for rsi in rsi_values)
        
        # Test batch function
        rsi_batch = rsi(prices, period=14)
        valid_rsi = rsi_batch[~np.isnan(rsi_batch)]
        assert len(valid_rsi) > 0
        assert all(0 <= r <= 100 for r in valid_rsi)
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        prices = TestDataGenerator.generate_price_series(100)
        
        macd_indicator = MACD(fast_period=12, slow_period=26, signal_period=9)
        
        # Update with prices
        macd_values = []
        for price in prices:
            result = macd_indicator.update(price)
            if result is not None and 'macd' in result:
                macd_values.append(result)
        
        assert len(macd_values) > 0
        
        # Check that all required fields are present
        last_macd = macd_values[-1]
        assert 'macd' in last_macd
        assert 'signal' in last_macd
        assert 'histogram' in last_macd
        
        # Test batch function
        macd_batch = macd(prices, fast=12, slow=26, signal=9)
        assert 'macd' in macd_batch
        assert 'signal' in macd_batch
        assert 'histogram' in macd_batch
    
    def test_stochastic_calculation(self):
        """Test Stochastic Oscillator calculation."""
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(50)
        
        stoch_indicator = StochasticOscillator(k_period=14, d_period=3)
        
        # Update with OHLCV data
        stoch_values = []
        for i in range(len(ohlcv_data['close'])):
            result = stoch_indicator.update_ohlc(
                ohlcv_data['open'][i],
                ohlcv_data['high'][i],
                ohlcv_data['low'][i],
                ohlcv_data['close'][i],
                ohlcv_data['volume'][i]
            )
            if result is not None:
                stoch_values.append(result)
        
        # Stochastic %K and %D should be between 0 and 100
        for result in stoch_values:
            assert 0 <= result['k'] <= 100
            assert 0 <= result['d'] <= 100


class TestVolatilityIndicators:
    """Test volatility indicators."""
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(50)
        
        atr_indicator = ATR(period=14)
        
        # Update with OHLCV data
        atr_values = []
        for i in range(len(ohlcv_data['close'])):
            result = atr_indicator.update_ohlc(
                ohlcv_data['open'][i],
                ohlcv_data['high'][i],
                ohlcv_data['low'][i],
                ohlcv_data['close'][i],
                ohlcv_data['volume'][i]
            )
            if result is not None:
                atr_values.append(result)
        
        # ATR should be positive
        assert all(atr > 0 for atr in atr_values)
        
        # Test batch function
        atr_batch = atr(
            ohlcv_data['high'],
            ohlcv_data['low'],
            ohlcv_data['close'],
            period=14
        )
        valid_atr = atr_batch[~np.isnan(atr_batch)]
        assert len(valid_atr) > 0
        assert all(a > 0 for a in valid_atr)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        prices = TestDataGenerator.generate_price_series(50)
        
        bb_indicator = BollingerBands(period=20, std_dev=2.0)
        
        # Update with prices
        bb_values = []
        for price in prices:
            result = bb_indicator.update(price)
            if result is not None:
                bb_values.append(result)
        
        # Check band relationships
        for bands in bb_values:
            assert bands['lower'] < bands['middle'] < bands['upper']
        
        # Test batch function
        bb_batch = bollinger_bands(prices, period=20, std_dev=2.0)
        valid_indices = ~np.isnan(bb_batch['middle'])
        
        for i in np.where(valid_indices)[0]:
            assert bb_batch['lower'][i] < bb_batch['middle'][i] < bb_batch['upper'][i]


class TestVolumeIndicators:
    """Test volume indicators."""
    
    def test_obv_calculation(self):
        """Test On-Balance Volume calculation."""
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(30)
        
        obv_indicator = OBV()
        
        # Update with price and volume
        obv_values = []
        for i in range(len(ohlcv_data['close'])):
            result = obv_indicator.update_price_volume(
                ohlcv_data['close'][i],
                ohlcv_data['volume'][i]
            )
            obv_values.append(result)
        
        # OBV should change with price movements
        assert len(set(obv_values)) > 1  # Should have different values
    
    def test_vwap_calculation(self):
        """Test VWAP calculation."""
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(30)
        
        # Test cumulative VWAP
        vwap_indicator = VWAP()
        
        vwap_values = []
        for i in range(len(ohlcv_data['close'])):
            result = vwap_indicator.update_price_volume(
                ohlcv_data['close'][i],
                ohlcv_data['volume'][i]
            )
            if result is not None:
                vwap_values.append(result)
        
        # VWAP should be positive
        assert all(vwap > 0 for vwap in vwap_values)
        
        # Test rolling VWAP
        vwap_rolling = VWAP(period=10)
        rolling_values = []
        for i in range(len(ohlcv_data['close'])):
            result = vwap_rolling.update_price_volume(
                ohlcv_data['close'][i],
                ohlcv_data['volume'][i]
            )
            if result is not None:
                rolling_values.append(result)
        
        assert all(vwap > 0 for vwap in rolling_values)


class TestIndicatorBase:
    """Test base indicator functionality."""
    
    def test_indicator_state_management(self):
        """Test indicator state management."""
        sma_indicator = SMA(period=5)
        
        # Test is_ready method
        assert not sma_indicator.is_ready()
        
        for i in range(5):
            sma_indicator.update(i + 1)
        
        assert sma_indicator.is_ready()
        
        # Test reset
        sma_indicator.reset()
        assert not sma_indicator.is_ready()
        assert len(sma_indicator.values) == 0
    
    def test_data_validation(self):
        """Test data validation functions."""
        from . import validate_data
        
        # Test valid data
        valid_data = [1, 2, 3, 4, 5]
        result = validate_data(valid_data, min_length=3)
        assert len(result) == 5
        
        # Test insufficient data
        with pytest.raises(ValueError):
            validate_data([1, 2], min_length=5)
        
        # Test NaN data
        with pytest.raises(ValueError):
            validate_data([1, 2, np.nan, 4, 5], min_length=3)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_volume(self):
        """Test indicators with zero volume."""
        vwap_indicator = VWAP(period=5)
        
        # Update with zero volume
        result = vwap_indicator.update_price_volume(100.0, 0.0)
        
        # Should handle gracefully
        assert result is None or result >= 0
    
    def test_identical_prices(self):
        """Test indicators with identical prices."""
        prices = [100.0] * 20
        
        # RSI with no price movement should be around 50
        rsi_values = rsi(prices, period=14)
        valid_rsi = rsi_values[~np.isnan(rsi_values)]
        
        # Should be close to 50 (neutral)
        if len(valid_rsi) > 0:
            assert abs(valid_rsi[-1] - 50.0) < 10.0
    
    def test_extreme_price_movements(self):
        """Test indicators with extreme price movements."""
        # Create series with extreme movements
        prices = [100, 200, 50, 300, 25, 400, 10]
        
        sma_indicator = SMA(period=3)
        results = []
        
        for price in prices:
            result = sma_indicator.update(price)
            if result is not None:
                results.append(result)
        
        # Should handle extreme movements without errors
        assert len(results) > 0
        assert all(r > 0 for r in results)


def run_performance_test():
    """Run performance test on indicators."""
    import time
    
    # Generate large dataset
    prices = TestDataGenerator.generate_price_series(10000)
    
    # Test EMA performance
    start_time = time.time()
    ema_values = ema(prices, period=20)
    ema_time = time.time() - start_time
    
    # Test RSI performance
    start_time = time.time()
    rsi_values = rsi(prices, period=14)
    rsi_time = time.time() - start_time
    
    print(f"Performance test with 10,000 data points:")
    print(f"EMA calculation: {ema_time:.4f} seconds")
    print(f"RSI calculation: {rsi_time:.4f} seconds")
    
    assert ema_time < 1.0  # Should complete in reasonable time
    assert rsi_time < 1.0


if __name__ == "__main__":
    # Run basic tests
    print("Running indicator tests...")
    
    # Test data generator
    test_data = TestDataGenerator()
    prices = test_data.generate_price_series(50)
    print(f"Generated {len(prices)} price points")
    
    # Test moving averages
    ma_tests = TestMovingAverages()
    ma_tests.test_sma_calculation()
    ma_tests.test_ema_calculation()
    print("Moving average tests passed ✓")
    
    # Test oscillators
    osc_tests = TestOscillators()
    osc_tests.test_rsi_calculation()
    osc_tests.test_macd_calculation()
    print("Oscillator tests passed ✓")
    
    # Test volatility indicators
    vol_tests = TestVolatilityIndicators()
    vol_tests.test_bollinger_bands_calculation()
    print("Volatility indicator tests passed ✓")
    
    # Test volume indicators
    vol_tests = TestVolumeIndicators()
    vol_tests.test_vwap_calculation()
    print("Volume indicator tests passed ✓")
    
    # Run performance test
    run_performance_test()
    
    print("\nAll tests completed successfully! ✓")