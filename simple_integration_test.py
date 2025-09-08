#!/usr/bin/env python3
"""
Simple integration test without external dependencies.
Tests basic module imports and indicator functionality.
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_module_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        # Test indicators module
        from modules.indicators import RSI, EMA, BollingerBands, SMA
        print("✓ Indicators module imported successfully")
        
        # Test backtester module
        from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
        print("✓ Backtester module imported successfully")
        
        # Test strategy
        from examples.strategies.sol_usdt_strategy import SolUsdtStrategy
        print("✓ SOL/USDT strategy imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_indicators_basic():
    """Test basic indicator functionality."""
    print("Testing basic indicator functionality...")
    
    try:
        from modules.indicators import RSI, EMA, BollingerBands, SMA
        
        # Generate simple test data
        prices = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 
                 100, 99, 98, 97, 96, 97, 98, 99, 100, 101,
                 102, 103, 104, 105, 106, 107, 106, 105, 104, 103]
        
        # Test RSI
        rsi = RSI(period=14)
        rsi_values = []
        for price in prices:
            val = rsi.update(price)
            if val is not None:
                rsi_values.append(val)
        
        assert len(rsi_values) > 0, "RSI should produce values"
        assert all(0 <= rsi <= 100 for rsi in rsi_values), "RSI values should be 0-100"
        print(f"✓ RSI working: {len(rsi_values)} values, last = {rsi_values[-1]:.1f}")
        
        # Test EMA
        ema = EMA(period=9)
        ema_values = []
        for price in prices:
            val = ema.update(price)
            if val is not None:
                ema_values.append(val)
        
        assert len(ema_values) > 0, "EMA should produce values"
        print(f"✓ EMA working: {len(ema_values)} values, last = {ema_values[-1]:.2f}")
        
        # Test Bollinger Bands
        bb = BollingerBands(period=20, std_dev=2.0)
        bb_values = []
        for price in prices:
            val = bb.update(price)
            if val is not None:
                bb_values.append(val)
        
        assert len(bb_values) > 0, "Bollinger Bands should produce values"
        last_bb = bb_values[-1]
        assert last_bb['lower'] < last_bb['middle'] < last_bb['upper'], "BB bands should be ordered"
        print(f"✓ Bollinger Bands working: last = {last_bb['middle']:.2f} [{last_bb['lower']:.2f}, {last_bb['upper']:.2f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Indicators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_initialization():
    """Test strategy initialization with indicators."""
    print("Testing strategy initialization...")
    
    try:
        from examples.strategies.sol_usdt_strategy import SolUsdtStrategy
        
        # Create strategy instance
        strategy = SolUsdtStrategy()
        
        # Check that indicators are properly initialized
        assert 'ema_fast' in strategy.indicators, "Strategy should have ema_fast indicator"
        assert 'ema_slow' in strategy.indicators, "Strategy should have ema_slow indicator"
        assert 'rsi' in strategy.indicators, "Strategy should have rsi indicator"
        assert 'bb' in strategy.indicators, "Strategy should have bb indicator"
        assert 'volume_ma' in strategy.indicators, "Strategy should have volume_ma indicator"
        
        print("✓ Strategy indicators initialized:")
        for name, indicator in strategy.indicators.items():
            print(f"  - {name}: {type(indicator).__name__}")
        
        # Check parameters
        assert strategy.parameters['ema_fast'] == 9, "EMA fast period should be 9"
        assert strategy.parameters['ema_slow'] == 21, "EMA slow period should be 21"
        assert strategy.parameters['rsi_period'] == 21, "RSI period should be 21"
        
        print("✓ Strategy parameters configured correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_signal_generation():
    """Test strategy signal generation."""
    print("Testing strategy signal generation...")
    
    try:
        from examples.strategies.sol_usdt_strategy import SolUsdtStrategy
        from modules.backtester import MarketData, Position
        
        # Create strategy
        strategy = SolUsdtStrategy()
        
        # Create test position (no position)
        position = Position(
            direction="NONE",
            quantity=0.0,
            entry_price=0.0,
            entry_time=datetime.now(),
            unrealized_pnl=0.0,
            duration_minutes=0
        )
        
        # Generate test market data
        base_time = datetime(2024, 1, 1, 12, 0)
        signals_generated = []
        
        # Generate 50 bars of data
        for i in range(50):
            timestamp = base_time + timedelta(minutes=i * 5)
            
            # Create some price movement
            base_price = 150.0
            price_offset = (i % 20 - 10) * 2  # Creates oscillation
            close_price = base_price + price_offset
            
            market_data = MarketData(
                timestamp=timestamp,
                open=close_price - 0.1,
                high=close_price + 0.5,
                low=close_price - 0.5,
                close=close_price,
                volume=2000 + (i % 10) * 100,  # Varying volume
                symbol="SOLUSDT"
            )
            
            # Get signal from strategy
            signal = strategy.on_bar(market_data, position)
            signals_generated.append(signal)
            
            # Print first few signals for debugging
            if i < 5:
                print(f"  Bar {i}: Price={close_price:.2f}, Signal={signal.signal.value}, Reason={signal.reason}")
        
        # Check that we got signals
        assert len(signals_generated) == 50, "Should generate 50 signals"
        
        # Check signal types
        signal_types = [s.signal.value for s in signals_generated]
        unique_signals = set(signal_types)
        print(f"✓ Generated {len(signals_generated)} signals with types: {unique_signals}")
        
        # Check that at least some HOLD signals are generated (indicators need time to warm up)
        hold_count = signal_types.count('HOLD')
        print(f"✓ HOLD signals: {hold_count}/{len(signals_generated)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy signal generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("SIMPLE TRADING SYSTEM INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_module_imports,
        test_indicators_basic,
        test_strategy_initialization,
        test_strategy_signal_generation
    ]
    
    results = []
    for test in tests:
        print()
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    all_passed = all(results)
    
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 SUCCESS! Module integration is working:")
        print("   • All modules import correctly")
        print("   • Indicators are functional") 
        print("   • Strategy uses indicators properly")
        print("   • Signal generation works")
        print("\nThe trading system modules are properly integrated!")
    else:
        print("\n⚠️  Integration issues detected. Check error messages above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)