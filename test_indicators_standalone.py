#!/usr/bin/env python3
"""
Test indicators standalone to see when they become ready
"""

import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, '.')

def test_indicators():
    """Test when indicators become ready with real data."""
    
    try:
        from modules.indicators import RSI, EMA, BollingerBands, SMA
        
        # Load real data
        df = pd.read_csv('./data/raw/SOLUSDT_5m_20250829_20250905.csv')
        print(f"Loaded {len(df)} bars of data")
        
        # Create indicators
        indicators = {
            'ema_fast': EMA(period=9),
            'ema_slow': EMA(period=21),
            'rsi': RSI(period=21),
            'bb': BollingerBands(period=20, std_dev=2.0),
            'volume_ma': SMA(period=20)
        }
        
        print("\nTesting indicator readiness...")
        
        for i, row in df.iterrows():
            price = row['close']
            volume = row['volume']
            
            # Update indicators
            results = {}
            results['ema_fast'] = indicators['ema_fast'].update(price)
            results['ema_slow'] = indicators['ema_slow'].update(price)
            results['rsi'] = indicators['rsi'].update(price)
            results['bb'] = indicators['bb'].update(price)
            results['volume_ma'] = indicators['volume_ma'].update(volume)
            
            # Check readiness
            ready_status = {name: ind.is_ready() for name, ind in indicators.items()}
            all_ready = all(ready_status.values())
            
            if i < 30 or i % 50 == 0 or all_ready:
                print(f"\nBar {i+1}: Price=${price:.2f}, Volume={volume:.0f}")
                print(f"Ready status: {ready_status}")
                
                if all_ready:
                    print("VALUES:")
                    print(f"  EMA Fast: {results['ema_fast']:.2f}")
                    print(f"  EMA Slow: {results['ema_slow']:.2f}")
                    print(f"  RSI: {results['rsi']:.1f}")
                    if results['bb']:
                        print(f"  BB: {results['bb']['lower']:.2f} | {results['bb']['middle']:.2f} | {results['bb']['upper']:.2f}")
                    print(f"  Volume MA: {results['volume_ma']:.0f}")
                    
                    # Test entry conditions with first ready values
                    if i < 100:  # Only for first 100 bars
                        test_entry_conditions(price, volume, results)
                    
                    if i == 50:  # Stop detailed logging after first ready + few bars
                        print(f"\n✅ All indicators ready at bar {i+1}. Stopping detailed logs...")
                        break
        
        print(f"\n✅ Indicator test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing indicators: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_entry_conditions(price, volume, indicator_results):
    """Test entry conditions with current values."""
    
    # Strategy parameters
    rsi_oversold = 25
    rsi_overbought = 75
    volume_spike_multiplier = 1.5
    ema_crossover_threshold = 0.002
    
    if not indicator_results['bb'] or not indicator_results['volume_ma']:
        return
    
    # Calculate conditions
    volume_spike = volume > indicator_results['volume_ma'] * volume_spike_multiplier
    ema_ratio = indicator_results['ema_fast'] / indicator_results['ema_slow']
    
    # LONG conditions
    long_bb = price <= indicator_results['bb']['lower']
    long_rsi = indicator_results['rsi'] <= rsi_oversold
    long_ema = ema_ratio > (1 - ema_crossover_threshold)
    
    # SHORT conditions
    short_bb = price >= indicator_results['bb']['upper']
    short_rsi = indicator_results['rsi'] >= rsi_overbought
    short_ema = ema_ratio < (1 + ema_crossover_threshold)
    
    long_score = sum([long_bb, long_rsi, long_ema, volume_spike])
    short_score = sum([short_bb, short_rsi, short_ema, volume_spike])
    
    if long_score >= 2 or short_score >= 2:
        print(f"  🔥 POTENTIAL SIGNALS:")
        print(f"    LONG score: {long_score}/4 (BB:{long_bb}, RSI:{long_rsi}, EMA:{long_ema}, Vol:{volume_spike})")
        print(f"    SHORT score: {short_score}/4 (BB:{short_bb}, RSI:{short_rsi}, EMA:{short_ema}, Vol:{volume_spike})")
        print(f"    RSI: {indicator_results['rsi']:.1f}, EMA ratio: {ema_ratio:.6f}")
        print(f"    BB: {indicator_results['bb']['lower']:.2f} <= {price:.2f} <= {indicator_results['bb']['upper']:.2f}")
        print(f"    Volume: {volume:.0f} vs MA {indicator_results['volume_ma']:.0f} (ratio: {volume/indicator_results['volume_ma']:.2f})")

if __name__ == "__main__":
    test_indicators()