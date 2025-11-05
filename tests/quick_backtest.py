#!/usr/bin/env python3
"""
Quick backtest demo with smaller dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_small_test_data():
    """Create smaller test dataset for quick testing."""
    
    # Use existing test data but make it smaller
    input_file = "data/processed/test_backtest.csv"
    
    if not Path(input_file).exists():
        print("❌ Test data not found, creating...")
        
        # Use optimized dataset
        full_data = "data/processed/SOLUSDT_5m_complete_top25.csv"
        if not Path(full_data).exists():
            print(f"❌ No data found: {full_data}")
            return None
        
        df = pd.read_csv(full_data)
        # Take last 1000 rows for quick test
        test_data = df.tail(1000).copy()
        test_data['target_10'] = (test_data['close'].shift(-10) > test_data['close']).astype(int)
        test_data = test_data.dropna()
        test_data.to_csv(input_file, index=False)
        print(f"✅ Created quick test data: {len(test_data)} rows")
    
    else:
        # Make existing test data smaller
        df = pd.read_csv(input_file)
        small_df = df.head(2000).copy()  # Just 2000 rows for quick test
        
        small_file = "data/processed/test_small.csv"
        small_df.to_csv(small_file, index=False)
        print(f"✅ Created small test data: {len(small_df)} rows")
        return small_file
    
    return input_file

def run_quick_test():
    """Run a quick backtester test."""
    
    print("⚡ QUICK BACKTEST TEST")
    print("=" * 30)
    
    # Create small test data
    test_file = create_small_test_data()
    if not test_file:
        return
    
    # Check model exists
    model_file = "models/optimized/rf_horizon_10_optimized.pkl"
    if not Path(model_file).exists():
        print(f"❌ Model not found: {model_file}")
        return
    
    # Run backtester with small data
    import subprocess
    
    cmd = [
        'python3', 'ml_backtester.py',
        '--model', model_file,
        '--data', test_file,
        '--initial_balance', '1000',
        '--fee', '0.001',
        '--buy_threshold', '0.6',
        '--sell_threshold', '0.4'
    ]
    
    print("🚀 Running quick backtest...")
    print(f"📊 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Quick backtest completed!")
        else:
            print(f"❌ Backtest failed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Quick backtest timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_quick_test()