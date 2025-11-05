#!/usr/bin/env python3
"""
Test the ML backtester with optimized model on test data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def create_test_data():
    """Create test data from the optimized dataset."""
    
    # Use the optimized dataset
    input_file = "data/processed/SOLUSDT_5m_complete_top25.csv"
    
    if not Path(input_file).exists():
        print(f"❌ Optimized dataset not found: {input_file}")
        return None
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"📊 Loaded {len(df)} rows from optimized dataset")
    
    # Take last 20% as test data (similar to train/test split)
    test_size = int(len(df) * 0.2)
    test_data = df.tail(test_size).copy()
    
    # Add target_10 for backtesting (horizon 10)
    test_data['target_10'] = (test_data['close'].shift(-10) > test_data['close']).astype(int)
    
    # Remove NaN rows
    test_data = test_data.dropna()
    
    # Save test data
    test_file = "data/processed/test_backtest.csv"
    test_data.to_csv(test_file, index=False)
    
    print(f"✅ Created test data: {test_file}")
    print(f"📈 Test data size: {len(test_data)} rows")
    print(f"📊 Price range: {test_data['close'].min():.4f} - {test_data['close'].max():.4f}")
    
    return test_file

def run_backtest_example():
    """Run backtester example with optimized model."""
    
    print("🧪 TESTING ML BACKTESTER")
    print("=" * 40)
    
    # Create test data
    test_file = create_test_data()
    if not test_file:
        return
    
    # Find optimized model
    model_file = "models/optimized/rf_horizon_10_optimized.pkl"
    
    if not Path(model_file).exists():
        print(f"❌ Model not found: {model_file}")
        print("   Run train_optimized_models.py first")
        return
    
    # Test different configurations
    test_configs = [
        {
            'name': 'Conservative',
            'buy_threshold': 0.7,
            'sell_threshold': 0.3,
            'fee': 0.001
        },
        {
            'name': 'Moderate', 
            'buy_threshold': 0.6,
            'sell_threshold': 0.4,
            'fee': 0.001
        },
        {
            'name': 'Aggressive',
            'buy_threshold': 0.55,
            'sell_threshold': 0.45,
            'fee': 0.001
        }
    ]
    
    print(f"\n🎯 Testing {len(test_configs)} configurations...")
    
    results = []
    
    for config in test_configs:
        print(f"\n📊 Testing {config['name']} strategy:")
        print(f"   Buy threshold: {config['buy_threshold']}")
        print(f"   Sell threshold: {config['sell_threshold']}")
        print(f"   Fee: {config['fee']:.1%}")
        
        # Run backtester
        import subprocess
        
        cmd = [
            'python3', 'ml_backtester.py',
            '--model', model_file,
            '--data', test_file,
            '--initial_balance', '1000',
            '--fee', str(config['fee']),
            '--buy_threshold', str(config['buy_threshold']),
            '--sell_threshold', str(config['sell_threshold']),
            '--output_dir', f"results/backtest_{config['name'].lower()}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"✅ {config['name']} completed successfully")
                
                # Extract key metrics from output
                output_lines = result.stdout.split('\n')
                metrics = {}
                
                for line in output_lines:
                    if 'Final balance:' in line:
                        metrics['final_balance'] = float(line.split('$')[1].replace(',', ''))
                    elif 'Total return:' in line:
                        metrics['total_return'] = line.split(':')[1].strip()
                    elif 'Win rate:' in line:
                        metrics['win_rate'] = line.split('(')[1].split(')')[0]
                    elif 'Total trades:' in line:
                        metrics['total_trades'] = int(line.split(':')[1].strip())
                
                results.append({
                    'config': config['name'],
                    **metrics
                })
                
            else:
                print(f"❌ {config['name']} failed:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {config['name']} timed out")
        except Exception as e:
            print(f"❌ Error running {config['name']}: {e}")
    
    # Summary
    if results:
        print(f"\n📊 BACKTEST COMPARISON")
        print("=" * 60)
        print(f"{'Strategy':<12} {'Final $':<10} {'Return':<10} {'Win Rate':<10} {'Trades':<8}")
        print("-" * 60)
        
        for result in results:
            final_balance = result.get('final_balance', 0)
            total_return = result.get('total_return', 'N/A')
            win_rate = result.get('win_rate', 'N/A')
            total_trades = result.get('total_trades', 0)
            
            print(f"{result['config']:<12} ${final_balance:<9.2f} {total_return:<10} {win_rate:<10} {total_trades:<8}")
        
        # Find best performer
        if len(results) > 1:
            best_result = max(results, key=lambda x: x.get('final_balance', 0))
            print(f"\n🏆 Best performer: {best_result['config']}")
    
    print(f"\n💾 Results saved in results/backtest_*/ folders")
    print(f"📄 Check backtest_trades.csv and backtest_equity_curve.csv for detailed analysis")

def main():
    """Main test function."""
    
    # Check if backtester exists
    if not Path("ml_backtester.py").exists():
        print("❌ ml_backtester.py not found")
        return
    
    # Run test
    run_backtest_example()

if __name__ == "__main__":
    main()