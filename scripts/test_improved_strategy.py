#!/usr/bin/env python3
"""Test improved strategy with relaxed parameters."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.enhanced_backtest import EnhancedBacktestEngine

def test_improved_parameters():
    """Test backtest with improved parameters."""
    print("🔧 Testing Improved Strategy Parameters")
    print("=" * 50)
    
    # Create backtest engine
    engine = EnhancedBacktestEngine(initial_balance=10000)
    
    # Update strategy parameters
    improved_params = {
        'min_confidence': 0.10,      # Lowered from 0.15
        'min_volume_ratio': 1.0,     # Lowered from 1.2  
        'min_adx': 15.0,             # Lowered from 20.0
        'atr_sl_multiplier': 1.2,
        'atr_tp_multiplier': 2.0,
        'max_position_time_hours': 4,
        'position_size_pct': 0.02,
    }
    
    engine.strategy_params.update(improved_params)
    
    print("📋 Updated Strategy Parameters:")
    for key, value in improved_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Run backtest
    try:
        data = engine.load_data_from_block("august_12_single_day")
        results = engine.run_backtest(symbol="SOLUSDT", data=data)
        
        print("📊 IMPROVED STRATEGY RESULTS")
        print("-" * 40)
        
        if "error" in results:
            print(f"❌ {results['error']}")
        else:
            print(f"✅ Total trades: {results['total_trades']}")
            print(f"💰 Total P&L: ${results['total_pnl']:.2f}")
            print(f"📈 Final balance: ${results['final_balance']:.2f}")
            print(f"📊 Return: {results['total_return_pct']:.2f}%")
            print(f"🎯 Win rate: {results['win_rate']:.1%}")
            print(f"📉 Max drawdown: {results['max_drawdown']:.2%}")
            
            print("\n🔍 Top Trades:")
            for i, trade in enumerate(engine.trades[:5], 1):
                pnl = trade.get('net_pnl', trade.get('pnl', 0))
                print(f"  {i}. {trade['signal_type'].value if hasattr(trade['signal_type'], 'value') else trade['signal_type']} @ "
                      f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
                      f"P&L: ${pnl:.2f} | Reason: {trade['exit_reason']}")
            
        # Save results
        if "error" not in results and results.get('total_trades', 0) > 0:
            import json
            from datetime import datetime
            
            output_file = f"Output/improved_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                # Convert non-serializable objects
                serializable_results = {}
                for key, value in results.items():
                    if key == 'trades':
                        serializable_results[key] = [
                            {k: (v.value if hasattr(v, 'value') else v) for k, v in trade.items()}
                            for trade in value
                        ]
                    else:
                        serializable_results[key] = value
                        
                json.dump(serializable_results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")
            
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_parameters()