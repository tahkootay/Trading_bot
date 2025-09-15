#!/usr/bin/env python3
"""
Create compact strategy comparison CSV with only essential metrics.
"""

import pandas as pd

def create_compact_comparison():
    """Create a compact version with only key metrics."""
    
    # Load full comparison data
    full_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/strategy_comparison.csv'
    df_full = pd.read_csv(full_file)
    
    # Select only essential columns for comparison
    compact_columns = [
        # Strategy identification
        'strategy_name', 'test_date',
        
        # Key parameters (most likely to change)
        'macd_fast', 'macd_slow', 'macd_signal', 
        'rsi_period', 'ema_period',
        'take_profit_pct', 'stop_loss_pct',
        
        # Performance results
        'total_return_pct', 'win_rate_pct', 'total_trades',
        'profit_factor', 'sharpe_ratio', 'max_drawdown_pct',
        
        # Signal efficiency
        'signal_filter_efficiency_pct',
        
        # Notes
        'notes'
    ]
    
    # Create compact DataFrame
    df_compact = df_full[compact_columns].copy()
    
    # Save compact version
    compact_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/strategy_comparison_compact.csv'
    df_compact.to_csv(compact_file, index=False)
    
    print(f"📄 Created compact comparison file: {compact_file}")
    print(f"📊 Columns: {len(compact_columns)} (vs {len(df_full.columns)} in full version)")
    
    # Display compact table
    print(f"\n📋 Compact Strategy Comparison:")
    print("=" * 120)
    
    # Show in readable format
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_compact.round(2).to_string(index=False))
    
    return compact_file

def main():
    """Main execution."""
    
    print("🚀 Creating compact strategy comparison...")
    
    compact_file = create_compact_comparison()
    
    print(f"\n✅ Compact comparison ready!")
    print(f"📂 File: {compact_file}")
    print(f"💡 Use this file for quick parameter comparison")
    print(f"📊 When testing new parameters, run create_strategy_comparison.py to add them")

if __name__ == "__main__":
    main()