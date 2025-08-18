#!/usr/bin/env python3
"""Multi-strategy backtesting with strategy selection support."""

import sys
import click
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.enhanced_backtest import EnhancedBacktestEngine
from scripts.final_aggressive_strategy import FinalAggressiveStrategy

AVAILABLE_STRATEGIES = {
    'conservative': {
        'class': EnhancedBacktestEngine,
        'description': 'Conservative strategy with strict filters (original)',
        'params': {
            'min_confidence': 0.15,
            'min_volume_ratio': 1.2,
            'min_adx': 20.0,
        }
    },
    'aggressive': {
        'class': FinalAggressiveStrategy,
        'description': 'Aggressive momentum strategy with relaxed filters',
        'params': {
            'min_confidence': 0.03,
            'min_volume_ratio': 0.3,
            'min_adx': 5.0,
            'immediate_entry_threshold': 0.006,
        }
    }
}

@click.command()
@click.option('--strategy', 
              type=click.Choice(list(AVAILABLE_STRATEGIES.keys())), 
              default='aggressive',
              help='Strategy to use for backtesting')
@click.option('--block-id', 
              default='august_12_single_day',
              help='Data block to use for testing')
@click.option('--balance', 
              type=float, 
              default=10000.0,
              help='Initial balance')
@click.option('--save-results', 
              is_flag=True,
              help='Save results to JSON file')
@click.option('--list-strategies', 
              is_flag=True,
              help='List available strategies')
def main(strategy, block_id, balance, save_results, list_strategies):
    """Multi-strategy backtesting tool."""
    
    if list_strategies:
        print("📋 Available Strategies:")
        print("=" * 60)
        for name, config in AVAILABLE_STRATEGIES.items():
            print(f"🎯 {name}:")
            print(f"   Description: {config['description']}")
            print(f"   Parameters: {config['params']}")
            print()
        return
    
    print(f"🧪 Multi-Strategy Backtest")
    print(f"📊 Strategy: {strategy}")
    print(f"📦 Block: {block_id}")
    print(f"💰 Balance: ${balance:,.2f}")
    print("=" * 50)
    
    # Get strategy configuration
    strategy_config = AVAILABLE_STRATEGIES[strategy]
    strategy_class = strategy_config['class']
    
    # Initialize strategy
    if strategy == 'aggressive':
        # Use special method for aggressive strategy
        engine = strategy_class(initial_balance=balance)
        
        print("🚀 Running aggressive momentum strategy...")
        results = engine.run_enhanced_backtest_with_momentum(block_id)
        
        # Print results
        if results['trades']:
            print(f"\n✅ Strategy completed successfully!")
            print(f"📊 Total trades: {len(results['trades'])}")
            print(f"📶 Signals generated: {results['signals_generated']}")
            print(f"💰 Final balance: ${results['final_balance']:.2f}")
            print(f"📈 Return: {results['total_return']:+.2f}%")
            
            if save_results:
                import json
                output_file = f"Output/multi_strategy_{strategy}_{block_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Results saved to: {output_file}")
        else:
            print("❌ No trades executed")
    
    else:
        # Use standard backtest for conservative strategy
        engine = strategy_class(initial_balance=balance)
        
        # Override strategy parameters
        engine.strategy_params.update(strategy_config['params'])
        
        print("🔄 Running conservative strategy...")
        data = engine.load_data_from_block(block_id)
        results = engine.run_backtest(symbol="SOLUSDT", data=data)
        
        # Print results
        if "error" in results:
            print(f"❌ {results['error']}")
        else:
            print(f"✅ Strategy completed!")
            print(f"📊 Results: {results}")
        
        if save_results:
            import json
            output_file = f"Output/multi_strategy_{strategy}_{block_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()