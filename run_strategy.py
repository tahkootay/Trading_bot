#!/usr/bin/env python3
"""Simple wrapper for running different trading strategies."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Trading Strategy Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run aggressive strategy (recommended)
  python run_strategy.py --strategy aggressive
  
  # Run conservative strategy
  python run_strategy.py --strategy conservative
  
  # Test on different blocks
  python run_strategy.py --strategy aggressive --block august_10_17_full
  
  # Save results
  python run_strategy.py --strategy aggressive --save-results
  
  # List available options
  python run_strategy.py --list
        """
    )
    
    parser.add_argument('--strategy', 
                       choices=['conservative', 'aggressive'], 
                       default='aggressive',
                       help='Strategy to run (default: aggressive)')
    
    parser.add_argument('--block', 
                       default='august_12_single_day',
                       help='Data block to use (default: august_12_single_day)')
    
    parser.add_argument('--balance', 
                       type=float, 
                       default=10000.0,
                       help='Initial balance (default: 10000)')
    
    parser.add_argument('--save-results', 
                       action='store_true',
                       help='Save results to Output/ directory')
    
    parser.add_argument('--html-report', 
                       action='store_true',
                       help='Generate HTML report (only for aggressive strategy)')
    
    parser.add_argument('--list', 
                       action='store_true',
                       help='List available strategies and blocks')
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 Available Options:")
        print("=" * 50)
        
        print("🎯 Strategies:")
        print("  conservative - Original strategy (0 trades on test data)")
        print("  aggressive   - New momentum strategy (19 trades, works!)")
        print()
        
        print("📦 Data Blocks:")
        print("  august_12_single_day  - 1 day, +8.7% move (fast test)")
        print("  august_10_13_trend    - 4 days, trending market")
        print("  august_14_17_volatile - 4 days, volatile market")
        print("  august_10_17_full     - 7 days, mixed conditions")
        print()
        
        print("💡 Recommended command:")
        print("  python run_strategy.py --strategy aggressive")
        return
    
    # Run the selected strategy
    if args.strategy == 'aggressive':
        print("🚀 Running Aggressive Momentum Strategy")
        print("=" * 50)
        
        # Use HTML-enabled version if requested
        if args.html_report:
            from scripts.aggressive_strategy_with_html import AggressiveStrategyWithHTML
            strategy = AggressiveStrategyWithHTML(initial_balance=args.balance)
            results = strategy.run_enhanced_backtest_with_html(args.block)
        else:
            from scripts.final_aggressive_strategy import FinalAggressiveStrategy
            strategy = FinalAggressiveStrategy(initial_balance=args.balance)
            results = strategy.run_enhanced_backtest_with_momentum(args.block)
        
        # Save results if requested
        if args.save_results and results.get('trades'):
            import json
            from datetime import datetime
            
            output_file = f"Output/strategy_aggressive_{args.block}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")
    
    elif args.strategy == 'conservative':
        print("🔄 Running Conservative Strategy")
        print("=" * 50)
        
        from scripts.enhanced_backtest import EnhancedBacktestEngine
        
        strategy = EnhancedBacktestEngine(initial_balance=args.balance)
        data = strategy.load_data_from_block(args.block)
        results = strategy.run_backtest(symbol="SOLUSDT", data=data)
        
        print("\n📊 Conservative Strategy Results:")
        if "error" in results:
            print(f"❌ {results['error']}")
        else:
            print("✅ Strategy completed (unlikely to have trades)")
        
        # Save results if requested
        if args.save_results:
            import json
            from datetime import datetime
            
            output_file = f"Output/strategy_conservative_{args.block}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()