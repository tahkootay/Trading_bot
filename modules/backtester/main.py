#!/usr/bin/env python3
"""
Module 2: Backtesting System

Run trading algorithms on historical data with standardized interfaces.
Generates structured results for further analysis.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.util

import pandas as pd

from strategy_base import StrategyBase, BacktestConfig
from backtest_engine import BacktestEngine
from result_formatter import ResultFormatter, BacktestResults


class BacktestRunner:
    """Main backtesting system orchestrator."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.engine = BacktestEngine(config)
        self.formatter = ResultFormatter()
    
    async def run_backtest(
        self,
        strategy: StrategyBase,
        data_file: str,
        output_file: Optional[str] = None
    ) -> BacktestResults:
        """
        Run backtest with specified strategy and data.
        
        Args:
            strategy: Strategy instance implementing StrategyBase
            data_file: Path to historical data file (CSV/JSON)
            output_file: Optional path for results file
            
        Returns:
            BacktestResults object with all results
        """
        
        print(f"🔄 Starting backtest: {strategy.__class__.__name__}")
        print(f"📊 Data file: {data_file}")
        print(f"💰 Initial capital: ${self.config.initial_capital:,.2f}")
        print(f"🏪 Commission: {self.config.commission_rate:.4f}")
        
        # Load data
        data = await self._load_data(data_file)
        if data.empty:
            raise ValueError(f"No data loaded from {data_file}")
        
        print(f"📈 Loaded {len(data)} candles from {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
        
        # Run backtest
        results = await self.engine.run_backtest(strategy, data)
        
        # Format results
        formatted_results = self.formatter.format_results(results)
        
        # Save results if output file specified
        if output_file:
            await self._save_results(formatted_results, output_file)
            print(f"💾 Results saved to: {output_file}")
        
        # Print summary
        self._print_summary(formatted_results)
        
        return formatted_results
    
    async def _load_data(self, data_file: str) -> pd.DataFrame:
        """Load historical data from file."""
        
        file_path = Path(data_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                if 'data' in json_data:
                    df = pd.DataFrame(json_data['data'])
                else:
                    df = pd.DataFrame(json_data)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Validate required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load data from {data_file}: {e}")
    
    async def _save_results(self, results: BacktestResults, output_file: str) -> None:
        """Save backtest results to file."""
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        results_dict = {
            'backtest_info': {
                'strategy_name': results.strategy_name,
                'start_date': results.start_date.isoformat(),
                'end_date': results.end_date.isoformat(),
                'initial_capital': results.initial_capital,
                'final_capital': results.final_capital,
                'backtest_timestamp': datetime.now().isoformat()
            },
            'performance_metrics': results.performance_metrics,
            'trades': [
                {
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'commission': trade.commission,
                    'duration_minutes': trade.duration_minutes,
                    'entry_reason': trade.entry_reason,
                    'exit_reason': trade.exit_reason
                }
                for trade in results.trades
            ],
            'equity_curve': [
                {
                    'timestamp': point.timestamp.isoformat(),
                    'equity': point.equity,
                    'drawdown': point.drawdown
                }
                for point in results.equity_curve
            ],
            'daily_returns': [
                {
                    'date': point.date.isoformat(),
                    'return_pct': point.return_pct,
                    'cumulative_return': point.cumulative_return
                }
                for point in results.daily_returns
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def _print_summary(self, results: BacktestResults) -> None:
        """Print backtest summary to console."""
        
        metrics = results.performance_metrics
        
        print("\n" + "="*60)
        print(f"📊 BACKTEST SUMMARY: {results.strategy_name}")
        print("="*60)
        
        print(f"📅 Period: {results.start_date.date()} to {results.end_date.date()}")
        print(f"💰 Initial Capital: ${results.initial_capital:,.2f}")
        print(f"💰 Final Capital: ${results.final_capital:,.2f}")
        print(f"📈 Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"📉 Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Winning Trades: {metrics['winning_trades']}")
        print(f"   Losing Trades: {metrics['losing_trades']}")
        print(f"   Win Rate: {metrics['win_rate_pct']:.2f}%")
        print(f"   Average Trade: ${metrics['avg_trade_pnl']:.2f}")
        print(f"   Best Trade: ${metrics['best_trade_pnl']:.2f}")
        print(f"   Worst Trade: ${metrics['worst_trade_pnl']:.2f}")
        
        print(f"\n📈 RISK METRICS:")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"   Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"   Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        
        print("="*60)


def load_strategy_from_file(strategy_file: str) -> StrategyBase:
    """Load strategy class from Python file."""
    
    file_path = Path(strategy_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_file}")
    
    # Load module
    spec = importlib.util.spec_from_file_location("strategy_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy from {strategy_file}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find strategy class
    strategy_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and 
            issubclass(obj, StrategyBase) and 
            obj is not StrategyBase):
            strategy_class = obj
            break
    
    if strategy_class is None:
        raise ValueError(f"No StrategyBase subclass found in {strategy_file}")
    
    return strategy_class()


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Run backtests on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest with strategy file
  python main.py --strategy ./strategies/sma_crossover.py --data ./data/SOLUSDT_5m.csv
  
  # Backtest with custom settings
  python main.py --strategy ./strategies/rsi.py --data ./data/BTCUSDT_1h.csv --capital 10000 --commission 0.001
  
  # Save results to specific file
  python main.py --strategy ./strategies/bollinger.py --data ./data/ETHUSDT_15m.csv --output ./results/backtest_results.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--strategy", "-s",
        required=True,
        help="Path to strategy Python file"
    )
    
    parser.add_argument(
        "--data", "-d", 
        required=True,
        help="Path to historical data file (CSV or JSON)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (defaults to auto-generated)"
    )
    
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0006,
        help="Commission rate per trade (default: 0.0006)"
    )
    
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.001,
        help="Slippage rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=1.0,
        help="Maximum position size as fraction of capital (default: 1.0)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (YAML)"
    )
    
    return parser


async def main():
    """Main entry point."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = BacktestConfig.from_file(args.config)
        else:
            config = BacktestConfig(
                initial_capital=args.capital,
                commission_rate=args.commission,
                slippage_rate=args.slippage,
                max_position_size=args.max_position_size
            )
        
        # Load strategy
        strategy = load_strategy_from_file(args.strategy)
        print(f"✅ Loaded strategy: {strategy.__class__.__name__}")
        
        # Generate output filename if not provided
        output_file = args.output
        if not output_file:
            strategy_name = strategy.__class__.__name__.lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"./output/backtests/{strategy_name}_backtest_{timestamp}.json"
        
        # Create runner and execute backtest
        runner = BacktestRunner(config)
        results = await runner.run_backtest(
            strategy=strategy,
            data_file=args.data,
            output_file=output_file
        )
        
        print(f"\n✅ Backtest completed successfully!")
        print(f"📊 Results: {output_file}")
        
    except KeyboardInterrupt:
        print("\n🛑 Backtest cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())