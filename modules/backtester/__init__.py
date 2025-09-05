"""
Backtesting Module

This module provides a standardized backtesting system for trading strategies.
Strategies must implement the StrategyBase interface for compatibility.

Main components:
- StrategyBase: Base class for all trading strategies
- BacktestEngine: Core backtesting execution engine
- ResultFormatter: Result formatting and validation
- BacktestRunner: High-level orchestrator

Example usage:
```python
from modules.backtester import BacktestRunner, BacktestConfig
from modules.backtester.strategy_base import SimpleMovingAverageStrategy
import asyncio

# Configure backtest
config = BacktestConfig(
    initial_capital=10000,
    commission_rate=0.0006
)

# Create strategy and runner
strategy = SimpleMovingAverageStrategy()
runner = BacktestRunner(config)

# Run backtest
results = asyncio.run(runner.run_backtest(
    strategy=strategy,
    data_file="./data/SOLUSDT_5m.csv"
))
```

CLI usage:
```bash
python -m modules.backtester --strategy ./strategies/my_strategy.py --data ./data/SOLUSDT_5m.csv
```
"""

from .strategy_base import (
    StrategyBase, 
    BacktestConfig, 
    Signal, 
    MarketData, 
    Position, 
    TradeSignal,
    SimpleMovingAverageStrategy,
    RSIStrategy
)
from .backtest_engine import BacktestEngine, Trade, EquityPoint, DailyReturn
from .result_formatter import ResultFormatter, BacktestResults
from .main import BacktestRunner

__version__ = "1.0.0"
__all__ = [
    "StrategyBase",
    "BacktestConfig", 
    "BacktestEngine",
    "ResultFormatter",
    "BacktestRunner",
    "BacktestResults",
    "Signal",
    "MarketData",
    "Position",
    "TradeSignal",
    "Trade",
    "EquityPoint", 
    "DailyReturn",
    "SimpleMovingAverageStrategy",
    "RSIStrategy"
]