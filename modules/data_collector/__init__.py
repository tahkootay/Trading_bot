"""
Historical Data Collection Module

This module provides automated collection of historical futures data from Bybit exchange.
Supports multiple timeframes, date ranges, and output formats (CSV, JSON, Parquet).

Main components:
- BybitDataCollector: Core data collection from Bybit API
- DataFormatter: Output format handling and validation
- DataCollectorConfig: Configuration management
- HistoricalDataCollector: High-level collection orchestrator

Example usage:
```python
from modules.data_collector import HistoricalDataCollector, DataCollectorConfig
import asyncio

config = DataCollectorConfig(output_directory="./data")
collector = HistoricalDataCollector(config)

# Collect 7 days of 5m data
results = asyncio.run(collector.collect_data(
    symbol="SOLUSDT",
    timeframes=["5m"],
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
))
```

CLI usage:
```bash
python -m modules.data_collector --symbol SOLUSDT --timeframe 5m --period week
```
"""

from .bybit_client import BybitDataCollector, TimeFrame, Candle
from .data_formats import DataFormatter, OutputFormat
from .config import DataCollectorConfig
from .main import HistoricalDataCollector

__version__ = "1.0.0"
__all__ = [
    "BybitDataCollector",
    "DataFormatter", 
    "DataCollectorConfig",
    "HistoricalDataCollector",
    "TimeFrame",
    "Candle",
    "OutputFormat"
]