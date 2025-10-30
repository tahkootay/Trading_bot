"""
Data Collection and Technical Indicators Module

This module provides automated collection of historical futures data from Bybit exchange
and comprehensive technical analysis indicators.

Main components:
- BybitDataCollector: Core data collection from Bybit API
- DataFormatter: Output format handling and validation
- DataCollectorConfig: Configuration management
- HistoricalDataCollector: High-level collection orchestrator
- TechnicalIndicators: Technical analysis calculations
- IndicatorConfig: Indicator configuration management

Example usage:
```python
from modules.data_collector import HistoricalDataCollector, DataCollectorConfig, TechnicalIndicators
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Data collection
config = DataCollectorConfig(output_directory="./data")
collector = HistoricalDataCollector(config)

results = asyncio.run(collector.collect_data(
    symbol="SOLUSDT",
    timeframes=["5m"],
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
))

# Technical indicators
df = pd.read_csv(results["5m"])
indicators = TechnicalIndicators()
df_with_indicators = indicators.add_indicators_to_dataframe(df)
```

CLI usage:
```bash
# Data collection
python -m modules.data_collector --symbol SOLUSDT --timeframe 5m --period week

# Add indicators to existing data
python -m modules.data_collector.indicators data.csv data_with_indicators.csv
```
"""

from .bybit_client import BybitDataCollector, TimeFrame, Candle
from .data_formats import DataFormatter, OutputFormat
from .config import DataCollectorConfig
from .main import HistoricalDataCollector
from .indicators import TechnicalIndicators, IndicatorConfig, calculate_indicators_for_file

__version__ = "1.0.0"
__all__ = [
    "BybitDataCollector",
    "DataFormatter", 
    "DataCollectorConfig",
    "HistoricalDataCollector",
    "TechnicalIndicators",
    "IndicatorConfig",
    "calculate_indicators_for_file",
    "TimeFrame",
    "Candle",
    "OutputFormat"
]