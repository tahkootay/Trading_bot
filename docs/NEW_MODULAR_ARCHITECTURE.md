# New Modular Architecture Design

## Overview
The trading bot system is being refactored into 4 independent modules that can be developed, tested, and run separately.

## Module Structure

### Module 1: Historical Data Collection (`modules/data_collector/`)
- **Purpose**: Automated collection of historical futures data from Bybit
- **Input**: Trading pair, timeframes, date ranges
- **Output**: Structured data files (CSV/JSON/Parquet)
- **Interface**: CLI with standardized parameters

### Module 2: Backtesting System (`modules/backtester/`)
- **Purpose**: Run trading algorithms on historical data
- **Input**: Strategy class + historical data
- **Output**: Raw results in JSON format
- **Interface**: Standardized strategy base class

### Module 3: Report Generator (`modules/reporter/`)
- **Purpose**: Convert backtest results into HTML reports
- **Input**: JSON results from Module 2
- **Output**: Interactive HTML report with charts
- **Interface**: Template-based report generation

### Module 4: Live Trading Bot (`modules/trading_bot/`)
- **Purpose**: Execute live trades (placeholder structure)
- **Input**: Strategy configuration
- **Output**: Trade execution and monitoring
- **Interface**: Strategy interface compatible with Module 2

## Key Design Principles

1. **Independence**: Each module can run without others
2. **Standardized Interfaces**: Clear input/output contracts
3. **File-based Communication**: JSON/CSV for data exchange
4. **Extensibility**: Easy to add new strategies/features
5. **Clean Architecture**: No circular dependencies

## Directory Structure
```
modules/
├── data_collector/          # Module 1
│   ├── main.py             # CLI entry point
│   ├── bybit_client.py     # Exchange API wrapper
│   ├── data_formats.py     # Output format handlers
│   └── config.yaml         # Module configuration
├── backtester/             # Module 2
│   ├── main.py             # CLI entry point
│   ├── strategy_base.py    # Base strategy class
│   ├── backtest_engine.py  # Core backtesting logic
│   └── result_formatter.py # Output standardization
├── reporter/               # Module 3
│   ├── main.py             # CLI entry point
│   ├── html_generator.py   # HTML report creation
│   ├── chart_builder.py    # Interactive charts
│   └── templates/          # HTML templates
└── trading_bot/            # Module 4 (placeholder)
    ├── main.py             # CLI entry point
    ├── strategy_runner.py  # Live strategy execution
    └── config.yaml         # Bot configuration
```

## Implementation Plan

1. Clean up root directory
2. Create modular structure
3. Move relevant existing code to modules
4. Remove ML/ensemble dependencies
5. Create standardized interfaces
6. Update documentation