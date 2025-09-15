# Indicators Module

The indicators module calculates technical indicators for trading data. It loads CSV data with OHLCV columns, calculates specified indicators, and saves enriched data to a new CSV file.

## Features

- **Standalone operation**: Works without external dependencies (numpy, pandas)
- **Selective calculation**: Choose which indicators to calculate
- **Standard CSV format**: Input/output via CSV files
- **Multiple indicators**: RSI, SMA, EMA, MACD, Bollinger Bands

## Usage

### Module Command
```bash
# Calculate all default indicators
python -m modules.indicators --input data.csv --output enriched.csv

# Calculate specific indicators
python -m modules.indicators --input data.csv --indicators RSI,MACD --output custom.csv

# List available indicators
python -m modules.indicators --list
```

### Standalone Script (Recommended)
```bash
# Calculate all indicators
python3 calculate_indicators.py --input data/raw/SOLUSDT_1h_august.csv --output data/processed/indicators.csv

# Calculate specific indicators
python3 calculate_indicators.py --input data.csv --indicators RSI,SMA,EMA --output custom.csv

# List available indicators
python3 calculate_indicators.py --list
```

## Available Indicators

| Indicator | Description | Parameters | Output Columns |
|-----------|-------------|------------|----------------|
| RSI | Relative Strength Index | Period: 14 | `rsi_14` |
| SMA | Simple Moving Average | Period: 20 | `sma_20` |
| EMA | Exponential Moving Average | Period: 20 | `ema_20` |
| MACD | Moving Average Convergence Divergence | 12/26/9 | `macd_line`, `macd_signal`, `macd_histogram` |
| BOLLINGER/BB | Bollinger Bands | Period: 20, Std Dev: 2 | `bb_upper`, `bb_middle`, `bb_lower` |

## Input Format

CSV file with the following columns:
- `timestamp` (optional)
- `open` (required for some indicators)
- `high` (required for some indicators) 
- `low` (required for some indicators)
- `close` (required)
- `volume` (optional)

## Examples

### Calculate All Indicators
```bash
python3 calculate_indicators.py \
  --input data/raw/SOLUSDT_1h_august.csv \
  --output data/processed/SOLUSDT_all_indicators.csv \
  --indicators RSI,SMA,EMA,MACD,BOLLINGER
```

### Calculate Only RSI and Moving Averages
```bash
python3 calculate_indicators.py \
  --input data/raw/SOLUSDT_1h_august.csv \
  --output data/processed/SOLUSDT_basic.csv \
  --indicators RSI,SMA,EMA
```

## Output

The output CSV contains all original columns plus calculated indicator columns. Indicators that cannot be calculated for initial periods (due to insufficient data) will have empty values.

Example output columns:
```
timestamp,open,high,low,close,volume,rsi_14,sma_20,ema_20,macd_line,macd_signal,macd_histogram,bb_upper,bb_middle,bb_lower
```

## Integration with Other Modules

The indicators module follows the modular architecture:
- **Input**: CSV files from data collector or other sources
- **Output**: Enhanced CSV files with calculated indicators
- **Usage**: Can be chained with backtester for strategy testing

Example workflow:
```bash
# 1. Collect data
python -m modules.data_collector --symbol SOLUSDT --timeframe 1h --period week

# 2. Calculate indicators  
python3 calculate_indicators.py --input data/raw/SOLUSDT_1h_week.csv --output data/processed/SOLUSDT_indicators.csv

# 3. Run backtesting
python -m modules.backtester --data data/processed/SOLUSDT_indicators.csv --strategy strategy.py
```