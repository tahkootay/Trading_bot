# CLAUDE.md - Data Collection and Technical Indicators System

## System Overview
Focused system for cryptocurrency data collection and technical analysis indicators.

```
modules/data_collector/ → CSV/JSON → Technical Indicators → Enhanced Data
```

## Commands

```bash
# Collect data
python -m modules.data_collector --symbol SOLUSDT --timeframe 5m --period week

# Add indicators to existing data
python -m modules.data_collector.indicators input_data.csv output_with_indicators.csv
```

## Available Indicators
- **SMA/EMA** - Moving averages
- **RSI** - Relative Strength Index
- **MACD** - MACD with signal and histogram
- **Bollinger Bands** - Upper, middle, lower bands
- **ATR** - Average True Range
- **KDJ** - Stochastic KDJ

## File Organization Rules

### ✅ CORRECT
- Module code → `modules/data_collector/`
- Data → `data/raw/`, `data/processed/`

### 🚫 FORBIDDEN
- ❌ NO files in project root (except essentials)
- ❌ NO additional modules without request
- ❌ NO modification of module structure

## Development Rules