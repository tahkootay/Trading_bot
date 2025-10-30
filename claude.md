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

# Add basic indicators
python -m modules.data_collector.indicators input_data.csv output_with_indicators.csv

# Add advanced indicators (50+ features)
python -m modules.data_collector.advanced_indicators input_data.csv output_advanced.csv

# Prepare data for ML (target, lags, splits, scaling)
python -m modules.data_collector.ml_data_prep input_advanced.csv data/processed 3
```

## File Organization Rules

### ✅ CORRECT
- Module code → `modules`
- Data → `data/raw/`, `data/processed/`

### 🚫 FORBIDDEN
- ❌ NO files in project root (except essentials)
- ❌ NO additional modules without request
- ❌ NO modification of module structure

## Development Rules
Соблюдай модульность проекта. Каждый существенный блок задач должен быть выполнен в виде отдельного модуля с возможностью независимой доработки.