# CLAUDE.md - Trading Bot System

## System Overview
Modular cryptocurrency trading analysis system with data collection, ML training, and prediction capabilities.

```
Data Collection → Processing → ML Training → Predictions
modules/data_collector/ → data/processed/ → modules/ml_training/ → Trading Signals
```

## Project Structure

```
Trading_bot/
├── main.py                    # Main entry point
├── CLAUDE.md                  # This file
├── pyproject.toml            # Dependencies
├── Makefile                  # Build commands
├── modules/                  # Core modules
│   ├── data_collector/       # Data collection from exchanges
│   └── ml_training/          # ML model training and management
├── config/                   # Configuration files
├── data/                     # Data storage
│   ├── raw/                  # Raw collected data
│   └── processed/            # Processed data for ML
├── models/                   # Trained ML models
├── scripts/                  # Utility scripts
├── tests/                    # Test files
├── demos/                    # Demo and example files
├── docs/                     # Documentation
└── examples/                 # Usage examples
```

## Commands

### Main Interface
```bash
# Main entry point - collect data
python main.py collect --symbol SOLUSDT --timeframe 5m --period week

# Train model
python main.py train --data data/processed --version v2

# Generate predictions
python main.py predict --symbol SOLUSDT --model v1
```

### Data Collection Module
```bash
# Collect data directly
python -m modules.data_collector --symbol SOLUSDT --timeframe 5m --period week

# Add basic indicators
python -m modules.data_collector.indicators input_data.csv output_with_indicators.csv

# Add advanced indicators (50+ features)
python -m modules.data_collector.advanced_indicators input_data.csv output_advanced.csv

# Prepare data for ML (target, lags, splits, scaling)
python -m modules.data_collector.ml_data_prep input_advanced.csv data/processed --targets 3

# Multiple target horizons for different prediction windows
python -m modules.data_collector.ml_data_prep input_advanced.csv data/processed --targets 1 3 5 10
```

### ML Training Module
```bash
# List available models
python -m modules.ml_training list

# Compare model versions
python -m modules.ml_training compare v1 v2

# Train model (use script for now)
python scripts/train_model.py
```

### Utility Scripts
```bash
# Train ML model
python scripts/train_model.py

# Analyze predictions
python scripts/analyze_predictions.py
```

## File Organization Rules

### ✅ CORRECT Structure
- **Root**: Only essential files (main.py, CLAUDE.md, pyproject.toml, Makefile)
- **modules/**: Core functionality modules
- **scripts/**: Utility and training scripts
- **tests/**: All test files
- **demos/**: Demo and example usage files
- **data/**: All data files (raw/, processed/)
- **models/**: Trained ML models and metadata
- **config/**: Configuration files

### 🚫 FORBIDDEN
- ❌ NO non-essential files in project root
- ❌ NO mixing of module responsibilities
- ❌ NO training/demo files in modules
- ❌ NO test files outside tests/

## Module Responsibilities

### modules/data_collector/
- Historical data collection from exchanges
- Technical indicators calculation
- Data preprocessing and formatting
- ML data preparation

### modules/ml_training/
- Model training and evaluation
- Model management (save/load/version)
- Performance analysis and comparison
- Trading signal generation

## Development Rules
1. **Модульность**: Каждый блок задач - отдельный модуль
2. **Чистота корня**: Минимум файлов в корне проекта
3. **Разделение**: Тесты, демо, скрипты в своих папках
4. **Независимость**: Модули должны работать автономно

## Core Rule
**NEVER create simplified versions without explicit user permission.**

## When Problems Occur

### Mandatory Algorithm:
1. **Diagnose** - identify the exact cause of the error
2. **Attempt to fix** - minimum 3-5 different approaches:
   - Different installation methods (pip/pip3/conda/system packages)
   - Alternative library versions
   - System dependencies
   - Permission fixes
   - Alternative methods to achieve the goal
3. **Document** - show the user all attempts
4. **Request permission** - only after exhausting options

### Permission Request Format:
```
Attempted to solve the problem:
1. [attempt] - result
2. [attempt] - result
3. [attempt] - result

Failed. Can offer [alternative with description of differences].
Continue searching for solution or use alternative?
```

## Prohibited
- ❌ Silently removing functionality
- ❌ Simplifying after first failure
- ❌ Replacing technologies without discussion
- ❌ Saying "let's simplify" without attempting solutions

## Priorities
1. Complete the task **fully**
2. Find an alternative **complete** solution
3. Suggest compromise (only with permission)

**Principle**: The user wants what they asked for, not "something that works".
