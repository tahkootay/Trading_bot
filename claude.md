# CLAUDE.md - Modular Trading System

## Architecture Overview
Four independent modules communicate through standardized file formats (CSV, JSON) without direct imports.

```
modules/data_collector/  → CSV/JSON → modules/backtester/ → JSON → modules/reporter/ → HTML
                                           ↓
                                    modules/trading_bot/ (placeholder)
```

## Module Commands

```bash
# Data Collection (Module 1)
python -m modules.data_collector --symbol SOLUSDT --timeframe 5m --period week

# Backtesting (Module 2)
python -m modules.backtester --strategy ./strategy.py --data ./data.csv

# Report Generation (Module 3)
python -m modules.reporter --results ./backtest_results.json

# Trading Bot (Module 4) - Placeholder
python -m modules.trading_bot --strategy ./strategy.py --paper-trading
```

## Strategy Template

```python
from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal

class MyStrategy(StrategyBase):
    def _initialize(self):
        self.parameters = {'period': 20, 'position_size_pct': 0.1}
        self.state = {'history': []}
    
    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        # Implement trading logic here
        return TradeSignal(signal=Signal.HOLD)
```

## File Organization Rules

### ✅ CORRECT Placement
- Module code → `modules/{module_name}/`
- Strategies → `examples/strategies/`
- Data → `data/raw/`, `data/test/`
- Results → `output/backtests/`, `output/reports/`
- Docs → `docs/project_management/`

### 🚫 FORBIDDEN
- ❌ NO cross-module imports
- ❌ NO files in project root (except essentials)
- ❌ NO module modifications when working on another module

## Development Rules (CRITICAL)

### Module Independence (MANDATORY)
1. **NEVER import code between modules**
2. **ALWAYS use file-based communication (JSON/CSV)**
3. **Each module MUST work standalone**
4. **NEVER modify other modules when working on one**
5. **Test each module independently**

### Code Standards
- Type hints required
- PEP 8 with 88 char limit
- Functions <50 lines
- Google-style docstrings

## Notes for Claude Code

### STRICT Module Isolation
- **Working on Module 1**: NEVER modify modules/backtester/, modules/reporter/, modules/trading_bot/
- **Working on Module 2**: NEVER modify modules/data_collector/, modules/reporter/, modules/trading_bot/
- **Working on Module 3**: NEVER modify modules/data_collector/, modules/backtester/, modules/trading_bot/
- **Working on Module 4**: NEVER modify modules/data_collector/, modules/backtester/, modules/reporter/

### File Organization (ENFORCE)
- Module code → ONLY in appropriate `modules/{name}/` directory
- NO temporary files in root
- NO cross-references in code
- ALWAYS maintain clean separation

**System Priority**: Module independence over convenience. Each module is a separate product.