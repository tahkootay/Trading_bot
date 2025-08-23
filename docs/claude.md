# CLAUDE.md - Intraday Trading Bot for SOL/USDT

## Project Overview
Professional intraday trading bot designed to capture $2+ price movements on SOL/USDT futures (Bybit). The system uses ML models, technical analysis, and market microstructure to generate high-probability trading signals.

**Critical Goal**: Capture 3-7 movements of $2+ daily on SOL/USDT with risk-controlled approach.

## Tech Stack
- **Language**: Python 3.11+
- **Async Framework**: asyncio, aiohttp
- **ML Libraries**: XGBoost, LightGBM, scikit-learn
- **Data Processing**: pandas, numpy, numba
- **Technical Analysis**: ta-lib, pandas-ta
- **Exchange API**: pybit (Bybit), ccxt (backup)
- **Database**: PostgreSQL (history), Redis (cache), InfluxDB (metrics)
- **Message Queue**: Apache Kafka
- **Monitoring**: Prometheus + Grafana
- **Notifications**: python-telegram-bot
- **Testing**: pytest, pytest-asyncio, pytest-mock
- **Container**: Docker, docker-compose
- **Config**: pydantic-settings, python-dotenv

## Project Structure & Organization Rules

### CRITICAL: Directory Structure Rules
**ALWAYS maintain clean project structure. Root directory MUST contain ONLY essential files:**

```
trading-bot/                        # ROOT: Essential files only
├── README.md                       # Project overview
├── requirements.txt                # Dependencies  
├── pyproject.toml                  # Python project config
├── Makefile                        # Build commands
├── .gitignore                      # Git ignore patterns
├── src/                           # CORE SOURCE CODE
│   ├── data_collector/            # Real-time data collection
│   ├── feature_engine/            # Technical indicators & features
│   ├── models/                    # ML models and predictions
│   ├── signal_generator/          # Trading signal logic
│   ├── execution/                 # Order management
│   ├── risk_manager/              # Risk controls
│   ├── notifications/             # Alerts and reporting
│   └── utils/                     # Shared utilities
├── tools/                         # DEVELOPMENT TOOLS
│   ├── backtest/                  # Backtest execution scripts
│   ├── testing/                   # Test and validation scripts
│   └── data_collection/           # Data gathering utilities
├── docs/                          # ALL DOCUMENTATION
│   ├── claude.md                  # This file (project instructions)
│   ├── project_management/        # Project reports, reorganization docs
│   ├── QUICK_START.md            # Getting started guide
│   └── TESTING_GUIDE.md          # Testing procedures
├── examples/                      # DEMO & EXAMPLE CODE
│   └── demo_scripts/              # Demonstration utilities
├── output/                        # ORGANIZED RESULTS
│   ├── reports/                   # HTML and Markdown reports
│   ├── backtests/                 # JSON backtest results
│   └── models_archive/            # Archived model versions
├── data/                          # DATA STORAGE (organized)
│   ├── raw/                       # Raw CSV files
│   ├── processed/                 # Processed data
│   ├── bybit_futures/             # Bybit futures data (priority)
│   ├── blocks/                    # Fixed test data blocks
│   ├── enhanced/                  # Enhanced data with features
│   └── test/                      # Test datasets
├── archive/                       # TEMPORARY STORAGE
│   ├── temp_data/                 # Temporary/old data files
│   └── old_runs/                  # Archive of old executions
├── models/                        # TRAINED ML MODELS
├── scripts/                       # CORE UTILITY SCRIPTS
├── config/                        # CONFIGURATION FILES
├── tests/                         # TEST SUITE
├── docker/                        # DOCKER CONFIGURATIONS
└── web_interface/                 # WEB UI COMPONENTS
```

### 🚫 NEVER CREATE FILES IN ROOT
- ❌ NO run_*.py files in root
- ❌ NO test_*.py files in root  
- ❌ NO temp_*.py or debug_*.py files in root
- ❌ NO .md files in root (except README.md)
- ❌ NO data files (.csv, .json) in root
- ❌ NO report files in root

### ✅ ALWAYS Place Files in Logical Locations
- **Backtest scripts** → `tools/backtest/`
- **Test scripts** → `tools/testing/`
- **Demo/example code** → `examples/demo_scripts/`
- **Documentation** → `docs/`
- **Project reports** → `docs/project_management/`
- **Generated reports** → `output/reports/`
- **Backtest results** → `output/backtests/`
- **Temporary files** → `archive/temp_data/`

### File Placement Decision Tree
```
New file created? → Ask:
├── Is it documentation? → docs/
├── Is it a backtest script? → tools/backtest/
├── Is it a test/validation script? → tools/testing/
├── Is it demo/example code? → examples/
├── Is it a generated report? → output/reports/
├── Is it backtest results? → output/backtests/
├── Is it temporary/analysis? → archive/temp_data/
└── Is it core functionality? → src/
```

## Key Commands
```bash
# Development
make dev          # Start development environment
make test         # Run all tests
make lint         # Run linting (ruff, mypy)
make format       # Format code (black, isort)
make typecheck    # Run type checking

# Docker
docker-compose up -d          # Start all services
docker-compose logs -f bot    # View bot logs
docker-compose down           # Stop all services

# Bot Control
python -m src.main --config config/prod.yaml    # Start bot
python scripts/backtest.py --pair SOLUSDT       # Run backtest
python scripts/train_model.py --data-days 90    # Train models
python scripts/emergency_stop.py                # Emergency shutdown

# Analysis & Reports (all saved to Output/)
python scripts/enhanced_backtest.py --save-results  # Enhanced backtest with JSON results
python generate_detailed_report.py              # Comprehensive HTML analysis report
python scripts/collect_data.py --symbol SOLUSDT --days 7  # Collect market data
```

## Code Style & Best Practices

### Python Standards
- Use type hints for ALL function signatures
- Follow PEP 8 with 88 char line limit (Black)
- Docstrings for all public functions (Google style)
- Use dataclasses or Pydantic for data structures
- Prefer composition over inheritance
- Keep functions small and testable (<50 lines)

### Async Programming
- Use `async/await` for all I/O operations
- Implement proper connection pooling
- Handle reconnections gracefully
- Use asyncio.gather() for parallel operations
- Implement circuit breakers for external APIs

### Error Handling
```python
# ALWAYS use specific exception handling
try:
    result = await exchange_api.get_balance()
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    await reconnect()
except APIError as e:
    logger.error(f"API error: {e}")
    await handle_api_error(e)
```

### Logging
- Use structured logging (JSON format)
- Include context in all log messages
- Log levels: DEBUG (dev only), INFO (important events), WARNING (issues), ERROR (failures)
- NEVER log sensitive data (API keys, positions details in prod)

## Trading Logic Rules

### Entry Conditions (BUY Signal)
1. P(UP) - P(DOWN) > 0.15 (ML confidence)
2. EMA8 > EMA20 > EMA50 (trend alignment)
3. Price >= VWAP (momentum confirmation)
4. ADX >= 20 (trend strength)
5. Volume z-score > 1.0 (volume confirmation)
6. No existing position in same direction

### Entry Conditions (SELL Signal)
1. P(DOWN) - P(UP) > 0.15
2. EMA8 < EMA20 < EMA50
3. Price <= VWAP
4. ADX >= 20
5. Volume z-score > 1.0
6. No existing position in same direction

### Risk Management Rules
- **Position Size**: Max 2% of account per trade
- **Stop Loss**: 1.0-1.5 ATR from entry
- **Take Profit**: Multi-level (40% at 1 ATR, 30% at 1.5 ATR, 20% at 2.5 ATR, 10% runner)
- **Daily Loss Limit**: -3% → pause until next day
- **Consecutive Losses**: 3 → pause 2 hours
- **Max Drawdown**: 10% → manual intervention required
- **Time Stop**: Close if no movement in 2 hours

## Critical Safety Rules

### NEVER Do
- ❌ Trade without stop loss
- ❌ Override risk limits programmatically
- ❌ Ignore exchange maintenance windows
- ❌ Use market orders in low liquidity
- ❌ Keep positions during major news events without confirmation
- ❌ Trust single data source (always validate)
- ❌ Deploy untested code to production

### ALWAYS Do
- ✅ Validate all external data
- ✅ Log every trade decision with reasoning
- ✅ Implement graceful shutdown
- ✅ Keep audit trail of all actions
- ✅ Test order execution in testnet first
- ✅ Monitor latency and slippage
- ✅ Have manual override capability

## Testing Requirements

### Before ANY Deployment
1. Unit tests: >90% coverage
2. Integration tests with mock exchange
3. Backtest on 6 months of data
4. Paper trade for minimum 2 weeks
5. Gradual capital scaling in production

### Test Scenarios Must Include
- Flash crash (>5% move in 1 minute)
- Exchange API degradation
- Network disconnection
- Partial order fills
- Rapid succession of signals
- Maximum position limits
- Drawdown scenarios

## Performance Optimization

### Data Processing
- Use vectorized operations (numpy/pandas)
- Cache frequently accessed data in Redis
- Implement sliding window for indicators
- Use numba @jit for computation-heavy functions
- Batch database writes

### Latency Targets
- Exchange API response: <100ms
- Feature calculation: <50ms
- Model inference: <20ms
- Signal generation: <10ms
- Total decision time: <200ms

## Monitoring & Alerts

### Critical Alerts (Immediate)
- Position without stop loss
- Disconnection >30 seconds
- Daily loss >2%
- Unexpected exception
- Model confidence degradation >30%

### Important Alerts (5 min)
- Approaching daily loss limit
- High slippage detected
- Low liquidity warning
- Correlation breakdown

### Heartbeat (Every 5 min)
- Current P&L
- Open positions
- Connection status
- Last signal time
- System resources

## Workflow Patterns

### Feature Implementation
```
1. Create feature branch from develop
2. Write tests first (TDD)
3. Implement feature
4. Run full test suite
5. Test in paper trading
6. Code review
7. Merge to develop
8. Deploy to staging
9. Monitor for 24h
10. Deploy to production
```

### Emergency Procedures
```
IF critical_error:
    1. Close all positions immediately
    2. Cancel all pending orders
    3. Send emergency alert
    4. Log full system state
    5. Pause all trading
    6. Await manual intervention
```

## ML Model Management

### Feature Engineering
- Calculate all features in consistent order
- Handle NaN values explicitly
- Normalize features using saved scalers
- Version all feature transformations

### Model Updates
- Retrain weekly with latest data
- A/B test new models for 48h
- Track performance degradation
- Keep last 3 model versions
- Document all model changes

## Environment Variables
```bash
# Required
BYBIT_API_KEY=xxx
BYBIT_API_SECRET=xxx
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
DATABASE_URL=postgresql://...
REDIS_URL=redis://...

# Optional
LOG_LEVEL=INFO
MAX_POSITION_SIZE=0.02
PAPER_TRADING=false
MODEL_VERSION=v1.2.3
```

## Common Issues & Solutions

### Issue: High slippage on orders
- Switch to limit orders
- Reduce position size
- Check liquidity before entry
- Use iceberg orders for large positions

### Issue: Model performance degradation
- Check for data drift
- Verify feature calculation
- Review recent market regime changes
- Retrain with recent data

### Issue: Missed profitable moves
- Review signal thresholds
- Check if filters too restrictive
- Analyze feature importance
- Consider market microstructure

## Development Guidelines

### Git Workflow
- Branch naming: `feature/description`, `fix/description`, `hotfix/description`
- Commit messages: Conventional commits format
- PR requires: Tests passing, code review, staging tested
- Protected branches: main, develop

### Code Review Checklist
- [ ] Tests coverage >90%
- [ ] Type hints complete
- [ ] Docstrings updated
- [ ] Error handling comprehensive
- [ ] Logging appropriate
- [ ] Performance acceptable
- [ ] Risk checks in place

## Quick Fixes & Shortcuts

### Restart Bot Safely
```bash
# Send graceful shutdown signal
kill -SIGTERM $(pgrep -f "python.*main.py")
# Wait for positions to close
sleep 30
# Start bot
./scripts/start_bot.sh
```

### Check System Health
```bash
# Quick health check
curl localhost:8080/health

# Detailed metrics
curl localhost:8080/metrics

# Current positions
python scripts/check_positions.py
```

### Manual Trade Override
```python
# Emergency close all positions
python scripts/emergency_close.py --confirm

# Manual trade entry (use with caution)
python scripts/manual_trade.py --side BUY --size 0.01 --pair SOLUSDT
```

## Notes for Claude Code

### File Organization (CRITICAL)
- **NEVER create files in project root** - always use appropriate subdirectory
- **ALWAYS use decision tree** above to determine correct file location
- **Project reports/summaries** → `docs/project_management/`
- **Generated reports** → `output/reports/`
- **Follow structure religiously** - clean organization is critical

### Backtest Requirements (MANDATORY)
- **ALWAYS generate comprehensive HTML report** after ANY backtest execution
- **Report must be generated** regardless of trades executed or results
- **Use template from** `docs/backtest_report_concept.md`
- **Save reports to** `output/reports/`
- **Include interactive charts** with Plotly.js

When implementing new features:
1. ALWAYS write tests first
2. ALWAYS handle edge cases  
3. ALWAYS add appropriate logging
4. ALWAYS update documentation
5. ALWAYS consider risk implications
6. ALWAYS maintain clean file structure
7. NEVER remove safety checks
8. NEVER increase risk limits without explicit confirmation
9. NEVER create files in project root

When fixing bugs:
1. First reproduce in test
2. Fix the issue
3. Verify test passes
4. Check for similar issues elsewhere
5. Update documentation if needed

When optimizing performance:
1. Profile first, optimize second
2. Maintain readability
3. Document any complex optimizations
4. Ensure tests still pass
5. Benchmark improvements

## Remember
This is REAL MONEY trading bot. Every decision must prioritize capital preservation over profit. When in doubt, don't trade. It's better to miss opportunities than to lose capital on bad trades.