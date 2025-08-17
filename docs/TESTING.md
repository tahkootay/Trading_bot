# 🧪 Testing Guide for SOL/USDT Trading Bot

## Quick Start Testing

### 1. Verify Environment Setup

```bash
# Check if dependencies are installed
python -c "import pandas, numpy, aiohttp; print('✅ Dependencies OK')"

# Test API credentials
make test-connection
```

### 2. Collect Historical Data

```bash
# Collect 30 days of data (recommended for testing)
make collect-data

# Or collect extended data (90 days)
make collect-data-extended

# Manual collection with custom parameters
python scripts/collect_data.py --symbol SOLUSDT --days 30 --timeframes "1m,5m,15m,1h"
```

### 3. Run Backtesting

```bash
# Full backtest with results saved
make backtest

# Quick backtest (last 7 days)
make backtest-quick

# Custom backtest
python scripts/enhanced_backtest.py --symbol SOLUSDT --balance 10000 --days 14
```

### 4. Complete Testing Pipeline

```bash
# Run all tests in sequence
make test-all
```

## 📊 Expected Results

### ✅ Good Performance Indicators

- **Win Rate**: ≥55%
- **Profit Factor**: ≥1.3
- **Max Drawdown**: ≤10%
- **Movements ≥$2**: ≥30% of trades
- **Sharpe Ratio**: ≥1.0

### ⚠️ Warning Signs

- Win rate <50%
- Profit factor <1.0
- Max drawdown >15%
- Very few trades generated
- High commission costs (>20% of P&L)

## 🔧 Configuration Testing

### Test Different Parameters

Edit `scripts/enhanced_backtest.py` to test different strategies:

```python
# In _generate_enhanced_signal method, modify:
strategy_params = {
    'min_confidence': 0.20,  # Higher confidence threshold
    'min_volume_ratio': 1.5,  # Higher volume requirement
    'atr_tp_multiplier': 1.5,  # Lower take profit target
}
```

### Test Different Timeframes

```python
# In run_backtest, change primary timeframe:
primary_timeframe = TimeFrame.M15  # Use 15m instead of 5m
```

## 📈 Data Quality Checks

### Verify Data Collection

```bash
# Check collected data
ls -la data/
cat data/SOLUSDT_metadata.json

# Verify data completeness
python -c "
import pandas as pd
df = pd.read_csv('data/SOLUSDT_5m_30d.csv')
print(f'Total candles: {len(df)}')
print(f'Date range: {df.timestamp.min()} → {df.timestamp.max()}')
print(f'Missing data: {df.isnull().sum().sum()}')
"
```

### Expected Data Volumes

- **1m data**: ~43,200 candles per 30 days
- **5m data**: ~8,640 candles per 30 days  
- **15m data**: ~2,880 candles per 30 days
- **1h data**: ~720 candles per 30 days

## 🚨 Troubleshooting

### Connection Issues

```bash
# Check API credentials
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('API Key:', os.getenv('BYBIT_API_KEY', 'NOT FOUND')[:8] + '...')
print('API Secret:', 'SET' if os.getenv('BYBIT_API_SECRET') else 'NOT FOUND')
"

# Test basic HTTP connectivity
curl -s "https://api-testnet.bybit.com/v5/market/tickers?category=linear&symbol=SOLUSDT" | jq '.result.list[0].lastPrice'
```

### Data Collection Issues

```bash
# Check rate limits
python scripts/collect_data.py --test-only

# Collect smaller chunks
python scripts/collect_data.py --days 7 --timeframes "5m"
```

### Backtest Issues

```bash
# Run with debug info
python scripts/enhanced_backtest.py --symbol SOLUSDT --days 7 2>&1 | tee backtest_debug.log

# Check data files
python -c "
import glob
files = glob.glob('data/SOLUSDT_*.csv')
print('Available data files:')
for f in files: print(f'  {f}')
"
```

## 📊 Performance Analysis

### Analyze Results

```bash
# View last backtest results
ls -la backtest_results_*.json | tail -1 | xargs cat | jq '.'

# Extract key metrics
cat backtest_results_*.json | jq '{
  win_rate: .win_rate,
  profit_factor: .profit_factor,
  max_drawdown: .max_drawdown,
  total_return_pct: .total_return_pct,
  total_trades: .total_trades
}'
```

### Compare Different Configurations

```bash
# Test conservative settings
python scripts/enhanced_backtest.py --commission 0.002 --balance 5000

# Test aggressive settings  
python scripts/enhanced_backtest.py --commission 0.0005 --balance 20000
```

## 🎯 Strategy Validation

### Key Questions to Answer

1. **Consistency**: Does strategy work across different market conditions?
2. **Scalability**: Performance with different position sizes?
3. **Robustness**: Sensitive to parameter changes?
4. **Realistic**: Accounts for slippage and commissions?

### Market Condition Testing

```bash
# Test on different periods
python scripts/enhanced_backtest.py --days 7   # Recent volatile period
python scripts/enhanced_backtest.py --days 30  # Medium term
python scripts/enhanced_backtest.py --days 90  # Long term trend
```

### Risk Scenario Testing

```bash
# High commission environment
python scripts/enhanced_backtest.py --commission 0.003

# High slippage simulation (edit slippage in script)
# Smaller position sizes
```

## 📋 Testing Checklist

### Before Live Trading

- [ ] ✅ API connection test passed
- [ ] ✅ Historical data collected (30+ days)
- [ ] ✅ Backtest shows positive results
- [ ] ✅ Win rate ≥55%
- [ ] ✅ Profit factor ≥1.3
- [ ] ✅ Max drawdown ≤10%
- [ ] ✅ Strategy captures $2+ movements
- [ ] ✅ Commission impact <20% of profits
- [ ] ✅ Tested across different market conditions
- [ ] ✅ Paper trading setup verified

### Red Flags - Don't Proceed If:

- [ ] ❌ Connection tests fail repeatedly
- [ ] ❌ Data collection incomplete (<80% expected candles)
- [ ] ❌ Win rate <45%
- [ ] ❌ Profit factor <0.8
- [ ] ❌ Max drawdown >20%
- [ ] ❌ Very few signals generated (<0.5 per day)
- [ ] ❌ Strategy loses money after commissions

## 📞 Getting Help

### Debug Information to Collect

```bash
# System info
python --version
pip list | grep -E "(pandas|numpy|aiohttp|bybit)"

# Environment check
env | grep -E "(BYBIT|TELEGRAM)" | sed 's/=.*/=***/'

# Data status
find data/ -name "*.csv" -exec wc -l {} \;

# Recent logs
tail -50 logs/trading.log
```

### Common Solutions

1. **"No data files found"**: Run `make collect-data` first
2. **"API error 10001"**: Check API key permissions
3. **"Not enough data"**: Collect more historical data
4. **"No trades executed"**: Lower confidence thresholds
5. **"High slippage"**: Check position sizes

---

## 🎉 Success Metrics

If your testing shows:
- ✅ Win rate >55%
- ✅ Profit factor >1.3  
- ✅ Max drawdown <10%
- ✅ Consistent across different periods
- ✅ Realistic commission/slippage handling

**You're ready for paper trading!** 🚀

Run: `make run-paper` to start paper trading mode.