# Deployment Guide for SOL/USDT Trading Bot

## 🚀 Quick Start Guide

### Prerequisites

1. **Python 3.11+** installed
2. **Bybit account** with API access
3. **8GB+ RAM** recommended for production
4. **Stable internet connection** (low latency preferred)

### Step 1: Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
```

### Step 2: Configure API Credentials

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Bybit credentials
nano .env
```

Required variables:
```bash
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token  # Optional
TELEGRAM_CHAT_ID=your_chat_id              # Optional
```

### Step 3: Test Setup

```bash
# Run paper trading to test setup
make run-paper

# Should see output like:
# 🤖 Starting SOL/USDT Trading Bot...
# 🔧 Environment: development
# 📊 Paper Trading: True
```

### Step 4: Backtesting (Recommended)

```bash
# Run backtest on historical data
python scripts/backtest.py --days 30 --balance 10000

# Should show performance metrics:
# ✅ GOOD PERFORMANCE - Strategy meets target criteria
```

### Step 5: Live Trading (After Testing)

```bash
# Edit config for live trading
nano config/default.yaml

# Set paper_trading: false
# Adjust position sizes appropriately

# Start live trading
make run
```

## 📊 Configuration Tuning

### Risk Management Settings

Key parameters in `config/default.yaml`:

```yaml
trading:
  max_position_size: 0.02  # 2% of account per trade
  leverage: 3              # Conservative leverage
  
risk:
  max_daily_loss: 0.03     # Stop trading at -3% daily loss
  max_drawdown: 0.10       # Emergency stop at 10% drawdown
  max_consecutive_losses: 3 # Pause after 3 losses
```

### Signal Generation Tuning

```yaml
trading:
  min_signal_confidence: 0.15  # Minimum ML confidence
  min_volume_ratio: 1.0        # Volume confirmation
  min_adx: 20.0               # Trend strength requirement
```

### Execution Optimization

```yaml
execution:
  max_slippage: 0.002     # 0.2% maximum slippage
  order_timeout: 30       # 30 seconds order timeout
  default_strategy: "SMART" # Smart order routing
```

## 🔐 Security Best Practices

### API Key Security

1. **Use trading-only API keys** (no withdrawal permissions)
2. **Enable IP whitelist** in Bybit settings
3. **Store keys in environment variables** (never in code)
4. **Rotate keys regularly** (monthly recommended)

### Infrastructure Security

```bash
# Set proper file permissions
chmod 600 .env
chmod 600 config/*.yaml

# Use separate user for trading bot
sudo useradd -m tradingbot
sudo -u tradingbot python src/main.py
```

### Monitoring Security

- Monitor for unusual API activity
- Set up alerts for unexpected behavior
- Keep logs for audit trail
- Regularly review trade performance

## 📈 Monitoring & Alerts

### Telegram Notifications (Recommended)

1. Create Telegram bot with @BotFather
2. Get bot token and your chat ID
3. Configure in `.env`
4. Test with: `make run-paper`

### Performance Monitoring

Key metrics to watch:
- **Daily P&L**: Should align with targets
- **Win Rate**: Target ≥55%
- **Drawdown**: Should stay <10%
- **Execution Quality**: Low slippage, high fill rates

### System Health Checks

```bash
# Check bot status
curl localhost:8000/health

# View recent performance
tail -f logs/trading.log

# Monitor resource usage
htop
```

## 🚨 Emergency Procedures

### Emergency Stop

```bash
# Method 1: Graceful shutdown (recommended)
pkill -SIGTERM -f "python.*main.py"

# Method 2: Force stop all positions
python scripts/emergency_stop.py --confirm

# Method 3: Manual intervention via Bybit web interface
```

### Common Issues & Solutions

#### Bot won't start
```bash
# Check dependencies
pip install -r requirements.txt

# Verify API credentials
python -c "from src.data_collector.bybit_client import BybitHTTPClient; print('OK')"

# Check configuration
python -c "from src.utils.config import load_config; load_config()"
```

#### No signals generated
- Check market hours (bot is less active on weekends)
- Verify minimum confidence thresholds
- Check if risk limits are preventing trades
- Review market volatility (bot needs movement)

#### High slippage
- Reduce position sizes
- Use limit orders instead of market
- Check for low liquidity periods
- Consider smaller, more frequent trades

## 📊 Performance Optimization

### ML Model Management

```bash
# Retrain models weekly
python scripts/train_model.py --retrain

# Update model version in config
model_version: "v1.1.0"

# Monitor model performance
python scripts/model_analysis.py
```

### Strategy Optimization

1. **Backtest with different parameters**
2. **A/B test new features in paper trading**
3. **Monitor feature importance changes**
4. **Adjust thresholds based on market conditions**

### Infrastructure Scaling

For high-frequency trading:
```bash
# Use faster VPS near exchange
# Optimize Python code with Cython
# Use Redis for caching
# Implement connection pooling
```

## 🛡️ Risk Management Checklist

### Pre-Launch Checklist

- [ ] ✅ Paper trading successful for 2+ weeks
- [ ] ✅ Backtest shows positive results
- [ ] ✅ Risk limits properly configured
- [ ] ✅ Emergency procedures tested
- [ ] ✅ Monitoring systems active
- [ ] ✅ API keys restricted to trading only
- [ ] ✅ Start with small position sizes
- [ ] ✅ Team trained on emergency procedures

### Daily Monitoring

- [ ] Check overnight performance
- [ ] Review risk metrics
- [ ] Verify system connectivity
- [ ] Monitor for unusual patterns
- [ ] Check execution quality

### Weekly Reviews

- [ ] Analyze strategy performance
- [ ] Review risk-adjusted returns
- [ ] Update ML models if needed
- [ ] Adjust parameters based on results
- [ ] Backup configuration and logs

## 📞 Support & Troubleshooting

### Log Analysis

```bash
# View trading decisions
grep "Trading signal" logs/trading.log

# Check for errors
grep "ERROR" logs/trading.log

# Monitor performance
grep "Performance update" logs/trading.log
```

### Debug Mode

```bash
# Run with debug logging
make run-debug

# Increased verbosity shows:
# - Feature calculations
# - Signal reasoning
# - Risk checks
# - Order execution details
```

### Common Debugging Commands

```bash
# Test API connection
python -c "import asyncio; from src.data_collector.bybit_client import BybitHTTPClient; asyncio.run(BybitHTTPClient('key', 'secret').get_ticker('SOLUSDT'))"

# Validate configuration
python -c "from src.utils.config import load_config; print(load_config())"

# Test feature generation
python scripts/test_features.py --symbol SOLUSDT
```

## 🔄 Maintenance Schedule

### Daily
- Monitor performance and alerts
- Check system health
- Review overnight results

### Weekly
- Analyze trading performance
- Update ML models if needed
- Review and adjust parameters

### Monthly
- Full system audit
- Rotate API keys
- Update dependencies
- Review strategy effectiveness

### Quarterly
- Complete strategy review
- Infrastructure upgrade planning
- Risk management assessment
- Performance benchmarking

---

## ⚠️ Final Warnings

1. **Start Small**: Begin with minimal position sizes
2. **Monitor Closely**: Watch the bot especially in first weeks
3. **Have Limits**: Never risk more than you can afford to lose
4. **Stay Updated**: Keep monitoring market conditions
5. **Manual Override**: Always be ready to intervene manually

**Remember**: This bot is a tool to assist trading, not a guarantee of profits. Success requires proper configuration, monitoring, and risk management.