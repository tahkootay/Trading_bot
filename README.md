# SOL/USDT Professional Intraday Trading Bot

A sophisticated trading bot designed to capture $2+ price movements on SOL/USDT futures using machine learning models, advanced technical analysis, and comprehensive risk management.

## 🎯 Key Features

- **ML-Powered Signals**: Ensemble of XGBoost, LightGBM, and CatBoost models
- **Multi-Timeframe Analysis**: 1m, 3m, 5m, 15m, 1h data integration
- **Advanced Risk Management**: Kelly Criterion, correlation limits, dynamic stops
- **Smart Order Execution**: TWAP, Iceberg, Market Impact optimization
- **Real-Time Data**: WebSocket integration with Bybit
- **Comprehensive Logging**: Structured logging with performance tracking
- **Paper Trading**: Safe testing environment
- **Backtesting**: Historical performance validation

## 📊 Target Performance

- **Target Movement**: $2+ captures on SOL/USDT
- **Win Rate**: ≥55%
- **Risk/Reward**: 1:1.5 minimum, 1:2 target
- **Max Drawdown**: <10%
- **Daily Trades**: 5-15 signals
- **Position Duration**: 5 minutes - 4 hours

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Bybit API credentials
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd trading-bot
```

2. **Setup environment**
```bash
make setup
```

3. **Configure credentials**
```bash
cp .env.example .env
# Edit .env with your API credentials
```

4. **Start paper trading**
```bash
make run-paper
```

## 📁 Project Structure

```
trading-bot/
├── src/
│   ├── data_collector/      # Real-time data collection
│   ├── feature_engine/      # Technical indicators & features
│   ├── models/             # ML models and predictions
│   ├── signal_generator/   # Trading signal logic
│   ├── execution/          # Order management
│   ├── risk_manager/       # Risk controls
│   ├── notifications/      # Alerts and reporting
│   └── utils/             # Shared utilities
├── config/                 # Configuration files
├── models/                # Trained ML models
├── tests/                 # Test suite
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## ⚙️ Configuration

### Environment Variables

Create `.env` file with:

```bash
# Bybit API
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/trading_bot
REDIS_URL=redis://localhost:6379/0

# Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Trading Configuration

Edit `config/default.yaml`:

```yaml
trading:
  symbol: "SOLUSDT"
  max_position_size: 0.02  # 2% of account
  min_signal_confidence: 0.15
  leverage: 3

risk:
  max_daily_loss: 0.03     # 3%
  max_drawdown: 0.10       # 10%
  max_consecutive_losses: 3
```

## 🛠️ Commands

```bash
# Development
make dev          # Setup development environment
make test         # Run tests
make lint         # Run linting
make format       # Format code

# Trading
make run          # Start live trading
make run-paper    # Start paper trading
make run-debug    # Start with debug logging

# Docker
make docker-build # Build Docker image
make docker-run   # Run in container
```

## 📈 Trading Logic

### Signal Generation

1. **Market Data**: Real-time collection from Bybit WebSocket
2. **Feature Engineering**: 100+ technical and microstructure features
3. **ML Prediction**: Ensemble model confidence scoring
4. **Multi-Level Filtering**:
   - ML confidence > 15%
   - Volume confirmation
   - Trend strength (ADX > 20)
   - Technical alignment
   - Risk limits

### Entry Conditions (BUY)

- P(UP) - P(DOWN) > 0.15
- EMA8 > EMA20 > EMA50
- Price >= VWAP
- ADX >= 20
- Volume z-score > 1.0
- No conflicting positions

### Risk Management

- **Position Sizing**: Kelly Criterion with 25% fraction
- **Stop Loss**: 1.0-1.5 ATR from entry
- **Take Profit**: Multi-level (40% at 1 ATR, 30% at 1.5 ATR, etc.)
- **Time Stop**: Close if no movement in 2 hours
- **Daily Limits**: -3% daily loss, 10% max drawdown

## 🔒 Safety Features

- **Paper Trading Mode**: Test without real money
- **Emergency Stops**: Automatic shutdown on critical events
- **Position Limits**: Maximum 3 concurrent positions
- **Correlation Checks**: Prevent over-concentration
- **API Rate Limiting**: Respect exchange limits
- **Graceful Shutdown**: Clean position closing

## 📊 Monitoring

### Real-Time Metrics

- Current P&L and positions
- Signal performance statistics
- Risk metrics and exposure
- Model confidence trends
- Execution quality (slippage, fill rates)

### Alerts

**Critical** (Immediate):
- Position without stop loss
- Connection lost >30s
- Daily loss >2%
- Model failure

**Important** (5 min):
- Approaching limits
- High slippage
- Low liquidity

**Heartbeat** (5 min):
- System status
- Current positions
- Performance summary

## 🧪 Testing

### Unit Tests
```bash
make test
```

### Backtesting
```bash
python scripts/backtest.py --symbol SOLUSDT --days 30
```

### Paper Trading
```bash
make run-paper
```

Start with paper trading for at least 2 weeks before live trading.

## 📝 Model Training

### Data Requirements

- Minimum 3 months of historical data
- 1000+ labeled examples
- Multiple market conditions (trend/range/high volatility)

### Training Process

```bash
python scripts/train_model.py --data-days 90 --retrain
```

### Model Validation

- Walk-forward analysis
- Out-of-sample testing
- Monte Carlo simulation
- Stress testing scenarios

## 🚨 Risk Warnings

⚠️ **This is real money trading software**

- Start with paper trading
- Use small position sizes initially
- Monitor performance closely
- Never exceed your risk tolerance
- Past performance doesn't guarantee future results

### Before Live Trading

- [ ] Complete 2+ weeks paper trading
- [ ] Verify all risk limits
- [ ] Test emergency procedures
- [ ] Validate API credentials
- [ ] Confirm sufficient account balance
- [ ] Setup monitoring alerts

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Standards

- Write tests for new features
- Follow type hints
- Update documentation
- Ensure risk safety
- Test in paper mode first

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. The developers are not responsible for any financial losses incurred through the use of this software. Always trade responsibly and within your means.

## 📞 Support

- 📧 Email: [your-email]
- 💬 Discord: [discord-link]
- 📖 Documentation: [docs-link]
- 🐛 Issues: GitHub Issues

---

**Remember**: Successful trading requires discipline, patience, and continuous learning. This bot is a tool to assist your trading, not a guarantee of profits.