"""
Configuration management for trading bot module.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class TradingBotConfig:
    """Configuration for live trading bot."""
    
    # Exchange settings
    exchange: str = "bybit"
    symbol: str = "SOLUSDT"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = False
    paper_trading: bool = True
    
    # Trading settings
    max_position_size: float = 0.1  # 10% of capital
    initial_capital: float = 10000.0
    
    # Risk management
    daily_loss_limit_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_consecutive_losses: int = 3
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    
    # Operational settings
    loop_interval_seconds: int = 5
    heartbeat_interval_seconds: int = 30
    log_level: str = "INFO"
    
    # Data settings
    timeframe: str = "5m"
    lookback_periods: int = 200
    
    # Notifications
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    email_notifications: bool = False
    
    # Safety settings
    trading_hours_start: str = "00:00"
    trading_hours_end: str = "23:59"
    emergency_stop: bool = False
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TradingBotConfig':
        """Load configuration from YAML file."""
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create instance with loaded data
        return cls(**data.get('trading_bot', {}))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'trading_bot': {
                'exchange': self.exchange,
                'symbol': self.symbol,
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'testnet': self.testnet,
                'paper_trading': self.paper_trading,
                'max_position_size': self.max_position_size,
                'initial_capital': self.initial_capital,
                'daily_loss_limit_pct': self.daily_loss_limit_pct,
                'max_drawdown_pct': self.max_drawdown_pct,
                'max_consecutive_losses': self.max_consecutive_losses,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'loop_interval_seconds': self.loop_interval_seconds,
                'heartbeat_interval_seconds': self.heartbeat_interval_seconds,
                'log_level': self.log_level,
                'timeframe': self.timeframe,
                'lookback_periods': self.lookback_periods,
                'telegram_bot_token': self.telegram_bot_token,
                'telegram_chat_id': self.telegram_chat_id,
                'email_notifications': self.email_notifications,
                'trading_hours_start': self.trading_hours_start,
                'trading_hours_end': self.trading_hours_end,
                'emergency_stop': self.emergency_stop
            }
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """Validate configuration settings."""
        
        if self.exchange not in ["bybit", "binance"]:
            raise ValueError("Unsupported exchange")
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")
        
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if self.daily_loss_limit_pct <= 0 or self.daily_loss_limit_pct > 50:
            raise ValueError("Daily loss limit must be between 0 and 50%")
        
        if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 50:
            raise ValueError("Max drawdown must be between 0 and 50%")
        
        if self.loop_interval_seconds <= 0:
            raise ValueError("Loop interval must be positive")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("Invalid log level")
        
        if not self.paper_trading and not self.api_key:
            raise ValueError("API key required for live trading")


# Default configuration template
DEFAULT_CONFIG = """
trading_bot:
  # Exchange settings
  exchange: "bybit"
  symbol: "SOLUSDT"
  api_key: null  # Set your API key here
  api_secret: null  # Set your API secret here
  testnet: false
  paper_trading: true  # IMPORTANT: Set to false only for live trading
  
  # Trading settings
  max_position_size: 0.1  # 10% of capital per trade
  initial_capital: 10000.0
  
  # Risk management
  daily_loss_limit_pct: 3.0
  max_drawdown_pct: 10.0
  max_consecutive_losses: 3
  stop_loss_pct: 2.0
  take_profit_pct: 4.0
  
  # Operational settings
  loop_interval_seconds: 5
  heartbeat_interval_seconds: 30
  log_level: "INFO"
  
  # Data settings
  timeframe: "5m"
  lookback_periods: 200
  
  # Notifications
  telegram_bot_token: null
  telegram_chat_id: null
  email_notifications: false
  
  # Safety settings
  trading_hours_start: "00:00"
  trading_hours_end: "23:59"
  emergency_stop: false

# Environment-specific configurations

# Development configuration
development:
  paper_trading: true
  testnet: true
  log_level: "DEBUG"
  loop_interval_seconds: 10

# Production configuration  
production:
  paper_trading: false  # WARNING: This enables live trading
  testnet: false
  log_level: "INFO"
  loop_interval_seconds: 5
  
  # Enhanced risk controls for production
  daily_loss_limit_pct: 2.0
  max_drawdown_pct: 5.0
  max_position_size: 0.05  # More conservative position sizing

# Paper trading configuration
paper_trading:
  paper_trading: true
  testnet: true
  initial_capital: 100000.0  # Larger capital for testing
  max_position_size: 0.2
  log_level: "INFO"
"""


def create_default_config(config_path: str) -> None:
    """Create default configuration file."""
    
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(DEFAULT_CONFIG)
    
    print(f"✅ Created default trading bot configuration: {config_path}")
    print("⚠️  Remember to set your API keys before live trading!")


if __name__ == "__main__":
    # Create default config file
    create_default_config("./trading_bot_config.yaml")