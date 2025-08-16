"""Configuration management using Pydantic settings."""

from typing import Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class ExchangeConfig(BaseModel):
    """Exchange configuration."""
    name: str = "bybit"
    api_key: str = Field(..., description="Exchange API key")
    api_secret: str = Field(..., description="Exchange API secret")
    testnet: bool = Field(False, description="Use testnet")
    rate_limit: int = Field(10, description="Max requests per second")


class TradingConfig(BaseModel):
    """Trading configuration."""
    symbol: str = Field("SOLUSDT", description="Trading symbol")
    timeframes: List[str] = Field(
        ["1m", "3m", "5m", "15m", "1h"],
        description="Timeframes to monitor"
    )
    max_position_size: float = Field(0.02, description="Max position size as % of balance")
    min_order_size: float = Field(10.0, description="Minimum order size in USDT")
    leverage: int = Field(3, description="Leverage to use")
    
    # Entry conditions
    min_signal_confidence: float = Field(0.15, description="Minimum signal confidence")
    min_volume_ratio: float = Field(1.0, description="Minimum volume ratio")
    min_adx: float = Field(20.0, description="Minimum ADX for trend strength")
    
    # Exit conditions
    take_profit_levels: List[float] = Field(
        [1.0, 1.5, 2.5],
        description="Take profit levels in ATR"
    )
    take_profit_ratios: List[float] = Field(
        [0.4, 0.3, 0.2],
        description="Portion of position to close at each TP level"
    )
    stop_loss_atr: float = Field(1.2, description="Stop loss in ATR")
    trailing_stop_enabled: bool = Field(True, description="Enable trailing stop")
    time_stop_hours: int = Field(2, description="Time stop in hours")


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_daily_loss: float = Field(0.03, description="Max daily loss as % of balance")
    max_drawdown: float = Field(0.10, description="Max drawdown as % of balance")
    max_consecutive_losses: int = Field(3, description="Max consecutive losses")
    pause_after_losses_hours: int = Field(2, description="Pause hours after max losses")
    max_correlation: float = Field(0.8, description="Max correlation between positions")
    max_positions: int = Field(3, description="Maximum concurrent positions")
    position_sizing_method: str = Field("kelly", description="Position sizing method")
    kelly_fraction: float = Field(0.25, description="Kelly fraction multiplier")


class MLConfig(BaseModel):
    """Machine learning configuration."""
    model_version: str = Field("v1.0.0", description="Model version")
    retrain_interval_hours: int = Field(168, description="Model retrain interval")
    min_training_samples: int = Field(1000, description="Minimum samples for training")
    feature_importance_threshold: float = Field(0.001, description="Min feature importance")
    prediction_threshold: float = Field(0.6, description="Minimum prediction confidence")
    ensemble_weights: Dict[str, float] = Field(
        {"xgboost": 0.4, "lightgbm": 0.3, "catboost": 0.3},
        description="Ensemble model weights"
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""
    postgres_url: str = Field(..., description="PostgreSQL connection URL")
    redis_url: str = Field(..., description="Redis connection URL")
    influxdb_url: Optional[str] = Field(None, description="InfluxDB URL for metrics")
    influxdb_token: Optional[str] = Field(None, description="InfluxDB token")
    influxdb_org: Optional[str] = Field(None, description="InfluxDB organization")
    influxdb_bucket: str = Field("trading-bot", description="InfluxDB bucket")


class NotificationConfig(BaseModel):
    """Notification configuration."""
    telegram_enabled: bool = Field(True, description="Enable Telegram notifications")
    telegram_bot_token: Optional[str] = Field(None, description="Telegram bot token")
    telegram_chat_id: Optional[str] = Field(None, description="Telegram chat ID")
    
    # Alert thresholds
    critical_alerts: List[str] = Field(
        ["position_without_sl", "disconnection", "daily_loss_limit"],
        description="Critical alert types"
    )
    important_alerts: List[str] = Field(
        ["approaching_limits", "high_slippage", "low_liquidity"],
        description="Important alert types"
    )


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    prometheus_enabled: bool = Field(True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(8000, description="Prometheus metrics port")
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("json", description="Log format (json/text)")
    heartbeat_interval: int = Field(300, description="Heartbeat interval in seconds")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field("development", description="Environment (dev/staging/prod)")
    debug: bool = Field(False, description="Debug mode")
    paper_trading: bool = Field(True, description="Paper trading mode")
    
    # Component configurations
    exchange: ExchangeConfig
    trading: TradingConfig
    risk: RiskConfig
    ml: MLConfig
    database: DatabaseConfig
    notifications: NotificationConfig
    monitoring: MonitoringConfig
    
    # Performance settings
    max_workers: int = Field(4, description="Max worker threads")
    cache_ttl: int = Field(300, description="Cache TTL in seconds")
    
    # Paths
    models_path: Path = Field("models/", description="Path to ML models")
    logs_path: Path = Field("logs/", description="Path to log files")
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment value."""
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of {allowed}')
        return v
    
    @validator('max_position_size')
    def validate_position_size(cls, v):
        """Validate position size."""
        if not 0 < v <= 0.1:  # Max 10% of balance
            raise ValueError('Position size must be between 0 and 0.1')
        return v
    
    @validator('leverage')
    def validate_leverage(cls, v):
        """Validate leverage."""
        if not 1 <= v <= 10:
            raise ValueError('Leverage must be between 1 and 10')
        return v
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration from file or environment."""
    if config_path:
        # Load from YAML/JSON file if needed
        pass
    
    return Settings()