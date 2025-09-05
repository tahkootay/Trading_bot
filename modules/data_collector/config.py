"""
Configuration management for data collector module.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class DataCollectorConfig:
    """Configuration for historical data collector."""
    
    # Output settings
    output_directory: str = "./data/raw"
    default_output_format: str = "csv"
    
    # API settings
    testnet: bool = False
    rate_limit: int = 10  # requests per second
    
    # Data collection settings
    max_candles_per_request: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Validation settings
    validate_data: bool = True
    allow_gaps: bool = True
    max_gap_hours: int = 24
    
    # Logging settings
    log_level: str = "INFO"
    log_api_requests: bool = False
    
    @classmethod
    def from_file(cls, config_path: str) -> 'DataCollectorConfig':
        """Load configuration from YAML file."""
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create instance with loaded data
        return cls(**data.get('data_collector', {}))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_collector': {
                'output_directory': self.output_directory,
                'default_output_format': self.default_output_format,
                'testnet': self.testnet,
                'rate_limit': self.rate_limit,
                'max_candles_per_request': self.max_candles_per_request,
                'retry_attempts': self.retry_attempts,
                'retry_delay': self.retry_delay,
                'validate_data': self.validate_data,
                'allow_gaps': self.allow_gaps,
                'max_gap_hours': self.max_gap_hours,
                'log_level': self.log_level,
                'log_api_requests': self.log_api_requests
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
        
        if self.rate_limit <= 0:
            raise ValueError("Rate limit must be positive")
        
        if self.max_candles_per_request <= 0 or self.max_candles_per_request > 1000:
            raise ValueError("Max candles per request must be between 1 and 1000")
        
        if self.retry_attempts < 0:
            raise ValueError("Retry attempts cannot be negative")
        
        if self.retry_delay < 0:
            raise ValueError("Retry delay cannot be negative")
        
        if self.default_output_format.lower() not in ['csv', 'json', 'parquet']:
            raise ValueError("Default output format must be csv, json, or parquet")
        
        if self.log_level.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            raise ValueError("Log level must be DEBUG, INFO, WARNING, or ERROR")


# Default configuration template
DEFAULT_CONFIG = """
data_collector:
  # Output settings
  output_directory: "./data/raw"
  default_output_format: "csv"
  
  # API settings
  testnet: false
  rate_limit: 10  # requests per second
  
  # Data collection settings
  max_candles_per_request: 1000
  retry_attempts: 3
  retry_delay: 1.0
  
  # Validation settings
  validate_data: true
  allow_gaps: true
  max_gap_hours: 24
  
  # Logging settings
  log_level: "INFO"
  log_api_requests: false

# Example usage configurations for different scenarios

# High-frequency collection (for recent data)
high_frequency:
  rate_limit: 5  # Lower rate to avoid limits
  max_candles_per_request: 500
  validate_data: true

# Bulk historical collection (for large date ranges)
bulk_collection:
  rate_limit: 8
  max_candles_per_request: 1000
  retry_attempts: 5
  retry_delay: 2.0
  allow_gaps: true

# Testnet configuration
testnet:
  testnet: true
  rate_limit: 5
  output_directory: "./data/testnet"
  log_api_requests: true
"""


def create_default_config(config_path: str) -> None:
    """Create default configuration file."""
    
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(DEFAULT_CONFIG)
    
    print(f"✅ Created default configuration file: {config_path}")


if __name__ == "__main__":
    # Create default config file
    create_default_config("./config.yaml")