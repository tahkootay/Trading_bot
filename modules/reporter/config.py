"""
Configuration management for reporter module.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class ReporterConfig:
    """Configuration for HTML report generator."""
    
    # Output settings
    output_directory: str = "./output/reports"
    default_template: str = "comprehensive"
    theme: str = "professional"
    
    # Chart settings
    interactive_charts: bool = True
    chart_width: int = 800
    chart_height: int = 400
    charts_only: bool = False
    
    # Content settings
    include_trade_details: bool = True
    max_trades_in_table: int = 100
    include_risk_analysis: bool = True
    include_monthly_breakdown: bool = True
    
    # Styling settings
    custom_logo_path: Optional[str] = None
    custom_css_path: Optional[str] = None
    font_family: str = "Segoe UI, Tahoma, Geneva, Verdana, sans-serif"
    
    # Performance settings
    lazy_load_charts: bool = False
    compress_output: bool = False
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ReporterConfig':
        """Load configuration from YAML file."""
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create instance with loaded data
        return cls(**data.get('reporter', {}))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'reporter': {
                'output_directory': self.output_directory,
                'default_template': self.default_template,
                'theme': self.theme,
                'interactive_charts': self.interactive_charts,
                'chart_width': self.chart_width,
                'chart_height': self.chart_height,
                'charts_only': self.charts_only,
                'include_trade_details': self.include_trade_details,
                'max_trades_in_table': self.max_trades_in_table,
                'include_risk_analysis': self.include_risk_analysis,
                'include_monthly_breakdown': self.include_monthly_breakdown,
                'custom_logo_path': self.custom_logo_path,
                'custom_css_path': self.custom_css_path,
                'font_family': self.font_family,
                'lazy_load_charts': self.lazy_load_charts,
                'compress_output': self.compress_output
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
        
        if self.default_template not in ['summary', 'comprehensive', 'professional', 'minimal']:
            raise ValueError("Invalid default template")
        
        if self.theme not in ['light', 'dark', 'blue', 'professional']:
            raise ValueError("Invalid theme")
        
        if self.chart_width <= 0 or self.chart_height <= 0:
            raise ValueError("Chart dimensions must be positive")
        
        if self.max_trades_in_table < 0:
            raise ValueError("Max trades in table cannot be negative")
        
        if self.custom_logo_path and not Path(self.custom_logo_path).exists():
            raise ValueError(f"Custom logo file not found: {self.custom_logo_path}")
        
        if self.custom_css_path and not Path(self.custom_css_path).exists():
            raise ValueError(f"Custom CSS file not found: {self.custom_css_path}")


# Default configuration template
DEFAULT_CONFIG = """
reporter:
  # Output settings
  output_directory: "./output/reports"
  default_template: "comprehensive"  # summary, comprehensive, professional, minimal
  theme: "professional"  # light, dark, blue, professional
  
  # Chart settings
  interactive_charts: true
  chart_width: 800
  chart_height: 400
  charts_only: false  # Generate only charts for debugging
  
  # Content settings
  include_trade_details: true
  max_trades_in_table: 100
  include_risk_analysis: true
  include_monthly_breakdown: true
  
  # Styling settings
  custom_logo_path: null  # Path to custom logo image
  custom_css_path: null   # Path to custom CSS file
  font_family: "Segoe UI, Tahoma, Geneva, Verdana, sans-serif"
  
  # Performance settings
  lazy_load_charts: false
  compress_output: false

# Template-specific configurations
templates:
  summary:
    include_trade_details: false
    include_risk_analysis: false
    max_trades_in_table: 10
  
  comprehensive:
    include_trade_details: true
    include_risk_analysis: true
    include_monthly_breakdown: true
    max_trades_in_table: 100
  
  professional:
    interactive_charts: true
    theme: "professional"
    include_trade_details: true
    include_risk_analysis: true
  
  minimal:
    interactive_charts: false
    include_trade_details: false
    include_risk_analysis: false
    max_trades_in_table: 0

# Theme configurations
themes:
  light:
    primary_color: "#74b9ff"
    background_color: "#ffffff"
    text_color: "#2d3436"
  
  dark:
    primary_color: "#6c5ce7"
    background_color: "#2d3436"
    text_color: "#ddd"
  
  blue:
    primary_color: "#0984e3"
    background_color: "#ffffff" 
    text_color: "#2d3436"
  
  professional:
    primary_color: "#2c3e50"
    background_color: "#ffffff"
    text_color: "#2c3e50"
"""


def create_default_config(config_path: str) -> None:
    """Create default configuration file."""
    
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(DEFAULT_CONFIG)
    
    print(f"✅ Created default reporter configuration: {config_path}")


if __name__ == "__main__":
    # Create default config file
    create_default_config("./reporter_config.yaml")