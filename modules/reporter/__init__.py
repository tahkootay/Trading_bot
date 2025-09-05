"""
HTML Report Generation Module

This module converts backtest results into professional HTML reports with interactive charts.
Takes JSON results from the backtester and generates comprehensive analysis reports.

Main components:
- HTMLReportGenerator: Core HTML report generation
- ChartBuilder: Interactive Plotly.js chart creation  
- TemplateManager: HTML template and styling management
- ReporterConfig: Configuration management
- ReportGenerator: High-level orchestrator

Example usage:
```python
from modules.reporter import ReportGenerator, ReporterConfig
import asyncio

config = ReporterConfig(theme="professional")
generator = ReportGenerator(config)

# Generate report from backtest results
report_path = asyncio.run(generator.generate_report(
    results_file="./output/backtests/my_backtest.json",
    template="comprehensive"
))
```

CLI usage:
```bash
python -m modules.reporter --results ./output/backtests/backtest_results.json --template professional
```
"""

from .html_generator import HTMLReportGenerator
from .chart_builder import ChartBuilder
from .template_manager import TemplateManager
from .config import ReporterConfig
from .main import ReportGenerator

__version__ = "1.0.0"
__all__ = [
    "HTMLReportGenerator",
    "ChartBuilder", 
    "TemplateManager",
    "ReporterConfig",
    "ReportGenerator"
]