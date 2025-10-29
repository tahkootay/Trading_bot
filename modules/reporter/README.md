# Reporter Module

The reporter module provides comprehensive reporting capabilities for trading strategies, including HTML reports and Excel exports with color-coded visualization.

## Components

### Excel Exporter (`excel_exporter.py`)

Professional Excel export functionality with color-coded visualization for trading strategy analysis.

**Features:**
- Color-coded position and signal visualization
- Automatic column width adjustment  
- Legend worksheet with strategy parameters
- Support for multiple data formats
- Professional Excel formatting

**Usage:**

```python
from modules.reporter.excel_exporter import ExcelExporter

# Create exporter
exporter = ExcelExporter()

# Export strategy signals
data = [
    {
        'timestamp': '2025-06-01 12:00',
        'signal': 'BUY', 
        'position': 'NONE',
        'k_value': 45.2,
        'd_value': 43.1,
        'entry_price': 0.0
    }
]

strategy_info = {
    'strategy_name': 'KDJ Strategy',
    'parameters': {'profit_target': 2.0}
}

output_path = exporter.export_strategy_signals(
    data=data,
    output_path='output/strategy_signals.xlsx', 
    strategy_info=strategy_info
)
```

**Color Scheme:**
- **LONG Position**: Light Green - Currently holding long position
- **BUY Signal**: Lime Green - Entry signal generated  
- **SELL Signal**: Light Red - Exit signal generated
- **HOLD Signal**: Gold - Holding existing position
- **NONE**: Light Gray - No position, no signal
- **Headers**: Steel Blue with white text

### Integration with Strategies

The Excel exporter is integrated into strategy classes via the `save_excel_output()` method:

```python
# In strategy class
def save_excel_output(self, output_path: str = None) -> str:
    from modules.reporter.excel_exporter import ExcelExporter
    
    exporter = ExcelExporter()
    strategy_info = self.get_strategy_info()
    
    return exporter.export_strategy_signals(
        data=self.state['csv_data'],
        output_path=output_path,
        strategy_info=strategy_info
    )
```

## Dependencies

The Excel export functionality requires:
- `openpyxl` for Excel file creation
- `pandas` for data manipulation

Install with:
```bash
pip install openpyxl pandas
```

## Output Structure

Excel files contain:
1. **Main Sheet**: Strategy signals with color-coded rows
2. **Legend Sheet**: Color explanations and strategy parameters

## File Organization

- `excel_exporter.py` - Core Excel export functionality
- `html_generator.py` - HTML report generation (existing)
- `chart_builder.py` - Chart creation (existing)
- `template_manager.py` - Template management (existing)

## Migration Notes

The Excel export functionality was extracted from individual strategy files into this centralized module to:
- Reduce code duplication
- Provide consistent formatting
- Enable easier maintenance
- Support multiple export formats
- Follow modular architecture principles

All existing strategy Excel export methods now delegate to this module while maintaining the same API.