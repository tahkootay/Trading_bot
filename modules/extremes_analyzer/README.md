# Extremes Analyzer Module v1.0

Statistical analysis module for detecting and analyzing price extremes in cryptocurrency trading data, with focus on extremes followed by significant price movements (≥3 USDT).

## Features (Version 1.0 - MVP)

✅ **Extremes Detection**
- Local minima/maxima detection using sliding window algorithm
- Configurable window size for extremes detection
- Filter extremes by minimum price movement threshold (default: 3 USDT)

✅ **Technical Indicators**
- EMA (20, 50, 200 periods)
- RSI (14 periods)
- Volume analysis with 20-period average
- Trend context relative to EMA200

✅ **Data Analysis**
- Direction and strength of price movements after extremes
- Future price tracking (max/min after extreme)
- Comprehensive DataFrame structure for further analysis

✅ **Visualization**
- Price chart with marked extremes (maxima/minima)
- Summary statistics plots (RSI distribution, movement strength, trend context)
- Export plots as PNG files

✅ **Export Capabilities**
- CSV format for data analysis
- JSON format with metadata
- Configurable output directory

## Installation

Install required dependencies:
```bash
pip install pandas numpy matplotlib
```

## Usage

### Command Line Interface

```bash
# Basic usage
python -m modules.extremes_analyzer --data data/SOLUSDT_5m.csv

# Custom parameters
python -m modules.extremes_analyzer \
  --data data/SOLUSDT_5m.csv \
  --symbol SOLUSDT \
  --timeframe 5m \
  --threshold 3.0 \
  --window 5

# Save results without displaying plots
python -m modules.extremes_analyzer \
  --data data/SOLUSDT_5m.csv \
  --save-only

# Quiet mode for automated analysis
python -m modules.extremes_analyzer \
  --data data/SOLUSDT_5m.csv \
  --no-plots \
  --quiet
```

### Programmatic Usage

```python
from modules.extremes_analyzer import ExtremesAnalyzer

# Initialize analyzer
analyzer = ExtremesAnalyzer(
    min_threshold_usdt=3.0,
    window_size=5,
    output_dir="output/extremes"
)

# Run complete analysis
results = analyzer.run_full_analysis(
    data_path="data/SOLUSDT_5m.csv",
    symbol="SOLUSDT",
    timeframe="5m",
    save_results=True,
    show_plots=True
)

print(f"Found {results['metadata']['extremes_found']} extremes")
```

## Data Format

Input CSV file must contain these columns:
- `timestamp`: DateTime or string timestamp
- `open`: Open price
- `high`: High price  
- `low`: Low price
- `close`: Close price
- `volume`: Volume

Example:
```csv
timestamp,open,high,low,close,volume,symbol,timeframe
2025-09-05 12:40:00,149.87,149.92,149.73,149.91,10611.2,SOLUSDT,5m
2025-09-05 12:45:00,149.91,150.01,149.83,149.94,11822.9,SOLUSDT,5m
```

## Output Format

### DataFrame Structure

```python
columns = [
    "id", "timestamp", "symbol", "timeframe",
    "extreme_type", "extreme_price",
    "ema20", "ema50", "ema200",
    "rsi14",
    "bb_upper", "bb_middle", "bb_lower",      # v1.1
    "macd", "macd_signal", "macd_hist",      # v1.1
    "volume", "volume_avg20",
    "atr",                                    # v1.1
    "max_price_after", "min_price_after",
    "price_change", "direction", "strength",
    "hit_threshold", "time_to_hit",          # v1.1
    "trend_context", "volatility_context"    # v1.1
]
```

### Example Output

```
🎯 Detected Extremes Summary:
================================================================================
📅 2025-09-05 13:15:00
   Type: MIN    Price: $203.79
   Movement: $6.20
   RSI: 93.8
   Trend: unknown

📅 2025-09-05 14:55:00  
   Type: MAX    Price: $208.45
   Movement: $6.98
   RSI: 64.5
   Trend: unknown
```

## Algorithm Details

### Extremes Detection
1. **Local Maxima**: Point higher than all points within window_size on both sides
2. **Local Minima**: Point lower than all points within window_size on both sides
3. **Threshold Filter**: Only include extremes followed by ≥ threshold USDT movement

### Performance
- Optimized for datasets up to 50K candles
- Uses numpy vectorization for price calculations
- Sliding window approach instead of scipy dependencies

## File Structure

```
modules/extremes_analyzer/
├── __init__.py              # Module exports
├── analyzer.py              # Main analyzer class
├── extremes_detector.py     # Core detection algorithms  
├── visualizer.py            # Plotting and visualization
├── __main__.py              # Command-line interface
└── README.md                # This file
```

## Development Roadmap

### 🚧 Version 1.1 - Enhanced Analytics
- [ ] Bollinger Bands indicators
- [ ] MACD indicators  
- [ ] ATR volatility analysis
- [ ] Time-to-threshold calculation
- [ ] Volatility context classification

### 🚧 Version 2.0 - Statistical Analysis
- [ ] Correlation heatmaps
- [ ] Automated PDF/HTML reports
- [ ] Multi-timeframe analysis

### 🚧 Version 3.0 - Machine Learning Integration
- [ ] Feature engineering for ML models
- [ ] Predictive models for movement direction
- [ ] Model performance metrics

## Examples

See `examples/` directory for:
- Custom strategy integration examples
- Batch processing scripts
- Advanced analysis workflows

## Contributing

Follow the project's modular architecture:
- No cross-module imports
- File-based communication (CSV/JSON)
- Independent module testing
- PEP 8 compliance with 88-character limit

## License

Part of the modular trading system architecture.