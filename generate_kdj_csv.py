#!/usr/bin/env python3
"""
Generate CSV output with KDJ signals and indicator values
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from examples.strategies.kdj_csv_output import KDJCsvOutput
from modules.backtester.backtest_engine import BacktestEngine, BacktestConfig
from modules.backtester.strategy_base import MarketData, Position


def run_kdj_csv_generation(data_file: str, output_file: str = None):
    """Run KDJ strategy and generate CSV with signals."""
    
    print(f"📊 Loading data from: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"📈 Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize strategy
    strategy = KDJCsvOutput()
    strategy.reset()
    
    # Initialize position
    position = Position(
        direction="NONE",
        quantity=0.0,
        entry_price=0.0,
        entry_time=datetime.now(),
        unrealized_pnl=0.0,
        duration_minutes=0
    )
    
    print("🔄 Processing bars and generating signals...")
    
    # Process each bar
    for idx, row in df.iterrows():
        market_data = MarketData(
            timestamp=row['timestamp'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
            symbol='SOLUSDT'
        )
        
        # Get signal from strategy
        signal = strategy.on_bar(market_data, position)
        
        # Update position based on signal (simplified simulation)
        if signal.signal.name == 'BUY' and position.direction == "NONE":
            position.direction = "LONG"
            position.entry_price = market_data.close
            position.entry_time = market_data.timestamp
        elif signal.signal.name == 'CLOSE_LONG' and position.direction == "LONG":
            position.direction = "NONE"
            position.entry_price = 0.0
            position.entry_time = None
    
    # Save CSV output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output/csv_reports/kdj_signals_{timestamp}.csv"
    
    output_path = strategy.save_csv_output(output_file)
    
    # Get strategy info
    info = strategy.get_strategy_info()
    
    print(f"\n✅ CSV generation completed!")
    print(f"📁 Output file: {output_path}")
    print(f"📊 Total bars processed: {info['statistics']['total_bars_processed']}")
    print(f"🔢 Total trades: {info['statistics']['total_trades']}")
    print(f"\n📋 CSV columns:")
    for col in info['csv_columns']:
        print(f"   - {col}")
    
    return output_path


if __name__ == "__main__":
    # Default data file
    data_file = "data/test/SOLUSDT_1h.csv"
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        print("Please provide a valid data file path.")
        sys.exit(1)
    
    # Generate CSV
    try:
        output_path = run_kdj_csv_generation(data_file)
        print(f"\n🎉 Success! Open the CSV file: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)