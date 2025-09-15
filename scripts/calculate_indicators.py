#!/usr/bin/env python3
"""
Standalone Indicators Calculator - Works without external dependencies.

This script loads CSV data, calculates technical indicators,
and saves enriched data to a new CSV file.

Usage:
    python3 calculate_indicators.py --input data.csv --output enriched.csv
    python3 calculate_indicators.py --input data.csv --indicators RSI,SMA --output custom.csv
"""

import argparse
import sys
import csv
import math
from pathlib import Path
from typing import List, Optional, Dict, Any


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate technical indicators for trading data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate all indicators
  python3 calculate_indicators.py --input data/raw/SOLUSDT_1h_august.csv --output data/processed/SOLUSDT_indicators.csv
  
  # Calculate specific indicators only
  python3 calculate_indicators.py --input data/raw/SOLUSDT_1h_august.csv --indicators RSI,SMA --output data/processed/SOLUSDT_custom.csv
  
  # List available indicators
  python3 calculate_indicators.py --list
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input CSV file with OHLCV data'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV file with calculated indicators'
    )
    
    parser.add_argument(
        '--indicators',
        type=str,
        help='Comma-separated list of indicators to calculate (default: RSI,SMA,EMA)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available indicators'
    )
    
    return parser.parse_args()


def simple_moving_average(values: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average."""
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(float('nan'))
        else:
            window = values[i - period + 1:i + 1]
            avg = sum(window) / len(window)
            result.append(round(avg, 4))
    return result


def exponential_moving_average(values: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average."""
    result = []
    multiplier = 2 / (period + 1)
    
    for i, value in enumerate(values):
        if i == 0:
            result.append(value)
        else:
            ema = (value * multiplier) + (result[i-1] * (1 - multiplier))
            result.append(round(ema, 4))
    
    return result


def rsi(values: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index."""
    if len(values) < period + 1:
        return [float('nan')] * len(values)
    
    # Calculate price changes
    changes = []
    for i in range(1, len(values)):
        change = values[i] - values[i-1]
        changes.append(change)
    
    result = [float('nan')] * (period)  # First 'period' values are NaN
    
    # Calculate initial averages
    gains = [max(0, change) for change in changes[:period]]
    losses = [abs(min(0, change)) for change in changes[:period]]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    # Calculate RSI for initial period
    if avg_loss == 0:
        rsi_value = 100
    else:
        rs = avg_gain / avg_loss
        rsi_value = 100 - (100 / (1 + rs))
    result.append(round(rsi_value, 2))
    
    # Calculate RSI for remaining periods using smoothed averages
    for i in range(period, len(changes)):
        change = changes[i]
        gain = max(0, change)
        loss = abs(min(0, change))
        
        # Smoothed averages
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            rsi_value = 100
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100 - (100 / (1 + rs))
        
        result.append(round(rsi_value, 2))
    
    return result


def bollinger_bands(values: List[float], period: int = 20, std_multiplier: float = 2.0) -> Dict[str, List[float]]:
    """Calculate Bollinger Bands."""
    sma_values = simple_moving_average(values, period)
    
    upper_band = []
    lower_band = []
    
    for i in range(len(values)):
        if i < period - 1:
            upper_band.append(float('nan'))
            lower_band.append(float('nan'))
        else:
            # Calculate standard deviation for the period
            window = values[i - period + 1:i + 1]
            mean = sum(window) / len(window)
            
            # Calculate sample standard deviation (divide by n-1 for sample, n for population)
            # Most trading platforms use sample standard deviation
            variance = sum((x - mean) ** 2 for x in window) / (len(window) - 1)
            std_dev = math.sqrt(variance)
            
            
            upper = sma_values[i] + (std_multiplier * std_dev)
            lower = sma_values[i] - (std_multiplier * std_dev)
            
            upper_band.append(round(upper, 4))
            lower_band.append(round(lower, 4))
    
    return {
        'middle': sma_values,
        'upper': upper_band,
        'lower': lower_band
    }


def macd(values: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[float]]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = exponential_moving_average(values, fast_period)
    ema_slow = exponential_moving_average(values, slow_period)
    
    # MACD line = EMA(fast) - EMA(slow)
    macd_line = []
    for i in range(len(values)):
        if math.isnan(ema_fast[i]) or math.isnan(ema_slow[i]):
            macd_line.append(float('nan'))
        else:
            macd_line.append(round(ema_fast[i] - ema_slow[i], 4))
    
    # Signal line = EMA of MACD line
    # Filter out NaN values for signal calculation
    macd_non_nan = [x for x in macd_line if not math.isnan(x)]
    if len(macd_non_nan) >= signal_period:
        signal_ema = exponential_moving_average(macd_non_nan, signal_period)
        
        # Map back to original length
        signal_line = [float('nan')] * len(values)
        signal_start_idx = len(values) - len(signal_ema)
        for i, sig_val in enumerate(signal_ema):
            signal_line[signal_start_idx + i] = sig_val
    else:
        signal_line = [float('nan')] * len(values)
    
    # Histogram = MACD - Signal
    histogram = []
    for i in range(len(values)):
        if math.isnan(macd_line[i]) or math.isnan(signal_line[i]):
            histogram.append(float('nan'))
        else:
            histogram.append(round(macd_line[i] - signal_line[i], 4))
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def read_csv_data(file_path: str) -> tuple[List[str], List[Dict[str, str]]]:
    """Read CSV file and return headers and data."""
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames
        data = list(reader)
    return headers, data


def write_csv_data(file_path: str, headers: List[str], data: List[Dict[str, Any]]):
    """Write data to CSV file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)


def calculate_indicators(data: List[Dict[str, str]], indicators: List[str]) -> List[Dict[str, Any]]:
    """Calculate selected indicators for the data."""
    # Extract close prices
    close_prices = []
    for row in data:
        try:
            price = float(row.get('close', row.get('Close', 0)))
            close_prices.append(price)
        except (ValueError, TypeError):
            close_prices.append(0.0)
    
    print(f"Calculating indicators for {len(close_prices)} data points...")
    
    # Calculate each indicator
    indicator_data = {}
    
    for indicator in indicators:
        indicator_upper = indicator.upper()
        print(f"  - Calculating {indicator_upper}...")
        
        if indicator_upper == 'SMA':
            indicator_data['sma_20'] = simple_moving_average(close_prices, 20)
        elif indicator_upper == 'EMA':
            indicator_data['ema_20'] = exponential_moving_average(close_prices, 20)
        elif indicator_upper == 'RSI':
            indicator_data['rsi_14'] = rsi(close_prices, 14)
        elif indicator_upper == 'BOLLINGER' or indicator_upper == 'BB':
            bb_data = bollinger_bands(close_prices, 20, 2.0)
            indicator_data['bb_upper'] = bb_data['upper']
            indicator_data['bb_middle'] = bb_data['middle'] 
            indicator_data['bb_lower'] = bb_data['lower']
        elif indicator_upper == 'MACD':
            macd_data = macd(close_prices, 12, 26, 9)
            indicator_data['macd_line'] = macd_data['macd']
            indicator_data['macd_signal'] = macd_data['signal']
            indicator_data['macd_histogram'] = macd_data['histogram']
        else:
            print(f"    Warning: Indicator {indicator} not implemented")
    
    # Combine original data with indicators
    result = []
    for i, row in enumerate(data):
        new_row = dict(row)  # Copy original data
        
        # Add indicator values
        for indicator_name, values in indicator_data.items():
            if i < len(values):
                value = values[i]
                new_row[indicator_name] = value if not math.isnan(value) else ''
            else:
                new_row[indicator_name] = ''
        
        result.append(new_row)
    
    return result


def list_indicators():
    """List all available indicators."""
    print("Available Technical Indicators:")
    print("\nBasic Indicators:")
    print("  - RSI (Relative Strength Index, 14 period)")
    print("  - SMA (Simple Moving Average, 20 period)")
    print("  - EMA (Exponential Moving Average, 20 period)")
    print("  - MACD (Moving Average Convergence Divergence, 12/26/9)")
    print("  - BOLLINGER/BB (Bollinger Bands, 20 period, 2 std dev)")
    print("\nUsage examples:")
    print("  --indicators RSI")
    print("  --indicators RSI,SMA,EMA")
    print("  --indicators MACD,BOLLINGER")


def main():
    """Main function."""
    args = parse_arguments()
    
    # List indicators and exit
    if args.list:
        list_indicators()
        return
    
    # Validate arguments
    if not args.input or not args.output:
        print("Error: Both --input and --output are required")
        print("Use --help for more information")
        sys.exit(1)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)
    
    # Parse selected indicators
    if args.indicators:
        selected_indicators = [ind.strip() for ind in args.indicators.split(',')]
    else:
        selected_indicators = ['RSI', 'SMA', 'EMA']
    
    try:
        # Load data
        print(f"Loading data from {input_path}...")
        headers, data = read_csv_data(str(input_path))
        print(f"Loaded {len(data)} rows with columns: {headers}")
        
        # Validate required columns
        required_columns = ['close']
        available_columns = [col.lower() for col in headers]
        
        if not any(col in available_columns for col in required_columns):
            print(f"Error: No 'close' price column found. Available: {headers}")
            sys.exit(1)
        
        # Calculate indicators
        enriched_data = calculate_indicators(data, selected_indicators)
        
        # Determine output headers
        output_headers = list(headers)
        
        # Add headers based on selected indicators
        for indicator in selected_indicators:
            if indicator.upper() == 'SMA':
                output_headers.append('sma_20')
            elif indicator.upper() == 'EMA':
                output_headers.append('ema_20')
            elif indicator.upper() == 'RSI':
                output_headers.append('rsi_14')
            elif indicator.upper() in ['BOLLINGER', 'BB']:
                output_headers.extend(['bb_upper', 'bb_middle', 'bb_lower'])
            elif indicator.upper() == 'MACD':
                output_headers.extend(['macd_line', 'macd_signal', 'macd_histogram'])
        
        # Save results
        print(f"Saving results to {output_path}...")
        write_csv_data(str(output_path), output_headers, enriched_data)
        
        # Print summary
        original_columns = len(headers)
        new_columns = len(output_headers)
        added_columns = new_columns - original_columns
        
        print(f"\nSuccess!")
        print(f"  - Original columns: {original_columns}")
        print(f"  - New columns: {new_columns}")
        print(f"  - Added indicators: {added_columns}")
        print(f"  - Selected indicators: {', '.join(selected_indicators)}")
        print(f"  - Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()