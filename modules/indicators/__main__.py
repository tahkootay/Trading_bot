"""
Indicators Module CLI - Calculate technical indicators for trading data.

This module loads CSV data, calculates specified technical indicators,
and saves the enriched data with indicators to a new CSV file.

Usage:
    python -m modules.indicators --input data.csv --output data_with_indicators.csv
    python -m modules.indicators --input data.csv --indicators RSI,MACD,SMA --output enriched.csv
"""

import argparse
import sys
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any

# Check for required dependencies
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
    HAS_NUMPY = True
except ImportError:
    print("Warning: pandas and/or numpy not available. Using basic CSV processing.")
    HAS_PANDAS = False
    HAS_NUMPY = False
    
    # Define minimal numpy-like functions
    class MockNumpy:
        @staticmethod
        def nan():
            return float('nan')
    
    np = MockNumpy()

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Try to import indicators, fall back to basic functionality if not available
try:
    from . import (
        ALL_INDICATORS, get_indicator, list_indicators
    )
    HAS_INDICATORS = True
except ImportError:
    print("Warning: Indicator modules not available. Limited functionality.")
    HAS_INDICATORS = False
    ALL_INDICATORS = ['RSI', 'SMA', 'EMA', 'MACD']


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate technical indicators for trading data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate all indicators
  python -m modules.indicators --input data/raw/SOLUSDT_1h_august.csv --output data/processed/SOLUSDT_indicators.csv
  
  # Calculate specific indicators only
  python -m modules.indicators --input data/raw/SOLUSDT_1h_august.csv --indicators RSI,MACD,SMA --output data/processed/SOLUSDT_custom.csv
  
  # List available indicators
  python -m modules.indicators --list
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
        help='Comma-separated list of indicators to calculate (default: all)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available indicators'
    )
    
    return parser.parse_args()


def validate_csv_columns(df: pd.DataFrame) -> bool:
    """Validate that CSV has required OHLCV columns."""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Check if all required columns exist (case insensitive)
    df_columns_lower = [col.lower() for col in df.columns]
    missing_columns = []
    
    for col in required_columns:
        if col not in df_columns_lower:
            missing_columns.append(col)
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    return True


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase for consistency."""
    df.columns = df.columns.str.lower()
    return df


def calculate_indicator_batch(df: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame:
    """Calculate indicator for entire dataframe."""
    result_df = df.copy()
    
    try:
        # Create indicator instance
        indicator = get_indicator(indicator_name, **kwargs)
        
        # Calculate values for entire series
        values = []
        
        # Determine input data based on indicator type
        if indicator_name.upper() in ['RSI', 'SMA', 'EMA', 'WMA']:
            # Price-based indicators using close price
            input_data = df['close'].values
            for i, price in enumerate(input_data):
                value = indicator.update(price)
                values.append(value if value is not None else np.nan)
        
        elif indicator_name.upper() in ['MACD']:
            # MACD returns multiple values
            input_data = df['close'].values
            macd_line = []
            signal_line = []
            histogram = []
            
            for price in input_data:
                result = indicator.update(price)
                if result is not None:
                    macd_line.append(result['macd'])
                    signal_line.append(result['signal'])
                    histogram.append(result['histogram'])
                else:
                    macd_line.append(np.nan)
                    signal_line.append(np.nan)
                    histogram.append(np.nan)
            
            result_df[f'{indicator_name.lower()}_line'] = macd_line
            result_df[f'{indicator_name.lower()}_signal'] = signal_line
            result_df[f'{indicator_name.lower()}_histogram'] = histogram
            return result_df
        
        elif indicator_name.upper() in ['ATR']:
            # OHLC-based indicators
            for i in range(len(df)):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low'] 
                close = df.iloc[i]['close']
                prev_close = df.iloc[i-1]['close'] if i > 0 else close
                
                value = indicator.update_ohlc(high, low, close, prev_close)
                values.append(value if value is not None else np.nan)
        
        elif indicator_name.upper() in ['BOLLINGER', 'BB']:
            # Bollinger Bands return multiple values
            input_data = df['close'].values
            upper_band = []
            middle_band = []
            lower_band = []
            
            for price in input_data:
                result = indicator.update(price)
                if result is not None:
                    upper_band.append(result['upper'])
                    middle_band.append(result['middle'])
                    lower_band.append(result['lower'])
                else:
                    upper_band.append(np.nan)
                    middle_band.append(np.nan)
                    lower_band.append(np.nan)
            
            result_df[f'{indicator_name.lower()}_upper'] = upper_band
            result_df[f'{indicator_name.lower()}_middle'] = middle_band  
            result_df[f'{indicator_name.lower()}_lower'] = lower_band
            return result_df
        
        elif indicator_name.upper() in ['STOCHASTIC']:
            # Stochastic needs high, low, close
            k_values = []
            d_values = []
            
            for i in range(len(df)):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low']
                close = df.iloc[i]['close']
                
                result = indicator.update_hlc(high, low, close)
                if result is not None:
                    k_values.append(result['%k'])
                    d_values.append(result['%d'])
                else:
                    k_values.append(np.nan)
                    d_values.append(np.nan)
            
            result_df[f'{indicator_name.lower()}_k'] = k_values
            result_df[f'{indicator_name.lower()}_d'] = d_values
            return result_df
        
        else:
            # Default: price-based indicators
            input_data = df['close'].values
            for price in input_data:
                value = indicator.update(price)
                values.append(value if value is not None else np.nan)
        
        # Add single column for simple indicators
        if values:
            result_df[indicator_name.lower()] = values
            
    except Exception as e:
        print(f"Error calculating {indicator_name}: {e}")
        return result_df
    
    return result_df


def calculate_all_indicators(df: pd.DataFrame, selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate all or selected indicators for the dataframe."""
    result_df = df.copy()
    
    # Default parameters for indicators
    indicator_params = {
        'RSI': {'period': 14},
        'SMA': {'period': 20},
        'EMA': {'period': 20},
        'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'ATR': {'period': 14},
        'BOLLINGER': {'period': 20, 'std_dev': 2.0},
        'BB': {'period': 20, 'std_dev': 2.0},
        'STOCHASTIC': {'k_period': 14, 'd_period': 3}
    }
    
    # Use selected indicators or all available ones
    indicators_to_calculate = selected_indicators if selected_indicators else ['RSI', 'SMA', 'EMA', 'MACD', 'ATR', 'BOLLINGER']
    
    print(f"Calculating indicators: {', '.join(indicators_to_calculate)}")
    
    for indicator_name in indicators_to_calculate:
        # Handle aliases
        normalized_name = indicator_name.upper()
        if normalized_name == 'BB':
            normalized_name = 'BOLLINGER'
        elif normalized_name == 'STOCHASTIC':
            normalized_name = 'STOCHASTICOSCILLATOR'
        
        if (normalized_name in [ind.upper() for ind in ALL_INDICATORS] or 
            indicator_name.upper() in ['BB', 'BOLLINGER', 'STOCHASTIC']):
            print(f"  - Calculating {indicator_name}...")
            params = indicator_params.get(normalized_name, {})
            result_df = calculate_indicator_batch(result_df, indicator_name, **params)
        else:
            print(f"  - Warning: Unknown indicator '{indicator_name}', skipping...")
    
    return result_df


def main():
    """Main function."""
    args = parse_arguments()
    
    # List indicators and exit
    if args.list:
        if HAS_INDICATORS:
            list_indicators()
        else:
            print("Available indicators (basic implementation):")
            print("  - RSI (Relative Strength Index)")
            print("  - SMA (Simple Moving Average)")
            print("  - EMA (Exponential Moving Average)")
        return
    
    # Validate arguments
    if not args.input or not args.output:
        print("Error: Both --input and --output are required")
        print("Use --help for more information")
        sys.exit(1)
    
    # If dependencies are missing, use standalone implementation
    if not HAS_PANDAS or not HAS_INDICATORS:
        print("Using standalone implementation due to missing dependencies...")
        import subprocess
        
        # Use the standalone calculator
        calc_path = Path(__file__).parent.parent.parent / "calculate_indicators.py"
        
        cmd = [sys.executable, str(calc_path)]
        cmd.extend(['--input', args.input])
        cmd.extend(['--output', args.output])
        if args.indicators:
            cmd.extend(['--indicators', args.indicators])
        
        result = subprocess.run(cmd, capture_output=False)
        sys.exit(result.returncode)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse selected indicators
    selected_indicators = None
    if args.indicators:
        selected_indicators = [ind.strip() for ind in args.indicators.split(',')]
        print(f"Selected indicators: {selected_indicators}")
    
    try:
        # Load data
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows")
        
        # Validate and normalize column names
        df = normalize_column_names(df)
        if not validate_csv_columns(df):
            sys.exit(1)
        
        # Calculate indicators
        print("Calculating indicators...")
        enriched_df = calculate_all_indicators(df, selected_indicators)
        
        # Save results
        print(f"Saving results to {output_path}...")
        enriched_df.to_csv(output_path, index=False)
        
        # Print summary
        original_columns = len(df.columns)
        new_columns = len(enriched_df.columns)
        added_columns = new_columns - original_columns
        
        print(f"\nSuccess!")
        print(f"  - Original columns: {original_columns}")
        print(f"  - New columns: {new_columns}")
        print(f"  - Added indicators: {added_columns}")
        print(f"  - Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()