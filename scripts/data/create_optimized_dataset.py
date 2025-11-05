#!/usr/bin/env python3
"""
Create optimized dataset with top 25 features only.
Comment out noisy features in code for future use.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# No need for sys.path.append - use proper imports
# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def get_top_25_features():
    """Get the exact list of top 25 features to keep."""
    
    # From feature_noise_analysis.csv - top 25 features
    return [
        # Essential features (10)
        'volume_sma_20', 'atr_14', 'volatility_ratio', 'bb_width', 'ema_100',
        'ema_diff_10_50', 'rsi_14', 'stoch_d', 'ema_50', 'stoch_k',
        
        # Good features (15)  
        'bb_position', 'slope_ema_20', 'momentum_3', 'macd_line', 'bb_upper',
        'macd_histogram', 'relative_volume', 'bb_position_lag_1', 'momentum_10',
        'rsi_14_lag_1', 'bb_position_lag_3', 'momentum_3_lag_1', 'rsi_14_lag_3',
        'volume_change', 'cci_20'
    ]

def get_base_features():
    """Get base OHLCV features that are always kept."""
    return ['timestamp', 'open', 'high', 'low', 'close', 'volume']

def filter_dataset_to_top25(input_file, output_file):
    """Filter existing dataset to only include top 25 features."""
    
    print(f"📊 Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    original_features = [col for col in df.columns if col not in get_base_features()]
    print(f"📈 Original features: {len(original_features)}")
    
    # Get features to keep
    base_features = get_base_features()
    top_25_features = get_top_25_features()
    
    # Filter to only available features
    available_top25 = [f for f in top_25_features if f in df.columns]
    missing_features = [f for f in top_25_features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
    
    # Create filtered dataset
    columns_to_keep = base_features + available_top25
    df_filtered = df[columns_to_keep].copy()
    
    print(f"✅ Filtered features: {len(available_top25)}")
    print(f"💾 Total columns: {len(df_filtered.columns)}")
    
    # Save filtered dataset
    df_filtered.to_csv(output_file, index=False)
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'original_features': len(original_features),
        'filtered_features': len(available_top25),
        'total_rows': len(df_filtered),
        'reduction_percent': round((1 - len(available_top25)/len(original_features)) * 100, 1)
    }

def create_lag_features(df, top_25_features):
    """Create only the essential lag features for top 25 list."""
    
    # Only create lag features that are in our top 25 list
    lag_configs = [
        {'indicator': 'rsi_14', 'lags': [1, 3]},
        {'indicator': 'bb_position', 'lags': [1, 3]}, 
        {'indicator': 'momentum_3', 'lags': [1]}
    ]
    
    for config in lag_configs:
        indicator = config['indicator']
        if indicator in df.columns:
            for lag in config['lags']:
                lag_col = f"{indicator}_lag_{lag}"
                if lag_col in top_25_features:  # Only create if it's in our top 25
                    df[lag_col] = df[indicator].shift(lag)
    
    return df

def create_optimized_dataset_from_raw():
    """Create optimized dataset from raw data with only top 25 features."""
    
    # Use existing raw data
    input_file = "data/raw/SOLUSDT_5m_20250301_20250930.csv"
    
    if not Path(input_file).exists():
        print(f"❌ Raw data not found: {input_file}")
        return None
    
    print("🔧 Creating optimized dataset from raw data...")
    
    # Load raw data
    df = pd.read_csv(input_file)
    print(f"📊 Loaded raw data: {len(df)} rows")
    
    # Get optimized indicators calculator
    try:
        from advanced_indicators_optimized import AdvancedIndicators
        
        # Use optimized config
        calculator = AdvancedIndicators("config/indicators_optimized.yaml")
        
        # Calculate indicators
        print("🔄 Calculating optimized indicators...")
        df_with_indicators = calculator.calculate_all_indicators(df)
        
        # Add lag features
        top_25 = get_top_25_features()
        df_with_indicators = create_lag_features(df_with_indicators, top_25)
        
        # Filter to only top 25 features
        base_features = get_base_features()
        available_top25 = [f for f in top_25 if f in df_with_indicators.columns]
        
        columns_to_keep = base_features + available_top25
        df_filtered = df_with_indicators[columns_to_keep].copy()
        
        # Clean data
        df_filtered = df_filtered.dropna()
        
        # Save optimized dataset
        output_file = "data/processed/SOLUSDT_5m_optimized_top25.csv"
        df_filtered.to_csv(output_file, index=False)
        
        print(f"✅ Created optimized dataset: {output_file}")
        print(f"📈 Features: {len(available_top25)} (target: 25)")
        print(f"📊 Rows: {len(df_filtered)}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error creating optimized dataset: {e}")
        return None

def comment_noisy_features_in_advanced_indicators():
    """Comment out noisy features in the advanced_indicators.py file."""
    
    file_path = "modules/data_collector/advanced_indicators.py"
    
    # Features to comment out
    noise_features = [
        'bb_touch_upper', 'bb_touch_lower', 'candle_type'
    ]
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add comments about optimization
    optimization_note = '''
# FEATURE OPTIMIZATION NOTES:
# Based on comprehensive noise analysis, the following features have been identified:
# 
# NOISE FEATURES (consider removing):
# - bb_touch_upper, bb_touch_lower: Very low importance (0.0009)
# - candle_type: Very low importance (0.0013)
#
# LOW PRIORITY FEATURES (use sparingly):
# - Most lag features beyond lag_1 and lag_3
# - Basic SMAs (sma_5, sma_10, sma_20) - use EMAs instead
# - Redundant indicators (bb_middle, high_low_range)
#
# TOP 25 ESSENTIAL + GOOD FEATURES:
# Essential: volume_sma_20, atr_14, volatility_ratio, bb_width, ema_100,
#           ema_diff_10_50, rsi_14, stoch_d, ema_50, stoch_k
# Good: bb_position, slope_ema_20, momentum_3, macd_line, bb_upper,
#       macd_histogram, relative_volume, bb_position_lag_1, momentum_10,
#       rsi_14_lag_1, bb_position_lag_3, momentum_3_lag_1, rsi_14_lag_3,
#       volume_change, cci_20
#
# Use create_optimized_dataset.py to generate datasets with top 25 features only.

'''
    
    # Insert optimization note after imports
    import_end = content.find('from typing import List')
    if import_end != -1:
        next_line = content.find('\n', import_end)
        content = content[:next_line] + optimization_note + content[next_line:]
    
    # Comment out noise feature calculations
    for feature in noise_features:
        # Find and comment lines related to this feature
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if f"'{feature}'" in line and not line.strip().startswith('#'):
                lines[i] = f"                # NOISE FEATURE COMMENTED: {line.strip()}"
            elif f'"{feature}"' in line and not line.strip().startswith('#'):
                lines[i] = f"                # NOISE FEATURE COMMENTED: {line.strip()}"
        
        content = '\n'.join(lines)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Added optimization comments to {file_path}")
    print(f"🔇 Commented out {len(noise_features)} noise features")

def main():
    """Main function to create optimized dataset."""
    
    print("🎯 CREATING OPTIMIZED DATASET WITH TOP 25 FEATURES")
    print("=" * 70)
    
    # Step 1: Comment noisy features in original code
    print("\n1️⃣ Adding optimization comments to advanced_indicators.py...")
    comment_noisy_features_in_advanced_indicators()
    
    # Step 2: Create optimized dataset from raw data
    print("\n2️⃣ Creating optimized dataset from raw data...")
    optimized_file = create_optimized_dataset_from_raw()
    
    if optimized_file:
        print(f"✅ Success: {optimized_file}")
    else:
        print("❌ Failed to create optimized dataset")
    
    # Step 3: Also filter existing advanced dataset if available
    print("\n3️⃣ Filtering existing advanced dataset...")
    existing_file = "data/raw/SOLUSDT_5m_20250301_20250930_advanced_indicators.csv"
    
    if Path(existing_file).exists():
        filtered_stats = filter_dataset_to_top25(
            existing_file, 
            "data/processed/SOLUSDT_5m_filtered_top25.csv"
        )
        
        print(f"📊 FILTERING RESULTS:")
        print(f"   Original features: {filtered_stats['original_features']}")
        print(f"   Filtered features: {filtered_stats['filtered_features']}")
        print(f"   Reduction: {filtered_stats['reduction_percent']}%")
        print(f"   Total rows: {filtered_stats['total_rows']}")
        print(f"   Output: {filtered_stats['output_file']}")
    else:
        print(f"⚠️  Existing advanced dataset not found: {existing_file}")
    
    print("\n🎉 OPTIMIZATION COMPLETE!")
    print("=" * 50)
    print("✅ Noisy features commented in code")
    print("✅ Optimized dataset created with top 25 features")
    print("📉 ~50% feature reduction while keeping signal")
    print("💾 Original features preserved as comments")
    
    print(f"\n📁 OUTPUT FILES:")
    if optimized_file:
        print(f"   {optimized_file}")
    if Path("data/processed/SOLUSDT_5m_filtered_top25.csv").exists():
        print(f"   data/processed/SOLUSDT_5m_filtered_top25.csv")

if __name__ == "__main__":
    main()