#!/usr/bin/env python3
"""
Add missing lag features to the filtered dataset to complete top 25 features.
"""

import pandas as pd
import numpy as np

def add_missing_lag_features():
    """Add missing lag features to complete the top 25 feature set."""
    
    input_file = "data/processed/SOLUSDT_5m_filtered_top25.csv"
    output_file = "data/processed/SOLUSDT_5m_complete_top25.csv"
    
    print(f"📊 Loading filtered dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"📈 Original columns: {len(df.columns)}")
    
    # Add missing lag features
    lag_configs = [
        {'indicator': 'rsi_14', 'lags': [1, 3]},
        {'indicator': 'bb_position', 'lags': [1, 3]},
        {'indicator': 'momentum_3', 'lags': [1]}
    ]
    
    added_features = []
    
    for config in lag_configs:
        indicator = config['indicator']
        if indicator in df.columns:
            for lag in config['lags']:
                lag_col = f"{indicator}_lag_{lag}"
                if lag_col not in df.columns:
                    df[lag_col] = df[indicator].shift(lag)
                    added_features.append(lag_col)
                    print(f"✅ Added: {lag_col}")
                else:
                    print(f"⚠️  Already exists: {lag_col}")
        else:
            print(f"❌ Missing base indicator: {indicator}")
    
    # Clean data (remove rows with NaN values)
    original_rows = len(df)
    df_clean = df.dropna()
    final_rows = len(df_clean)
    
    print(f"🧹 Cleaned data: {original_rows} -> {final_rows} rows ({original_rows - final_rows} removed)")
    
    # Get final feature count
    base_features = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    technical_features = [col for col in df_clean.columns if col not in base_features]
    
    print(f"📈 Final features: {len(technical_features)}")
    print(f"💾 Total columns: {len(df_clean.columns)}")
    
    # Save complete dataset
    df_clean.to_csv(output_file, index=False)
    
    print(f"✅ Saved complete top 25 dataset: {output_file}")
    
    return {
        'added_features': added_features,
        'total_features': len(technical_features),
        'total_rows': final_rows,
        'output_file': output_file
    }

def list_final_features():
    """List all features in the final optimized dataset."""
    
    file_path = "data/processed/SOLUSDT_5m_complete_top25.csv"
    df = pd.read_csv(file_path, nrows=1)  # Just read headers
    
    base_features = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    technical_features = [col for col in df.columns if col not in base_features]
    
    print(f"\n📋 FINAL TECHNICAL FEATURES ({len(technical_features)}):")
    print("=" * 50)
    
    # Group features by type for better readability
    feature_groups = {
        'Volume': [],
        'Bollinger Bands': [],
        'RSI & Momentum': [],
        'Moving Averages': [],
        'MACD': [],
        'Stochastic': [],
        'Price Action': [],
        'Other': []
    }
    
    for feature in sorted(technical_features):
        if 'volume' in feature:
            feature_groups['Volume'].append(feature)
        elif 'bb_' in feature:
            feature_groups['Bollinger Bands'].append(feature)
        elif 'rsi' in feature or 'momentum' in feature:
            feature_groups['RSI & Momentum'].append(feature)
        elif 'ema' in feature or 'sma' in feature or 'slope' in feature:
            feature_groups['Moving Averages'].append(feature)
        elif 'macd' in feature:
            feature_groups['MACD'].append(feature)
        elif 'stoch' in feature:
            feature_groups['Stochastic'].append(feature)
        elif 'atr' in feature or 'volatility' in feature:
            feature_groups['Price Action'].append(feature)
        else:
            feature_groups['Other'].append(feature)
    
    for group_name, features in feature_groups.items():
        if features:
            print(f"\n{group_name}:")
            for feature in features:
                print(f"  • {feature}")
    
    return technical_features

def main():
    """Main function to complete the top 25 feature dataset."""
    
    print("🔧 COMPLETING TOP 25 FEATURE DATASET")
    print("=" * 50)
    
    # Add missing lag features
    result = add_missing_lag_features()
    
    print(f"\n📊 RESULTS:")
    print(f"✅ Added features: {len(result['added_features'])}")
    if result['added_features']:
        for feature in result['added_features']:
            print(f"   • {feature}")
    
    print(f"📈 Total technical features: {result['total_features']}")
    print(f"📊 Total rows: {result['total_rows']}")
    print(f"💾 Output file: {result['output_file']}")
    
    # List all final features
    final_features = list_final_features()
    
    print(f"\n🎯 OPTIMIZATION SUMMARY:")
    print(f"   Original features: ~50")
    print(f"   Optimized features: {len(final_features)}")
    print(f"   Reduction: {round((1 - len(final_features)/50) * 100, 1)}%")
    
    if len(final_features) <= 25:
        print("✅ Successfully achieved target of ≤25 features!")
    else:
        print(f"⚠️  Still {len(final_features) - 25} features over target")

if __name__ == "__main__":
    main()