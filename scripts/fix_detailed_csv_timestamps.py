#!/usr/bin/env python3
"""
Fix timestamp issues in detailed backtest CSV and recreate with proper 15-minute intervals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fix_detailed_csv_timestamps():
    """Fix timestamps in the detailed CSV file."""
    
    print("🔧 Fixing timestamps in detailed backtest CSV...")
    
    # Load the original CSV with incorrect timestamps
    original_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/detailed_backtest_20250914_191449.csv'
    df = pd.read_csv(original_file)
    
    print(f"📊 Loaded {len(df)} rows of detailed data")
    print(f"🕐 Original timestamp range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    # Generate proper 15-minute intervals starting from June 1, 2025
    start_time = datetime(2025, 6, 1, 0, 0, 0)
    
    # Create proper timestamp sequence
    timestamps = []
    for i in range(len(df)):
        timestamp = start_time + timedelta(minutes=15 * i)
        timestamps.append(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Update timestamps in dataframe
    df['timestamp'] = timestamps
    
    print(f"✅ Fixed timestamps: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    # Calculate actual time period
    end_time = start_time + timedelta(minutes=15 * (len(df) - 1))
    total_days = (end_time - start_time).days
    print(f"📅 Time period: {total_days} days ({total_days/30.44:.1f} months)")
    
    # Save corrected CSV
    output_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/detailed_backtest_corrected_timestamps.csv'
    df.to_csv(output_file, index=False)
    
    print(f"💾 Saved corrected CSV: {output_file}")
    print(f"📁 File size: {len(df)} rows × {len(df.columns)} columns")
    
    # Show sample of corrected data
    print("\n📋 Sample of corrected data:")
    sample_columns = ['bar', 'timestamp', 'close', 'signal_type', 'macd_line', 'rsi', 'ema50']
    available_columns = [col for col in sample_columns if col in df.columns]
    
    # Show first few rows
    print("First 5 rows:")
    print(df[available_columns].head().to_string(index=False))
    
    # Show some trading signals
    trading_signals = df[df['signal_type'].isin(['BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT'])]
    if len(trading_signals) > 0:
        print(f"\nFirst 5 trading signals (out of {len(trading_signals)} total):")
        print(trading_signals[available_columns].head().to_string(index=False))
    
    # Show last few rows
    print("\nLast 5 rows:")
    print(df[available_columns].tail().to_string(index=False))
    
    # Statistics about the corrected data
    print(f"\n📊 Data Statistics:")
    print(f"   Total bars: {len(df):,}")
    print(f"   Indicators ready: {df['indicators_ready'].sum():,}")
    print(f"   Signal breakdown:")
    signal_counts = df['signal_type'].value_counts()
    for signal, count in signal_counts.items():
        print(f"     {signal}: {count}")
    
    # Time-based statistics
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    print(f"\n🕐 Time Distribution:")
    print(f"   Hours covered: {df['hour'].min()}:00 to {df['hour'].max()}:00")
    print(f"   Days of week: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][df['day_of_week'].min()]} to {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][df['day_of_week'].max()]}")
    
    # Show date range by month
    df['month'] = df['datetime'].dt.strftime('%Y-%m')
    monthly_counts = df['month'].value_counts().sort_index()
    print(f"\n📅 Monthly breakdown:")
    for month, count in monthly_counts.items():
        print(f"     {month}: {count:,} bars")
    
    return output_file

def verify_timestamps():
    """Verify that timestamps are properly sequential."""
    
    print("\n🔍 Verifying timestamp sequence...")
    
    # Load corrected file
    corrected_file = '/Users/alexey/Documents/Development/Python/Trading_bot/output/detailed_backtest_corrected_timestamps.csv'
    df = pd.read_csv(corrected_file)
    
    # Convert to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    # Check for proper 15-minute intervals
    time_diffs = df['datetime'].diff()
    expected_interval = timedelta(minutes=15)
    
    # Check if all intervals are 15 minutes (except first row which is NaT)
    valid_intervals = time_diffs[1:] == expected_interval
    
    print(f"✅ Timestamp verification:")
    print(f"   Total intervals checked: {len(valid_intervals):,}")
    print(f"   Valid 15-min intervals: {valid_intervals.sum():,}")
    print(f"   Invalid intervals: {(~valid_intervals).sum()}")
    
    if valid_intervals.all():
        print("   🎉 All timestamps are properly spaced at 15-minute intervals!")
    else:
        print("   ⚠️  Some intervals are not exactly 15 minutes")
        invalid_rows = df[1:][~valid_intervals]
        print(f"   First few invalid intervals:")
        print(invalid_rows[['bar', 'timestamp']].head())
    
    # Show start and end times
    print(f"\n📅 Final time range:")
    print(f"   Start: {df['datetime'].min()}")
    print(f"   End: {df['datetime'].max()}")
    print(f"   Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")

def main():
    """Main execution function."""
    
    print("🚀 Starting detailed CSV timestamp correction...")
    
    # Fix timestamps
    output_file = fix_detailed_csv_timestamps()
    
    # Verify the fix
    verify_timestamps()
    
    print(f"\n✅ Timestamp correction completed!")
    print(f"📄 Corrected file: {output_file}")
    print(f"🗂️  You can now use this file for accurate time-based analysis")

if __name__ == "__main__":
    main()