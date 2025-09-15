"""
Data filtering module for specific trading conditions.

This module provides functionality to filter enriched data based on specific
technical analysis conditions like Bollinger Bands touches, RSI levels, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime


class DataFilter:
    """Filter enriched data based on technical analysis conditions."""
    
    def __init__(self, output_dir: str = "output/filtered_data"):
        """
        Initialize data filter.
        
        Args:
            output_dir: Directory for saving filtered results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load enriched data from CSV file.
        
        Args:
            file_path: Path to enriched CSV file
            
        Returns:
            Loaded DataFrame
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"✅ Loaded {len(self.df)} rows for filtering")
        
        return self.df
    
    def filter_bollinger_lower_touch(
        self, 
        output_path: Optional[str] = None,
        include_near_misses: bool = False,
        tolerance: float = 0.001
    ) -> str:
        """
        Filter rows where low <= bb_lower (Bollinger Bands lower touch).
        
        Args:
            output_path: Custom output file path
            include_near_misses: Include rows where low is very close to bb_lower
            tolerance: Tolerance for near misses (as percentage, e.g., 0.001 = 0.1%)
            
        Returns:
            Path to filtered output file
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Check required columns
        required_columns = ['low', 'bb_lower']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Apply filter condition
        if include_near_misses:
            # Include touches and near misses
            condition = (self.df['low'] <= self.df['bb_lower'] * (1 + tolerance))
            condition_desc = f"low <= bb_lower * (1 + {tolerance})"
        else:
            # Strict condition: only actual touches
            condition = (self.df['low'] <= self.df['bb_lower'])
            condition_desc = "low <= bb_lower"
        
        # Filter data
        filtered_df = self.df[condition].copy()
        
        if filtered_df.empty:
            print(f"⚠️ No rows found matching condition: {condition_desc}")
            return ""
        
        # Add additional analysis columns
        filtered_df = self._add_touch_analysis(filtered_df, 'bb_lower')
        
        # Generate output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = self.df['symbol'].iloc[0] if 'symbol' in self.df.columns else 'DATA'
            timeframe = self.df['timeframe'].iloc[0] if 'timeframe' in self.df.columns else 'UNKNOWN'
            output_path = self.output_dir / f"{symbol}_{timeframe}_bb_lower_touches_{timestamp}.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save filtered data
        filtered_df.to_csv(output_path, index=False)
        
        print(f"🎯 Bollinger Lower Band touches filtered:")
        print(f"   Condition: {condition_desc}")
        print(f"   Found: {len(filtered_df)} rows out of {len(self.df)} ({len(filtered_df)/len(self.df)*100:.1f}%)")
        print(f"   Saved to: {output_path}")
        
        # Print summary statistics
        self._print_touch_summary(filtered_df, 'Bollinger Lower Band')
        
        return str(output_path)
    
    def filter_bollinger_upper_touch(
        self, 
        output_path: Optional[str] = None,
        include_near_misses: bool = False,
        tolerance: float = 0.001
    ) -> str:
        """
        Filter rows where high >= bb_upper (Bollinger Bands upper touch).
        
        Args:
            output_path: Custom output file path
            include_near_misses: Include rows where high is very close to bb_upper
            tolerance: Tolerance for near misses (as percentage)
            
        Returns:
            Path to filtered output file
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Check required columns
        required_columns = ['high', 'bb_upper']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Apply filter condition
        if include_near_misses:
            condition = (self.df['high'] >= self.df['bb_upper'] * (1 - tolerance))
            condition_desc = f"high >= bb_upper * (1 - {tolerance})"
        else:
            condition = (self.df['high'] >= self.df['bb_upper'])
            condition_desc = "high >= bb_upper"
        
        # Filter data
        filtered_df = self.df[condition].copy()
        
        if filtered_df.empty:
            print(f"⚠️ No rows found matching condition: {condition_desc}")
            return ""
        
        # Add additional analysis columns
        filtered_df = self._add_touch_analysis(filtered_df, 'bb_upper')
        
        # Generate output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = self.df['symbol'].iloc[0] if 'symbol' in self.df.columns else 'DATA'
            timeframe = self.df['timeframe'].iloc[0] if 'timeframe' in self.df.columns else 'UNKNOWN'
            output_path = self.output_dir / f"{symbol}_{timeframe}_bb_upper_touches_{timestamp}.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save filtered data
        filtered_df.to_csv(output_path, index=False)
        
        print(f"🎯 Bollinger Upper Band touches filtered:")
        print(f"   Condition: {condition_desc}")
        print(f"   Found: {len(filtered_df)} rows out of {len(self.df)} ({len(filtered_df)/len(self.df)*100:.1f}%)")
        print(f"   Saved to: {output_path}")
        
        # Print summary statistics
        self._print_touch_summary(filtered_df, 'Bollinger Upper Band')
        
        return str(output_path)
    
    def filter_rsi_oversold(
        self, 
        output_path: Optional[str] = None,
        rsi_threshold: float = 30.0
    ) -> str:
        """
        Filter rows where RSI <= threshold (oversold condition).
        
        Args:
            output_path: Custom output file path
            rsi_threshold: RSI threshold for oversold condition
            
        Returns:
            Path to filtered output file
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'rsi' not in self.df.columns:
            raise ValueError("RSI column not found in data")
        
        # Apply filter condition
        condition = (self.df['rsi'] <= rsi_threshold) & (self.df['rsi'].notna())
        filtered_df = self.df[condition].copy()
        
        if filtered_df.empty:
            print(f"⚠️ No rows found with RSI <= {rsi_threshold}")
            return ""
        
        # Generate output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = self.df['symbol'].iloc[0] if 'symbol' in self.df.columns else 'DATA'
            timeframe = self.df['timeframe'].iloc[0] if 'timeframe' in self.df.columns else 'UNKNOWN'
            output_path = self.output_dir / f"{symbol}_{timeframe}_rsi_oversold_{timestamp}.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save filtered data
        filtered_df.to_csv(output_path, index=False)
        
        print(f"🎯 RSI Oversold conditions filtered:")
        print(f"   Condition: RSI <= {rsi_threshold}")
        print(f"   Found: {len(filtered_df)} rows out of {len(self.df)} ({len(filtered_df)/len(self.df)*100:.1f}%)")
        print(f"   Saved to: {output_path}")
        
        if 'rsi' in filtered_df.columns:
            rsi_stats = filtered_df['rsi'].describe()
            print(f"   RSI Statistics:")
            print(f"     Min: {rsi_stats['min']:.2f}")
            print(f"     Mean: {rsi_stats['mean']:.2f}")
            print(f"     Max: {rsi_stats['max']:.2f}")
        
        return str(output_path)
    
    def filter_rsi_overbought(
        self, 
        output_path: Optional[str] = None,
        rsi_threshold: float = 70.0
    ) -> str:
        """
        Filter rows where RSI >= threshold (overbought condition).
        
        Args:
            output_path: Custom output file path
            rsi_threshold: RSI threshold for overbought condition
            
        Returns:
            Path to filtered output file
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'rsi' not in self.df.columns:
            raise ValueError("RSI column not found in data")
        
        # Apply filter condition
        condition = (self.df['rsi'] >= rsi_threshold) & (self.df['rsi'].notna())
        filtered_df = self.df[condition].copy()
        
        if filtered_df.empty:
            print(f"⚠️ No rows found with RSI >= {rsi_threshold}")
            return ""
        
        # Generate output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = self.df['symbol'].iloc[0] if 'symbol' in self.df.columns else 'DATA'
            timeframe = self.df['timeframe'].iloc[0] if 'timeframe' in self.df.columns else 'UNKNOWN'
            output_path = self.output_dir / f"{symbol}_{timeframe}_rsi_overbought_{timestamp}.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save filtered data
        filtered_df.to_csv(output_path, index=False)
        
        print(f"🎯 RSI Overbought conditions filtered:")
        print(f"   Condition: RSI >= {rsi_threshold}")
        print(f"   Found: {len(filtered_df)} rows out of {len(self.df)} ({len(filtered_df)/len(self.df)*100:.1f}%)")
        print(f"   Saved to: {output_path}")
        
        if 'rsi' in filtered_df.columns:
            rsi_stats = filtered_df['rsi'].describe()
            print(f"   RSI Statistics:")
            print(f"     Min: {rsi_stats['min']:.2f}")
            print(f"     Mean: {rsi_stats['mean']:.2f}")
            print(f"     Max: {rsi_stats['max']:.2f}")
        
        return str(output_path)
    
    def filter_custom_condition(
        self, 
        condition_string: str,
        output_path: Optional[str] = None,
        description: str = "Custom condition"
    ) -> str:
        """
        Filter data using a custom pandas condition string.
        
        Args:
            condition_string: Pandas condition string (e.g., "(low <= bb_lower) & (rsi < 30)")
            output_path: Custom output file path
            description: Description of the condition for output
            
        Returns:
            Path to filtered output file
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        try:
            # Evaluate the condition
            condition = self.df.eval(condition_string)
            filtered_df = self.df[condition].copy()
            
            if filtered_df.empty:
                print(f"⚠️ No rows found matching condition: {condition_string}")
                return ""
            
            # Generate output path
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                symbol = self.df['symbol'].iloc[0] if 'symbol' in self.df.columns else 'DATA'
                timeframe = self.df['timeframe'].iloc[0] if 'timeframe' in self.df.columns else 'UNKNOWN'
                output_path = self.output_dir / f"{symbol}_{timeframe}_custom_filter_{timestamp}.csv"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save filtered data
            filtered_df.to_csv(output_path, index=False)
            
            print(f"🎯 Custom condition filtered:")
            print(f"   Condition: {condition_string}")
            print(f"   Description: {description}")
            print(f"   Found: {len(filtered_df)} rows out of {len(self.df)} ({len(filtered_df)/len(self.df)*100:.1f}%)")
            print(f"   Saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            raise ValueError(f"Error evaluating condition '{condition_string}': {str(e)}")
    
    def _add_touch_analysis(self, df: pd.DataFrame, band_column: str) -> pd.DataFrame:
        """Add additional analysis columns for band touches."""
        
        # Calculate distance from band
        if band_column == 'bb_lower':
            df['distance_from_band'] = df['low'] - df[band_column]
            df['distance_percent'] = (df['distance_from_band'] / df[band_column]) * 100
            df['penetration_depth'] = np.where(df['distance_from_band'] < 0, abs(df['distance_from_band']), 0)
        elif band_column == 'bb_upper':
            df['distance_from_band'] = df['high'] - df[band_column]
            df['distance_percent'] = (df['distance_from_band'] / df[band_column]) * 100
            df['penetration_depth'] = np.where(df['distance_from_band'] > 0, df['distance_from_band'], 0)
        
        # Add touch type classification
        if band_column == 'bb_lower':
            conditions = [
                df['distance_from_band'] < -0.01,  # Significant penetration
                (df['distance_from_band'] >= -0.01) & (df['distance_from_band'] < 0),  # Minor penetration
                df['distance_from_band'] == 0,  # Exact touch
                (df['distance_from_band'] > 0) & (df['distance_from_band'] <= 0.01)  # Near miss
            ]
            choices = ['Strong Penetration', 'Minor Penetration', 'Exact Touch', 'Near Miss']
        else:
            conditions = [
                df['distance_from_band'] > 0.01,  # Significant penetration
                (df['distance_from_band'] <= 0.01) & (df['distance_from_band'] > 0),  # Minor penetration
                df['distance_from_band'] == 0,  # Exact touch
                (df['distance_from_band'] < 0) & (df['distance_from_band'] >= -0.01)  # Near miss
            ]
            choices = ['Strong Penetration', 'Minor Penetration', 'Exact Touch', 'Near Miss']
        
        df['touch_type'] = np.select(conditions, choices, default='Other')
        
        return df
    
    def _print_touch_summary(self, df: pd.DataFrame, band_name: str):
        """Print summary statistics for band touches."""
        
        print(f"\n📊 {band_name} Touch Analysis:")
        
        if 'touch_type' in df.columns:
            touch_counts = df['touch_type'].value_counts()
            for touch_type, count in touch_counts.items():
                print(f"   {touch_type}: {count} ({count/len(df)*100:.1f}%)")
        
        if 'distance_percent' in df.columns:
            print(f"\n📏 Distance Statistics:")
            dist_stats = df['distance_percent'].describe()
            print(f"   Mean Distance: {dist_stats['mean']:.3f}%")
            print(f"   Max Distance: {dist_stats['max']:.3f}%")
            print(f"   Min Distance: {dist_stats['min']:.3f}%")
        
        # Show some additional context if available
        if all(col in df.columns for col in ['rsi', 'volume_spike', 'is_extreme']):
            print(f"\n🔍 Context Analysis:")
            rsi_mean = df['rsi'].mean()
            volume_spikes = df['volume_spike'].sum()
            extremes = df['is_extreme'].sum()
            
            print(f"   Average RSI at touches: {rsi_mean:.1f}")
            print(f"   Volume spikes: {volume_spikes} ({volume_spikes/len(df)*100:.1f}%)")
            print(f"   Price extremes: {extremes} ({extremes/len(df)*100:.1f}%)")