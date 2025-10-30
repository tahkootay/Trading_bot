#!/usr/bin/env python3
"""
ML Data Preparation Module

Prepares historical cryptocurrency data with technical indicators for machine learning.
Creates target variables, lag features, and splits data for time series prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class MLDataPreparator:
    """Prepares trading data for machine learning models."""
    
    def __init__(self, prediction_horizon: int = 3):
        """
        Initialize ML data preparator.
        
        Args:
            prediction_horizon: Number of bars ahead to predict (N)
        """
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.lag_features = ['rsi_14', 'macd_line', 'ema_20', 'bb_position', 'momentum_3']
        self.lag_periods = [1, 2, 3]
        
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable: 1 if price goes up in N bars, 0 otherwise.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added target column
        """
        print(f"📊 Creating target variable with prediction horizon = {self.prediction_horizon}")
        
        df = df.copy()
        
        # Create target: 1 if close(t+N) > close(t), else 0
        df['future_close'] = df['close'].shift(-self.prediction_horizon)
        df['target'] = (df['future_close'] > df['close']).astype(int)
        
        # Remove the temporary future_close column
        df = df.drop('future_close', axis=1)
        
        # Remove last N rows as they don't have future prices
        valid_rows_before = len(df)
        df = df.iloc[:-self.prediction_horizon]
        valid_rows_after = len(df)
        
        print(f"   ✅ Target created: {valid_rows_before} → {valid_rows_after} rows")
        print(f"   📈 Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for specified indicators.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with added lag features
        """
        print(f"🔄 Creating lag features for: {self.lag_features}")
        print(f"   Lag periods: {self.lag_periods}")
        
        df = df.copy()
        
        for feature in self.lag_features:
            if feature not in df.columns:
                print(f"   ⚠️ Warning: Feature '{feature}' not found in data")
                continue
                
            for lag in self.lag_periods:
                lag_col_name = f"{feature}_lag_{lag}"
                df[lag_col_name] = df[feature].shift(lag)
                
        lag_features_added = [f"{feature}_lag_{lag}" 
                            for feature in self.lag_features 
                            for lag in self.lag_periods
                            if feature in df.columns]
        
        print(f"   ✅ Added {len(lag_features_added)} lag features")
        
        return df
    
    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing values and report statistics.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        print("🧹 Cleaning missing values...")
        
        initial_rows = len(df)
        
        # Count NaN values per column
        nan_counts = df.isnull().sum()
        columns_with_nan = nan_counts[nan_counts > 0]
        
        if len(columns_with_nan) > 0:
            print("   📊 NaN values per column:")
            for col, count in columns_with_nan.items():
                print(f"      {col}: {count} ({count/initial_rows*100:.1f}%)")
        
        # Remove rows with any NaN values
        df_clean = df.dropna()
        final_rows = len(df_clean)
        rows_removed = initial_rows - final_rows
        
        print(f"   ✅ Removed {rows_removed} rows ({rows_removed/initial_rows*100:.1f}%)")
        print(f"   📈 Final dataset: {final_rows} rows")
        
        return df_clean
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature columns for ML (exclude non-feature columns).
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            Tuple of (features_df, feature_column_names)
        """
        print("⚙️ Preparing features for ML...")
        
        # Columns to exclude from features
        exclude_columns = [
            'timestamp', 'symbol', 'timeframe', 'target',
            'open', 'high', 'low', 'close', 'volume'  # Raw OHLCV data
        ]
        
        # Get feature columns (everything except excluded)
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        print(f"   📊 Total columns: {len(df.columns)}")
        print(f"   🎯 Feature columns: {len(feature_columns)}")
        print(f"   ❌ Excluded: {exclude_columns}")
        
        # Create features DataFrame
        features_df = df[feature_columns + ['target']].copy()
        
        self.feature_columns = feature_columns
        
        return features_df, feature_columns
    
    def split_data_temporal(self, df: pd.DataFrame, 
                          train_ratio: float = 0.7, 
                          val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (no shuffle) for time series.
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("📅 Splitting data temporally (no shuffle)...")
        
        total_rows = len(df)
        train_size = int(total_rows * train_ratio)
        val_size = int(total_rows * val_ratio)
        
        # Temporal split (chronological order)
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()
        
        print(f"   🚂 Train: {len(train_df)} rows ({len(train_df)/total_rows*100:.1f}%)")
        print(f"   🔍 Validation: {len(val_df)} rows ({len(val_df)/total_rows*100:.1f}%)")
        print(f"   🧪 Test: {len(test_df)} rows ({len(test_df)/total_rows*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def scale_features(self, train_df: pd.DataFrame, 
                      val_df: pd.DataFrame, 
                      test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler (fit on train only).
        
        Args:
            train_df, val_df, test_df: DataFrames to scale
            
        Returns:
            Tuple of scaled DataFrames
        """
        print("📏 Scaling features with StandardScaler...")
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col != 'target']
        
        # Fit scaler on training data only
        X_train = train_df[feature_cols]
        self.scaler.fit(X_train)
        
        # Apply scaling to all sets
        train_scaled = train_df.copy()
        val_scaled = val_df.copy() 
        test_scaled = test_df.copy()
        
        train_scaled[feature_cols] = self.scaler.transform(train_df[feature_cols])
        val_scaled[feature_cols] = self.scaler.transform(val_df[feature_cols])
        test_scaled[feature_cols] = self.scaler.transform(test_df[feature_cols])
        
        print(f"   ✅ Scaled {len(feature_cols)} features")
        print(f"   📊 Feature means (train): {train_scaled[feature_cols].mean().mean():.6f}")
        print(f"   📊 Feature stds (train): {train_scaled[feature_cols].std().mean():.6f}")
        
        return train_scaled, val_scaled, test_scaled
    
    def analyze_correlations(self, df: pd.DataFrame, top_n: int = 10) -> pd.Series:
        """
        Analyze feature correlations with target.
        
        Args:
            df: DataFrame with features and target
            top_n: Number of top correlations to return
            
        Returns:
            Series with top correlations
        """
        print(f"🔍 Analyzing correlations with target (top {top_n})...")
        
        # Calculate correlations with target
        correlations = df.corr()['target'].abs().sort_values(ascending=False)
        
        # Exclude target itself
        correlations = correlations.drop('target')
        top_correlations = correlations.head(top_n)
        
        print("   📊 Top correlations with target:")
        for feature, corr in top_correlations.items():
            print(f"      {feature}: {corr:.4f}")
        
        return top_correlations
    
    def check_class_balance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check target class balance.
        
        Args:
            df: DataFrame with target column
            
        Returns:
            Dictionary with balance statistics
        """
        print("⚖️ Checking class balance...")
        
        target_counts = df['target'].value_counts()
        total = len(df)
        
        balance_stats = {
            'class_0': target_counts.get(0, 0),
            'class_1': target_counts.get(1, 0),
            'class_0_pct': target_counts.get(0, 0) / total * 100,
            'class_1_pct': target_counts.get(1, 0) / total * 100,
            'balance_ratio': min(target_counts) / max(target_counts) if len(target_counts) == 2 else 0
        }
        
        print(f"   📊 Class 0 (price down): {balance_stats['class_0']} ({balance_stats['class_0_pct']:.1f}%)")
        print(f"   📊 Class 1 (price up): {balance_stats['class_1']} ({balance_stats['class_1_pct']:.1f}%)")
        print(f"   ⚖️ Balance ratio: {balance_stats['balance_ratio']:.3f}")
        
        if balance_stats['balance_ratio'] < 0.8:
            print("   ⚠️ WARNING: Classes are imbalanced (ratio < 0.8)")
        else:
            print("   ✅ Classes are reasonably balanced")
        
        return balance_stats
    
    def get_date_ranges(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, str]:
        """
        Get date range for a dataset.
        
        Args:
            df: DataFrame with timestamp column
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary with min and max dates
        """
        if 'timestamp' in df.columns:
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            print(f"   📅 {dataset_name}: {min_date} → {max_date}")
            return {'min_date': str(min_date), 'max_date': str(max_date)}
        else:
            print(f"   ⚠️ {dataset_name}: No timestamp column found")
            return {'min_date': 'N/A', 'max_date': 'N/A'}
    
    def save_datasets(self, train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, 
                     test_df: pd.DataFrame, 
                     output_dir: str = "data/processed") -> Dict[str, str]:
        """
        Save datasets to CSV files.
        
        Args:
            train_df, val_df, test_df: DataFrames to save
            output_dir: Output directory path
            
        Returns:
            Dictionary with file paths
        """
        print(f"💾 Saving datasets to {output_dir}...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save files
        file_paths = {
            'train': output_path / "train.csv",
            'val': output_path / "val.csv", 
            'test': output_path / "test.csv"
        }
        
        train_df.to_csv(file_paths['train'], index=False)
        val_df.to_csv(file_paths['val'], index=False)
        test_df.to_csv(file_paths['test'], index=False)
        
        print(f"   ✅ Saved train.csv: {len(train_df)} rows")
        print(f"   ✅ Saved val.csv: {len(val_df)} rows") 
        print(f"   ✅ Saved test.csv: {len(test_df)} rows")
        
        return {k: str(v) for k, v in file_paths.items()}
    
    def prepare_ml_data(self, input_file: str, output_dir: str = "data/processed") -> Dict[str, Any]:
        """
        Complete ML data preparation pipeline.
        
        Args:
            input_file: Path to input CSV file with indicators
            output_dir: Output directory for processed files
            
        Returns:
            Dictionary with preparation statistics
        """
        print("🚀 Starting ML Data Preparation Pipeline")
        print("=" * 50)
        
        # 1. Load data
        print("1️⃣ Loading data...")
        df = pd.read_csv(input_file)
        print(f"   📁 Loaded: {input_file}")
        print(f"   📊 Shape: {df.shape}")
        print(f"   📅 Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        # 2. Create target variable
        print("\n2️⃣ Creating target variable...")
        df = self.create_target_variable(df)
        
        # 3. Add lag features
        print("\n3️⃣ Adding lag features...")
        df = self.create_lag_features(df)
        
        # 4. Clean missing values
        print("\n4️⃣ Cleaning missing values...")
        initial_rows = len(df)
        df_clean = self.clean_missing_values(df)
        rows_removed = initial_rows - len(df_clean)
        
        # 5. Prepare features
        print("\n5️⃣ Preparing features...")
        features_df, feature_columns = self.prepare_features(df_clean)
        
        # 6. Split data temporally
        print("\n6️⃣ Splitting data...")
        train_df, val_df, test_df = self.split_data_temporal(features_df)
        
        # 7. Scale features
        print("\n7️⃣ Scaling features...")
        train_scaled, val_scaled, test_scaled = self.scale_features(train_df, val_df, test_df)
        
        # 8. Analyze data
        print("\n8️⃣ Analyzing data...")
        top_correlations = self.analyze_correlations(train_scaled)
        balance_stats = self.check_class_balance(train_scaled)
        
        # 9. Get date ranges
        print("\n9️⃣ Date ranges:")
        train_dates = self.get_date_ranges(train_df, "Train")
        val_dates = self.get_date_ranges(val_df, "Validation") 
        test_dates = self.get_date_ranges(test_df, "Test")
        
        # 10. Save datasets
        print("\n🔟 Saving datasets...")
        file_paths = self.save_datasets(train_scaled, val_scaled, test_scaled, output_dir)
        
        # Summary statistics
        print("\n📋 PREPARATION SUMMARY")
        print("=" * 50)
        print(f"📊 Rows removed due to NaN: {rows_removed}")
        print(f"📊 Features count: {len(feature_columns)}")
        print(f"📊 Train rows: {len(train_scaled)}")
        print(f"📊 Validation rows: {len(val_scaled)}")
        print(f"📊 Test rows: {len(test_scaled)}")
        print(f"📊 Prediction horizon: {self.prediction_horizon} bars")
        print(f"💾 Files saved to: {output_dir}")
        
        return {
            'initial_rows': len(df),
            'rows_removed': rows_removed,
            'final_rows': len(df_clean),
            'feature_count': len(feature_columns),
            'train_rows': len(train_scaled),
            'val_rows': len(val_scaled),
            'test_rows': len(test_scaled),
            'prediction_horizon': self.prediction_horizon,
            'top_correlations': top_correlations.to_dict(),
            'balance_stats': balance_stats,
            'date_ranges': {
                'train': train_dates,
                'val': val_dates,
                'test': test_dates
            },
            'file_paths': file_paths,
            'feature_columns': feature_columns
        }


def prepare_ml_data_cli(input_file: str, output_dir: str = "data/processed", 
                       prediction_horizon: int = 3) -> Dict[str, Any]:
    """
    CLI function for ML data preparation.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Output directory for processed files
        prediction_horizon: Number of bars ahead to predict
        
    Returns:
        Preparation statistics
    """
    preparator = MLDataPreparator(prediction_horizon=prediction_horizon)
    return preparator.prepare_ml_data(input_file, output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ml_data_prep.py <input_file.csv> [output_dir] [prediction_horizon]")
        print("Example: python ml_data_prep.py data/raw/SOLUSDT_5m_advanced_indicators.csv data/processed 3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    prediction_horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    # Run preparation
    stats = prepare_ml_data_cli(input_file, output_dir, prediction_horizon)
    
    print(f"\n🎉 ML data preparation completed successfully!")
    print(f"📁 Check files in: {output_dir}")