#!/usr/bin/env python3
"""
Analyze Feature Importance for trained Random Forest models across different horizons.

This script:
1. Trains RF models for each horizon (1, 3, 5, 10)
2. Extracts feature importances from each model
3. Creates comparison tables and visualizations
4. Identifies most important features for each prediction horizon
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def load_horizon_data(horizon):
    """Load train data for specific horizon."""
    try:
        train_df = pd.read_csv(f"data/processed/train_target{horizon}.csv")
        print(f"📁 Loaded horizon {horizon}: {len(train_df)} samples")
        return train_df
    except FileNotFoundError as e:
        print(f"❌ Error loading data for horizon {horizon}: {e}")
        return None

def prepare_features_target(train_df):
    """Separate features and target from dataframe."""
    feature_columns = [col for col in train_df.columns if col != 'target']
    
    X_train = train_df[feature_columns].values
    y_train = train_df['target'].values
    
    return X_train, y_train, feature_columns

def train_and_extract_importance(X_train, y_train, feature_columns, horizon):
    """Train RF model and extract feature importance."""
    print(f"🌲 Training Random Forest for horizon {horizon}...")
    
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Extract feature importance
    importance_scores = model.feature_importances_
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance_scores,
        'horizon': horizon
    }).sort_values('importance', ascending=False)
    
    print(f"✅ Model trained, feature importance extracted")
    
    return importance_df, model

def create_importance_comparison(all_importance_dfs):
    """Create comparison table of top features across horizons."""
    print("\n📊 TOP-10 FEATURE IMPORTANCE BY HORIZON")
    print("=" * 80)
    
    # Show top 10 for each horizon
    for horizon in [1, 3, 5, 10]:
        horizon_df = all_importance_dfs[horizon]
        print(f"\n🎯 HORIZON {horizon} - Top 10 Features:")
        print("-" * 50)
        
        for i, row in horizon_df.head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")
    
    return True

def create_feature_ranking_matrix(all_importance_dfs):
    """Create matrix showing feature rankings across all horizons."""
    print(f"\n📋 FEATURE RANKING MATRIX (Top 20)")
    print("=" * 80)
    
    # Get all unique features
    all_features = set()
    for df in all_importance_dfs.values():
        all_features.update(df['feature'].tolist())
    
    # Create ranking matrix
    ranking_data = []
    
    for feature in all_features:
        row = {'feature': feature}
        
        for horizon in [1, 3, 5, 10]:
            horizon_df = all_importance_dfs[horizon]
            feature_row = horizon_df[horizon_df['feature'] == feature]
            
            if not feature_row.empty:
                # Get rank (1-based)
                rank = horizon_df.index[horizon_df['feature'] == feature].tolist()[0] + 1
                importance = feature_row['importance'].iloc[0]
                row[f'rank_h{horizon}'] = rank
                row[f'imp_h{horizon}'] = importance
            else:
                row[f'rank_h{horizon}'] = 999  # Not found
                row[f'imp_h{horizon}'] = 0.0
        
        # Calculate average rank
        ranks = [row[f'rank_h{h}'] for h in [1, 3, 5, 10]]
        row['avg_rank'] = np.mean([r for r in ranks if r < 999])
        
        ranking_data.append(row)
    
    # Sort by average rank
    ranking_df = pd.DataFrame(ranking_data).sort_values('avg_rank')
    
    # Display top 20 features
    print(f"{'Feature':<25} {'H1':<8} {'H3':<8} {'H5':<8} {'H10':<8} {'Avg':<8}")
    print("-" * 80)
    
    for _, row in ranking_df.head(20).iterrows():
        feature = row['feature'][:24]  # Truncate long names
        h1_rank = int(row['rank_h1']) if row['rank_h1'] < 999 else '-'
        h3_rank = int(row['rank_h3']) if row['rank_h3'] < 999 else '-'
        h5_rank = int(row['rank_h5']) if row['rank_h5'] < 999 else '-'
        h10_rank = int(row['rank_h10']) if row['rank_h10'] < 999 else '-'
        avg_rank = f"{row['avg_rank']:.1f}" if row['avg_rank'] < 999 else '-'
        
        print(f"{feature:<25} {h1_rank:<8} {h3_rank:<8} {h5_rank:<8} {h10_rank:<8} {avg_rank:<8}")
    
    return ranking_df

def identify_consistent_features(ranking_df):
    """Identify features that are consistently important across horizons."""
    print(f"\n🏆 CONSISTENTLY IMPORTANT FEATURES")
    print("=" * 50)
    
    # Features in top 10 for all horizons
    top_10_all = ranking_df[
        (ranking_df['rank_h1'] <= 10) & 
        (ranking_df['rank_h3'] <= 10) & 
        (ranking_df['rank_h5'] <= 10) & 
        (ranking_df['rank_h10'] <= 10)
    ]
    
    if not top_10_all.empty:
        print("📈 Features in TOP-10 for ALL horizons:")
        for _, row in top_10_all.iterrows():
            print(f"   {row['feature']:<30} (avg rank: {row['avg_rank']:.1f})")
    else:
        print("📈 No features in TOP-10 for ALL horizons")
    
    # Features in top 15 for all horizons
    top_15_all = ranking_df[
        (ranking_df['rank_h1'] <= 15) & 
        (ranking_df['rank_h3'] <= 15) & 
        (ranking_df['rank_h5'] <= 15) & 
        (ranking_df['rank_h10'] <= 15)
    ]
    
    print(f"\n📈 Features in TOP-15 for ALL horizons:")
    for _, row in top_15_all.head(10).iterrows():
        print(f"   {row['feature']:<30} (avg rank: {row['avg_rank']:.1f})")
    
    return top_10_all, top_15_all

def analyze_feature_categories(ranking_df):
    """Analyze feature importance by categories."""
    print(f"\n📊 FEATURE CATEGORY ANALYSIS")
    print("=" * 50)
    
    # Define feature categories
    categories = {
        'Moving Averages': ['sma_', 'ema_', 'slope_ema', 'ema_diff'],
        'Momentum': ['rsi_', 'momentum_', 'stoch_'],
        'Bollinger Bands': ['bb_'],
        'MACD': ['macd_'],
        'Volume': ['volume_', 'relative_volume'],
        'Price Action': ['high_low_range', 'body_to_range', 'candle_', 'volatility'],
        'Lag Features': ['_lag_'],
        'Other': ['cci_', 'atr_']
    }
    
    # Analyze top 20 features by category
    top_20_features = ranking_df.head(20)['feature'].tolist()
    
    category_counts = {cat: 0 for cat in categories.keys()}
    
    for feature in top_20_features:
        categorized = False
        for category, keywords in categories.items():
            if any(keyword in feature for keyword in keywords):
                category_counts[category] += 1
                categorized = True
                break
        if not categorized:
            category_counts['Other'] += 1
    
    print("Top 20 features by category:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = count / 20 * 100
            print(f"   {category:<20}: {count:2d} features ({percentage:4.1f}%)")
    
    return category_counts

def save_importance_results(all_importance_dfs, ranking_df, consistent_features):
    """Save feature importance analysis to files."""
    output_dir = Path("data/processed")
    
    # Save individual horizon importance
    for horizon, importance_df in all_importance_dfs.items():
        output_file = output_dir / f"feature_importance_horizon{horizon}.csv"
        importance_df.to_csv(output_file, index=False)
        print(f"💾 Saved {output_file}")
    
    # Save ranking matrix
    ranking_file = output_dir / "feature_ranking_matrix.csv"
    ranking_df.to_csv(ranking_file, index=False)
    print(f"💾 Saved {ranking_file}")
    
    # Save summary report
    summary_file = output_dir / "feature_importance_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Feature Importance Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("TOP-5 FEATURES BY HORIZON:\n")
        f.write("-" * 30 + "\n")
        
        for horizon in [1, 3, 5, 10]:
            f.write(f"\nHorizon {horizon}:\n")
            for i, row in all_importance_dfs[horizon].head(5).iterrows():
                f.write(f"  {i+1}. {row['feature']}: {row['importance']:.4f}\n")
        
        f.write(f"\n\nCONSISTENTLY IMPORTANT FEATURES (Top-15 all horizons):\n")
        f.write("-" * 50 + "\n")
        
        if len(consistent_features[1]) > 0:
            for _, row in consistent_features[1].iterrows():
                f.write(f"  {row['feature']}: avg rank {row['avg_rank']:.1f}\n")
        else:
            f.write("  None found in top-15 for all horizons\n")
    
    print(f"💾 Saved {summary_file}")
    
    return True

def main():
    """Main function to analyze feature importance across horizons."""
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    print("🎯 Analyzing Random Forest feature importance for all horizons")
    print()
    
    horizons = [1, 3, 5, 10]
    all_importance_dfs = {}
    
    # Train models and extract importance for each horizon
    for horizon in horizons:
        print(f"\n{'='*20} HORIZON {horizon} {'='*20}")
        
        # Load data
        train_df = load_horizon_data(horizon)
        if train_df is None:
            continue
        
        # Prepare features
        X_train, y_train, feature_columns = prepare_features_target(train_df)
        
        # Train and extract importance
        importance_df, model = train_and_extract_importance(
            X_train, y_train, feature_columns, horizon
        )
        
        all_importance_dfs[horizon] = importance_df
    
    if all_importance_dfs:
        # Create comparison analysis
        create_importance_comparison(all_importance_dfs)
        
        # Create ranking matrix
        ranking_df = create_feature_ranking_matrix(all_importance_dfs)
        
        # Identify consistent features
        consistent_features = identify_consistent_features(ranking_df)
        
        # Analyze feature categories
        analyze_feature_categories(ranking_df)
        
        # Save results
        print(f"\n💾 SAVING RESULTS")
        print("-" * 30)
        save_importance_results(all_importance_dfs, ranking_df, consistent_features)
        
        print(f"\n🎉 FEATURE IMPORTANCE ANALYSIS COMPLETED!")
        print("=" * 60)
        print(f"✅ Analyzed feature importance for horizons: {', '.join(map(str, horizons))}")
        print(f"📊 Results saved to data/processed/feature_importance_*.csv")
        print(f"📋 Summary saved to data/processed/feature_importance_summary.txt")
        
    else:
        print("❌ No models were successfully analyzed")

if __name__ == "__main__":
    main()