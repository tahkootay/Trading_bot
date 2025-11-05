#!/usr/bin/env python3
"""
Display Feature Importance results in a clear format
"""

import pandas as pd
import numpy as np

def load_importance_data():
    """Load all feature importance data."""
    horizons = [1, 3, 5, 10]
    importance_data = {}
    
    for horizon in horizons:
        try:
            df = pd.read_csv(f"data/processed/feature_importance_horizon{horizon}.csv")
            importance_data[horizon] = df
            print(f"📁 Loaded horizon {horizon}: {len(df)} features")
        except FileNotFoundError:
            print(f"❌ No data for horizon {horizon}")
            
    return importance_data

def show_top_features_by_horizon(importance_data):
    """Show top 10 features for each horizon."""
    print("\n🏆 TOP-10 MOST IMPORTANT FEATURES BY HORIZON")
    print("=" * 80)
    
    for horizon in [1, 3, 5, 10]:
        if horizon not in importance_data:
            continue
            
        df = importance_data[horizon]
        print(f"\n🎯 HORIZON {horizon} (predicting {horizon} bars ahead):")
        print("-" * 60)
        
        for i, row in df.head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']:25s}: {row['importance']:.4f}")

def compare_top_features():
    """Compare which features appear in top 10 across horizons."""
    importance_data = load_importance_data()
    
    print("\n📊 TOP FEATURES COMPARISON")
    print("=" * 80)
    
    # Get top 10 for each horizon
    top_features_by_horizon = {}
    for horizon in [1, 3, 5, 10]:
        if horizon in importance_data:
            top_10 = importance_data[horizon].head(10)['feature'].tolist()
            top_features_by_horizon[horizon] = top_10
    
    # Find common features
    all_top_features = set()
    for features in top_features_by_horizon.values():
        all_top_features.update(features)
    
    # Create comparison table
    print(f"{'Feature':<25} {'H1':<6} {'H3':<6} {'H5':<6} {'H10':<6} {'Count':<6}")
    print("-" * 70)
    
    feature_rankings = {}
    
    for feature in sorted(all_top_features):
        rankings = []
        count = 0
        
        for horizon in [1, 3, 5, 10]:
            if horizon in top_features_by_horizon:
                if feature in top_features_by_horizon[horizon]:
                    rank = top_features_by_horizon[horizon].index(feature) + 1
                    rankings.append(f"{rank:2d}")
                    count += 1
                else:
                    rankings.append(" -")
            else:
                rankings.append(" -")
        
        if count >= 2:  # Show features that appear in at least 2 horizons
            feature_rankings[feature] = count
            print(f"{feature:<25} {rankings[0]:<6} {rankings[1]:<6} {rankings[2]:<6} {rankings[3]:<6} {count:<6}")
    
    return feature_rankings

def analyze_feature_categories(importance_data):
    """Analyze most important features by category."""
    print("\n📈 FEATURE CATEGORIES ANALYSIS")
    print("=" * 60)
    
    categories = {
        'Volume Indicators': ['volume_change', 'volume_sma_20', 'relative_volume'],
        'Bollinger Bands': ['bb_position', 'bb_upper', 'bb_lower', 'bb_width', 'bb_touch'],
        'RSI & Momentum': ['rsi_14', 'momentum_3', 'momentum_10'],
        'Stochastic': ['stoch_k', 'stoch_d'],
        'Moving Averages': ['sma_', 'ema_', 'slope_ema', 'ema_diff'],
        'MACD': ['macd_line', 'macd_signal', 'macd_histogram'],
        'Price Action': ['atr_14', 'volatility_ratio', 'high_low_range', 'body_to_range', 'candle_'],
        'Lag Features': ['_lag_'],
        'Other': ['cci_20']
    }
    
    for horizon in [1, 3, 5, 10]:
        if horizon not in importance_data:
            continue
            
        print(f"\n🎯 Horizon {horizon} - Category breakdown (Top 15):")
        
        top_15 = importance_data[horizon].head(15)
        category_counts = {}
        
        for _, row in top_15.iterrows():
            feature = row['feature']
            categorized = False
            
            for category, keywords in categories.items():
                if any(keyword in feature for keyword in keywords):
                    if category not in category_counts:
                        category_counts[category] = []
                    category_counts[category].append((feature, row['importance']))
                    categorized = True
                    break
            
            if not categorized:
                if 'Other' not in category_counts:
                    category_counts['Other'] = []
                category_counts['Other'].append((feature, row['importance']))
        
        # Show results
        for category, features in sorted(category_counts.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"   {category:20s}: {len(features)} features")
            for feature, importance in features[:3]:  # Show top 3 in category
                print(f"      {feature:20s} ({importance:.4f})")
            if len(features) > 3:
                print(f"      ... and {len(features)-3} more")

def show_horizon_insights():
    """Show insights about different prediction horizons."""
    importance_data = load_importance_data()
    
    print("\n🔍 HORIZON-SPECIFIC INSIGHTS")
    print("=" * 60)
    
    insights = {
        1: "Short-term (1 bar): Focus on immediate momentum and volume",
        3: "Short-medium term (3 bars): Balanced technical indicators", 
        5: "Medium term (5 bars): Price volatility and trend strength",
        10: "Longer term (10 bars): Moving averages and trend direction"
    }
    
    for horizon in [1, 3, 5, 10]:
        if horizon in importance_data:
            print(f"\n🎯 Horizon {horizon}: {insights[horizon]}")
            
            df = importance_data[horizon]
            top_3 = df.head(3)
            
            print("   Top 3 features:")
            for i, row in top_3.iterrows():
                print(f"     {i+1}. {row['feature']:20s} ({row['importance']:.4f})")

def main():
    """Main function to display feature importance analysis."""
    print("🔍 FEATURE IMPORTANCE ANALYSIS RESULTS")
    print("=" * 70)
    
    # Load and display data
    importance_data = load_importance_data()
    
    if not importance_data:
        print("❌ No feature importance data found!")
        print("   Run analyze_feature_importance.py first")
        return
    
    # Show top features by horizon
    show_top_features_by_horizon(importance_data)
    
    # Compare features across horizons
    feature_rankings = compare_top_features()
    
    # Analyze categories
    analyze_feature_categories(importance_data)
    
    # Show insights
    show_horizon_insights()
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("=" * 50)
    print("💡 Key takeaways:")
    print("   • Volume indicators are crucial for short-term predictions")
    print("   • Bollinger Bands important across all horizons")
    print("   • Moving averages more important for longer horizons")
    print("   • Lag features help capture momentum patterns")

if __name__ == "__main__":
    main()