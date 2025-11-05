#!/usr/bin/env python3
"""
Analyze feature importance across all horizons to identify noisy/redundant indicators
"""

import pandas as pd
import numpy as np

def load_all_importance_data():
    """Load feature importance from all horizons."""
    horizons = [1, 3, 5, 10]
    all_data = {}
    
    for horizon in horizons:
        try:
            df = pd.read_csv(f"data/processed/feature_importance_horizon{horizon}.csv")
            all_data[horizon] = df
            print(f"✅ Loaded horizon {horizon}: {len(df)} features")
        except FileNotFoundError:
            print(f"❌ Missing data for horizon {horizon}")
    
    return all_data

def calculate_aggregate_scores(importance_data):
    """Calculate aggregate importance scores across all horizons."""
    
    # Collect all unique features
    all_features = set()
    for df in importance_data.values():
        all_features.update(df['feature'].tolist())
    
    # Create aggregate dataframe
    results = []
    
    for feature in all_features:
        scores = []
        ranks = []
        
        for horizon in [1, 3, 5, 10]:
            if horizon in importance_data:
                df = importance_data[horizon]
                if feature in df['feature'].values:
                    feature_row = df[df['feature'] == feature].iloc[0]
                    scores.append(feature_row['importance'])
                    ranks.append(df[df['feature'] == feature].index[0] + 1)
                else:
                    scores.append(0.0)
                    ranks.append(999)  # Very low rank for missing features
        
        # Calculate aggregate metrics
        avg_importance = np.mean(scores)
        max_importance = np.max(scores)
        min_importance = np.min(scores)
        std_importance = np.std(scores)
        
        avg_rank = np.mean(ranks)
        best_rank = np.min(ranks)
        worst_rank = np.max(ranks)
        
        # Count how many horizons this feature appears in top-N
        top10_count = sum(1 for rank in ranks if rank <= 10)
        top20_count = sum(1 for rank in ranks if rank <= 20)
        top30_count = sum(1 for rank in ranks if rank <= 30)
        
        # Calculate consistency score (lower std_importance relative to avg is better)
        consistency = avg_importance / (std_importance + 1e-8)
        
        results.append({
            'feature': feature,
            'avg_importance': avg_importance,
            'max_importance': max_importance,
            'min_importance': min_importance,
            'std_importance': std_importance,
            'consistency': consistency,
            'avg_rank': avg_rank,
            'best_rank': best_rank,
            'worst_rank': worst_rank,
            'top10_count': top10_count,
            'top20_count': top20_count,
            'top30_count': top30_count,
            'h1_imp': scores[0] if len(scores) > 0 else 0,
            'h3_imp': scores[1] if len(scores) > 1 else 0,
            'h5_imp': scores[2] if len(scores) > 2 else 0,
            'h10_imp': scores[3] if len(scores) > 3 else 0,
        })
    
    return pd.DataFrame(results).sort_values('avg_importance', ascending=False)

def categorize_features(df):
    """Categorize features by type for easier analysis."""
    
    categories = {
        'Volume': ['volume_change', 'volume_sma_20', 'relative_volume'],
        'Bollinger_Bands': ['bb_position', 'bb_upper', 'bb_lower', 'bb_width', 'bb_middle', 'bb_touch'],
        'RSI': ['rsi_14'],
        'Stochastic': ['stoch_k', 'stoch_d'],
        'MACD': ['macd_line', 'macd_signal', 'macd_histogram'],
        'Moving_Averages': ['sma_', 'ema_', 'slope_ema', 'ema_diff'],
        'Momentum': ['momentum_3', 'momentum_10'],
        'Price_Action': ['atr_14', 'volatility_ratio', 'high_low_range', 'body_to_range', 'candle_'],
        'Lag_Features': ['_lag_'],
        'CCI': ['cci_20']
    }
    
    def get_category(feature_name):
        for category, keywords in categories.items():
            if any(keyword in feature_name for keyword in keywords):
                return category
        return 'Other'
    
    df['category'] = df['feature'].apply(get_category)
    return df

def analyze_noise_candidates(df):
    """Identify features that might be adding noise."""
    
    print("\n🔍 NOISE ANALYSIS")
    print("=" * 80)
    
    # Very low importance features
    low_importance = df[df['avg_importance'] < 0.01]
    print(f"\n❌ VERY LOW IMPORTANCE (avg < 0.01): {len(low_importance)} features")
    print("These can likely be removed:")
    for _, row in low_importance.head(10).iterrows():
        print(f"   {row['feature']:25s}: avg={row['avg_importance']:.4f}, best_rank={row['best_rank']}")
    
    # Features that never make it to top 30
    never_top30 = df[df['top30_count'] == 0]
    print(f"\n❌ NEVER IN TOP-30: {len(never_top30)} features")
    print("Consider removing these:")
    for _, row in never_top30.head(10).iterrows():
        print(f"   {row['feature']:25s}: avg_rank={row['avg_rank']:.1f}, avg_imp={row['avg_importance']:.4f}")
    
    # High variance/inconsistent features  
    inconsistent = df[df['consistency'] < 5].sort_values('consistency')
    print(f"\n⚠️  INCONSISTENT FEATURES (low consistency): {len(inconsistent)} features")
    print("These vary a lot across horizons:")
    for _, row in inconsistent.head(10).iterrows():
        print(f"   {row['feature']:25s}: consistency={row['consistency']:.2f}, std={row['std_importance']:.4f}")
    
    return {
        'low_importance': low_importance,
        'never_top30': never_top30,
        'inconsistent': inconsistent
    }

def rank_features_for_removal(df):
    """Create a comprehensive ranking for feature removal."""
    
    # Create removal score (higher = more likely to remove)
    df['removal_score'] = 0
    
    # Penalize low average importance
    df['removal_score'] += (0.02 - df['avg_importance'].clip(0, 0.02)) * 50
    
    # Penalize high average rank (worse ranking)
    df['removal_score'] += (df['avg_rank'] - 25).clip(0, None) * 0.5
    
    # Penalize never being in top 30
    df['removal_score'] += (df['top30_count'] == 0) * 20
    
    # Penalize inconsistency
    df['removal_score'] += (5 - df['consistency'].clip(0, 5)) * 2
    
    # Sort by removal score (highest first = most likely to remove)
    removal_ranking = df.sort_values('removal_score', ascending=False)
    
    print("\n🗑️  REMOVAL CANDIDATES (top 20)")
    print("=" * 80)
    print(f"{'Feature':<25} {'Score':<8} {'AvgImp':<8} {'AvgRank':<8} {'Top30':<6} {'Consistency':<12}")
    print("-" * 80)
    
    for _, row in removal_ranking.head(20).iterrows():
        print(f"{row['feature']:<25} {row['removal_score']:<8.1f} {row['avg_importance']:<8.4f} "
              f"{row['avg_rank']:<8.1f} {row['top30_count']:<6} {row['consistency']:<12.2f}")
    
    return removal_ranking

def suggest_feature_sets(df):
    """Suggest different feature sets based on importance."""
    
    print("\n📊 SUGGESTED FEATURE SETS")
    print("=" * 60)
    
    # Essential features (consistently important)
    essential = df[(df['avg_importance'] >= 0.02) & (df['top10_count'] >= 2)]
    print(f"\n🔥 ESSENTIAL FEATURES ({len(essential)}): Always keep these")
    for _, row in essential.iterrows():
        print(f"   {row['feature']:25s}: avg_imp={row['avg_importance']:.4f}, top10_count={row['top10_count']}")
    
    # Good features (moderately important)
    good = df[(df['avg_importance'] >= 0.015) & (df['top20_count'] >= 2) & (~df.index.isin(essential.index))]
    print(f"\n✅ GOOD FEATURES ({len(good)}): Keep for better performance")
    for _, row in good.head(15).iterrows():
        print(f"   {row['feature']:25s}: avg_imp={row['avg_importance']:.4f}, top20_count={row['top20_count']}")
    
    # Optional features (sometimes useful)
    optional = df[(df['avg_importance'] >= 0.01) & (df['top30_count'] >= 1) & 
                  (~df.index.isin(essential.index)) & (~df.index.isin(good.index))]
    print(f"\n🤔 OPTIONAL FEATURES ({len(optional)}): Keep if you have computing capacity")
    
    # Noise features (likely to remove)
    noise = df[(df['avg_importance'] < 0.01) | (df['top30_count'] == 0)]
    print(f"\n❌ NOISE FEATURES ({len(noise)}): Consider removing")
    
    return {
        'essential': essential,
        'good': good, 
        'optional': optional,
        'noise': noise
    }

def analyze_by_category(df):
    """Analyze feature importance by category."""
    
    print("\n📈 ANALYSIS BY CATEGORY")
    print("=" * 60)
    
    category_stats = df.groupby('category').agg({
        'avg_importance': ['count', 'mean', 'std', 'max'],
        'top10_count': 'sum',
        'top30_count': 'sum'
    }).round(4)
    
    category_stats.columns = ['count', 'avg_imp_mean', 'avg_imp_std', 'max_imp', 'total_top10', 'total_top30']
    category_stats = category_stats.sort_values('avg_imp_mean', ascending=False)
    
    print("\nCategory performance:")
    print(category_stats)
    
    # Show best features from each category
    print("\n🏆 BEST FEATURE FROM EACH CATEGORY:")
    for category in df['category'].unique():
        cat_features = df[df['category'] == category].head(1)
        if len(cat_features) > 0:
            row = cat_features.iloc[0]
            print(f"   {category:15s}: {row['feature']:25s} (avg_imp={row['avg_importance']:.4f})")

def main():
    """Main analysis function."""
    
    print("🔍 COMPREHENSIVE FEATURE NOISE ANALYSIS")
    print("=" * 70)
    
    # Load data
    importance_data = load_all_importance_data()
    
    if not importance_data:
        print("❌ No importance data found. Run analyze_feature_importance.py first")
        return
    
    # Calculate aggregate scores
    print("\n📊 Calculating aggregate importance scores...")
    df = calculate_aggregate_scores(importance_data)
    
    # Categorize features
    df = categorize_features(df)
    
    # Save detailed results
    df.to_csv("data/processed/feature_noise_analysis.csv", index=False)
    print(f"💾 Saved detailed analysis to data/processed/feature_noise_analysis.csv")
    
    # Analyze noise candidates
    noise_analysis = analyze_noise_candidates(df)
    
    # Rank features for removal
    removal_ranking = rank_features_for_removal(df)
    
    # Suggest feature sets
    feature_sets = suggest_feature_sets(df)
    
    # Analyze by category
    analyze_by_category(df)
    
    # Summary recommendations
    print("\n🎯 SUMMARY RECOMMENDATIONS")
    print("=" * 50)
    print(f"📊 Total features analyzed: {len(df)}")
    print(f"🔥 Essential features: {len(feature_sets['essential'])}")
    print(f"✅ Good features: {len(feature_sets['good'])}")
    print(f"🤔 Optional features: {len(feature_sets['optional'])}")
    print(f"❌ Noise features: {len(feature_sets['noise'])}")
    print(f"\n💡 Recommended: Keep {len(feature_sets['essential']) + len(feature_sets['good'])} features")
    print(f"   Remove {len(feature_sets['noise'])} noise features")
    print(f"   Potential reduction: {len(feature_sets['noise'])/len(df)*100:.1f}%")

if __name__ == "__main__":
    main()