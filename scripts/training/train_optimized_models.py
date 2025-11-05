#!/usr/bin/env python3
"""
Train ML models on optimized top-25 features for multiple prediction horizons.
Test performance on target horizons: 3, 5, 7, 10
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class OptimizedModelTrainer:
    """Train models on optimized feature set."""
    
    def __init__(self, input_file="data/processed/SOLUSDT_5m_complete_top25.csv"):
        self.input_file = input_file
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_importance = {}
        
    def prepare_data_for_horizon(self, df, horizon):
        """Prepare data for specific prediction horizon."""
        
        print(f"📊 Preparing data for horizon {horizon}...")
        
        # Create target variable
        df_work = df.copy()
        target_col = f'target_{horizon}'
        df_work[target_col] = (df_work['close'].shift(-horizon) > df_work['close']).astype(int)
        
        # Remove rows where target is NaN
        df_clean = df_work.dropna()
        
        # Get features (exclude OHLCV and target)
        base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', target_col]
        feature_cols = [col for col in df_clean.columns if col not in base_cols]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        print(f"   📈 Features: {len(feature_cols)}")
        print(f"   📊 Samples: {len(X)}")
        print(f"   ⚖️  Class balance: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def train_model_for_horizon(self, horizon, test_size=0.2, random_state=42):
        """Train Random Forest model for specific horizon."""
        
        print(f"\n🚀 TRAINING MODEL FOR HORIZON {horizon}")
        print("=" * 50)
        
        # Load data
        df = pd.read_csv(self.input_file)
        
        # Prepare data
        X, y, feature_cols = self.prepare_data_for_horizon(df, horizon)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        print("🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = rf_model.predict(X_train_scaled)
        y_pred_test = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        # Cross validation
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        self.models[horizon] = rf_model
        self.scalers[horizon] = scaler
        self.feature_importance[horizon] = importance_df
        
        self.results[horizon] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'samples_train': len(X_train),
            'samples_test': len(X_test),
            'features_count': len(feature_cols),
            'class_balance': y.value_counts().to_dict()
        }
        
        # Print results
        print(f"✅ RESULTS FOR HORIZON {horizon}:")
        print(f"   🎯 Train Accuracy: {train_accuracy:.3f}")
        print(f"   🎯 Test Accuracy:  {test_accuracy:.3f}")
        print(f"   📊 Precision:      {precision:.3f}")
        print(f"   📊 Recall:         {recall:.3f}")
        print(f"   📊 F1-Score:       {f1:.3f}")
        print(f"   🔄 CV Accuracy:    {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_test))
        
        return {
            'model': rf_model,
            'scaler': scaler,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred_test,
            'y_pred_proba': y_pred_proba,
            'feature_cols': feature_cols
        }
    
    def train_all_horizons(self, horizons=[3, 5, 7, 10]):
        """Train models for all specified horizons."""
        
        print("🚀 TRAINING MODELS FOR ALL HORIZONS")
        print("=" * 60)
        
        trained_models = {}
        
        for horizon in horizons:
            try:
                model_info = self.train_model_for_horizon(horizon)
                trained_models[horizon] = model_info
                
                # Save model
                model_dir = Path("models/optimized")
                model_dir.mkdir(parents=True, exist_ok=True)
                
                model_file = model_dir / f"rf_horizon_{horizon}_optimized.pkl"
                scaler_file = model_dir / f"scaler_horizon_{horizon}_optimized.pkl"
                
                with open(model_file, 'wb') as f:
                    pickle.dump(model_info['model'], f)
                
                with open(scaler_file, 'wb') as f:
                    pickle.dump(model_info['scaler'], f)
                
                print(f"💾 Saved model: {model_file}")
                
            except Exception as e:
                print(f"❌ Error training horizon {horizon}: {e}")
                continue
        
        return trained_models
    
    def compare_results(self, horizons=[3, 5, 7, 10]):
        """Create comparison table of all models."""
        
        print("\n📊 MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for horizon in horizons:
            if horizon in self.results:
                result = self.results[horizon]
                comparison_data.append({
                    'Horizon': horizon,
                    'Test_Accuracy': f"{result['test_accuracy']:.3f}",
                    'Precision': f"{result['precision']:.3f}",
                    'Recall': f"{result['recall']:.3f}",
                    'F1_Score': f"{result['f1_score']:.3f}",
                    'CV_Accuracy': f"{result['cv_mean']:.3f} ± {result['cv_std']:.3f}",
                    'Features': result['features_count'],
                    'Train_Samples': result['samples_train'],
                    'Test_Samples': result['samples_test']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv("results/optimized_models_comparison.csv", index=False)
        
        # Find best performing models
        print(f"\n🏆 BEST PERFORMING MODELS:")
        
        metrics = ['Test_Accuracy', 'Precision', 'Recall', 'F1_Score']
        for metric in metrics:
            if metric in comparison_df.columns:
                # Convert string values back to float for comparison
                metric_values = comparison_df[metric].str.split(' ±').str[0].astype(float)
                best_idx = metric_values.idxmax()
                best_horizon = comparison_df.loc[best_idx, 'Horizon']
                best_value = comparison_df.loc[best_idx, metric]
                print(f"   📈 Best {metric}: Horizon {best_horizon} ({best_value})")
        
        return comparison_df
    
    def analyze_feature_importance_across_horizons(self, top_n=10):
        """Analyze feature importance across all horizons."""
        
        print(f"\n🔍 TOP-{top_n} FEATURE IMPORTANCE BY HORIZON")
        print("=" * 70)
        
        for horizon in sorted(self.feature_importance.keys()):
            print(f"\n🎯 HORIZON {horizon}:")
            print("-" * 30)
            
            top_features = self.feature_importance[horizon].head(top_n)
            for idx, row in top_features.iterrows():
                print(f"   {idx+1:2d}. {row['feature']:20s}: {row['importance']:.4f}")
        
        # Save detailed feature importance
        for horizon, importance_df in self.feature_importance.items():
            importance_df.to_csv(f"results/feature_importance_optimized_horizon_{horizon}.csv", index=False)
    
    def create_summary_report(self):
        """Create comprehensive summary report."""
        
        report_file = "results/optimized_models_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("OPTIMIZED MODELS TRAINING SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Input data: {self.input_file}\n")
            f.write(f"Feature set: Top 25 optimized features\n")
            f.write(f"Models trained: {len(self.results)}\n")
            f.write(f"Horizons tested: {list(self.results.keys())}\n\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-"*30 + "\n")
            
            for horizon in sorted(self.results.keys()):
                result = self.results[horizon]
                f.write(f"\nHorizon {horizon}:\n")
                f.write(f"  Test Accuracy: {result['test_accuracy']:.3f}\n")
                f.write(f"  Precision:     {result['precision']:.3f}\n")
                f.write(f"  Recall:        {result['recall']:.3f}\n")
                f.write(f"  F1-Score:      {result['f1_score']:.3f}\n")
                f.write(f"  CV Accuracy:   {result['cv_mean']:.3f} ± {result['cv_std']:.3f}\n")
            
            # Best models
            f.write(f"\nBEST MODELS:\n")
            f.write("-"*15 + "\n")
            
            best_accuracy = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
            best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
            
            f.write(f"Best Accuracy: Horizon {best_accuracy[0]} ({best_accuracy[1]['test_accuracy']:.3f})\n")
            f.write(f"Best F1-Score: Horizon {best_f1[0]} ({best_f1[1]['f1_score']:.3f})\n")
        
        print(f"📄 Summary report saved: {report_file}")

def main():
    """Main function to train optimized models."""
    
    print("🚀 TRAINING OPTIMIZED ML MODELS")
    print("=" * 60)
    
    # Check if data exists
    input_file = "data/processed/SOLUSDT_5m_complete_top25.csv"
    if not Path(input_file).exists():
        print(f"❌ Data file not found: {input_file}")
        print("   Run add_lag_features.py first to create optimized dataset")
        return
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    Path("models/optimized").mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = OptimizedModelTrainer(input_file)
    
    # Train models for all horizons
    horizons = [3, 5, 7, 10]
    print(f"🎯 Target horizons: {horizons}")
    
    trained_models = trainer.train_all_horizons(horizons)
    
    # Compare results
    if trainer.results:
        comparison_df = trainer.compare_results(horizons)
        
        # Analyze feature importance
        trainer.analyze_feature_importance_across_horizons(top_n=15)
        
        # Create summary report
        trainer.create_summary_report()
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print("=" * 40)
        print(f"✅ Models trained: {len(trained_models)}")
        print(f"✅ Features used: 25 (optimized set)")
        print(f"💾 Models saved to: models/optimized/")
        print(f"📊 Results saved to: results/")
        
        # Show key insights
        if len(trainer.results) > 1:
            accuracies = [result['test_accuracy'] for result in trainer.results.values()]
            avg_accuracy = np.mean(accuracies)
            print(f"\n📈 Average test accuracy: {avg_accuracy:.3f}")
            print(f"📈 Best accuracy: {max(accuracies):.3f}")
            print(f"📉 Feature reduction: 50% (25 vs ~50 original)")
    
    else:
        print("❌ No models were successfully trained")

if __name__ == "__main__":
    main()