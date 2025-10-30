#!/usr/bin/env python3
"""
Example: Training ML Model on Prepared Data

This example shows how to train Random Forest and XGBoost models 
on the prepared trading data for price direction prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional: XGBoost (install with: pip install xgboost)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")


def load_ml_data(data_dir: str = "data/processed") -> tuple:
    """Load prepared ML datasets."""
    print("📁 Loading ML datasets...")
    
    data_path = Path(data_dir)
    
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    
    # Separate features and targets
    feature_cols = [col for col in train_df.columns if col != 'target']
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"   ✅ Train: {len(X_train)} samples, {len(feature_cols)} features")
    print(f"   ✅ Val: {len(X_val)} samples")
    print(f"   ✅ Test: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model."""
    print("\n🌲 Training Random Forest...")
    
    # Random Forest with reasonable parameters for trading data
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    rf_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = rf_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    print(f"   ✅ Validation Accuracy: {val_accuracy:.4f}")
    
    return rf_model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    if not HAS_XGBOOST:
        print("\n⚠️ Skipping XGBoost (not installed)")
        return None
        
    print("\n🚀 Training XGBoost...")
    
    # XGBoost with reasonable parameters for trading data
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train model
    xgb_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = xgb_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    print(f"   ✅ Validation Accuracy: {val_accuracy:.4f}")
    
    return xgb_model


def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate model on test set."""
    print(f"\n📊 Evaluating {model_name} on Test Set...")
    
    # Predictions
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Accuracy
    accuracy = accuracy_score(y_test, test_pred)
    print(f"   🎯 Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\n   📋 Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['Down', 'Up']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    print(f"\n   🔢 Confusion Matrix:")
    print(f"        Predicted: Down  Up")
    print(f"   True Down:     {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"   True Up:       {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return accuracy, test_pred, test_proba


def get_feature_importance(model, feature_cols, top_n: int = 15):
    """Get and display feature importance."""
    if hasattr(model, 'feature_importances_'):
        print(f"\n🔍 Top {top_n} Most Important Features:")
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in importance_df.head(top_n).iterrows():
            print(f"   {row['feature']:20s}: {row['importance']:.4f}")
        
        return importance_df
    else:
        print("   ⚠️ Model doesn't support feature importance")
        return None


def main():
    """Main training and evaluation pipeline."""
    print("🤖 ML Model Training Example")
    print("=" * 50)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_ml_data()
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    
    # Train XGBoost (if available)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    print("\n" + "=" * 50)
    print("📊 MODEL EVALUATION")
    print("=" * 50)
    
    # Random Forest evaluation
    rf_accuracy, _, _ = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_importance = get_feature_importance(rf_model, feature_cols)
    
    # XGBoost evaluation
    if xgb_model is not None:
        xgb_accuracy, _, _ = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        xgb_importance = get_feature_importance(xgb_model, feature_cols)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print(f"🌲 Random Forest Test Accuracy: {rf_accuracy:.4f}")
    if xgb_model is not None:
        print(f"🚀 XGBoost Test Accuracy: {xgb_accuracy:.4f}")
    print(f"📊 Dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"🎯 Task: Predict price direction 3 bars ahead")
    print(f"📈 Features: {len(feature_cols)} technical indicators + lags")
    
    # Trading simulation example
    print("\n💰 TRADING SIMULATION EXAMPLE")
    print("=" * 30)
    
    # Use Random Forest predictions for example
    test_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Simple strategy: Buy when prediction > 0.6, Sell when < 0.4
    buy_threshold = 0.6
    sell_threshold = 0.4
    
    buy_signals = test_pred_proba > buy_threshold
    sell_signals = test_pred_proba < sell_threshold
    
    print(f"   📈 Buy signals (prob > {buy_threshold}): {buy_signals.sum()}")
    print(f"   📉 Sell signals (prob < {sell_threshold}): {sell_signals.sum()}")
    print(f"   ⏸️ Hold signals: {len(test_pred_proba) - buy_signals.sum() - sell_signals.sum()}")
    
    # Calculate accuracy for high-confidence predictions
    high_conf_mask = (test_pred_proba > buy_threshold) | (test_pred_proba < sell_threshold)
    if high_conf_mask.sum() > 0:
        high_conf_pred = (test_pred_proba[high_conf_mask] > 0.5).astype(int)
        high_conf_true = y_test[high_conf_mask]
        high_conf_accuracy = accuracy_score(high_conf_true, high_conf_pred)
        print(f"   🎯 High-confidence accuracy: {high_conf_accuracy:.4f}")
    
    print(f"\n🎉 Training completed! Models ready for trading.")


if __name__ == "__main__":
    main()