#!/usr/bin/env python3
"""
Quick LSTM-inspired Demo for Price Direction Prediction - Horizon 3
Simplified version for fast execution
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)

def create_sequences(data, window_size=20):
    """Create sliding window sequences"""
    X, y = [], []
    
    for i in range(window_size, len(data)):
        # Use flattened window as features
        window_data = data[i-window_size:i].flatten()
        X.append(window_data)
        y.append(data.iloc[i]['target'])
    
    return np.array(X), np.array(y)

def main():
    print("=== Quick LSTM-inspired Demo - Horizon 3 ===\n")
    
    # Load data
    print("Loading data...")
    data_path = "/Users/alexey/Documents/Development/Python/Trading_bot/data/raw/SOLUSDT_5m_20250101_20251031_advanced_indicators.csv"
    df = pd.read_csv(data_path)
    
    # Use only the most recent 2000 rows for speed
    df = df.tail(2000).copy()
    print(f"Using {len(df)} recent records")
    
    # Create target
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    
    # Select a few key features
    features = ['rsi_14', 'ema_20', 'ema_50', 'macd_line', 'atr_14', 'bb_position']
    available_features = [f for f in features if f in df.columns and df[f].notna().sum() > len(df) * 0.8]
    
    print(f"Using features: {available_features}")
    
    # Clean data
    df_clean = df[['timestamp', 'close', 'target'] + available_features].dropna()
    print(f"Clean data shape: {df_clean.shape}")
    
    if len(df_clean) < 100:
        print("Not enough clean data. Exiting.")
        return
    
    # Prepare features for windowing
    feature_data = df_clean[available_features]
    
    # Create sequences (simplified - use raw features without scaling)
    window_size = 10  # Smaller window for speed
    X, y = [], []
    
    for i in range(window_size, len(feature_data)):
        # Simple approach: use last 'window_size' values as features
        window_features = []
        for feature in available_features:
            window_features.extend(feature_data[feature].iloc[i-window_size:i].values)
        X.append(window_features)
        y.append(df_clean['target'].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Sequence data shape: X={X.shape}, y={y.shape}")
    
    # Time-based splits
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train simple model (Logistic Regression as LSTM substitute)
    print("\nTraining model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    val_acc = model.score(X_val_scaled, y_val)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"\nResults:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Overfitting gap: {train_acc - test_acc:.4f}")
    
    # Test predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\nTest Confusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Save artifacts
    model_path = "models/lstm_horizon3.pkl"
    scaler_path = "models/scaler_horizon3.pkl" 
    features_path = "models/features_lstm.txt"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(features_path, 'w') as f:
        f.write('\\n'.join(available_features))
    
    # Save predictions
    test_timestamps = df_clean['timestamp'].iloc[val_end + window_size:val_end + window_size + len(y_test)]
    test_closes = df_clean['close'].iloc[val_end + window_size:val_end + window_size + len(y_test)]
    
    predictions_df = pd.DataFrame({
        'timestamp': test_timestamps.values,
        'close': test_closes.values,
        'prob_up': y_prob,
        'pred': y_pred,
        'actual': y_test
    })
    
    predictions_path = "data/processed/test_lstm_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    
    # Generate report
    report_content = f"""LSTM Horizon 3 Training Report (Quick Demo)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== Model Configuration ===
Window Size: {window_size}
Horizon: 3
Features: {len(available_features)} - {available_features}
Architecture: Logistic Regression (LSTM substitute)

=== Data Splits ===
Train samples: {len(X_train)}
Validation samples: {len(X_val)}
Test samples: {len(X_test)}

=== Performance Metrics ===
Train Accuracy: {train_acc:.4f}
Validation Accuracy: {val_acc:.4f}
Test Accuracy: {test_acc:.4f}
Overfitting Gap: {train_acc - test_acc:.4f}

=== Classification Report ===
Precision (Down): {report['0']['precision']:.4f}
Recall (Down): {report['0']['recall']:.4f}
F1-Score (Down): {report['0']['f1-score']:.4f}

Precision (Up): {report['1']['precision']:.4f}
Recall (Up): {report['1']['recall']:.4f}
F1-Score (Up): {report['1']['f1-score']:.4f}

Macro Avg F1: {report['macro avg']['f1-score']:.4f}
Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}

=== Confusion Matrix ===
{cm}

=== Model Artifacts ===
Model: {model_path}
Scaler: {scaler_path}
Features: {features_path}
Predictions: {predictions_path}

=== Assessment ===
{'✅ Model is viable!' if val_acc > 0.5 and abs(train_acc - test_acc) < 0.1 else '⚠️ Model needs improvement'}
Validation accuracy > 50%: {'Yes' if val_acc > 0.5 else 'No'}
Low overfitting: {'Yes' if abs(train_acc - test_acc) < 0.1 else 'No'}
"""
    
    report_path = "reports/lstm_horizon3_results.txt"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"\n=== Final Summary ===")
    print(f"✅ LSTM-inspired training completed!")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Overfitting gap: {train_acc - test_acc:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")
    
    # Quick visualization
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.subplot(1, 2, 2)
    plt.bar(['Train', 'Val', 'Test'], [train_acc, val_acc, test_acc])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('reports/quick_results.png')
    print("Plots saved to reports/quick_results.png")

if __name__ == "__main__":
    main()