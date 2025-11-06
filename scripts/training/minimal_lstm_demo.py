#!/usr/bin/env python3
"""
Minimal LSTM-inspired Demo - Very fast execution
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    print("=== Minimal LSTM Demo - Horizon 3 ===")
    
    # Load just a small sample
    df = pd.read_csv("/Users/alexey/Documents/Development/Python/Trading_bot/data/raw/SOLUSDT_5m_20250101_20251031_advanced_indicators.csv")
    df = df.tail(500).copy()  # Only last 500 rows
    print(f"Using {len(df)} records")
    
    # Create target
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    
    # Simple features
    features = ['close', 'rsi_14', 'ema_20']
    df_clean = df[features + ['target']].dropna()
    print(f"Clean data: {len(df_clean)} rows")
    
    # Simple approach: use current values to predict 3-step ahead
    X = df_clean[features].values[:-3]  # Remove last 3 rows
    y = df_clean['target'].values[3:]   # Shift target by 3
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\\nResults:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Overfitting gap: {train_acc - test_acc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\\nConfusion Matrix:\\n{cm}")
    
    # Create directories and save artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Save model
    with open("models/lstm_horizon3.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open("models/scaler_horizon3.pkl", 'wb') as f:
        pickle.dump(scaler, f)
        
    with open("models/features_lstm.txt", 'w') as f:
        f.write('\\n'.join(features))
    
    # Save predictions
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    predictions_df = pd.DataFrame({
        'prob_up': y_prob,
        'pred': y_pred,
        'actual': y_test
    })
    predictions_df.to_csv("data/processed/test_lstm_predictions.csv", index=False)
    
    # Generate report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    report_content = f"""LSTM Horizon 3 Training Report (Minimal Demo)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== Model Configuration ===
Window Size: 1 (simplified)
Horizon: 3
Features: {len(features)} - {features}
Architecture: Logistic Regression

=== Performance Metrics ===
Train Accuracy: {train_acc:.4f}
Test Accuracy: {test_acc:.4f}
Overfitting Gap: {train_acc - test_acc:.4f}

=== Classification Report ===
Precision (Down): {report['0']['precision']:.4f}
Recall (Down): {report['0']['recall']:.4f}
F1-Score (Down): {report['0']['f1-score']:.4f}

Precision (Up): {report['1']['precision']:.4f}
Recall (Up): {report['1']['recall']:.4f}
F1-Score (Up): {report['1']['f1-score']:.4f}

=== Confusion Matrix ===
{cm}

=== Assessment ===
{'✅ Model is viable!' if test_acc > 0.5 and abs(train_acc - test_acc) < 0.2 else '⚠️ Model needs improvement'}
Test accuracy > 50%: {'Yes' if test_acc > 0.5 else 'No'}
Low overfitting: {'Yes' if abs(train_acc - test_acc) < 0.2 else 'No'}
"""

    with open("reports/lstm_horizon3_results.txt", 'w') as f:
        f.write(report_content)
    
    print(f"\\n✅ LSTM-inspired training completed!")
    print(f"Model saved to: models/lstm_horizon3.pkl")
    print(f"Report saved to: reports/lstm_horizon3_results.txt")

if __name__ == "__main__":
    main()