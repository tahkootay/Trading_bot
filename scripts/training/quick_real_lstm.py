#!/usr/bin/env python3
"""
Quick Real LSTM Demo - Fast execution for demonstration
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Set JAX backend for Keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set random seeds
np.random.seed(42)

def create_sequences(data, targets, window_size=20):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

def main():
    print("=== Quick Real LSTM Demo ===")
    print(f"Using Keras backend: {keras.config.backend()}")
    
    # Load sample data
    try:
        df = pd.read_csv("data/processed/test_oct_2025.csv")
        # Use only recent 2000 records for speed
        df = df.tail(2000).copy()
        print(f"Using {len(df)} records")
    except:
        print("Creating synthetic data...")
        n = 2000
        np.random.seed(42)
        df = pd.DataFrame({
            'close': 200 + np.cumsum(np.random.randn(n) * 0.1),
            'rsi_14': 50 + 30 * np.sin(np.arange(n) * 0.1) + np.random.randn(n) * 5,
            'ema_20': np.random.randn(n),
            'ema_50': np.random.randn(n),
            'macd_line': np.random.randn(n) * 0.5,
            'atr_14': np.abs(np.random.randn(n)) + 1,
        })
    
    # Create target
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    
    # Select features
    features = ['rsi_14', 'ema_20', 'ema_50', 'macd_line', 'atr_14']
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        available_features = ['close']
    
    print(f"Using features: {available_features}")
    
    # Clean data
    df_clean = df[available_features + ['target']].dropna()
    print(f"Clean data: {len(df_clean)} rows")
    
    # Prepare sequences
    window_size = 15  # Smaller for speed
    
    # Split data
    n = len(df_clean)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_df = df_clean[:train_end]
    val_df = df_clean[train_end:val_end]
    test_df = df_clean[val_end:]
    
    # Scale features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[available_features])
    val_scaled = scaler.transform(val_df[available_features])
    test_scaled = scaler.transform(test_df[available_features])
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, train_df['target'].values, window_size)
    X_val, y_val = create_sequences(val_scaled, val_df['target'].values, window_size)
    X_test, y_test = create_sequences(test_scaled, test_df['target'].values, window_size)
    
    print(f"Sequence shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Build LSTM model
    print("\\nBuilding LSTM model...")
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(window_size, len(available_features))),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Train model
    print("\\nTraining LSTM...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,  # Fewer epochs for speed
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=False
    )
    
    # Evaluate
    print("\\nEvaluating LSTM...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\\n=== LSTM Results ===")
    print(f"Backend: {keras.config.backend()}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Overfitting Gap: {train_acc - test_acc:.4f}")
    print(f"\\nConfusion Matrix:")
    print(cm)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/quick_real_lstm.keras")
    print("\\nModel saved to: models/quick_real_lstm.keras")
    
    # Compare with previous results
    print(f"\\n=== Comparison with Previous Model ===")
    print(f"Previous (Logistic): ~48% accuracy")
    print(f"Real LSTM: {test_acc:.1%} accuracy")
    print(f"Improvement: {((test_acc - 0.48) / 0.48 * 100):+.1f}%")
    
    # Quick simulation
    print(f"\\n=== Quick Trading Simulation ===")
    capital = 10000
    trades = 0
    profit = 0
    
    # Simple strategy: trade on high confidence predictions
    confidence_threshold = 0.6
    
    for i in range(len(y_pred_prob) - 3):
        prob = y_pred_prob[i][0]
        
        if prob > confidence_threshold or prob < (1 - confidence_threshold):
            current_price = 100  # Normalized
            future_price = 100 + np.random.randn() * 0.5  # Simulate price change
            
            if prob > confidence_threshold:  # Buy
                pnl = (future_price - current_price) / current_price * 0.1 * capital
            else:  # Sell
                pnl = (current_price - future_price) / current_price * 0.1 * capital
            
            profit += pnl
            trades += 1
    
    if trades > 0:
        total_return = profit / capital
        print(f"Simulated Return: {total_return:.2%}")
        print(f"Number of Trades: {trades}")
    else:
        print("No high-confidence trades generated")
    
    print("\\n✅ Quick Real LSTM demo completed!")

if __name__ == "__main__":
    main()