#!/usr/bin/env python3
"""
Real LSTM Model for Price Direction Prediction - Horizon 3
Implementation with proper Keras/JAX backend
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set JAX backend for Keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)

class RealLSTMPredictor:
    def __init__(self, window_size=30, horizon=3):
        self.window_size = window_size
        self.horizon = horizon
        self.scaler = StandardScaler()
        self.model = None
        self.features = None
        self.history = None
        
    def load_and_prepare_data(self, file_path, sample_size=None):
        """Load CSV data and prepare target variable"""
        print("Loading data...")
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        
        # Optionally sample data for faster training
        if sample_size and len(df) > sample_size:
            df = df.tail(sample_size).copy()
            print(f"Using sample of {sample_size} most recent records")
        
        # Create target variable: price direction in N bars
        df["target"] = (df["close"].shift(-self.horizon) > df["close"]).astype(int)
        
        # Remove rows with NaN targets
        df = df.dropna(subset=["target"])
        print(f"Data shape after target creation: {df.shape}")
        
        # Select enhanced features for LSTM
        self.features = [
            # Price-based indicators
            'rsi_14', 'rsi_14_lag_1', 'rsi_14_lag_3',
            # Moving averages
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
            'sma_5', 'sma_10', 'sma_20',
            # MACD family
            'macd_line', 'macd', 'macd_signal', 'macd_histogram',
            # Bollinger Bands
            'bb_position', 'bb_width', 'bb_position_lag_1', 'bb_position_lag_3',
            # Volatility
            'atr_14', 'volatility_ratio',
            # Momentum
            'momentum_3', 'momentum_10', 'momentum_3_lag_1',
            # Stochastic
            'stoch_k', 'stoch_d',
            # Volume
            'relative_volume', 'volume_change',
            # Price patterns
            'candle_ratio', 'body_to_range',
            # Other
            'cci_20', 'slope_ema_20', 'ema_diff_10_50'
        ]
        
        # Filter available features
        available_features = [f for f in self.features if f in df.columns]
        missing_features = [f for f in self.features if f not in df.columns]
        
        self.features = available_features
        print(f"Using {len(self.features)} features")
        if missing_features:
            print(f"Missing features: {missing_features}")
        
        # Remove rows with NaN in selected features
        df_clean = df[['timestamp', 'close', 'target'] + self.features].dropna()
        print(f"Data shape after cleaning: {df_clean.shape}")
        
        return df_clean
    
    def create_sequences(self, data, targets):
        """Create time sequences for LSTM"""
        X, y = [], []
        
        for i in range(self.window_size, len(data)):
            # Create sequence of length window_size
            X.append(data[i-self.window_size:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data_splits(self, df):
        """Split data chronologically and create sequences"""
        print("\\nPreparing data splits...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Time-based splits (70/15/15)
        n = len(df)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        print(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Fit scaler on training data
        train_features = train_df[self.features].values
        self.scaler.fit(train_features)
        
        # Transform all splits
        train_scaled = self.scaler.transform(train_df[self.features].values)
        val_scaled = self.scaler.transform(val_df[self.features].values)
        test_scaled = self.scaler.transform(test_df[self.features].values)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled, train_df['target'].values)
        X_val, y_val = self.create_sequences(val_scaled, val_df['target'].values)
        X_test, y_test = self.create_sequences(test_scaled, test_df['target'].values)
        
        print(f"Sequence shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Store test data for predictions
        self.test_df = test_df[self.window_size:].copy()
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def build_model(self, input_shape):
        """Build real LSTM model architecture"""
        print(f"\\nBuilding LSTM model with input shape: {input_shape}")
        
        model = Sequential([
            # First LSTM layer with return_sequences=True
            LSTM(64, return_sequences=True, input_shape=input_shape,
                 dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(32, return_sequences=False,
                 dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Dense layers
            Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            
            Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        print(model.summary())
        
        self.model = model
        return model
    
    def train_model(self, train_data, val_data, epochs=50, batch_size=32):
        """Train the LSTM model"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print(f"\\nTraining LSTM model for up to {epochs} epochs...")
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'models/lstm_best_weights.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series!
        )
        
        self.history = history
        return history
    
    def plot_training_history(self, save_path="reports/lstm_training_history.png"):
        """Plot training and validation metrics"""
        if not self.history:
            print("No training history available")
            return
            
        history = self.history.history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history saved to: {save_path}")
    
    def evaluate_model(self, test_data):
        """Evaluate model on test data"""
        X_test, y_test = test_data
        
        print("\\nEvaluating LSTM model on test data...")
        
        # Get predictions
        y_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_prob > 0.5).astype(int).flatten()
        y_prob = y_prob.flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\\n{cm}")
        print(f"Classification Report:\\n{classification_report(y_test, y_pred)}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.title('LSTM Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('reports/lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store predictions for CSV export
        self.test_predictions = {
            'timestamp': self.test_df['timestamp'].values,
            'close': self.test_df['close'].values,
            'prob_up': y_prob,
            'pred': y_pred,
            'actual': y_test
        }
        
        return accuracy, cm, report
    
    def save_artifacts(self, base_path="models"):
        """Save model, scaler, and features"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save model
        model_path = f"{base_path}/real_lstm_horizon3.keras"
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = f"{base_path}/scaler_real_lstm.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save features list
        features_path = f"{base_path}/features_real_lstm.txt"
        with open(features_path, 'w') as f:
            f.write('\\n'.join(self.features))
        print(f"Features saved to: {features_path}")
        
        # Save training history
        if self.history:
            history_path = f"{base_path}/training_history_real_lstm.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
            print(f"Training history saved to: {history_path}")
        
        return model_path, scaler_path, features_path

def main():
    print("=== Real LSTM Price Direction Prediction - Horizon 3 ===\\n")
    print(f"Using Keras backend: {keras.config.backend()}")
    
    # Initialize predictor
    predictor = RealLSTMPredictor(window_size=30, horizon=3)
    
    # Load and prepare data
    data_path = "/Users/alexey/Documents/Development/Python/Trading_bot/data/raw/SOLUSDT_5m_20250101_20251031_advanced_indicators.csv"
    df = predictor.load_and_prepare_data(data_path, sample_size=15000)  # Use more data but still manageable
    
    # Prepare data splits
    train_data, val_data, test_data = predictor.prepare_data_splits(df)
    
    # Build model
    input_shape = (predictor.window_size, len(predictor.features))
    predictor.build_model(input_shape)
    
    # Train model
    history = predictor.train_model(train_data, val_data, epochs=30, batch_size=64)
    
    # Plot training history
    os.makedirs("reports", exist_ok=True)
    predictor.plot_training_history()
    
    # Evaluate model
    test_accuracy, cm, report = predictor.evaluate_model(test_data)
    
    # Calculate training metrics for comparison
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    train_loss, train_accuracy_final = predictor.model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy_final = predictor.model.evaluate(X_val, y_val, verbose=0)
    overfitting_gap = train_accuracy_final - test_accuracy
    
    # Save artifacts
    model_path, scaler_path, features_path = predictor.save_artifacts()
    
    # Save predictions CSV
    os.makedirs("data/processed", exist_ok=True)
    predictions_df = pd.DataFrame(predictor.test_predictions)
    predictions_path = "data/processed/real_lstm_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Generate report
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = np.min(history.history['val_loss'])
    
    report_content = f"""Real LSTM Horizon 3 Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== Model Configuration ===
Backend: {keras.config.backend()}
Window Size: {predictor.window_size}
Horizon: {predictor.horizon}
Features: {len(predictor.features)}
Architecture: LSTM(64) -> LSTM(32) -> Dense(16) -> Dense(8) -> Dense(1)
Regularization: Dropout, BatchNorm, L2

=== Training Details ===
Total Epochs: {len(history.history['loss'])}
Best Epoch: {best_epoch}
Best Val Loss: {best_val_loss:.4f}
Early Stopping: {'Yes' if len(history.history['loss']) < 30 else 'No'}

=== Data Splits ===
Train samples: {len(train_data[0])}
Validation samples: {len(val_data[0])}
Test samples: {len(test_data[0])}

=== Performance Metrics ===
Train Accuracy: {train_accuracy_final:.4f}
Validation Accuracy: {val_accuracy_final:.4f}
Test Accuracy: {test_accuracy:.4f}
Overfitting Gap: {overfitting_gap:.4f}

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
{'✅ Real LSTM model shows improvement!' if test_accuracy > 0.52 and abs(overfitting_gap) < 0.1 else '⚠️ Model needs further optimization'}
Test accuracy > 52%: {'Yes' if test_accuracy > 0.52 else 'No'}
Low overfitting: {'Yes' if abs(overfitting_gap) < 0.1 else 'No'}
Proper LSTM: Yes
Backend: {keras.config.backend()}
"""
    
    report_path = "reports/real_lstm_results.txt"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"\\n=== Final Summary ===")
    print(f"✅ Real LSTM training completed!")
    print(f"Backend: {keras.config.backend()}")
    print(f"Train Accuracy: {train_accuracy_final:.4f}")
    print(f"Val Accuracy: {val_accuracy_final:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting gap: {overfitting_gap:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()