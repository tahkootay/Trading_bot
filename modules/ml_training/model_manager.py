#!/usr/bin/env python3
"""
Model Management Module

Handles saving, loading, and versioning of trained ML models for trading.
Includes model metadata, feature importance, and performance metrics.
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelManager:
    """Manages ML models: saving, loading, versioning, and metadata."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize ModelManager.
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "random_forest").mkdir(exist_ok=True)
        (self.models_dir / "xgboost").mkdir(exist_ok=True)
        (self.models_dir / "scalers").mkdir(exist_ok=True)
        (self.models_dir / "metadata").mkdir(exist_ok=True)
    
    def save_model(self, 
                  model: Any, 
                  scaler: StandardScaler,
                  model_type: str,
                  version: str,
                  feature_columns: List[str],
                  performance_metrics: Dict[str, Any],
                  hyperparameters: Dict[str, Any],
                  training_info: Dict[str, Any],
                  feature_importance: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Save trained model with all associated data.
        
        Args:
            model: Trained ML model
            scaler: Fitted StandardScaler
            model_type: Type of model ('random_forest', 'xgboost', etc.)
            version: Model version (e.g., 'v1', 'v2', 'v1.1')
            feature_columns: List of feature column names
            performance_metrics: Dict with accuracy, precision, recall, etc.
            hyperparameters: Dict with model hyperparameters
            training_info: Dict with training dataset info
            feature_importance: DataFrame with feature importance (optional)
            
        Returns:
            Dict with file paths
        """
        print(f"💾 Saving {model_type} model {version}...")
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File paths
        model_file = self.models_dir / model_type / f"{model_type}_{version}.pkl"
        scaler_file = self.models_dir / "scalers" / f"scaler_{version}.pkl" 
        metadata_file = self.models_dir / "metadata" / f"metadata_{version}.json"
        importance_file = self.models_dir / "metadata" / f"feature_importance_{version}.csv"
        
        # Save model
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✅ Model saved: {model_file}")
        
        # Save scaler
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"   ✅ Scaler saved: {scaler_file}")
        
        # Save feature importance
        if feature_importance is not None:
            feature_importance.to_csv(importance_file, index=False)
            print(f"   ✅ Feature importance saved: {importance_file}")
        
        # Create metadata
        metadata = {
            "model_info": {
                "model_type": model_type,
                "version": version,
                "timestamp": timestamp,
                "sklearn_version": self._get_sklearn_version(),
                "python_version": self._get_python_version()
            },
            "files": {
                "model": str(model_file),
                "scaler": str(scaler_file),
                "metadata": str(metadata_file),
                "feature_importance": str(importance_file) if feature_importance is not None else None
            },
            "features": {
                "count": len(feature_columns),
                "columns": feature_columns
            },
            "hyperparameters": hyperparameters,
            "performance": performance_metrics,
            "training": training_info
        }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"   ✅ Metadata saved: {metadata_file}")
        
        return {
            "model": str(model_file),
            "scaler": str(scaler_file),
            "metadata": str(metadata_file),
            "feature_importance": str(importance_file) if feature_importance is not None else None
        }
    
    def load_model(self, version: str) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
        """
        Load model, scaler, and metadata by version.
        
        Args:
            version: Model version to load
            
        Returns:
            Tuple of (model, scaler, metadata)
        """
        print(f"📁 Loading model {version}...")
        
        # Load metadata first to get model type
        metadata_file = self.models_dir / "metadata" / f"metadata_{version}.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        model_type = metadata["model_info"]["model_type"]
        
        # Load model
        model_file = Path(metadata["files"]["model"])
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"   ✅ Model loaded: {model_file}")
        
        # Load scaler
        scaler_file = Path(metadata["files"]["scaler"])
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        print(f"   ✅ Scaler loaded: {scaler_file}")
        
        return model, scaler, metadata
    
    def list_models(self) -> pd.DataFrame:
        """
        List all available models with their info.
        
        Returns:
            DataFrame with model information
        """
        models_info = []
        
        metadata_dir = self.models_dir / "metadata"
        for metadata_file in metadata_dir.glob("metadata_*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                models_info.append({
                    "version": metadata["model_info"]["version"],
                    "model_type": metadata["model_info"]["model_type"],
                    "timestamp": metadata["model_info"]["timestamp"],
                    "test_accuracy": metadata["performance"]["test_accuracy"],
                    "val_accuracy": metadata["performance"]["val_accuracy"],
                    "features_count": metadata["features"]["count"],
                    "train_samples": metadata["training"]["train_samples"]
                })
            except Exception as e:
                print(f"   ⚠️ Error reading {metadata_file}: {e}")
        
        if models_info:
            return pd.DataFrame(models_info).sort_values("timestamp", ascending=False)
        else:
            return pd.DataFrame()
    
    def predict(self, 
               data: pd.DataFrame, 
               version: str, 
               return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Make predictions using saved model.
        
        Args:
            data: DataFrame with features (must match training features)
            version: Model version to use
            return_probabilities: Whether to return probability scores
            
        Returns:
            Dict with predictions and probabilities
        """
        # Load model and scaler
        model, scaler, metadata = self.load_model(version)
        
        # Check features
        required_features = metadata["features"]["columns"]
        missing_features = [f for f in required_features if f not in data.columns]
        
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and scale features
        X = data[required_features]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        result = {
            "predictions": predictions,
            "model_version": version,
            "model_type": metadata["model_info"]["model_type"],
            "prediction_timestamp": datetime.now().isoformat()
        }
        
        if return_probabilities:
            probabilities = model.predict_proba(X_scaled)
            result["probabilities"] = probabilities
            result["probability_up"] = probabilities[:, 1]  # Probability of price going up
        
        return result
    
    def get_trading_signals(self, 
                          data: pd.DataFrame, 
                          version: str,
                          buy_threshold: float = 0.6,
                          sell_threshold: float = 0.4) -> pd.DataFrame:
        """
        Generate trading signals using saved model.
        
        Args:
            data: DataFrame with features
            version: Model version to use
            buy_threshold: Probability threshold for buy signals
            sell_threshold: Probability threshold for sell signals
            
        Returns:
            DataFrame with trading signals
        """
        predictions = self.predict(data, version, return_probabilities=True)
        
        signals_df = data.copy()
        signals_df['prediction'] = predictions["predictions"]
        signals_df['probability_up'] = predictions["probability_up"]
        
        # Generate signals
        signals_df['signal'] = 'HOLD'
        signals_df.loc[signals_df['probability_up'] > buy_threshold, 'signal'] = 'BUY'
        signals_df.loc[signals_df['probability_up'] < sell_threshold, 'signal'] = 'SELL'
        
        # Signal strength
        signals_df['signal_strength'] = 'WEAK'
        signals_df.loc[signals_df['probability_up'] > 0.7, 'signal_strength'] = 'STRONG'
        signals_df.loc[signals_df['probability_up'] < 0.3, 'signal_strength'] = 'STRONG'
        
        return signals_df[['prediction', 'probability_up', 'signal', 'signal_strength']]
    
    def compare_models(self, versions: List[str]) -> pd.DataFrame:
        """
        Compare performance of multiple model versions.
        
        Args:
            versions: List of model versions to compare
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for version in versions:
            try:
                metadata_file = self.models_dir / "metadata" / f"metadata_{version}.json"
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                comparison_data.append({
                    "version": version,
                    "model_type": metadata["model_info"]["model_type"],
                    "timestamp": metadata["model_info"]["timestamp"],
                    "train_accuracy": metadata["performance"]["train_accuracy"],
                    "val_accuracy": metadata["performance"]["val_accuracy"],
                    "test_accuracy": metadata["performance"]["test_accuracy"],
                    "overfitting": metadata["performance"]["train_accuracy"] - metadata["performance"]["val_accuracy"],
                    "features": metadata["features"]["count"],
                    "train_samples": metadata["training"]["train_samples"]
                })
            except Exception as e:
                print(f"   ⚠️ Error loading {version}: {e}")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            return df.sort_values("test_accuracy", ascending=False)
        else:
            return pd.DataFrame()
    
    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version."""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def save_trained_model_example():
    """Example: Save a trained model with all metadata."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd
    
    print("📚 ПРИМЕР: Сохранение обученной модели")
    print("=" * 50)
    
    # Load processed data
    print("1️⃣ Загружаем данные...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    feature_cols = [col for col in train_df.columns if col != 'target']
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"   ✅ Данные загружены: {len(X_train)} train, {len(X_test)} test")
    
    # Note: Data is already scaled, but we need the scaler for new predictions
    print("2️⃣ Создаём scaler (для новых данных)...")
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on original training data
    
    # Train model
    print("3️⃣ Обучаем модель...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("4️⃣ Оцениваем качество...")
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Val Accuracy: {val_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # Prepare metadata
    hyperparameters = {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": 42
    }
    
    performance_metrics = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "overfitting": train_acc - val_acc
    }
    
    training_info = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "prediction_horizon": 3,
        "data_period": "2025-03-01 to 2025-09-30",
        "symbol": "SOLUSDT",
        "timeframe": "5m"
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save model
    print("5️⃣ Сохраняем модель...")
    manager = ModelManager()
    
    files = manager.save_model(
        model=model,
        scaler=scaler,
        model_type="random_forest",
        version="v1",
        feature_columns=feature_cols,
        performance_metrics=performance_metrics,
        hyperparameters=hyperparameters,
        training_info=training_info,
        feature_importance=feature_importance
    )
    
    print(f"\n✅ Модель сохранена!")
    print(f"📁 Файлы:")
    for name, path in files.items():
        if path:
            print(f"   {name}: {path}")
    
    return manager


if __name__ == "__main__":
    # Run example
    manager = save_trained_model_example()
    
    print(f"\n📋 Список сохранённых моделей:")
    models_df = manager.list_models()
    if not models_df.empty:
        print(models_df.to_string(index=False))
    else:
        print("   Нет сохранённых моделей")