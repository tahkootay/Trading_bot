#!/usr/bin/env python3
"""
Test Model Manager - Simple Example

This script demonstrates the model saving/loading functionality
without requiring all dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock the missing modules to test the core functionality
class MockScaler:
    def __init__(self):
        self.mean_ = [0.0, 0.0]
        self.scale_ = [1.0, 1.0]
    
    def fit(self, X):
        return self
        
    def transform(self, X):
        return X

class MockModel:
    def __init__(self):
        self.feature_importances_ = [0.5, 0.3, 0.2]
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        import numpy as np
        return np.random.choice([0, 1], size=len(X))
        
    def predict_proba(self, X):
        import numpy as np
        proba = np.random.random((len(X), 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

def test_model_manager():
    """Test basic model manager functionality."""
    print("🧪 ТЕСТ MODEL MANAGER")
    print("=" * 50)
    
    # Import after setting up mocks
    try:
        from modules.data_collector.model_manager import ModelManager
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return
    
    # Create mock data
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)
    
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    print(f"📊 Создали тестовые данные: {n_samples} образцов, {n_features} признаков")
    
    # Create mock model and scaler
    model = MockModel()
    scaler = MockScaler()
    
    # Mock training
    model.fit(X, y)
    scaler.fit(X)
    
    print("✅ Обучили mock модель")
    
    # Create feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.random.random(n_features)
    }).sort_values('importance', ascending=False)
    
    # Performance metrics
    performance_metrics = {
        "train_accuracy": 0.85,
        "val_accuracy": 0.78,
        "test_accuracy": 0.76,
        "overfitting": 0.07
    }
    
    # Hyperparameters
    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 20,
        "random_state": 42
    }
    
    # Training info
    training_info = {
        "train_samples": 800,
        "val_samples": 100,
        "test_samples": 100,
        "prediction_horizon": 3,
        "data_period": "test_period",
        "symbol": "TESTUSDT",
        "timeframe": "5m"
    }
    
    # Initialize ModelManager
    manager = ModelManager()
    print("✅ Создали ModelManager")
    
    # Save model
    print("\n💾 Сохраняем модель...")
    try:
        files = manager.save_model(
            model=model,
            scaler=scaler,
            model_type="random_forest",
            version="test_v1",
            feature_columns=feature_cols,
            performance_metrics=performance_metrics,
            hyperparameters=hyperparameters,
            training_info=training_info,
            feature_importance=feature_importance
        )
        
        print(f"\n✅ Модель сохранена!")
        print(f"📁 Файлы:")
        for name, path in files.items():
            if path and os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                print(f"   {name}: {path} ({size_kb:.1f} KB)")
            else:
                print(f"   {name}: {path} (не найден)")
                
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        return
    
    # Test loading
    print(f"\n📁 Загружаем модель...")
    try:
        loaded_model, loaded_scaler, metadata = manager.load_model("test_v1")
        print(f"✅ Модель загружена!")
        print(f"   Тип модели: {metadata['model_info']['model_type']}")
        print(f"   Версия: {metadata['model_info']['version']}")
        print(f"   Признаков: {metadata['features']['count']}")
        print(f"   Test accuracy: {metadata['performance']['test_accuracy']}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # Test listing models
    print(f"\n📋 Список моделей:")
    try:
        models_df = manager.list_models()
        if not models_df.empty:
            print(models_df.to_string(index=False))
        else:
            print("   Нет сохранённых моделей")
    except Exception as e:
        print(f"❌ Ошибка получения списка: {e}")
    
    print(f"\n🎉 Тест завершён успешно!")
    print(f"📂 Проверь директорию 'models/' для сохранённых файлов")

if __name__ == "__main__":
    test_model_manager()