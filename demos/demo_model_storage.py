#!/usr/bin/env python3
"""
Демонстрация сохранения и загрузки моделей машинного обучения

Этот скрипт показывает, где хранятся обученные модели и как с ними работать.
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

class SimpleModelManager:
    """Упрощённый менеджер моделей для демонстрации."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Создаём поддиректории
        (self.models_dir / "random_forest").mkdir(exist_ok=True)
        (self.models_dir / "xgboost").mkdir(exist_ok=True)
        (self.models_dir / "scalers").mkdir(exist_ok=True)
        (self.models_dir / "metadata").mkdir(exist_ok=True)
    
    def save_model_demo(self):
        """Демонстрация сохранения модели."""
        print("💾 ДЕМОНСТРАЦИЯ СОХРАНЕНИЯ МОДЕЛИ")
        print("=" * 50)
        
        # Создаём mock модель (имитация обученной модели)
        model_data = {
            'type': 'RandomForest',
            'parameters': {'n_estimators': 150, 'max_depth': 15},
            'trained': True,
            'accuracy': 0.765
        }
        
        # Создаём mock scaler
        scaler_data = {
            'type': 'StandardScaler',
            'mean': [0.1, -0.05, 0.03],
            'std': [1.02, 0.98, 1.05],
            'fitted': True
        }
        
        version = "v1"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Пути файлов
        model_file = self.models_dir / "random_forest" / f"random_forest_{version}.pkl"
        scaler_file = self.models_dir / "scalers" / f"scaler_{version}.pkl"
        metadata_file = self.models_dir / "metadata" / f"metadata_{version}.json"
        importance_file = self.models_dir / "metadata" / f"feature_importance_{version}.csv"
        
        # Сохраняем модель
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Модель сохранена: {model_file}")
        
        # Сохраняем scaler
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler_data, f)
        print(f"✅ Scaler сохранён: {scaler_file}")
        
        # Создаём важность признаков
        feature_importance = pd.DataFrame({
            'feature': ['sma_20', 'rsi_14', 'bb_width', 'volume_sma', 'ema_50'],
            'importance': [0.25, 0.22, 0.18, 0.15, 0.12]
        })
        feature_importance.to_csv(importance_file, index=False)
        print(f"✅ Важность признаков сохранена: {importance_file}")
        
        # Создаём метаданные
        metadata = {
            "model_info": {
                "model_type": "random_forest",
                "version": version,
                "timestamp": timestamp,
                "sklearn_version": "1.3.0",
                "python_version": "3.11.0"
            },
            "files": {
                "model": str(model_file),
                "scaler": str(scaler_file),
                "metadata": str(metadata_file),
                "feature_importance": str(importance_file)
            },
            "features": {
                "count": 5,
                "columns": ['sma_20', 'rsi_14', 'bb_width', 'volume_sma', 'ema_50']
            },
            "hyperparameters": {
                "n_estimators": 150,
                "max_depth": 15,
                "min_samples_split": 20,
                "random_state": 42
            },
            "performance": {
                "train_accuracy": 0.823,
                "val_accuracy": 0.778,
                "test_accuracy": 0.765,
                "overfitting": 0.045
            },
            "training": {
                "train_samples": 15000,
                "val_samples": 3000,
                "test_samples": 3000,
                "prediction_horizon": 3,
                "data_period": "2025-03-01 to 2025-09-30",
                "symbol": "SOLUSDT",
                "timeframe": "5m"
            }
        }
        
        # Сохраняем метаданные
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Метаданные сохранены: {metadata_file}")
        
        return {
            "model": str(model_file),
            "scaler": str(scaler_file),
            "metadata": str(metadata_file),
            "feature_importance": str(importance_file)
        }
    
    def load_model_demo(self, version: str):
        """Демонстрация загрузки модели."""
        print(f"\n📁 ДЕМОНСТРАЦИЯ ЗАГРУЗКИ МОДЕЛИ {version}")
        print("=" * 50)
        
        # Загружаем метаданные
        metadata_file = self.models_dir / "metadata" / f"metadata_{version}.json"
        
        if not metadata_file.exists():
            print(f"❌ Файл метаданных не найден: {metadata_file}")
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"📋 Информация о модели:")
        print(f"   Тип: {metadata['model_info']['model_type']}")
        print(f"   Версия: {metadata['model_info']['version']}")
        print(f"   Дата создания: {metadata['model_info']['timestamp']}")
        print(f"   Признаков: {metadata['features']['count']}")
        print(f"   Test Accuracy: {metadata['performance']['test_accuracy']:.3f}")
        
        # Загружаем модель
        model_file = Path(metadata["files"]["model"])
        if model_file.exists():
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Модель загружена: {model_file}")
        else:
            print(f"❌ Файл модели не найден: {model_file}")
        
        # Загружаем scaler
        scaler_file = Path(metadata["files"]["scaler"])
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Scaler загружен: {scaler_file}")
        else:
            print(f"❌ Файл scaler не найден: {scaler_file}")
        
        # Показываем важность признаков
        importance_file = Path(metadata["files"]["feature_importance"])
        if importance_file.exists():
            importance_df = pd.read_csv(importance_file)
            print(f"\n🔍 Важность признаков:")
            for _, row in importance_df.iterrows():
                print(f"   {row['feature']:15s}: {row['importance']:.3f}")
        
        return model, scaler, metadata
    
    def show_directory_structure(self):
        """Показывает структуру директории с моделями."""
        print(f"\n📂 СТРУКТУРА ДИРЕКТОРИИ МОДЕЛЕЙ")
        print("=" * 50)
        
        if not self.models_dir.exists():
            print(f"❌ Директория {self.models_dir} не существует")
            return
        
        def show_tree(path, prefix=""):
            """Рекурсивно показывает дерево файлов."""
            items = sorted(path.iterdir())
            
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                
                if item.is_file():
                    size_kb = item.stat().st_size / 1024
                    print(f"{prefix}{current_prefix}{item.name} ({size_kb:.1f} KB)")
                else:
                    print(f"{prefix}{current_prefix}{item.name}/")
                    if item.is_dir():
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        show_tree(item, next_prefix)
        
        print(f"models/")
        show_tree(self.models_dir, "")


def main():
    """Главная функция демонстрации."""
    print("🗃️ ДЕМОНСТРАЦИЯ ХРАНЕНИЯ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
    print("=" * 70)
    
    # Создаём менеджер
    manager = SimpleModelManager()
    
    # Демонстрируем сохранение
    saved_files = manager.save_model_demo()
    
    # Показываем структуру директории
    manager.show_directory_structure()
    
    # Демонстрируем загрузку
    manager.load_model_demo("v1")
    
    print(f"\n🎯 ОТВЕТ НА ВОПРОС: 'Где хранятся данные по обученной модели?'")
    print("=" * 70)
    print(f"📂 Все файлы моделей хранятся в директории: models/")
    print(f"")
    print(f"📁 Структура:")
    print(f"   models/")
    print(f"   ├── random_forest/        # Файлы моделей Random Forest")
    print(f"   │   └── random_forest_v1.pkl")
    print(f"   ├── xgboost/             # Файлы моделей XGBoost")
    print(f"   ├── scalers/             # Файлы нормализаторов")
    print(f"   │   └── scaler_v1.pkl")
    print(f"   └── metadata/            # Метаданные и важность признаков")
    print(f"       ├── metadata_v1.json")
    print(f"       └── feature_importance_v1.csv")
    print(f"")
    print(f"🔍 Файлы содержат:")
    print(f"   📦 Модель (*.pkl)          - Обученная модель со всеми параметрами")
    print(f"   ⚖️ Scaler (*.pkl)          - Нормализатор для новых данных")
    print(f"   📋 Metadata (*.json)       - Информация о модели, качестве, параметрах")
    print(f"   📊 Feature importance (*.csv) - Важность каждого признака")
    
    print(f"\n💡 КАК ИСПОЛЬЗОВАТЬ:")
    print(f"   1. Обучи модель → автоматически сохраняется в models/")
    print(f"   2. Загрузи модель по версии → делай предсказания")
    print(f"   3. Сравнивай разные версии → выбирай лучшую")
    print(f"   4. Используй в торговом боте → загружай и предсказывай")


if __name__ == "__main__":
    main()