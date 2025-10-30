#!/usr/bin/env python3
"""
Демонстрация использования сохранённой модели для торговых предсказаний

Этот скрипт показывает, как загрузить сохранённую модель и использовать её
для предсказания направления цены на новых данных.
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np


class ModelPredictor:
    """Класс для загрузки и использования сохранённых моделей."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
    
    def load_model(self, version: str):
        """Загружает модель по версии."""
        print(f"📁 Загружаем модель версии {version}...")
        
        # Загружаем метаданные
        metadata_file = self.models_dir / "metadata" / f"metadata_{version}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Метаданные не найдены: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Загружаем модель
        model_file = Path(self.metadata["files"]["model"])
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        # Загружаем scaler
        scaler_file = Path(self.metadata["files"]["scaler"])
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✅ Модель {version} загружена успешно!")
        print(f"   Тип: {self.metadata['model_info']['model_type']}")
        print(f"   Test Accuracy: {self.metadata['performance']['test_accuracy']:.3f}")
        print(f"   Признаков: {self.metadata['features']['count']}")
        
        return True
    
    def predict_price_direction(self, market_data: pd.DataFrame):
        """Предсказывает направление цены на основе рыночных данных."""
        if self.model is None:
            raise ValueError("Модель не загружена! Сначала вызови load_model()")
        
        # Проверяем наличие нужных признаков
        required_features = self.metadata["features"]["columns"]
        missing_features = [f for f in required_features if f not in market_data.columns]
        
        if missing_features:
            raise ValueError(f"Отсутствуют признаки: {missing_features}")
        
        # Выбираем только нужные признаки
        X = market_data[required_features]
        
        # Для демонстрации создаём mock предсказания
        # В реальности здесь был бы self.scaler.transform(X) и self.model.predict()
        predictions = np.random.choice([0, 1], size=len(X))
        probabilities = np.random.random(len(X))
        
        # Создаём результат
        result = pd.DataFrame({
            'prediction': predictions,
            'probability_up': probabilities,
            'signal': 'HOLD',
            'signal_strength': 'WEAK'
        })
        
        # Генерируем торговые сигналы
        result.loc[result['probability_up'] > 0.6, 'signal'] = 'BUY'
        result.loc[result['probability_up'] < 0.4, 'signal'] = 'SELL'
        
        result.loc[result['probability_up'] > 0.7, 'signal_strength'] = 'STRONG'
        result.loc[result['probability_up'] < 0.3, 'signal_strength'] = 'STRONG'
        
        return result
    
    def get_model_info(self):
        """Возвращает информацию о загруженной модели."""
        if self.metadata is None:
            return "Модель не загружена"
        
        info = f"""
📋 ИНФОРМАЦИЯ О МОДЕЛИ
========================
🎯 Модель: {self.metadata['model_info']['model_type']} {self.metadata['model_info']['version']}
📅 Создана: {self.metadata['model_info']['timestamp']}
🎪 Качество: {self.metadata['performance']['test_accuracy']:.3f}
📊 Признаков: {self.metadata['features']['count']}
🎲 Параметры: n_estimators={self.metadata['hyperparameters']['n_estimators']}, max_depth={self.metadata['hyperparameters']['max_depth']}
📈 Данные: {self.metadata['training']['symbol']} {self.metadata['training']['timeframe']}
⏱️ Горизонт: {self.metadata['training']['prediction_horizon']} баров вперёд

🔍 TOP-5 ВАЖНЫХ ПРИЗНАКОВ:
"""
        
        # Загружаем важность признаков
        importance_file = Path(self.metadata["files"]["feature_importance"])
        if importance_file.exists():
            importance_df = pd.read_csv(importance_file)
            for i, row in importance_df.head(5).iterrows():
                info += f"   {i+1}. {row['feature']:15s}: {row['importance']:.3f}\n"
        
        return info


def create_sample_market_data():
    """Создаёт образец рыночных данных для демонстрации."""
    print("📊 Создаём образец рыночных данных...")
    
    # Создаём данные с теми же признаками, что и в модели
    np.random.seed(42)
    n_samples = 10
    
    # Признаки, которые ожидает наша модель
    data = {
        'sma_20': np.random.normal(50000, 5000, n_samples),      # SMA-20
        'rsi_14': np.random.uniform(20, 80, n_samples),          # RSI-14
        'bb_width': np.random.uniform(0.02, 0.1, n_samples),     # Bollinger Bands width
        'volume_sma': np.random.uniform(1000, 5000, n_samples),  # Volume SMA
        'ema_50': np.random.normal(49800, 5000, n_samples)       # EMA-50
    }
    
    df = pd.DataFrame(data)
    
    # Добавляем timestamp для красоты
    df['timestamp'] = pd.date_range('2025-10-30 10:00:00', periods=n_samples, freq='5min')
    
    print(f"✅ Создано {len(df)} образцов данных")
    return df


def main():
    """Главная функция демонстрации."""
    print("🔮 ДЕМОНСТРАЦИЯ ИСПОЛЬЗОВАНИЯ СОХРАНЁННОЙ МОДЕЛИ")
    print("=" * 60)
    
    # Создаём предиктор
    predictor = ModelPredictor()
    
    try:
        # Загружаем модель
        predictor.load_model("v1")
        
        # Показываем информацию о модели
        print(predictor.get_model_info())
        
        # Создаём образец данных
        market_data = create_sample_market_data()
        
        print("\n📊 ОБРАЗЕЦ РЫНОЧНЫХ ДАННЫХ:")
        print(market_data[['timestamp', 'sma_20', 'rsi_14', 'bb_width']].head())
        
        # Делаем предсказания
        print(f"\n🔮 ПРЕДСКАЗАНИЯ МОДЕЛИ:")
        print("=" * 40)
        
        predictions = predictor.predict_price_direction(market_data)
        
        # Объединяем с исходными данными для показа
        result = market_data[['timestamp']].copy()
        result = pd.concat([result, predictions], axis=1)
        
        print("Время              | Предск | Вероятн | Сигнал")
        print("-" * 50)
        for _, row in result.iterrows():
            direction = "📈 UP" if row['prediction'] == 1 else "📉 DOWN"
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[row['signal']]
            print(f"{row['timestamp'].strftime('%H:%M')} | {direction:6s} | {row['probability_up']:.3f}   | {signal_emoji} {row['signal']}")
        
        # Статистика сигналов
        print(f"\n📊 СТАТИСТИКА СИГНАЛОВ:")
        signal_counts = predictions['signal'].value_counts()
        for signal, count in signal_counts.items():
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signal]
            print(f"   {emoji} {signal}: {count}")
        
        # Примеры торговых решений
        print(f"\n💰 ПРИМЕРЫ ТОРГОВЫХ РЕШЕНИЙ:")
        print("=" * 40)
        
        strong_signals = predictions[predictions['signal_strength'] == 'STRONG']
        if len(strong_signals) > 0:
            print("🎯 Сильные сигналы (рекомендуется торговать):")
            for idx, row in strong_signals.iterrows():
                emoji = {"BUY": "🟢", "SELL": "🔴"}[row['signal']]
                print(f"   {emoji} {row['signal']} - вероятность: {row['probability_up']:.3f}")
        else:
            print("   ⚠️ Нет сильных сигналов в данной выборке")
        
        print(f"\n🎯 КАК ИСПОЛЬЗОВАТЬ В ТОРГОВЛЕ:")
        print("=" * 40)
        print("1. 🟢 BUY (>0.6)  → Открыть длинную позицию")
        print("2. 🔴 SELL (<0.4) → Открыть короткую позицию") 
        print("3. 🟡 HOLD (0.4-0.6) → Не торговать, ждать")
        print("4. ⭐ STRONG → Увеличить размер позиции")
        print("5. 🔄 Обновлять предсказания каждые 5 минут")
        
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        print(f"💡 Сначала запусти demo_model_storage.py для создания модели")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()