#!/usr/bin/env python3
"""
Тестовый скрипт для бэктестирования ансамбля на малых данных (5m за 1 день)
Включает оптимизированную версию с отложенной загрузкой моделей
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

class LazyEnsemblePredictor:
    """
    Оптимизированная версия EnsemblePredictor с отложенной загрузкой моделей
    """
    
    def __init__(self, models_dir: str = "models/ensemble_live"):
        self.models_dir = Path(models_dir)
        self.base_models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_names = []
        self.models_loaded = False
        self.logger = logging.getLogger("LazyEnsemblePredictor")
        
        # НЕ загружаем модели в конструкторе!
        self.logger.info("✅ Lazy predictor создан, модели будут загружены по требованию")
    
    def _find_models_directory(self):
        """Поиск директории с моделями."""
        if not self.models_dir.exists():
            self.logger.error(f"Models directory not found: {self.models_dir}")
            return None
        
        latest_link = self.models_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                target_dir = latest_link.resolve()
                if target_dir.exists():
                    return target_dir
            elif latest_link.is_dir():
                return latest_link
        
        # Если нет ссылки latest, ищем самую новую директорию
        subdirs = [d for d in self.models_dir.iterdir() if d.is_dir() and d.name != "latest"]
        if subdirs:
            subdirs.sort(key=lambda x: x.name, reverse=True)
            return subdirs[0]
        
        return None
    
    def load_models_on_demand(self):
        """Загрузка моделей по требованию с детальным логированием."""
        if self.models_loaded:
            return True
        
        self.logger.info("🔄 Начинаем загрузку моделей...")
        
        try:
            import joblib
            
            models_path = self._find_models_directory()
            if models_path is None:
                return False
            
            self.logger.info(f"📁 Загружаем из: {models_path}")
            
            # Загружаем базовые модели с таймингом
            model_files = {
                'random_forest': 'random_forest_intraday.joblib',
                'lightgbm': 'lightgbm_intraday.joblib',
                'xgboost': 'xgboost_intraday.joblib',
                'catboost': 'catboost_intraday.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    self.logger.info(f"🔄 Загружаем {model_name}...")
                    start_time = time.time()
                    
                    model = joblib.load(model_path)
                    self.base_models[model_name] = model
                    
                    load_time = time.time() - start_time
                    self.logger.info(f"✅ {model_name} загружена за {load_time:.2f}s")
            
            if not self.base_models:
                self.logger.error("❌ Не удалось загрузить базовые модели!")
                return False
            
            # Загружаем метамодель
            self.logger.info("🔄 Загружаем метамодель...")
            meta_path = models_path / "meta_intraday.joblib"
            if meta_path.exists():
                self.meta_model = joblib.load(meta_path)
                self.logger.info("✅ Метамодель загружена")
            else:
                self.logger.error("❌ Метамодель не найдена!")
                return False
            
            # Загружаем скейлер и признаки
            scaler_path = models_path / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info("✅ Скейлер загружен")
            
            features_path = models_path / "feature_names.joblib"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
                self.logger.info(f"✅ Загружены названия {len(self.feature_names)} признаков")
            
            self.models_loaded = True
            self.logger.info("🎉 Все модели загружены успешно!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки моделей: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_ensemble(self, features):
        """Получение предсказания ансамбля с автозагрузкой."""
        # Загружаем модели если не загружены
        if not self.models_loaded:
            if not self.load_models_on_demand():
                return None
        
        try:
            # Подготавливаем признаки
            if isinstance(features, pd.DataFrame):
                if self.feature_names:
                    available_features = [col for col in self.feature_names if col in features.columns]
                    features_array = features[available_features].values
                else:
                    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'confirm']
                    feature_cols = [col for col in features.columns if col not in exclude_cols]
                    features_array = features[feature_cols].values
            else:
                features_array = features
            
            if len(features_array.shape) == 1:
                features_array = features_array.reshape(1, -1)
            
            # Применяем скейлер
            if self.scaler is not None:
                features_array = self.scaler.transform(features_array)
            
            # Предсказания базовых моделей
            base_predictions = {}
            for model_name, model in self.base_models.items():
                pred_proba = model.predict_proba(features_array)
                if len(pred_proba.shape) == 2 and pred_proba.shape[1] >= 2:
                    probability = pred_proba[0, 1]
                else:
                    probability = pred_proba[0]
                base_predictions[model_name] = float(probability)
            
            # Метамодель
            expected_models = ['random_forest', 'lightgbm', 'xgboost', 'catboost']
            meta_features = []
            for model_name in expected_models:
                if model_name in base_predictions:
                    meta_features.append(base_predictions[model_name])
                else:
                    meta_features.append(0.5)
            
            meta_features_array = np.array([meta_features])
            final_probability = self.meta_model.predict_proba(meta_features_array)[0, 1]
            final_signal = self.meta_model.predict(meta_features_array)[0]
            
            return {
                'final_signal': int(final_signal),
                'final_probability': float(final_probability),
                'signal_strength': 'STRONG' if abs(final_probability - 0.5) > 0.3 else 'WEAK',
                'base_predictions': base_predictions,
                'meta_features': meta_features
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка предсказания: {e}")
            return None
    
    def is_ready(self):
        """Проверка готовности предсказателя."""
        return self.models_loaded and len(self.base_models) > 0 and self.meta_model is not None


def collect_small_data():
    """Сбор небольшого объема данных для тестирования (5m за 1 день)."""
    logger.info("📊 Собираем малые данные для тестирования...")
    
    try:
        from src.data_collector.bybit_client import BybitClient
        
        client = BybitClient()
        symbol = "SOLUSDT"
        
        # Собираем данные за последний день
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        logger.info(f"📅 Период: {start_time} - {end_time}")
        
        # Получаем 5-минутные данные
        data = client.get_klines(
            symbol=symbol,
            interval="5",
            start_time=start_time,
            end_time=end_time
        )
        
        if data is None or len(data) == 0:
            logger.error("❌ Не удалось получить данные")
            return None
        
        logger.info(f"✅ Получено {len(data)} свечей 5m")
        return data
        
    except Exception as e:
        logger.error(f"❌ Ошибка сбора данных: {e}")
        return None


def run_small_backtest():
    """Запуск бэктеста на малых данных."""
    logger.info("🚀 Запуск бэктеста на малых данных")
    
    # 1. Создаем ленивый предсказатель
    logger.info("1️⃣ Создаем ленивый предсказатель...")
    predictor = LazyEnsemblePredictor()
    
    # 2. Собираем данные
    logger.info("2️⃣ Собираем тестовые данные...")
    data = collect_small_data()
    if data is None:
        logger.error("❌ Не удалось получить данные для тестирования")
        return
    
    # 3. Генерируем признаки
    logger.info("3️⃣ Генерируем признаки...")
    try:
        from src.feature_engine.technical_indicators import TechnicalIndicatorCalculator
        
        calc = TechnicalIndicatorCalculator()
        
        # Добавляем базовые индикаторы
        data = calc.add_moving_averages(data, [5, 10, 20])
        data = calc.add_rsi(data)
        data = calc.add_macd(data)
        data = calc.add_bollinger_bands(data)
        data = calc.add_atr(data)
        data = calc.add_volume_indicators(data)
        
        # Очищаем NaN
        data = data.dropna()
        
        if len(data) == 0:
            logger.error("❌ Нет данных после добавления индикаторов")
            return
        
        logger.info(f"✅ Подготовлено {len(data)} строк с признаками")
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации признаков: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Тестируем предсказания
    logger.info("4️⃣ Тестируем предсказания...")
    try:
        # Берем последние 10 свечей для тестирования
        test_samples = data.tail(10)
        predictions = []
        
        for idx, row in test_samples.iterrows():
            # Формируем признаки для одной свечи
            features_df = pd.DataFrame([row])
            
            # Получаем предсказание
            prediction = predictor.predict_ensemble(features_df)
            
            if prediction:
                predictions.append({
                    'timestamp': idx,
                    'price': row['close'],
                    'signal': prediction['final_signal'],
                    'probability': prediction['final_probability'],
                    'strength': prediction['signal_strength']
                })
                
                logger.info(f"📈 {idx}: Price=${row['close']:.2f} | "
                          f"Signal={prediction['final_signal']} | "
                          f"Prob={prediction['final_probability']:.3f} | "
                          f"Strength={prediction['signal_strength']}")
            else:
                logger.warning(f"⚠️ Не удалось получить предсказание для {idx}")
        
        # 5. Результаты
        if predictions:
            buy_signals = sum(1 for p in predictions if p['signal'] == 1)
            strong_signals = sum(1 for p in predictions if p['strength'] == 'STRONG')
            avg_probability = np.mean([p['probability'] for p in predictions])
            
            logger.info("="*60)
            logger.info("📋 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
            logger.info(f"  Всего предсказаний: {len(predictions)}")
            logger.info(f"  BUY сигналов: {buy_signals}")
            logger.info(f"  STRONG сигналов: {strong_signals}")
            logger.info(f"  Средняя вероятность: {avg_probability:.3f}")
            logger.info("="*60)
            
            logger.info("🎉 Тестирование на малых данных УСПЕШНО завершено!")
            
        else:
            logger.error("❌ Не получено ни одного предсказания")
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования предсказаний: {e}")
        import traceback
        traceback.print_exc()


def test_initialization_speed():
    """Тест скорости инициализации."""
    logger.info("⏱️ Тестируем скорость инициализации...")
    
    # Тест обычной инициализации
    logger.info("1️⃣ Тестируем стандартный EnsemblePredictor...")
    try:
        from src.models.ensemble_predictor import EnsemblePredictor
        
        start_time = time.time()
        predictor_standard = EnsemblePredictor()
        standard_time = time.time() - start_time
        
        logger.info(f"⏱️ Стандартная инициализация: {standard_time:.2f}s")
        logger.info(f"   Готов: {predictor_standard.is_ready()}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка стандартной инициализации: {e}")
        standard_time = float('inf')
    
    # Тест ленивой инициализации
    logger.info("2️⃣ Тестируем ленивый LazyEnsemblePredictor...")
    try:
        start_time = time.time()
        predictor_lazy = LazyEnsemblePredictor()
        lazy_creation_time = time.time() - start_time
        
        logger.info(f"⏱️ Ленивое создание: {lazy_creation_time:.4f}s")
        logger.info(f"   Готов: {predictor_lazy.is_ready()}")
        
        # Тестируем первую загрузку
        start_time = time.time()
        predictor_lazy.load_models_on_demand()
        lazy_load_time = time.time() - start_time
        
        logger.info(f"⏱️ Первая загрузка: {lazy_load_time:.2f}s")
        logger.info(f"   Готов: {predictor_lazy.is_ready()}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка ленивой инициализации: {e}")
        lazy_creation_time = float('inf')
        lazy_load_time = float('inf')
    
    # Сравнение
    logger.info("="*60)
    logger.info("📊 СРАВНЕНИЕ ВРЕМЕНИ ИНИЦИАЛИЗАЦИИ:")
    logger.info(f"  Стандартная инициализация: {standard_time:.2f}s")
    logger.info(f"  Ленивое создание: {lazy_creation_time:.4f}s")
    logger.info(f"  Ленивая загрузка: {lazy_load_time:.2f}s")
    
    if lazy_creation_time < standard_time:
        speedup = standard_time / lazy_creation_time
        logger.info(f"🚀 Ускорение создания: {speedup:.0f}x")
    
    logger.info("="*60)


def main():
    """Главная функция."""
    logger.info("🎯 Тестирование ансамбля на малых данных")
    logger.info("="*60)
    
    try:
        # 1. Тест скорости инициализации
        test_initialization_speed()
        
        # 2. Запуск бэктеста
        run_small_backtest()
        
        logger.info("✅ Все тесты завершены успешно!")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Тест прерван пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()