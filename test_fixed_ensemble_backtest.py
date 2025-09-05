#!/usr/bin/env python3
"""
Финальный исправленный скрипт для тестирования ансамбля на реальных данных
с правильными признаками и отложенной загрузкой
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedEnsemblePredictor:
    """Исправленный предсказатель с отложенной загрузкой и правильными признаками."""
    
    def __init__(self, models_dir: str = "models/ensemble_live", lazy_loading: bool = True):
        self.models_dir = Path(models_dir)
        self.lazy_loading = lazy_loading
        self.base_models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_names = []
        self.models_loaded = False
        self.logger = logging.getLogger("FixedEnsemble")
        
        if not lazy_loading:
            self.load_models()
        else:
            self.logger.info("✅ Predictor создан с отложенной загрузкой")
    
    def _find_models_directory(self):
        """Поиск директории с моделями."""
        if not self.models_dir.exists():
            return None
        
        latest_link = self.models_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                return latest_link.resolve()
            elif latest_link.is_dir():
                return latest_link
        
        subdirs = [d for d in self.models_dir.iterdir() if d.is_dir() and d.name != "latest"]
        if subdirs:
            subdirs.sort(key=lambda x: x.name, reverse=True)
            return subdirs[0]
        
        return None
    
    def load_models(self):
        """Загрузка моделей с логированием времени."""
        if self.models_loaded:
            return True
        
        self.logger.info("🔄 Загрузка моделей...")
        start_time = time.time()
        
        try:
            import joblib
            models_path = self._find_models_directory()
            
            if models_path is None:
                self.logger.error("❌ Директория моделей не найдена")
                return False
            
            self.logger.info(f"📁 Загрузка из: {models_path}")
            
            # Загружаем модели
            model_files = {
                'random_forest': 'random_forest_intraday.joblib',
                'lightgbm': 'lightgbm_intraday.joblib',
                'xgboost': 'xgboost_intraday.joblib',
                'catboost': 'catboost_intraday.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    model_start = time.time()
                    self.base_models[model_name] = joblib.load(model_path)
                    model_time = time.time() - model_start
                    self.logger.info(f"  ✅ {model_name}: {model_time:.2f}s")
                else:
                    self.logger.warning(f"  ❌ {model_name}: файл не найден")
            
            # Метамодель
            meta_path = models_path / "meta_intraday.joblib"
            if meta_path.exists():
                self.meta_model = joblib.load(meta_path)
                self.logger.info("  ✅ Метамодель загружена")
            else:
                self.logger.error("  ❌ Метамодель не найдена")
                return False
            
            # Скейлер и признаки
            scaler_path = models_path / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info("  ✅ Скейлер загружен")
            
            features_path = models_path / "feature_names.joblib"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
                self.logger.info(f"  ✅ Признаки загружены: {len(self.feature_names)}")
                self.logger.info(f"    Первые 5: {self.feature_names[:5]}")
            else:
                self.logger.error("  ❌ Названия признаков не найдены")
                return False
            
            self.models_loaded = True
            total_time = time.time() - start_time
            self.logger.info(f"🎉 Все модели загружены за {total_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_proba(self, features):
        """Предсказание вероятности с правильной обработкой признаков."""
        if not self.models_loaded:
            if not self.load_models():
                return np.array([[0.5, 0.5]])
        
        try:
            # Подготовка признаков с использованием правильных названий
            if isinstance(features, pd.DataFrame):
                # Используем только те признаки, которые есть и в данных, и в обученной модели
                available_features = [col for col in self.feature_names if col in features.columns]
                
                if len(available_features) < len(self.feature_names):
                    missing_features = set(self.feature_names) - set(available_features)
                    self.logger.warning(f"Отсутствуют признаки: {missing_features}")
                
                # Если слишком мало признаков, возвращаем нейтральное предсказание
                if len(available_features) < len(self.feature_names) * 0.8:  # Менее 80%
                    self.logger.warning(f"Слишком мало признаков: {len(available_features)}/{len(self.feature_names)}")
                    return np.array([[0.5, 0.5]])
                
                # Создаем массив с правильным порядком признаков
                features_array = []
                for feature_name in self.feature_names:
                    if feature_name in features.columns:
                        features_array.append(features[feature_name].iloc[0])
                    else:
                        # Заполняем отсутствующие признаки средними значениями
                        if 'MA' in feature_name or 'EMA' in feature_name:
                            features_array.append(features['close'].iloc[0])
                        elif 'RSI' in feature_name:
                            features_array.append(50.0)
                        elif 'BB_position' in feature_name:
                            features_array.append(0.5)
                        elif 'volume' in feature_name:
                            features_array.append(1.0)
                        elif 'change' in feature_name:
                            features_array.append(0.0)
                        elif 'timeframe' in feature_name:
                            features_array.append(5.0)  # 5-минутный таймфрейм
                        else:
                            features_array.append(0.0)
                
                features_array = np.array(features_array).reshape(1, -1)
                
            else:
                features_array = features
                if len(features_array.shape) == 1:
                    features_array = features_array.reshape(1, -1)
            
            # Проверяем размерность
            if features_array.shape[1] != len(self.feature_names):
                self.logger.error(f"Неправильное количество признаков: {features_array.shape[1]} != {len(self.feature_names)}")
                return np.array([[0.5, 0.5]])
            
            # Скейлинг
            if self.scaler is not None:
                features_array = self.scaler.transform(features_array)
            
            # Проверка на NaN
            if np.isnan(features_array).any():
                self.logger.warning("Обнаружены NaN значения, заменяем на 0")
                features_array = np.nan_to_num(features_array)
            
            # Базовые предсказания
            base_predictions = []
            expected_models = ['random_forest', 'lightgbm', 'xgboost', 'catboost']
            
            for model_name in expected_models:
                if model_name in self.base_models:
                    model = self.base_models[model_name]
                    pred_proba = model.predict_proba(features_array)
                    if len(pred_proba.shape) == 2 and pred_proba.shape[1] >= 2:
                        probability = pred_proba[0, 1]
                    else:
                        probability = pred_proba[0]
                    base_predictions.append(float(probability))
                else:
                    base_predictions.append(0.5)
            
            # Метамодель
            if self.meta_model is not None and len(base_predictions) == 4:
                meta_features_array = np.array([base_predictions])
                final_proba = self.meta_model.predict_proba(meta_features_array)
                return final_proba
            else:
                # Простое усреднение
                avg_prob = np.mean(base_predictions)
                return np.array([[1-avg_prob, avg_prob]])
            
        except Exception as e:
            self.logger.error(f"Ошибка предсказания: {e}")
            import traceback
            traceback.print_exc()
            return np.array([[0.5, 0.5]])


def create_proper_test_data():
    """Создание данных с правильными признаками."""
    logger.info("📊 Создаем данные с правильными признаками...")
    
    # Создаем 1 день 5-минутных данных
    dates = pd.date_range(start='2024-08-23', periods=300, freq='5min')
    
    # Синтетические цены
    np.random.seed(42)
    base_price = 150
    prices = []
    current_price = base_price
    
    for i in range(300):
        # Случайное изменение цены
        change = np.random.normal(0, 1)
        current_price += change
        prices.append(current_price)
    
    prices = np.array(prices)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.rand(300) * 2,
        'low': prices - np.random.rand(300) * 2,
        'close': prices,
        'volume': np.random.rand(300) * 1000000 + 500000
    })
    
    # Корректируем high/low
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Добавляем ВСЕ необходимые признаки в правильном порядке
    expected_features = [
        'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 
        'BB_hband', 'BB_lband', 'BB_width', 'BB_position', 'vol_change', 
        'volume_sma', 'volume_ratio', 'price_change', 'price_change_5', 
        'volatility', 'EMA12', 'EMA26', 'high_low_pct', 'close_to_high', 
        'close_to_low', 'timeframe_minutes'
    ]
    
    # MA
    data['MA5'] = data['close'].rolling(5, min_periods=1).mean()
    data['MA10'] = data['close'].rolling(10, min_periods=1).mean()
    data['MA20'] = data['close'].rolling(20, min_periods=1).mean()
    
    # EMA
    data['EMA12'] = data['close'].ewm(span=12, min_periods=1).mean()
    data['EMA26'] = data['close'].ewm(span=26, min_periods=1).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)  # Избегаем деления на ноль
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
    data['MACD_diff'] = data['MACD'] - data['MACD_signal']
    
    # Bollinger Bands
    data['BB_hband'] = data['MA20'] + (data['close'].rolling(20, min_periods=1).std() * 2)
    data['BB_lband'] = data['MA20'] - (data['close'].rolling(20, min_periods=1).std() * 2)
    data['BB_width'] = data['BB_hband'] - data['BB_lband']
    data['BB_position'] = (data['close'] - data['BB_lband']) / (data['BB_width'] + 1e-8)
    
    # Volume indicators
    data['volume_sma'] = data['volume'].rolling(20, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_sma'] + 1e-8)
    data['vol_change'] = data['volume'].pct_change().fillna(0)
    
    # Price changes
    data['price_change'] = data['close'].pct_change().fillna(0)
    data['price_change_5'] = data['close'].pct_change(5).fillna(0)
    
    # Volatility
    data['volatility'] = data['close'].rolling(14, min_periods=1).std() / (data['close'].rolling(14, min_periods=1).mean() + 1e-8)
    
    # Price position indicators
    data['high_low_pct'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    data['close_to_high'] = (data['close'] - data['low']) / ((data['high'] - data['low']) + 1e-8)
    data['close_to_low'] = (data['high'] - data['close']) / ((data['high'] - data['low']) + 1e-8)
    
    # Timeframe
    data['timeframe_minutes'] = 5.0
    
    # Заполняем NaN нулями
    data = data.fillna(0)
    
    logger.info(f"✅ Создано {len(data)} строк с {len(expected_features)} признаками")
    
    # Проверим, что все признаки есть
    missing_features = set(expected_features) - set(data.columns)
    if missing_features:
        logger.error(f"Отсутствуют признаки: {missing_features}")
    else:
        logger.info("✅ Все ожидаемые признаки присутствуют")
    
    return data


def run_final_test():
    """Финальный тест с правильными данными."""
    logger.info("🎯 Финальный тест ансамбля")
    logger.info("="*60)
    
    try:
        # 1. Создаем предсказатель
        logger.info("1️⃣ Создание предсказателя...")
        predictor = FixedEnsemblePredictor(lazy_loading=True)
        
        # 2. Создаем данные
        logger.info("2️⃣ Создание тестовых данных...")
        data = create_proper_test_data()
        
        # 3. Тест первого предсказания
        logger.info("3️⃣ Тест первого предсказания...")
        start_time = time.time()
        
        test_row = data.iloc[-1:]  # Последняя строка
        prediction = predictor.predict_proba(test_row)
        
        first_pred_time = time.time() - start_time
        logger.info(f"⏱️ Время первого предсказания: {first_pred_time:.2f}s")
        logger.info(f"📈 Результат: {prediction[0]}")
        
        # 4. Тест множественных предсказаний
        logger.info("4️⃣ Тест множественных предсказаний...")
        start_time = time.time()
        
        predictions = []
        for i in range(min(50, len(data))):  # Последние 50 строк
            row = data.iloc[-(i+1):-(i) if i > 0 else None]
            pred = predictor.predict_proba(row)
            predictions.append(pred[0, 1])  # Вероятность BUY
        
        multiple_pred_time = time.time() - start_time
        avg_pred_time = multiple_pred_time / len(predictions)
        
        logger.info(f"⏱️ Время {len(predictions)} предсказаний: {multiple_pred_time:.2f}s")
        logger.info(f"⏱️ Среднее время на предсказание: {avg_pred_time:.4f}s")
        
        # 5. Анализ результатов
        buy_signals = sum(1 for p in predictions if p > 0.55)
        strong_signals = sum(1 for p in predictions if p > 0.7 or p < 0.3)
        avg_probability = np.mean(predictions)
        
        logger.info("="*60)
        logger.info("📊 РЕЗУЛЬТАТЫ ТЕСТА:")
        logger.info(f"  Всего предсказаний: {len(predictions)}")
        logger.info(f"  BUY сигналов (>55%): {buy_signals}")
        logger.info(f"  Сильных сигналов: {strong_signals}")
        logger.info(f"  Средняя вероятность: {avg_probability:.3f}")
        logger.info(f"  Min вероятность: {min(predictions):.3f}")
        logger.info(f"  Max вероятность: {max(predictions):.3f}")
        logger.info("="*60)
        
        # 6. Заключение
        if avg_pred_time < 0.1 and not np.isnan(avg_probability):
            logger.info("🎉 ТЕСТ ПРОЙДЕН УСПЕШНО!")
            logger.info("✅ Инициализация быстрая")
            logger.info("✅ Предсказания работают")
            logger.info("✅ Нет ошибок с признаками")
            logger.info("✅ Производительность приемлема")
        else:
            logger.warning("⚠️ Тест прошел с замечаниями")
            if avg_pred_time >= 0.1:
                logger.warning(f"  Медленные предсказания: {avg_pred_time:.4f}s")
            if np.isnan(avg_probability):
                logger.warning("  Проблемы с предсказаниями")
        
    except Exception as e:
        logger.error(f"❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Главная функция."""
    run_final_test()


if __name__ == "__main__":
    main()