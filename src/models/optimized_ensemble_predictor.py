#!/usr/bin/env python3
"""
Оптимизированный ансамбль ML-моделей с прогресс-баром и отложенной загрузкой
Решение проблемы медленной инициализации RandomForest (5+ секунд)
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging
import warnings
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Подавление предупреждений
warnings.filterwarnings('ignore')


class ProgressBar:
    """Простой прогресс-бар для терминала."""
    
    def __init__(self, total: int, description: str = "", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def update(self, delta: int = 1):
        """Обновление прогресса."""
        with self._lock:
            self.current += delta
            self._print_progress()
    
    def _print_progress(self):
        """Печать прогресс-бара."""
        if self.total == 0:
            return
            
        percent = (self.current / self.total) * 100
        filled_width = int(self.width * self.current // self.total)
        bar = '█' * filled_width + '░' * (self.width - filled_width)
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        print(f"\r{self.description} |{bar}| {percent:.1f}% ({self.current}/{self.total}) "
              f"[{elapsed:.1f}s<{eta:.1f}s, {rate:.1f}it/s]", end='', flush=True)
        
        if self.current >= self.total:
            print()  # Новая строка после завершения


class OptimizedEnsemblePredictor:
    """
    Оптимизированный предсказатель ансамбля с отложенной загрузкой,
    прогресс-баром и параллельной загрузкой моделей.
    """
    
    def __init__(self, models_dir: str = "models/ensemble_live", 
                 lazy_loading: bool = True, show_progress: bool = True):
        """
        Инициализация предсказателя.
        
        Args:
            models_dir: Директория с моделями
            lazy_loading: Отложенная загрузка (загружать только при первом предсказании)
            show_progress: Показывать прогресс-бар при загрузке
        """
        self.models_dir = Path(models_dir)
        self.lazy_loading = lazy_loading
        self.show_progress = show_progress
        
        # Модели
        self.base_models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_names = []
        
        # Статус загрузки
        self.models_loaded = False
        self.loading_in_progress = False
        self._loading_lock = threading.Lock()
        
        # Информация о моделях
        self.models_info = {}
        self.load_times = {}
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        # Автозагрузка только если не ленивая
        if not lazy_loading:
            self.load_models()
        else:
            self.logger.info("✅ Predictor создан с отложенной загрузкой")
    
    def _find_models_directory(self) -> Optional[Path]:
        """Поиск директории с моделями."""
        if not self.models_dir.exists():
            self.logger.error(f"Models directory not found: {self.models_dir}")
            return None
        
        # Проверяем ссылку latest
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
    
    def _load_single_model(self, model_name: str, file_path: Path) -> Tuple[str, Any, float]:
        """Загрузка одной модели с измерением времени."""
        start_time = time.time()
        try:
            model = joblib.load(file_path)
            load_time = time.time() - start_time
            return model_name, model, load_time
        except Exception as e:
            load_time = time.time() - start_time
            self.logger.error(f"Failed to load {model_name}: {e}")
            return model_name, None, load_time
    
    def load_models(self) -> bool:
        """Параллельная загрузка всех моделей ансамбля с прогресс-баром."""
        # Проверка на повторную загрузку
        if self.models_loaded:
            return True
        
        # Блокировка для предотвращения параллельной загрузки
        with self._loading_lock:
            if self.models_loaded:  # Двойная проверка
                return True
            
            if self.loading_in_progress:
                # Ждем завершения загрузки в другом потоке
                while self.loading_in_progress and not self.models_loaded:
                    time.sleep(0.1)
                return self.models_loaded
            
            self.loading_in_progress = True
        
        try:
            total_start_time = time.time()
            models_path = self._find_models_directory()
            
            if models_path is None:
                self.logger.error("❌ Директория моделей не найдена")
                return False
            
            self.logger.info(f"📁 Загрузка моделей из: {models_path}")
            
            # Определяем файлы для загрузки
            model_files = {
                'random_forest': 'random_forest_intraday.joblib',
                'lightgbm': 'lightgbm_intraday.joblib', 
                'xgboost': 'xgboost_intraday.joblib',
                'catboost': 'catboost_intraday.joblib'
            }
            
            auxiliary_files = {
                'meta_model': 'meta_intraday.joblib',
                'scaler': 'scaler.joblib',
                'feature_names': 'feature_names.joblib'
            }
            
            total_files = len(model_files) + len(auxiliary_files)
            
            # Прогресс-бар
            progress = None
            if self.show_progress:
                progress = ProgressBar(total_files, "🤖 Загрузка моделей")
            
            # 1. Параллельная загрузка базовых моделей
            base_models_loaded = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Запускаем загрузку всех базовых моделей параллельно
                future_to_model = {}
                for model_name, filename in model_files.items():
                    model_path = models_path / filename
                    if model_path.exists():
                        future = executor.submit(self._load_single_model, model_name, model_path)
                        future_to_model[future] = model_name
                    else:
                        self.logger.warning(f"⚠️ Файл не найден: {filename}")
                        if progress:
                            progress.update(1)
                
                # Собираем результаты по мере готовности
                for future in as_completed(future_to_model):
                    model_name, model, load_time = future.result()
                    
                    if model is not None:
                        base_models_loaded[model_name] = model
                        self.load_times[model_name] = load_time
                        self.logger.info(f"  ✅ {model_name}: {load_time:.2f}s")
                    else:
                        self.logger.error(f"  ❌ {model_name}: загрузка провалена")
                    
                    if progress:
                        progress.update(1)
            
            # Сохраняем загруженные модели
            self.base_models = base_models_loaded
            
            if not self.base_models:
                self.logger.error("❌ Ни одна базовая модель не загружена!")
                return False
            
            # 2. Загрузка вспомогательных файлов (последовательно, они быстрые)
            
            # Метамодель
            meta_path = models_path / auxiliary_files['meta_model']
            if meta_path.exists():
                try:
                    start_time = time.time()
                    self.meta_model = joblib.load(meta_path)
                    self.load_times['meta_model'] = time.time() - start_time
                    self.logger.info(f"  ✅ meta_model: {self.load_times['meta_model']:.3f}s")
                except Exception as e:
                    self.logger.error(f"❌ Ошибка загрузки метамодели: {e}")
                    return False
            else:
                self.logger.error("❌ Метамодель не найдена!")
                return False
            
            if progress:
                progress.update(1)
            
            # Скейлер
            scaler_path = models_path / auxiliary_files['scaler']
            if scaler_path.exists():
                try:
                    start_time = time.time()
                    self.scaler = joblib.load(scaler_path)
                    self.load_times['scaler'] = time.time() - start_time
                    self.logger.info(f"  ✅ scaler: {self.load_times['scaler']:.3f}s")
                except Exception as e:
                    self.logger.warning(f"⚠️ Ошибка загрузки скейлера: {e}")
            
            if progress:
                progress.update(1)
            
            # Названия признаков
            features_path = models_path / auxiliary_files['feature_names']
            if features_path.exists():
                try:
                    start_time = time.time()
                    self.feature_names = joblib.load(features_path)
                    self.load_times['feature_names'] = time.time() - start_time
                    self.logger.info(f"  ✅ feature_names ({len(self.feature_names)}): {self.load_times['feature_names']:.3f}s")
                except Exception as e:
                    self.logger.warning(f"⚠️ Ошибка загрузки названий признаков: {e}")
            
            if progress:
                progress.update(1)
            
            # Завершение
            total_time = time.time() - total_start_time
            self.models_loaded = True
            
            # Сохраняем информацию
            self.models_info = {
                'models_path': str(models_path),
                'base_models': list(self.base_models.keys()),
                'has_meta_model': self.meta_model is not None,
                'has_scaler': self.scaler is not None,
                'feature_count': len(self.feature_names),
                'total_load_time': total_time,
                'load_times': self.load_times.copy(),
                'loaded_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"🎉 Все модели загружены за {total_time:.2f}s!")
            self.logger.info(f"   Базовых моделей: {len(self.base_models)}")
            self.logger.info(f"   Метамодель: {'Да' if self.meta_model else 'Нет'}")
            self.logger.info(f"   Скейлер: {'Да' if self.scaler else 'Нет'}")
            self.logger.info(f"   Признаков: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки моделей: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
        finally:
            self.loading_in_progress = False
    
    def _prepare_features(self, features: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Подготовка признаков для предсказания."""
        try:
            # Преобразуем в numpy array
            if isinstance(features, pd.DataFrame):
                if self.feature_names:
                    # Используем только нужные колонки в правильном порядке
                    available_features = [col for col in self.feature_names if col in features.columns]
                    if len(available_features) != len(self.feature_names):
                        missing = set(self.feature_names) - set(available_features)
                        self.logger.warning(f"Отсутствующие признаки: {missing}")
                    
                    # Создаем массив с правильным порядком
                    features_array = []
                    for feature_name in self.feature_names:
                        if feature_name in features.columns:
                            features_array.append(features[feature_name].iloc[0])
                        else:
                            # Заполняем отсутствующие признаки разумными значениями
                            if 'MA' in feature_name or 'EMA' in feature_name:
                                features_array.append(features['close'].iloc[0])
                            elif 'RSI' in feature_name:
                                features_array.append(50.0)
                            elif 'BB_position' in feature_name:
                                features_array.append(0.5)
                            elif 'volume' in feature_name:
                                features_array.append(1.0)
                            else:
                                features_array.append(0.0)
                    
                    features_array = np.array(features_array).reshape(1, -1)
                else:
                    # Исключаем служебные колонки
                    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'confirm']
                    feature_cols = [col for col in features.columns if col not in exclude_cols]
                    features_array = features[feature_cols].values
            else:
                features_array = features
            
            # Проверяем размерность
            if len(features_array.shape) == 1:
                features_array = features_array.reshape(1, -1)
            
            # Применяем скейлер если есть
            if self.scaler is not None:
                features_array = self.scaler.transform(features_array)
            
            # Проверяем на NaN и Inf
            if np.isnan(features_array).any() or np.isinf(features_array).any():
                self.logger.warning("NaN/Inf values detected, replacing with 0")
                features_array = np.nan_to_num(features_array)
            
            return features_array
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    
    def predict_ensemble(self, features: Union[np.ndarray, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Получение финального предсказания ансамбля с автозагрузкой моделей.
        
        Returns:
            Dict с результатами: final_signal, final_probability, base_predictions
        """
        # Автозагрузка при первом обращении
        if not self.models_loaded:
            if not self.load_models():
                self.logger.error("❌ Не удалось загрузить модели!")
                return None
        
        if self.meta_model is None:
            self.logger.error("❌ Метамодель не загружена!")
            return None
        
        # Подготавливаем признаки
        prepared_features = self._prepare_features(features)
        if prepared_features is None:
            return None
        
        try:
            # Получаем предсказания от базовых моделей
            base_predictions = {}
            expected_models = ['random_forest', 'lightgbm', 'xgboost', 'catboost']
            
            for model_name in expected_models:
                if model_name in self.base_models:
                    model = self.base_models[model_name]
                    try:
                        # Получаем вероятность класса 1 (BUY)
                        pred_proba = model.predict_proba(prepared_features)
                        if len(pred_proba.shape) == 2 and pred_proba.shape[1] >= 2:
                            probability = pred_proba[0, 1]
                        else:
                            probability = pred_proba[0]
                        
                        base_predictions[model_name] = float(probability)
                        
                    except Exception as e:
                        self.logger.error(f"Error predicting with {model_name}: {e}")
                        base_predictions[model_name] = 0.5  # Нейтральное значение
                else:
                    base_predictions[model_name] = 0.5  # Нейтральное значение
            
            # Подготавливаем входные данные для метамодели
            meta_features = [base_predictions[model] for model in expected_models]
            meta_features_array = np.array([meta_features])
            
            # Предсказание метамодели
            final_probability = self.meta_model.predict_proba(meta_features_array)[0, 1]
            final_signal = self.meta_model.predict(meta_features_array)[0]
            
            return {
                'final_signal': int(final_signal),
                'final_probability': float(final_probability),
                'signal_strength': 'STRONG' if abs(final_probability - 0.5) > 0.3 else 'WEAK',
                'base_predictions': base_predictions,
                'meta_features': meta_features,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            return None
    
    def predict_proba(self, features: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Совместимость с sklearn - возвращает вероятности классов."""
        result = self.predict_ensemble(features)
        if result:
            prob_1 = result['final_probability']
            prob_0 = 1 - prob_1
            return np.array([[prob_0, prob_1]])
        return None
    
    def predict(self, features: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Совместимость с sklearn - возвращает предсказанные классы."""
        result = self.predict_ensemble(features)
        if result:
            return np.array([result['final_signal']])
        return None
    
    def is_ready(self) -> bool:
        """Проверка готовности предсказателя."""
        return (
            self.models_loaded and 
            len(self.base_models) > 0 and 
            self.meta_model is not None
        )
    
    def get_models_info(self) -> Dict[str, Any]:
        """Получение информации о загруженных моделях."""
        return self.models_info.copy()
    
    def get_prediction_summary(self, prediction_result: Dict[str, Any]) -> str:
        """Формирование краткой сводки по предсказанию."""
        if not prediction_result:
            return "No prediction available"
        
        signal = "BUY" if prediction_result['final_signal'] == 1 else "HOLD/SELL"
        prob = prediction_result['final_probability']
        strength = prediction_result['signal_strength']
        
        summary = f"Signal: {signal} | Probability: {prob:.3f} | Strength: {strength}"
        
        # Добавляем информацию о базовых моделях
        base_preds = prediction_result.get('base_predictions', {})
        if base_preds:
            base_summary = " | Base: " + ", ".join([f"{k}: {v:.3f}" for k, v in base_preds.items()])
            summary += base_summary
        
        return summary


# Пример использования
def example_usage():
    """Демонстрация использования оптимизированного ансамбля."""
    # Создаем предсказатель с отложенной загрузкой
    print("🚀 Создание оптимизированного предсказателя...")
    predictor = OptimizedEnsemblePredictor(lazy_loading=True)
    
    # Создаем тестовые данные
    test_features = pd.DataFrame({
        'MA5': [100.5], 'MA10': [100.3], 'MA20': [100.1],
        'RSI': [55.0], 'MACD': [0.1], 'MACD_signal': [0.05], 'MACD_diff': [0.05],
        'BB_hband': [102.0], 'BB_lband': [98.0], 'BB_width': [4.0], 'BB_position': [0.6],
        'vol_change': [0.02], 'volume_sma': [1000000], 'volume_ratio': [1.1],
        'price_change': [0.01], 'price_change_5': [0.05], 'volatility': [0.02],
        'EMA12': [100.4], 'EMA26': [100.2], 'high_low_pct': [0.02],
        'close_to_high': [0.8], 'close_to_low': [0.9], 'timeframe_minutes': [5.0]
    })
    
    print("📊 Первое предсказание (инициирует загрузку)...")
    start_time = time.time()
    prediction = predictor.predict_ensemble(test_features)
    first_pred_time = time.time() - start_time
    
    if prediction:
        print(f"⏱️ Время: {first_pred_time:.2f}s")
        print(f"📈 Результат: {predictor.get_prediction_summary(prediction)}")
        
        # Второе предсказание должно быть быстрым
        print("\n📊 Второе предсказание (модели уже загружены)...")
        start_time = time.time()
        prediction2 = predictor.predict_ensemble(test_features)
        second_pred_time = time.time() - start_time
        print(f"⏱️ Время: {second_pred_time:.4f}s")
        
        # Информация о моделях
        print(f"\n📋 Информация о моделях:")
        info = predictor.get_models_info()
        print(f"   Общее время загрузки: {info.get('total_load_time', 0):.2f}s")
        print(f"   Базовые модели: {info.get('base_models', [])}")
        print(f"   Время загрузки по моделям:")
        for model, load_time in info.get('load_times', {}).items():
            print(f"     {model}: {load_time:.3f}s")
        
    else:
        print("❌ Не удалось получить предсказание")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    example_usage()