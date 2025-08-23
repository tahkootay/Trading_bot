#!/usr/bin/env python3
"""
Модуль управления ансамблем ML-моделей для live торговли
Согласно спецификации: RF, LightGBM, XGBoost, CatBoost + Logistic Regression метамодель
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


# Подавление предупреждений
warnings.filterwarnings('ignore')


class EnsemblePredictor:
    """Предсказатель на основе ансамбля моделей."""
    
    def __init__(self, models_dir: str = "models/ensemble_live", use_latest: bool = True):
        """
        Инициализация ансамбля моделей.
        
        Args:
            models_dir: Директория с моделями
            use_latest: Использовать последнюю версию моделей
        """
        self.models_dir = Path(models_dir)
        self.use_latest = use_latest
        
        # Модели
        self.base_models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_names = []
        
        # Статус загрузки
        self.models_loaded = False
        self.models_info = {}
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        # Автозагрузка моделей
        self.load_models()
    
    def _find_models_directory(self) -> Optional[Path]:
        """Поиск директории с моделями."""
        if not self.models_dir.exists():
            self.logger.error(f"Models directory not found: {self.models_dir}")
            return None
        
        if self.use_latest:
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
                # Сортируем по имени (предполагается формат YYYYMMDD_HHMMSS)
                subdirs.sort(key=lambda x: x.name, reverse=True)
                return subdirs[0]
        
        return self.models_dir
    
    def load_models(self) -> bool:
        """Загрузка всех моделей ансамбля."""
        try:
            models_path = self._find_models_directory()
            if models_path is None:
                return False
            
            self.logger.info(f"Loading models from: {models_path}")
            
            # Определяем ожидаемые файлы моделей
            model_files = {
                'random_forest': 'random_forest_intraday.joblib',
                'lightgbm': 'lightgbm_intraday.joblib', 
                'gradient_boosting': 'gradient_boosting_intraday.joblib',  # fallback для LightGBM
                'xgboost': 'xgboost_intraday.joblib',
                'catboost': 'catboost_intraday.joblib'
            }
            
            # Загружаем базовые модели
            for model_name, filename in model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    try:
                        model = joblib.load(model_path)
                        self.base_models[model_name] = model
                        self.logger.info(f"✅ Loaded {model_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_name}: {e}")
            
            if not self.base_models:
                self.logger.error("No base models loaded!")
                return False
            
            # Загружаем метамодель
            meta_path = models_path / "meta_intraday.joblib"
            if meta_path.exists():
                try:
                    self.meta_model = joblib.load(meta_path)
                    self.logger.info("✅ Loaded meta model")
                except Exception as e:
                    self.logger.error(f"Failed to load meta model: {e}")
                    return False
            else:
                self.logger.error("Meta model not found!")
                return False
            
            # Загружаем скейлер
            scaler_path = models_path / "scaler.joblib"
            if scaler_path.exists():
                try:
                    self.scaler = joblib.load(scaler_path)
                    self.logger.info("✅ Loaded scaler")
                except Exception as e:
                    self.logger.warning(f"Failed to load scaler: {e}")
            
            # Загружаем названия признаков
            features_path = models_path / "feature_names.joblib"
            if features_path.exists():
                try:
                    self.feature_names = joblib.load(features_path)
                    self.logger.info(f"✅ Loaded {len(self.feature_names)} feature names")
                except Exception as e:
                    self.logger.warning(f"Failed to load feature names: {e}")
            
            self.models_loaded = True
            self.models_info = {
                'models_path': str(models_path),
                'base_models': list(self.base_models.keys()),
                'has_meta_model': self.meta_model is not None,
                'has_scaler': self.scaler is not None,
                'feature_count': len(self.feature_names),
                'loaded_at': datetime.now().isoformat()
            }
            
            self.logger.info("🎉 All models loaded successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def _prepare_features(self, features: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Подготовка признаков для предсказания."""
        try:
            # Преобразуем в numpy array
            if isinstance(features, pd.DataFrame):
                if self.feature_names:
                    # Используем только нужные колонки в правильном порядке
                    available_features = [col for col in self.feature_names if col in features.columns]
                    if len(available_features) != len(self.feature_names):
                        self.logger.warning(f"Missing features: {set(self.feature_names) - set(available_features)}")
                    features_array = features[available_features].values
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
                self.logger.error("NaN or Inf values in features")
                return None
            
            return features_array
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    
    def predict_base_models(self, features: Union[np.ndarray, pd.DataFrame]) -> Optional[Dict[str, float]]:
        """Получение предсказаний от всех базовых моделей."""
        if not self.models_loaded:
            self.logger.error("Models not loaded!")
            return None
        
        prepared_features = self._prepare_features(features)
        if prepared_features is None:
            return None
        
        try:
            base_predictions = {}
            
            for model_name, model in self.base_models.items():
                try:
                    # Получаем вероятность класса 1 (BUY)
                    pred_proba = model.predict_proba(prepared_features)
                    if len(pred_proba.shape) == 2 and pred_proba.shape[1] >= 2:
                        probability = pred_proba[0, 1]  # Вероятность класса 1
                    else:
                        probability = pred_proba[0]
                    
                    base_predictions[model_name] = float(probability)
                    
                except Exception as e:
                    self.logger.error(f"Error predicting with {model_name}: {e}")
                    continue
            
            return base_predictions if base_predictions else None
            
        except Exception as e:
            self.logger.error(f"Error in base models prediction: {e}")
            return None
    
    def predict_ensemble(self, features: Union[np.ndarray, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Получение финального предсказания ансамбля через метамодель.
        
        Returns:
            Dict с результатами: final_signal, final_probability, base_predictions
        """
        if not self.models_loaded or self.meta_model is None:
            self.logger.error("Models or meta-model not loaded!")
            return None
        
        # Получаем предсказания базовых моделей
        base_predictions = self.predict_base_models(features)
        if base_predictions is None:
            return None
        
        try:
            # Подготавливаем входные данные для метамодели
            # Важно: порядок должен соответствовать обучению
            expected_models = ['random_forest', 'lightgbm', 'xgboost', 'catboost']
            if 'lightgbm' not in base_predictions and 'gradient_boosting' in base_predictions:
                expected_models = ['random_forest', 'gradient_boosting', 'xgboost', 'catboost']
            
            meta_features = []
            for model_name in expected_models:
                if model_name in base_predictions:
                    meta_features.append(base_predictions[model_name])
                else:
                    self.logger.warning(f"Missing prediction from {model_name}")
                    meta_features.append(0.5)  # Нейтральное значение
            
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
    
    def get_models_info(self) -> Dict[str, Any]:
        """Получение информации о загруженных моделях."""
        return self.models_info.copy()
    
    def is_ready(self) -> bool:
        """Проверка готовности предсказателя."""
        return (
            self.models_loaded and 
            len(self.base_models) > 0 and 
            self.meta_model is not None
        )
    
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


class ModelPerformanceTracker:
    """Трекер производительности моделей в live режиме."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.predictions_history = []
        self.performance_stats = {}
    
    def add_prediction(self, prediction: Dict[str, Any], actual_result: Optional[bool] = None):
        """Добавление предсказания в историю."""
        record = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual_result': actual_result
        }
        
        self.predictions_history.append(record)
        
        # Ограничиваем размер истории
        if len(self.predictions_history) > self.max_history:
            self.predictions_history = self.predictions_history[-self.max_history:]
    
    def calculate_performance(self) -> Dict[str, Any]:
        """Расчет метрик производительности."""
        if not self.predictions_history:
            return {}
        
        # Фильтруем записи с известными результатами
        validated_predictions = [
            record for record in self.predictions_history 
            if record['actual_result'] is not None
        ]
        
        if not validated_predictions:
            return {'total_predictions': len(self.predictions_history)}
        
        # Считаем точность
        correct = sum(
            1 for record in validated_predictions
            if (record['prediction']['final_signal'] == 1) == record['actual_result']
        )
        
        accuracy = correct / len(validated_predictions)
        
        return {
            'total_predictions': len(self.predictions_history),
            'validated_predictions': len(validated_predictions),
            'accuracy': accuracy,
            'correct_predictions': correct
        }


# Пример использования
def example_usage():
    """Демонстрация использования ансамбля."""
    # Создаем предсказатель
    predictor = EnsemblePredictor()
    
    if not predictor.is_ready():
        print("❌ Predictor not ready - models not loaded")
        return
    
    print("✅ Predictor ready!")
    print(f"Models info: {predictor.get_models_info()}")
    
    # Создаем тестовые признаки
    feature_names = predictor.feature_names
    if feature_names:
        test_features = pd.DataFrame({
            name: [np.random.normal(0, 1)] for name in feature_names
        })
    else:
        # Если нет информации о признаках, создаем стандартный набор
        test_features = pd.DataFrame({
            'MA5': [100.5], 'MA10': [100.3], 'MA20': [100.1],
            'RSI': [55.0], 'MACD': [0.1], 'MACD_signal': [0.05],
            'BB_position': [0.6], 'vol_change': [0.02]
        })
    
    # Получаем предсказание
    prediction = predictor.predict_ensemble(test_features)
    if prediction:
        print(f"\nPrediction: {predictor.get_prediction_summary(prediction)}")
        print(f"Full result: {prediction}")
    else:
        print("❌ Failed to get prediction")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    example_usage()